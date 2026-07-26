"""
40/60 split test — 40% train, 60% test (an even larger test window than 50/50).

Continues the supervisor's hypothesis test: does LSTM overtake ARIMA as the test
window / horizon grows? Train on first 40% (~2015-2019), forecast last 60%
(~2019-2026, ~1,599 days). ARIMA/LSTM/Hybrid/Naive, one-step + multi-step.

Because the LSTM result is seed-sensitive (shown at 50/50), the LSTM is additionally
run across 5 seeds and reported as mean +/- std, so the ARIMA-vs-LSTM verdict is robust.

Does NOT touch the 80/20 or 50/50 outputs.

Artifacts:
  results/predictions/split4060_predictions.csv
  results/tables/split4060_comparison.csv
  results/figures/split4060_overlay.png
  results/predictions/split_info_4060.json
"""
import json
import random
import warnings
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
DATA = _ROOT / "data" / "processed"
PRED = _ROOT / "results" / "predictions"
FIG = _ROOT / "results" / "figures"
TAB = _ROOT / "results" / "tables"
for _d in (PRED, FIG, TAB):
    _d.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

warnings.filterwarnings("ignore")
SEED = 42
SEEDS = [42, 43, 44, 45, 46]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
UNITS, DROPOUT, MAX_EPOCHS, PATIENCE, VAL_FRAC, BATCH = 50, 0.2, 150, 12, 0.10, 32
TRAIN_FRAC = 0.40
device = torch.device("cpu")


def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return (float(np.mean(np.abs(a - p))), float(np.mean(np.abs((a - p) / a)) * 100),
            float(np.sqrt(np.mean((a - p) ** 2))))


def make_windows(arr, w):
    X, y = [], []
    for i in range(w, len(arr)):
        X.append(arr[i - w:i]); y.append(arr[i])
    return np.array(X, np.float32), np.array(y, np.float32)


class LSTMNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.lstm = nn.LSTM(1, UNITS, num_layers=2, batch_first=True, dropout=DROPOUT)
        s.head = nn.Linear(UNITS, 1)
    def forward(s, x):
        o, _ = s.lstm(x); return s.head(o[:, -1, :])


def train_lstm(scaled_train, w, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y = make_windows(scaled_train, w)
    nv = max(1, int(len(X) * VAL_FRAC))
    Xtr, ytr, Xv, yv = X[:-nv], y[:-nv], X[-nv:], y[-nv:]
    tt = lambda a: torch.tensor(a).unsqueeze(-1)
    Xtr, ytr = tt(Xtr), torch.tensor(ytr).unsqueeze(-1)
    Xv, yv = tt(Xv), torch.tensor(yv).unsqueeze(-1)
    m = LSTMNet(); opt = torch.optim.Adam(m.parameters(), 1e-3); lf = nn.MSELoss()
    best, bs, wait = 1e9, None, 0
    for _ in range(MAX_EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), BATCH):
            idx = perm[s:s + BATCH]; opt.zero_grad()
            lf(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lf(m(Xv), yv).item()
        if vl < best - 1e-9:
            best, bs, wait = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.load_state_dict(bs); return m, np.sqrt(best)


def lstm_onestep(m, scaled_full, w, split, n, scaler):
    m.eval()
    Xt = np.array([scaled_full[i - w:i] for i in range(split, n)], np.float32)
    with torch.no_grad():
        ps = m(torch.tensor(Xt).unsqueeze(-1)).cpu().numpy().ravel()
    return scaler.inverse_transform(ps.reshape(-1, 1)).ravel()


def lstm_recursive(m, seed_win, steps, w, scaler):
    m.eval(); win = list(seed_win); out = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(np.array(win[-w:], np.float32)).reshape(1, -1, 1)
            p = m(x).item(); out.append(p); win.append(p)
    return scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel()


# =============================================================== data + 40/60 split
df = pd.read_csv(DATA / "spsl20_trading_days_clean.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
n = len(df)
split = int(n * TRAIN_FRAC)
p_tr = df["spsl20_points"].astype(float).values[:split]
p_te = df["spsl20_points"].astype(float).values[split:]
dates_te = df["date"].iloc[split:]
full = np.concatenate([p_tr, p_te])

json.dump({"n": int(n), "split_index": int(split), "ratio": "40/60",
           "train_start": str(df.date.iloc[0].date()), "train_end": str(df.date.iloc[split-1].date()),
           "test_start": str(df.date.iloc[split].date()), "test_end": str(df.date.iloc[-1].date())},
          open(PRED / "split_info_4060.json", "w"), indent=2)

print("=" * 66)
print("40/60 SPLIT TEST  —  even larger test window; does LSTM beat ARIMA?")
print("=" * 66)
print(f"Train: {split} days ({df.date.iloc[0].date()} -> {df.date.iloc[split-1].date()})")
print(f"Test : {n-split} days ({df.date.iloc[split].date()} -> {df.date.iloc[-1].date()})")
print(f"Test index: {p_te[0]:.0f} -> {p_te[-1]:.0f} ({(p_te[-1]-p_te[0])/p_te[0]*100:+.1f}%)\n")

results = {}
results["Naive"] = {"1s": full[split - 1:n - 1], "ms": np.full_like(p_te, p_tr[-1])}

# ARIMA (deterministic)
tr_ser = pd.Series(p_tr, index=pd.RangeIndex(split))
auto1 = pm.auto_arima(tr_ser, start_p=0, start_q=0, max_p=5, max_q=5, d=1, seasonal=False,
                      stepwise=True, suppress_warnings=True, error_action="ignore")
res1 = ARIMA(tr_ser, order=auto1.order).fit()
res1e = res1.append(pd.Series(p_te, index=pd.RangeIndex(split, n)), refit=False)
arima_1s = res1e.predict(start=split, end=n - 1).values
logtr = pd.Series(np.log(p_tr), index=pd.RangeIndex(split))
auto2 = pm.auto_arima(logtr, start_p=0, start_q=0, max_p=5, max_q=5, d=1, seasonal=False,
                      stepwise=True, suppress_warnings=True, error_action="ignore")
res2 = ARIMA(logtr, order=auto2.order).fit()
arima_ms = np.exp(res2.get_forecast(steps=len(p_te)).predicted_mean.values)
results["ARIMA"] = {"1s": arima_1s, "ms": arima_ms}

# LSTM (main seed for table/plot)
scaler = MinMaxScaler((0, 1))
sc_tr = scaler.fit_transform(p_tr.reshape(-1, 1)).ravel()
sc_full = scaler.transform(full.reshape(-1, 1)).ravel()
cand = {w: train_lstm(sc_tr, w) for w in (60, 30)}
bw = min(cand, key=lambda k: cand[k][1]); lstm_m = cand[bw][0]
results["LSTM"] = {"1s": lstm_onestep(lstm_m, sc_full, bw, split, n, scaler),
                   "ms": lstm_recursive(lstm_m, sc_tr[-bw:], len(p_te), bw, scaler)}

# HYBRID
res_tr_1s = (p_tr - res1.fittedvalues.values)[1:]
full_res_1s = np.concatenate([res_tr_1s, p_te - arima_1s])
rs1 = MinMaxScaler((0, 1)); scr1_tr = rs1.fit_transform(res_tr_1s.reshape(-1, 1)).ravel()
scr1_full = rs1.transform(full_res_1s.reshape(-1, 1)).ravel()
mr1, _ = train_lstm(scr1_tr, 30)
hybrid_1s = arima_1s + lstm_onestep(mr1, scr1_full, 30, len(res_tr_1s), len(full_res_1s), rs1)
res_tr_ms = (p_tr - np.exp(res2.fittedvalues.values))[1:]
rs2 = MinMaxScaler((0, 1)); scr2_tr = rs2.fit_transform(res_tr_ms.reshape(-1, 1)).ravel()
mr2, _ = train_lstm(scr2_tr, 30)
hybrid_ms = arima_ms + lstm_recursive(mr2, scr2_tr[-30:], len(p_te), 30, rs2)
results["Hybrid"] = {"1s": hybrid_1s, "ms": hybrid_ms}

# score table (main seed)
rows = []
for name in ["Naive", "ARIMA", "LSTM", "Hybrid"]:
    m1 = metrics(p_te, results[name]["1s"]); m2 = metrics(p_te, results[name]["ms"])
    rows.append({"Model": name, "OneStep_MAE": m1[0], "OneStep_MAPE": m1[1], "OneStep_RMSE": m1[2],
                 "MultiStep_MAE": m2[0], "MultiStep_MAPE": m2[1], "MultiStep_RMSE": m2[2]})
table = pd.DataFrame(rows).set_index("Model")
table.round(4).to_csv(TAB / "split4060_comparison.csv")
print(f"ARIMA one-step {auto1.order} | multi-step(log) {auto2.order} | LSTM window {bw}\n")
disp = table.copy(); disp.columns = ["1s_MAE", "1s_MAPE", "1s_RMSE", "ms_MAE", "ms_MAPE", "ms_RMSE"]
print(disp.round(2).to_string())

# LSTM seed robustness (both protocols)
print("\n--- LSTM across seeds (robust ARIMA-vs-LSTM verdict) ---")
one_s, multi_s = [], []
for sd in SEEDS:
    m, _ = train_lstm(sc_tr, bw, seed=sd)
    one_s.append(metrics(p_te, lstm_onestep(m, sc_full, bw, split, n, scaler))[1])
    multi_s.append(metrics(p_te, lstm_recursive(m, sc_tr[-bw:], len(p_te), bw, scaler))[1])
a1 = metrics(p_te, arima_1s)[1]; a2 = metrics(p_te, arima_ms)[1]
print(f"  One-step  : ARIMA {a1:.2f}%  |  LSTM {np.mean(one_s):.2f}% +/- {np.std(one_s):.2f}%  "
      f"(range {min(one_s):.2f}-{max(one_s):.2f})")
print(f"  Multi-step: ARIMA {a2:.2f}%  |  LSTM {np.mean(multi_s):.2f}% +/- {np.std(multi_s):.2f}%  "
      f"(range {min(multi_s):.2f}-{max(multi_s):.2f})")
for label, a, s in [("One-step", a1, one_s), ("Multi-step", a2, multi_s)]:
    if a < min(s): v = "ARIMA wins (beats every LSTM seed)"
    elif a > max(s): v = "LSTM wins (beats ARIMA every seed)"
    else: v = "TIE (ARIMA within LSTM's seed range -> no robust winner)"
    print(f"    {label}: {v}")

# save preds + plot (main seed)
pred_df = pd.DataFrame({"date": dates_te.values, "actual": p_te})
for name in ["Naive", "ARIMA", "LSTM", "Hybrid"]:
    pred_df[f"{name.lower()}_1s"] = results[name]["1s"]
    pred_df[f"{name.lower()}_ms"] = results[name]["ms"]
pred_df.to_csv(PRED / "split4060_predictions.csv", index=False)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
colors = {"Naive": "gray", "ARIMA": "tab:blue", "LSTM": "tab:green", "Hybrid": "tab:red"}
for ax, proto, title in [(ax1, "1s", "One-step-ahead"), (ax2, "ms", "Multi-step")]:
    ax.plot(dates_te, p_te, color="black", linewidth=2, label="Actual")
    for name in ["Naive", "ARIMA", "LSTM", "Hybrid"]:
        ls = "--" if name == "Naive" else "-"
        ax.plot(dates_te, results[name][proto], color=colors[name], linewidth=1.1, alpha=0.85, linestyle=ls, label=name)
    ax.set_title(f"{title}  —  40/60 split, test {df.date.iloc[split].date()} -> {df.date.iloc[-1].date()}")
    ax.set_ylabel("S&P SL 20"); ax.grid(True, alpha=0.3); ax.legend(ncol=5, fontsize=8)
ax2.set_xlabel("Date")
fig.tight_layout(); fig.savefig(FIG / "split4060_overlay.png", dpi=150); plt.close(fig)
print("\nSaved: split4060_predictions.csv, split4060_comparison.csv, split4060_overlay.png, split_info_4060.json")
