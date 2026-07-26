"""
New test series — 50/50 chronological split (vs. the 80/20 used in Experiments 1-3).

Motivation (supervisor's hypothesis): ARIMA beat LSTM on the 80/20 test window; on a
LARGER test window / longer prediction horizon, LSTM might overtake ARIMA. This script
tests that by halving the data: train on the first 50% (~2015-mid2020), forecast the
last 50% (~mid2020-2026, ~1,332 days) — a test window 2.5x larger than the 80/20 one.

Models: ARIMA, LSTM, Hybrid, Naive — both one-step and multi-step. Same methodology as
Experiments 1-3. Does NOT touch the 80/20 split_info.json or any existing output.

Artifacts:
  results/predictions/split5050_predictions.csv
  results/tables/split5050_comparison.csv
  results/figures/split5050_overlay.png
  results/predictions/split_info_5050.json
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
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
UNITS, DROPOUT, MAX_EPOCHS, PATIENCE, VAL_FRAC, BATCH = 50, 0.2, 150, 12, 0.10, 32
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


# =============================================================== data + 50/50 split
df = pd.read_csv(DATA / "spsl20_trading_days_clean.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
n = len(df)
split = n // 2                                   # 50/50 chronological
p_tr = df["spsl20_points"].astype(float).values[:split]
p_te = df["spsl20_points"].astype(float).values[split:]
dates_te = df["date"].iloc[split:]
full = np.concatenate([p_tr, p_te])

json.dump({"n": int(n), "split_index": int(split), "ratio": "50/50",
           "train_start": str(df.date.iloc[0].date()), "train_end": str(df.date.iloc[split-1].date()),
           "test_start": str(df.date.iloc[split].date()), "test_end": str(df.date.iloc[-1].date())},
          open(PRED / "split_info_5050.json", "w"), indent=2)

print("=" * 66)
print("50/50 SPLIT TEST  —  does a larger test window let LSTM beat ARIMA?")
print("=" * 66)
print(f"Train: {split} days ({df.date.iloc[0].date()} -> {df.date.iloc[split-1].date()})")
print(f"Test : {n-split} days ({df.date.iloc[split].date()} -> {df.date.iloc[-1].date()})")
print(f"Test index: {p_te[0]:.0f} -> {p_te[-1]:.0f} ({(p_te[-1]-p_te[0])/p_te[0]*100:+.1f}%)\n")

results = {}

# NAIVE
results["Naive"] = {"1s": full[split - 1:n - 1], "ms": np.full_like(p_te, p_tr[-1])}

# ARIMA
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

# LSTM
scaler = MinMaxScaler((0, 1))
sc_tr = scaler.fit_transform(p_tr.reshape(-1, 1)).ravel()
sc_full = scaler.transform(full.reshape(-1, 1)).ravel()
cand = {w: train_lstm(sc_tr, w) for w in (60, 30)}
bw = min(cand, key=lambda k: cand[k][1]); lstm_m = cand[bw][0]
results["LSTM"] = {"1s": lstm_onestep(lstm_m, sc_full, bw, split, n, scaler),
                   "ms": lstm_recursive(lstm_m, sc_tr[-bw:], len(p_te), bw, scaler)}

# HYBRID
res_tr_1s = (p_tr - res1.fittedvalues.values)[1:]
res_te_1s = p_te - arima_1s
full_res_1s = np.concatenate([res_tr_1s, res_te_1s])
rs1 = MinMaxScaler((0, 1)); scr1_tr = rs1.fit_transform(res_tr_1s.reshape(-1, 1)).ravel()
scr1_full = rs1.transform(full_res_1s.reshape(-1, 1)).ravel()
mr1, _ = train_lstm(scr1_tr, 30)
hybrid_1s = arima_1s + lstm_onestep(mr1, scr1_full, 30, len(res_tr_1s), len(full_res_1s), rs1)
res_tr_ms = (p_tr - np.exp(res2.fittedvalues.values))[1:]
rs2 = MinMaxScaler((0, 1)); scr2_tr = rs2.fit_transform(res_tr_ms.reshape(-1, 1)).ravel()
mr2, _ = train_lstm(scr2_tr, 30)
hybrid_ms = arima_ms + lstm_recursive(mr2, scr2_tr[-30:], len(p_te), 30, rs2)
results["Hybrid"] = {"1s": hybrid_1s, "ms": hybrid_ms}

# score
rows = []
for name in ["Naive", "ARIMA", "LSTM", "Hybrid"]:
    m1 = metrics(p_te, results[name]["1s"]); m2 = metrics(p_te, results[name]["ms"])
    rows.append({"Model": name, "OneStep_MAE": m1[0], "OneStep_MAPE": m1[1], "OneStep_RMSE": m1[2],
                 "MultiStep_MAE": m2[0], "MultiStep_MAPE": m2[1], "MultiStep_RMSE": m2[2]})
table = pd.DataFrame(rows).set_index("Model")
table.round(4).to_csv(TAB / "split5050_comparison.csv")

print(f"ARIMA one-step {auto1.order} | multi-step(log) {auto2.order} | LSTM window {bw}\n")
disp = table.copy(); disp.columns = ["1s_MAE", "1s_MAPE", "1s_RMSE", "ms_MAE", "ms_MAPE", "ms_RMSE"]
print(disp.round(2).to_string())

# direct ARIMA vs LSTM verdict
print("\n--- Supervisor's hypothesis: does LSTM beat ARIMA on the larger window? ---")
for proto, label in [("1s", "One-step"), ("ms", "Multi-step")]:
    a = metrics(p_te, results["ARIMA"][proto])[1]; l = metrics(p_te, results["LSTM"][proto])[1]
    winner = "LSTM" if l < a else "ARIMA"
    print(f"  {label:<10}: ARIMA {a:.2f}%  vs  LSTM {l:.2f}%  ->  {winner} wins")

# save preds + plot
pred_df = pd.DataFrame({"date": dates_te.values, "actual": p_te})
for name in ["Naive", "ARIMA", "LSTM", "Hybrid"]:
    pred_df[f"{name.lower()}_1s"] = results[name]["1s"]
    pred_df[f"{name.lower()}_ms"] = results[name]["ms"]
pred_df.to_csv(PRED / "split5050_predictions.csv", index=False)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
colors = {"Naive": "gray", "ARIMA": "tab:blue", "LSTM": "tab:green", "Hybrid": "tab:red"}
for ax, proto, title in [(ax1, "1s", "One-step-ahead"), (ax2, "ms", "Multi-step")]:
    ax.plot(dates_te, p_te, color="black", linewidth=2, label="Actual")
    for name in ["Naive", "ARIMA", "LSTM", "Hybrid"]:
        ls = "--" if name == "Naive" else "-"
        ax.plot(dates_te, results[name][proto], color=colors[name], linewidth=1.1, alpha=0.85, linestyle=ls, label=name)
    ax.set_title(f"{title}  —  50/50 split, test {df.date.iloc[split].date()} -> {df.date.iloc[-1].date()}")
    ax.set_ylabel("S&P SL 20"); ax.grid(True, alpha=0.3); ax.legend(ncol=5, fontsize=8)
ax2.set_xlabel("Date")
fig.tight_layout(); fig.savefig(FIG / "split5050_overlay.png", dpi=150); plt.close(fig)

print("\nSaved: split5050_predictions.csv, split5050_comparison.csv, split5050_overlay.png, split_info_5050.json")
