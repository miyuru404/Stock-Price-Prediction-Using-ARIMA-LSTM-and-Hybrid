"""
Last-month hold-out test: train ARIMA & LSTM on all data except the final ~1 month,
forecast that month, and plot forecast vs actual. Short horizon (~22 trading days) is
where forecasting has a realistic chance. Both one-step and multi-step protocols + Naive.

Artifacts:
  results/figures/lastmonth_overlay.png
  results/tables/lastmonth_comparison.csv
"""
import warnings, random
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

_ROOT = _Path(__file__).resolve().parents[1]
DATA = _ROOT / "data" / "processed"
FIG = _ROOT / "results" / "figures"
TAB = _ROOT / "results" / "tables"
for _d in (FIG, TAB):
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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
TEST_DAYS = 22                       # ~1 trading month
WIN, UNITS, DROPOUT, EPOCHS, PAT, VF, BATCH = 60, 50, 0.2, 150, 12, 0.10, 32


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
        super().__init__(); s.lstm = nn.LSTM(1, UNITS, 2, batch_first=True, dropout=DROPOUT); s.head = nn.Linear(UNITS, 1)
    def forward(s, x):
        o, _ = s.lstm(x); return s.head(o[:, -1, :])


def train_lstm(sc_tr):
    torch.manual_seed(SEED); np.random.seed(SEED)
    X, y = make_windows(sc_tr, WIN); nv = int(len(X) * VF)
    Xtr, ytr, Xv, yv = X[:-nv], y[:-nv], X[-nv:], y[-nv:]
    tt = lambda a: torch.tensor(a).unsqueeze(-1)
    Xtr, ytr, Xv, yv = tt(Xtr), torch.tensor(ytr).unsqueeze(-1), tt(Xv), torch.tensor(yv).unsqueeze(-1)
    m = LSTMNet(); opt = torch.optim.Adam(m.parameters(), 1e-3); lf = nn.MSELoss()
    best, bs, wait = 1e9, None, 0
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), BATCH):
            idx = perm[s:s+BATCH]; opt.zero_grad(); lf(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = lf(m(Xv), yv).item()
        if vl < best - 1e-9: best, bs, wait = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PAT: break
    m.load_state_dict(bs); return m


# ---- data ----
df = pd.read_csv(DATA / "spsl20_trading_days_clean.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
prices = df["spsl20_points"].astype(float).values
dates = df["date"]
n = len(prices); split = n - TEST_DAYS
p_tr, p_te = prices[:split], prices[split:]
d_te = dates.iloc[split:]
full = prices

print(f"Train: {split} days (-> {dates.iloc[split-1].date()}) | Test: {TEST_DAYS} days "
      f"({d_te.iloc[0].date()} -> {d_te.iloc[-1].date()})")
print(f"Test index moved {p_te[0]:.0f} -> {p_te[-1]:.0f} ({(p_te[-1]-p_te[0])/p_te[0]*100:+.1f}%)")

res = {}
# NAIVE
res["Naive"] = {"1s": full[split-1:n-1], "ms": np.full(TEST_DAYS, p_tr[-1])}
# ARIMA
tr = pd.Series(p_tr, index=pd.RangeIndex(split))
o1 = pm.auto_arima(tr, d=1, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore").order
r1 = ARIMA(tr, order=o1).fit()
arima_1s = r1.append(pd.Series(p_te, index=pd.RangeIndex(split, n)), refit=False).predict(start=split, end=n-1).values
lg = pd.Series(np.log(p_tr), index=pd.RangeIndex(split))
o2 = pm.auto_arima(lg, d=1, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore").order
arima_ms = np.exp(ARIMA(lg, order=o2).fit().get_forecast(steps=TEST_DAYS).predicted_mean.values)
res["ARIMA"] = {"1s": arima_1s, "ms": arima_ms}
# LSTM
sca = MinMaxScaler((0, 1)); sct = sca.fit_transform(p_tr.reshape(-1, 1)).ravel()
scf = sca.transform(full.reshape(-1, 1)).ravel()
m = train_lstm(sct)
m.eval()
with torch.no_grad():
    Xt = np.array([scf[i-WIN:i] for i in range(split, n)], np.float32)
    lstm_1s = sca.inverse_transform(m(torch.tensor(Xt).unsqueeze(-1)).numpy()).ravel()
    win = list(sct[-WIN:]); out = []
    for _ in range(TEST_DAYS):
        x = torch.tensor(np.array(win[-WIN:], np.float32)).reshape(1, -1, 1); pv = m(x).item(); out.append(pv); win.append(pv)
    lstm_ms = sca.inverse_transform(np.array(out).reshape(-1, 1)).ravel()
res["LSTM"] = {"1s": lstm_1s, "ms": lstm_ms}

rows = []
for name in ["Naive", "ARIMA", "LSTM"]:
    for proto, lab in [("1s", "one-step"), ("ms", "multi-step")]:
        mae, mape, rmse = metrics(p_te, res[name][proto])
        rows.append({"model": name, "protocol": lab, "MAE": mae, "MAPE": mape, "RMSE": rmse})
tbl = pd.DataFrame(rows)
tbl.to_csv(TAB / "lastmonth_comparison.csv", index=False)
print("\n" + tbl.round(3).to_string(index=False))

# plot
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
for ax, proto, title in [(a1, "1s", "One-step-ahead"), (a2, "ms", "Multi-step (forecast the whole month)")]:
    ax.plot(d_te, p_te, "o-", color="black", lw=2, ms=4, label="Actual")
    ax.plot(d_te, res["ARIMA"][proto], "s--", color="tab:blue", ms=3, label="ARIMA")
    ax.plot(d_te, res["LSTM"][proto], "^--", color="tab:green", ms=3, label="LSTM")
    ax.plot(d_te, res["Naive"][proto], ":", color="gray", label="Naive")
    ax.set_title(f"{title}\ntest {d_te.iloc[0].date()} -> {d_te.iloc[-1].date()}")
    ax.set_ylabel("S&P SL 20"); ax.grid(True, alpha=0.3); ax.legend(); ax.tick_params(axis="x", rotation=45)
fig.suptitle("Last-month hold-out: ARIMA & LSTM forecast vs actual", fontsize=13)
fig.tight_layout(); fig.savefig(FIG / "lastmonth_overlay.png", dpi=150); plt.close(fig)
print("\nSaved: lastmonth_overlay.png, lastmonth_comparison.csv")
