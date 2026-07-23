"""
Paper-window replication check — ARIMA / LSTM / Hybrid / Naive, MULTI-STEP, on
train 2015-2016 -> test 2017-2018 (approximating paper 26's setup).

Purpose: show that our models reproduce the source paper's single-digit MAPE when
evaluated on its calm test window, proving that our much larger 2024-2026 multi-step
errors come from the harder (strongly trending) test period, not from the models.

NOTE: our dataset starts in 2015, so training is 2015-2016 (~475 days), not the
paper's 2010-2016 (~7 years). This approximates paper 26's setup rather than
replicating it exactly.

Mirrors the project's multi-step recipe: log-transform ARIMA one-shot forecast;
recursive LSTM on raw price; Hybrid = ARIMA multi-step + recursive residual-LSTM.

Artifact: results/tables/papercheck_2017_2018.csv
"""
import json
import random
import warnings
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
DATA = _ROOT / "data" / "processed"
TAB = _ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
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


def recursive(m, seed_win, steps, w, scaler):
    m.eval(); win = list(seed_win); out = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(np.array(win[-w:], np.float32)).reshape(1, -1, 1)
            p = m(x).item(); out.append(p); win.append(p)
    return scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel()


# ---- data: 2015-2016 train, 2017-2018 test ----
df = pd.read_csv(DATA / "spsl20_trading_days_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df[(df.date >= "2015-01-01") & (df.date <= "2018-12-31")].reset_index(drop=True)
tr = df[df.date <= "2016-12-31"]; te = df[df.date >= "2017-01-01"]
p_tr = tr["spsl20_points"].astype(float).values
p_te = te["spsl20_points"].astype(float).values

print("=" * 60)
print("PAPER-WINDOW CHECK  |  train 2015-2016 -> test 2017-2018")
print("=" * 60)
print(f"Train: {len(p_tr)} days   Test: {len(p_te)} days")
print(f"Index over test: {p_te[0]:.0f} -> {p_te[-1]:.0f} "
      f"({(p_te[-1] - p_te[0]) / p_te[0] * 100:+.1f}%)\n")

# ARIMA multi-step (log)
logtr = pd.Series(np.log(p_tr), index=pd.RangeIndex(len(p_tr)))
auto = pm.auto_arima(logtr, start_p=0, start_q=0, max_p=5, max_q=5, d=1, seasonal=False,
                     stepwise=True, suppress_warnings=True, error_action="ignore")
res = ARIMA(logtr, order=auto.order).fit()
arima_ms = np.exp(res.get_forecast(steps=len(p_te)).predicted_mean.values)

# LSTM recursive multi-step, window selected by validation RMSE
sc = MinMaxScaler((0, 1)); sc_tr = sc.fit_transform(p_tr.reshape(-1, 1)).ravel()
cand = {w: train_lstm(sc_tr, w) for w in (60, 30)}
bw = min(cand, key=lambda k: cand[k][1]); lstm_model = cand[bw][0]
lstm_ms = recursive(lstm_model, sc_tr[-bw:], len(p_te), bw, sc)

# Hybrid: ARIMA(log) multi-step + recursive residual-LSTM
resid = (p_tr - np.exp(res.fittedvalues.values))[1:]      # drop day-0 Kalman artifact
rsc = MinMaxScaler((0, 1)); sc_res = rsc.fit_transform(resid.reshape(-1, 1)).ravel()
mr, _ = train_lstm(sc_res, 30)
hybrid_ms = arima_ms + recursive(mr, sc_res[-30:], len(p_te), 30, rsc)

# Naive flat
naive_ms = np.full_like(p_te, p_tr[-1])

rows = []
for name, pred in [("Naive", naive_ms), ("ARIMA", arima_ms),
                   ("LSTM", lstm_ms), ("Hybrid", hybrid_ms)]:
    mae, mape, rmse = metrics(p_te, pred)
    rows.append({"Model": name, "MAE": mae, "MAPE": mape, "RMSE": rmse})
tbl = pd.DataFrame(rows).set_index("Model")

print(f"ARIMA order {auto.order} | LSTM window {bw} | residual window 30\n")
print(tbl.round(2).to_string())
print("\nPaper 26 (2017-2018): ARIMA 233.96/7.04%/269.57 | LSTM 249.37/6.96%/269.86")

tbl.round(4).to_csv(TAB / "papercheck_2017_2018.csv")
print("\nSaved: results/tables/papercheck_2017_2018.csv")
