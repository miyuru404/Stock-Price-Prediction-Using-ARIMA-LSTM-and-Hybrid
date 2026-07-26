"""
Classical models at the SAME protocols as the transformer benchmark, for a fair fight.

Protocol B (fixed 60-day horizon, rolling): ARIMA (log, dynamic) and LSTM (recursive)
forecast 60 days ahead from real history at each cutoff across the test window.
Protocol A (whole test window): pulled from the already-computed multi-step results.

Outputs results/tables/classical_benchmark_AB.csv in the same schema as
transformer_benchmark_AB.csv so the two can be merged.
"""
import json, warnings, random
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

_ROOT = _Path(__file__).resolve().parents[1]
DATA = _ROOT / "data" / "processed"
PRED = _ROOT / "results" / "predictions"
TAB = _ROOT / "results" / "tables"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
H = 60
UNITS, DROPOUT, MAX_EPOCHS, PATIENCE, VAL_FRAC, BATCH, WIN = 50, 0.2, 150, 12, 0.10, 32, 60
SPLITS = {"80/20": 0.8, "50/50": 0.5, "40/60": 0.4}


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


def train_lstm(scaled_train, w):
    torch.manual_seed(SEED); np.random.seed(SEED)
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
            idx = perm[s:s+BATCH]; opt.zero_grad(); lf(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = lf(m(Xv), yv).item()
        if vl < best - 1e-9: best, bs, wait = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    m.load_state_dict(bs); return m


def lstm_roll(m, seed_win, steps, scaler):
    m.eval(); win = list(seed_win); out = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(np.array(win[-WIN:], np.float32)).reshape(1, -1, 1)
            p = m(x).item(); out.append(p); win.append(p)
    return scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel()


# ---- whole-window (A) classical numbers from existing result tables ----
A_src = {
    "80/20": TAB / "comparison_multistep.csv",   # index Model, col ms_MAPE etc (MultiStep)
    "50/50": TAB / "split5050_comparison.csv",
    "40/60": TAB / "split4060_comparison.csv",
}
rows = []
for sname, path in A_src.items():
    t = pd.read_csv(path, index_col=0)
    # unify column names for multi-step MAE/MAPE/RMSE
    cols = {c.lower(): c for c in t.columns}
    def col(key):
        for c in t.columns:
            if key in c.lower().replace("multistep", "ms").replace("multi", "ms"):
                return c
        return None
    mae_c = [c for c in t.columns if "ms" in c.lower().replace("multistep","ms").replace("multi","ms") and "mae" in c.lower()][0]
    mape_c = [c for c in t.columns if "ms" in c.lower().replace("multistep","ms").replace("multi","ms") and "mape" in c.lower()][0]
    rmse_c = [c for c in t.columns if "ms" in c.lower().replace("multistep","ms").replace("multi","ms") and "rmse" in c.lower()][0]
    for model in ["Naive", "ARIMA", "LSTM", "Hybrid"]:
        if model in t.index:
            rows.append({"split": sname, "model": model, "protocol": "A_wholewindow",
                         "MAE": float(t.loc[model, mae_c]), "MAPE": float(t.loc[model, mape_c]),
                         "RMSE": float(t.loc[model, rmse_c])})

# ---- Protocol B (60-day rolling) for ARIMA + LSTM ----
df = pd.read_csv(DATA / "spsl20_trading_days_clean.csv").sort_values("date").reset_index(drop=True)
prices = df["spsl20_points"].astype(float).values
n = len(prices)

for sname, frac in SPLITS.items():
    split = int(n * frac)
    cutoffs = list(range(split, n, H))
    print(f"[{sname}] classical Protocol B ...", flush=True)

    # ARIMA (log, dynamic 60-ahead, fixed params, real history)
    logtr = pd.Series(np.log(prices[:split]), index=pd.RangeIndex(split))
    order = pm.auto_arima(logtr, start_p=0, start_q=0, max_p=5, max_q=5, d=1, seasonal=False,
                          stepwise=True, suppress_warnings=True, error_action="ignore").order
    res = ARIMA(logtr, order=order).fit()
    res_c = res
    pB, aB = [], []
    for c in cutoffs:
        end = min(c + H, n); hh = end - c
        pB.extend(np.exp(res_c.get_forecast(steps=hh).predicted_mean.values)); aB.extend(prices[c:end])
        res_c = res_c.append(pd.Series(np.log(prices[c:end]), index=pd.RangeIndex(c, end)), refit=False)
    mB = metrics(aB, pB)
    rows.append({"split": sname, "model": "ARIMA", "protocol": "B_60day",
                 "MAE": mB[0], "MAPE": mB[1], "RMSE": mB[2]})

    # LSTM (recursive 60-ahead from real context, fixed model)
    scaler = MinMaxScaler((0, 1)); sc = scaler.fit_transform(prices[:split].reshape(-1, 1)).ravel()
    sc_full = scaler.transform(prices.reshape(-1, 1)).ravel()
    m = train_lstm(sc, WIN)
    pB, aB = [], []
    for c in cutoffs:
        end = min(c + H, n); hh = end - c
        pB.extend(lstm_roll(m, sc_full[c - WIN:c], hh, scaler)); aB.extend(prices[c:end])
    mB = metrics(aB, pB)
    rows.append({"split": sname, "model": "LSTM", "protocol": "B_60day",
                 "MAE": mB[0], "MAPE": mB[1], "RMSE": mB[2]})

    # Naive B (persistence per window)
    nb, ab = [], []
    for c in cutoffs:
        end = min(c + H, n); nb.extend([prices[c - 1]] * (end - c)); ab.extend(prices[c:end])
    mB = metrics(ab, nb)
    rows.append({"split": sname, "model": "Naive", "protocol": "B_60day",
                 "MAE": mB[0], "MAPE": mB[1], "RMSE": mB[2]})
    print(f"  ARIMA B {[r['MAPE'] for r in rows if r['split']==sname and r['model']=='ARIMA' and r['protocol']=='B_60day'][0]:.2f}%  "
          f"LSTM B {[r['MAPE'] for r in rows if r['split']==sname and r['model']=='LSTM' and r['protocol']=='B_60day'][0]:.2f}%", flush=True)

pd.DataFrame(rows).to_csv(TAB / "classical_benchmark_AB.csv", index=False)
print("\nSaved classical_benchmark_AB.csv")
