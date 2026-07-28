"""
Multi-step METHOD comparison: Recursive vs Direct vs Multi-output (LSTM), + Naive.

Answers: does avoiding error-compounding (Direct / Multi-output) beat the Recursive
method we used everywhere before? Evaluated at fixed horizons with rolling (walk-
forward) origins, on HNB daily and HNB 1-hour data. 80/20 split.

- Recursive : one 1-step LSTM, rolled forward h times feeding its own predictions.
- Direct    : a separate LSTM per horizon h (predicts t+h directly from real data).
- Multi     : one LSTM with a length-|H| output head (all horizons at once).
- Naive     : persistence (value at origin) for every horizon.

Artifacts: results/tables/multistep_methods.csv
"""
import warnings, random
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

_ROOT = _Path(__file__).resolve().parents[1]
DATA = _ROOT / "data" / "processed"
TAB = _ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
UNITS, DROPOUT, EPOCHS, PAT, VF, BATCH = 50, 0.2, 120, 12, 0.10, 32


def mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(np.mean(np.abs((a - p) / a)) * 100)


class LSTMNet(nn.Module):
    def __init__(s, n_out=1):
        super().__init__(); s.lstm = nn.LSTM(1, UNITS, 2, batch_first=True, dropout=DROPOUT); s.head = nn.Linear(UNITS, n_out)
    def forward(s, x):
        o, _ = s.lstm(x); return s.head(o[:, -1, :])


def train(X, Y, n_out):
    torch.manual_seed(SEED); np.random.seed(SEED)
    nv = max(1, int(len(X) * VF))
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    Yt = torch.tensor(Y, dtype=torch.float32).reshape(len(Y), n_out)
    Xtr, Ytr, Xv, Yv = Xt[:-nv], Yt[:-nv], Xt[-nv:], Yt[-nv:]
    m = LSTMNet(n_out); opt = torch.optim.Adam(m.parameters(), 1e-3); lf = nn.MSELoss()
    best, bs, wait = 1e9, None, 0
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), BATCH):
            idx = perm[s:s+BATCH]; opt.zero_grad(); lf(m(Xtr[idx]), Ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = lf(m(Xv), Yv).item()
        if vl < best - 1e-9: best, bs, wait = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PAT: break
    m.load_state_dict(bs); m.eval(); return m


def run_dataset(name, prices, W, horizons, stride):
    n = len(prices); split = int(n * 0.8); maxh = max(horizons)
    sca = MinMaxScaler((0, 1)); sc_tr = sca.fit_transform(prices[:split].reshape(-1, 1)).ravel()
    scf = sca.transform(prices.reshape(-1, 1)).ravel()
    # eval origins: last known index o in test, needs o+maxh valid
    origins = list(range(split, n - maxh, stride))
    print(f"[{name}] n={n} train={split} horizons={horizons} eval_origins={len(origins)}", flush=True)

    # ---- build training windows on TRAIN (origin o = last known index) ----
    def windows(target_h_list):
        X, Y = [], []
        for o in range(W - 1, split - maxh):
            X.append(sc_tr[o - W + 1:o + 1]); Y.append([sc_tr[o + h] for h in target_h_list])
        return np.array(X, np.float32), np.array(Y, np.float32)

    rows = []
    actuals = {h: np.array([prices[o + h] for o in origins]) for h in horizons}
    seeds = np.array([scf[o - W + 1:o + 1] for o in origins], np.float32)

    # NAIVE (persistence)
    for h in horizons:
        rows.append({"dataset": name, "method": "Naive", "horizon": h,
                     "MAPE": mape(actuals[h], [prices[o] for o in origins])})

    # RECURSIVE: one 1-step model, roll maxh steps per origin
    Xr, Yr = windows([1]); mr = train(Xr, Yr, 1)
    rec = {h: [] for h in horizons}
    with torch.no_grad():
        for o in origins:
            win = list(scf[o - W + 1:o + 1]); path = {}
            for step in range(1, maxh + 1):
                x = torch.tensor(np.array(win[-W:], np.float32)).reshape(1, -1, 1)
                p = mr(x).item(); win.append(p)
                if step in horizons: path[step] = p
            for h in horizons: rec[h].append(path[h])
    for h in horizons:
        pred = sca.inverse_transform(np.array(rec[h]).reshape(-1, 1)).ravel()
        rows.append({"dataset": name, "method": "Recursive", "horizon": h, "MAPE": mape(actuals[h], pred)})

    # DIRECT: separate model per horizon
    for h in horizons:
        Xd, Yd = windows([h]); md = train(Xd, Yd, 1)
        with torch.no_grad():
            ps = md(torch.tensor(seeds).unsqueeze(-1)).numpy().ravel()
        pred = sca.inverse_transform(ps.reshape(-1, 1)).ravel()
        rows.append({"dataset": name, "method": "Direct", "horizon": h, "MAPE": mape(actuals[h], pred)})

    # MULTI-OUTPUT: one model, |H| heads
    Xm, Ym = windows(horizons); mm = train(Xm, Ym, len(horizons))
    with torch.no_grad():
        out = mm(torch.tensor(seeds).unsqueeze(-1)).numpy()  # (origins, |H|)
    for j, h in enumerate(horizons):
        pred = sca.inverse_transform(out[:, j].reshape(-1, 1)).ravel()
        rows.append({"dataset": name, "method": "Multi-output", "horizon": h, "MAPE": mape(actuals[h], pred)})

    return rows


all_rows = []
# HNB daily (88 pts) — short, small horizons, noisy
dly = pd.read_csv(DATA / "hnb_eod_daily.csv")["close"].astype(float).values
all_rows += run_dataset("HNB-daily", dly, W=10, horizons=[1, 3, 5, 10], stride=1)
# HNB hourly (7528 pts) — the meaningful one
hr = pd.read_csv(DATA / "hnb_hourly.csv")["close"].astype(float).values
all_rows += run_dataset("HNB-hourly", hr, W=60, horizons=[1, 5, 25, 110], stride=5)

df = pd.DataFrame(all_rows)
df.to_csv(TAB / "multistep_methods.csv", index=False)
print("\n=== MAPE (%) by method x horizon ===")
for ds in ["HNB-daily", "HNB-hourly"]:
    piv = df[df.dataset == ds].pivot(index="method", columns="horizon", values="MAPE")
    piv = piv.reindex(["Naive", "Recursive", "Direct", "Multi-output"])
    print(f"\n{ds}:"); print(piv.round(2).to_string())
print("\nSaved results/tables/multistep_methods.csv")
