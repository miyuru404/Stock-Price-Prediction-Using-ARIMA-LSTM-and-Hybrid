"""
Experiment 3 — Model B: GRU (recurrent neural network, LSTM's lighter cousin).

Reuses the LSTM pipeline exactly; only nn.LSTM -> nn.GRU. Architecture: 2 stacked GRU
layers x 50 units, dropout 0.2, Dense(1) head, MSE loss, Adam, early stopping on a
validation slice carved from the end of train. MinMax scaler fit on train only.

Both protocols (as in Experiments 1 & 2):
  - One-step : each test day predicted from the true prior w days.
  - Multi-step: seed with the last w training days, feed each prediction back
                recursively across the whole horizon (never real test values).

Windows 60 and 30 are tried; the lower-validation-RMSE window is selected. The model
is stochastic, so 3 seeds are run and metrics reported as mean +/- std. The saved
prediction files hold the mean prediction across seeds.

Artifacts (new filenames — nothing earlier is overwritten):
  gru_predictions_onestep.csv    date, actual, gru_onestep_pred (seed-mean)
  gru_predictions_multistep.csv  date, actual, gru_multistep_pred (seed-mean)
  gru_info.json                  per-seed metrics + mean/std, both protocols
"""

import json
import random
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
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

SEEDS = [42, 43, 44]
WINDOWS = [60, 30]
UNITS, DROPOUT = 50, 0.2
MAX_EPOCHS, PATIENCE, VAL_FRAC, BATCH = 150, 12, 0.10, 32
device = torch.device("cpu")


def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return (float(np.mean(np.abs(a - p))),
            float(np.mean(np.abs((a - p) / a)) * 100),
            float(np.sqrt(np.mean((a - p) ** 2))))


def make_windows(arr, w):
    X, y = [], []
    for i in range(w, len(arr)):
        X.append(arr[i - w:i])
        y.append(arr[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class GRUNet(nn.Module):
    def __init__(self, units=UNITS, dropout=DROPOUT):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=units, num_layers=2,
                          batch_first=True, dropout=dropout)
        self.head = nn.Linear(units, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def train_gru(scaled_train, w, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y = make_windows(scaled_train, w)
    n_val = int(len(X) * VAL_FRAC)
    Xtr, ytr = X[:-n_val], y[:-n_val]
    Xval, yval = X[-n_val:], y[-n_val:]
    to_t = lambda a: torch.tensor(a, device=device).unsqueeze(-1)
    Xtr_t, ytr_t = to_t(Xtr), torch.tensor(ytr, device=device).unsqueeze(-1)
    Xval_t, yval_t = to_t(Xval), torch.tensor(yval, device=device).unsqueeze(-1)
    model = GRUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_val, best_state, wait = float("inf"), None, 0
    n_tr = len(Xtr_t)
    for _ in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xval_t), yval_t).item()
        if vloss < best_val - 1e-9:
            best_val, best_state, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, np.sqrt(best_val)


def predict_onestep(model, scaled_full, w, split, n, scaler):
    model.eval()
    Xt = np.array([scaled_full[i - w:i] for i in range(split, n)], dtype=np.float32)
    with torch.no_grad():
        ps = model(torch.tensor(Xt, device=device).unsqueeze(-1)).cpu().numpy().ravel()
    return scaler.inverse_transform(ps.reshape(-1, 1)).ravel()


def predict_multistep(model, seed_window, steps, w, scaler):
    model.eval()
    window = list(seed_window)
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(np.array(window[-w:], dtype=np.float32),
                             device=device).reshape(1, -1, 1)
            p = model(x).item()
            preds.append(p)
            window.append(p)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()


# ---------------------------------------------------- load + reuse saved split
df = pd.read_csv(DATA / "spsl20_trading_days_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
with open(PRED / "split_info.json") as f:
    split = json.load(f)["split_index"]
n = len(df)
values = df["spsl20_points"].astype(float).values.reshape(-1, 1)
test_dates = df["date"].iloc[split:]
actual_test = values[split:].ravel()

print("=" * 68)
print("EXPERIMENT 3 — MODEL B: GRU (2x50, 3 seeds, one-step + multi-step)")
print("=" * 68)
print(f"Train: {split}  Test: {n - split}  seeds={SEEDS}")

scaler = MinMaxScaler((0, 1))
scaled_train = scaler.fit_transform(values[:split]).ravel()   # fit on TRAIN only
scaled_full = scaler.transform(values).ravel()

per_seed = []
onestep_preds, multistep_preds = [], []
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # try both windows, keep the lower-validation-RMSE model
    cand = {}
    for w in WINDOWS:
        model, vr = train_gru(scaled_train, w, seed)
        cand[w] = (model, vr)
    best_w = min(cand, key=lambda k: cand[k][1])
    model = cand[best_w][0]

    p1 = predict_onestep(model, scaled_full, best_w, split, n, scaler)
    p2 = predict_multistep(model, scaled_train[-best_w:], len(actual_test), best_w, scaler)
    onestep_preds.append(p1)
    multistep_preds.append(p2)

    m1 = metrics(actual_test, p1)
    m2 = metrics(actual_test, p2)
    per_seed.append({"seed": seed, "window": best_w,
                     "onestep": {"MAE": m1[0], "MAPE": m1[1], "RMSE": m1[2]},
                     "multistep": {"MAE": m2[0], "MAPE": m2[1], "RMSE": m2[2]}})
    print(f"\nseed {seed}: window {best_w}")
    print(f"  one-step   MAE {m1[0]:.3f}  MAPE {m1[1]:.3f}%  RMSE {m1[2]:.3f}")
    print(f"  multi-step MAE {m2[0]:.3f}  MAPE {m2[1]:.3f}%  RMSE {m2[2]:.3f}")


def mean_std(key, metric):
    vals = [s[key][metric] for s in per_seed]
    return float(np.mean(vals)), float(np.std(vals))


summary = {}
for proto in ["onestep", "multistep"]:
    summary[proto] = {m: {"mean": mean_std(proto, m)[0], "std": mean_std(proto, m)[1]}
                      for m in ["MAE", "MAPE", "RMSE"]}

print("\n--- GRU mean +/- std across seeds ---")
for proto in ["onestep", "multistep"]:
    s = summary[proto]
    print(f"{proto:>9}: MAE {s['MAE']['mean']:.3f}+/-{s['MAE']['std']:.3f}   "
          f"MAPE {s['MAPE']['mean']:.3f}+/-{s['MAPE']['std']:.3f}%   "
          f"RMSE {s['RMSE']['mean']:.3f}+/-{s['RMSE']['std']:.3f}")

# seed-mean predictions for the saved files / overlay
p1_mean = np.mean(onestep_preds, axis=0)
p2_mean = np.mean(multistep_preds, axis=0)
pd.DataFrame({"date": test_dates.values, "actual": actual_test,
              "gru_onestep_pred": p1_mean}).to_csv(
    PRED / "gru_predictions_onestep.csv", index=False)
pd.DataFrame({"date": test_dates.values, "actual": actual_test,
              "gru_multistep_pred": p2_mean}).to_csv(
    PRED / "gru_predictions_multistep.csv", index=False)
with open(PRED / "gru_info.json", "w") as f:
    json.dump({"seeds": SEEDS, "windows_tried": WINDOWS,
               "per_seed": per_seed, "summary_mean_std": summary,
               "note": "saved CSVs hold the mean prediction across seeds"}, f, indent=2)
print("\nSaved: gru_predictions_onestep.csv, gru_predictions_multistep.csv, gru_info.json")
