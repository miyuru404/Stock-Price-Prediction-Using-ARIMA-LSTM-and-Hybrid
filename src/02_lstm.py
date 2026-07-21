"""
Setup 2 — LSTM on the S&P SL 20 index.

Implemented in PyTorch (TensorFlow has no wheel for the Python 3.14 interpreter on
this machine). The architecture matches CLAUDE.md: two stacked LSTM layers of 50
units with dropout 0.2 feeding a Dense(1) head, MSE loss, Adam, and early stopping
on a validation slice carved from the end of the training set.

Split is reused from split_info.json (identical dates to ARIM A/Hybrid). The MinMax
scaler is fit on the TRAINING portion only, then applied to everything. Test
predictions are one-step-ahead: each next-day prediction uses the true prior `w`
days as input (comparable to the ARIMA one-step-ahead forecast).

Two window sizes (60 and 30) are trained; the one with the lower VALIDATION RMSE
is selected (no test peeking) and its test predictions are saved.

Artifacts:
  lstm_predictions.csv   test set: date, actual, lstm_pred
  lstm_info.json         chosen window + test metrics
"""

import json
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
DATA = _ROOT / "data" / "processed"
PRED = _ROOT / "results" / "predictions"
FIG = _ROOT / "results" / "figures"
TAB = _ROOT / "results" / "tables"
REPORT = _ROOT / "report"
for _d in (PRED, FIG, TAB, REPORT):
    _d.mkdir(parents=True, exist_ok=True)
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CSV = str(DATA / "spsl20_trading_days_clean.csv")
WINDOWS = [60, 30]
UNITS = 50
DROPOUT = 0.2
MAX_EPOCHS = 150
PATIENCE = 12
VAL_FRAC = 0.10        # validation slice carved from the END of train
BATCH = 32
device = torch.device("cpu")


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return mae, mape, rmse


def make_windows(arr, w):
    X, y = [], []
    for i in range(w, len(arr)):
        X.append(arr[i - w:i])
        y.append(arr[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMNet(nn.Module):
    def __init__(self, units=UNITS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=units, num_layers=2,
            batch_first=True, dropout=dropout,
        )
        self.head = nn.Linear(units, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, seq, units)
        return self.head(out[:, -1, :])  # last timestep -> Dense(1)


def train_and_predict(w, scaled_train, scaled_full, split, n):
    """Train an LSTM for window `w`; return (val_rmse_scaled, test_pred_scaled)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, y = make_windows(scaled_train, w)
    n_val = int(len(X) * VAL_FRAC)
    Xtr, ytr = X[:-n_val], y[:-n_val]      # chronological split — no shuffle
    Xval, yval = X[-n_val:], y[-n_val:]

    to_t = lambda a: torch.tensor(a, device=device).unsqueeze(-1)  # (N, w, 1)
    Xtr_t, ytr_t = to_t(Xtr), torch.tensor(ytr, device=device).unsqueeze(-1)
    Xval_t, yval_t = to_t(Xval), torch.tensor(yval, device=device).unsqueeze(-1)

    model = LSTMNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val, best_state, wait = float("inf"), None, 0
    n_train = len(Xtr_t)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)     # shuffle only the mini-batch order
        for s in range(0, n_train, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xval_t), yval_t).item()
        if vloss < best_val - 1e-7:
            best_val, best_state, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)

    # One-step-ahead test predictions: window of TRUE prior days for each test day.
    model.eval()
    Xtest = np.array([scaled_full[i - w:i] for i in range(split, n)], dtype=np.float32)
    with torch.no_grad():
        pred_scaled = model(torch.tensor(Xtest, device=device).unsqueeze(-1)).cpu().numpy().ravel()
    return np.sqrt(best_val), pred_scaled


# ---------------------------------------------------------------- load + split
df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
with open(str(PRED / "split_info.json")) as f:
    split_info = json.load(f)
split = split_info["split_index"]
n = len(df)
assert n == split_info["n"], "CSV length changed since ARIMA split was saved"

values = df["spsl20_points"].astype(float).values.reshape(-1, 1)
dates = df["date"]
test_dates = dates.iloc[split:]
actual_test = values[split:].ravel()

print("=" * 70)
print("SETUP 2 — LSTM (PyTorch)")
print("=" * 70)
print(f"Train: {split}  Test: {n - split}  "
      f"({split_info['test_start_date']} -> {split_info['test_end_date']})")

# Scaler fit on TRAIN ONLY, then applied to everything.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(values[:split]).ravel()
scaled_full = scaler.transform(values).ravel()

results = {}
for w in WINDOWS:
    val_rmse, pred_scaled = train_and_predict(w, scaled_train, scaled_full, split, n)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    mae, mape, rmse = metrics(actual_test, pred)
    results[w] = {"val_rmse": val_rmse, "pred": pred,
                  "mae": mae, "mape": mape, "rmse": rmse}
    print(f"\nWindow {w:>2}: val_RMSE(scaled)={val_rmse:.5f}  "
          f"| test MAE={mae:.4f}  MAPE={mape:.4f}%  RMSE={rmse:.4f}")

# Select by validation RMSE (no test peeking).
best_w = min(results, key=lambda k: results[k]["val_rmse"])
best = results[best_w]
print(f"\n==> Selected window = {best_w} (lowest validation RMSE)")
print(f"    Test  MAE={best['mae']:.4f}  MAPE={best['mape']:.4f}%  RMSE={best['rmse']:.4f}")

pd.DataFrame(
    {"date": test_dates.values, "actual": actual_test, "lstm_pred": best["pred"]}
).to_csv(str(PRED / "lstm_predictions.csv"), index=False)

with open(str(PRED / "lstm_info.json"), "w") as f:
    json.dump(
        {"window": int(best_w),
         "test_MAE": float(best["mae"]), "test_MAPE": float(best["mape"]),
         "test_RMSE": float(best["rmse"]),
         "windows_tried": {str(k): {"val_rmse": float(v["val_rmse"]),
                                    "test_RMSE": float(v["rmse"])}
                           for k, v in results.items()}},
        f, indent=2,
    )
print("\nSaved: lstm_predictions.csv, lstm_info.json")
