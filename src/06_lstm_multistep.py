"""
Experiment 2 — Setup 2: LSTM, recursive MULTI-STEP forecast.

Same architecture and training recipe as Experiment 1 (2 stacked LSTM layers of 50
units, dropout 0.2, Dense(1), MSE, Adam, early stopping on a validation slice, MinMax
scaler fit on train only, seeds set). Implemented in PyTorch.

Difference from Experiment 1: the test window is forecast RECURSIVELY. The model is
seeded with the last `w` days of training, predicts day 1, then that prediction is fed
back as the newest input (oldest dropped) to predict day 2, and so on across the whole
horizon. Real test values are never used as input — only the model's own guesses.

Window w = 60 (the size Experiment 1 selected for the standalone LSTM), kept fixed for
a like-for-like multi-step comparison.

Artifacts:
  ms_lstm_predictions.csv   test: date, actual, ms_lstm_pred
  ms_lstm_info.json         window + test metrics
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
WINDOW = 60
UNITS, DROPOUT = 50, 0.2
MAX_EPOCHS, PATIENCE, VAL_FRAC, BATCH = 150, 12, 0.10, 32
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
        self.lstm = nn.LSTM(input_size=1, hidden_size=units, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(units, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_model(scaled_train, w):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X, y = make_windows(scaled_train, w)
    n_val = int(len(X) * VAL_FRAC)
    Xtr, ytr = X[:-n_val], y[:-n_val]
    Xval, yval = X[-n_val:], y[-n_val:]

    to_t = lambda a: torch.tensor(a, device=device).unsqueeze(-1)
    Xtr_t, ytr_t = to_t(Xtr), torch.tensor(ytr, device=device).unsqueeze(-1)
    Xval_t, yval_t = to_t(Xval), torch.tensor(yval, device=device).unsqueeze(-1)

    model = LSTMNet().to(device)
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


def recursive_forecast(model, seed_window, steps):
    """Roll forward `steps` days feeding each prediction back in. All in scaled space."""
    model.eval()
    window = list(seed_window)  # last w scaled training values
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(np.array(window[-len(seed_window):], dtype=np.float32),
                             device=device).reshape(1, -1, 1)
            p = model(x).item()
            preds.append(p)
            window.append(p)
    return np.array(preds, dtype=np.float32)


# ---------------------------------------------------- load + reuse saved split
df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
with open(str(PRED / "split_info.json")) as f:
    split_info = json.load(f)
split = split_info["split_index"]
n = len(df)
assert n == split_info["n"]

values = df["spsl20_points"].astype(float).values.reshape(-1, 1)
test_dates = df["date"].iloc[split:]
actual_test = values[split:].ravel()

print("=" * 70)
print("EXPERIMENT 2 — SETUP 2: LSTM (recursive multi-step)")
print("=" * 70)
print(f"Train: {split}  Test: {n - split}  window={WINDOW}")

scaler = MinMaxScaler((0, 1))
scaled_train = scaler.fit_transform(values[:split]).ravel()   # fit on TRAIN only

model, val_rmse = train_model(scaled_train, WINDOW)
seed_window = scaled_train[-WINDOW:]                            # last w training days
pred_scaled = recursive_forecast(model, seed_window, len(actual_test))
ms_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

mae, mape, rmse = metrics(actual_test, ms_pred)
print(f"\nval_RMSE(scaled)={val_rmse:.5f}")
print("--- Test-set accuracy (recursive multi-step) ---")
print(f"MAE  = {mae:.4f}")
print(f"MAPE = {mape:.4f}%")
print(f"RMSE = {rmse:.4f}")

pd.DataFrame(
    {"date": test_dates.values, "actual": actual_test, "ms_lstm_pred": ms_pred}
).to_csv(str(PRED / "ms_lstm_predictions.csv"), index=False)
with open(str(PRED / "ms_lstm_info.json"), "w") as f:
    json.dump({"window": WINDOW, "test_MAE": float(mae),
               "test_MAPE": float(mape), "test_RMSE": float(rmse)}, f, indent=2)
print("\nSaved: ms_lstm_predictions.csv, ms_lstm_info.json")
