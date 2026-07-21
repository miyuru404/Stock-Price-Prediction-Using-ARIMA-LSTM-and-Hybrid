"""
Experiment 2 — Setup 3: Hybrid ARIMA + LSTM, MULTI-STEP.

  ARIMA part  = the dynamic multi-step ARIMA forecast from script 05 (L̂ₜ over the
                whole horizon).
  LSTM part   = an LSTM trained on the Experiment-2 ARIMA's TRAINING residuals
                (price space, from ms_arima_train_residuals.csv), then rolled forward
                RECURSIVELY (self-feeding, same scheme as script 06) to predict the
                residuals N̂ₜ across the test horizon.
  Final       = L̂ₜ + N̂ₜ.

Residual window w = 30 (the size Experiment 1 selected for the hybrid's residual LSTM).
Scaler fit on train residuals only; seeds set.

Artifacts:
  ms_hybrid_predictions.csv   test: date, actual, ms_arima_pred, ms_lstm_resid, ms_hybrid_pred
  ms_hybrid_info.json         window + test metrics
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

WINDOW = 30
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


def recursive_forecast(model, seed_window, steps, w):
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
    return np.array(preds, dtype=np.float32)


# ------------------------------------------- reuse Experiment-2 ARIMA artifacts
with open(str(PRED / "split_info.json")) as f:
    split_info = json.load(f)

arima_ms = pd.read_csv(str(PRED / "ms_arima_predictions.csv"), parse_dates=["date"])
actual_test = arima_ms["actual"].values
arima_pred = arima_ms["ms_arima_pred"].values   # L̂ₜ (whole horizon)

train_res = pd.read_csv(str(PRED / "ms_arima_train_residuals.csv"))  # day-0 artifact already dropped
train_resid = train_res["residual"].values.reshape(-1, 1)

print("=" * 70)
print("EXPERIMENT 2 — SETUP 3: HYBRID ARIMA + LSTM (multi-step)")
print("=" * 70)
print(f"Train residuals: {len(train_resid)}  Test: {len(actual_test)}  resid_window={WINDOW}")

# Scale residuals with scaler fit on TRAIN residuals only.
rscaler = MinMaxScaler((0, 1))
scaled_train_resid = rscaler.fit_transform(train_resid).ravel()

model, val_rmse = train_model(scaled_train_resid, WINDOW)
seed_window = scaled_train_resid[-WINDOW:]
pred_scaled = recursive_forecast(model, seed_window, len(actual_test), WINDOW)
lstm_resid = rscaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

hybrid_pred = arima_pred + lstm_resid          # Ŷₜ = L̂ₜ + N̂ₜ

mae, mape, rmse = metrics(actual_test, hybrid_pred)
print(f"\nresid val_RMSE(scaled)={val_rmse:.5f}")
print("--- Test-set accuracy (multi-step hybrid) ---")
print(f"MAE  = {mae:.4f}")
print(f"MAPE = {mape:.4f}%")
print(f"RMSE = {rmse:.4f}")

pd.DataFrame({
    "date": arima_ms["date"].values, "actual": actual_test,
    "ms_arima_pred": arima_pred, "ms_lstm_resid": lstm_resid,
    "ms_hybrid_pred": hybrid_pred,
}).to_csv(str(PRED / "ms_hybrid_predictions.csv"), index=False)
with open(str(PRED / "ms_hybrid_info.json"), "w") as f:
    json.dump({"residual_window": WINDOW, "test_MAE": float(mae),
               "test_MAPE": float(mape), "test_RMSE": float(rmse)}, f, indent=2)
print("\nSaved: ms_hybrid_predictions.csv, ms_hybrid_info.json")
