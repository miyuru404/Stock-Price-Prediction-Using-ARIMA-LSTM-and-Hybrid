"""
Setup 3 — Hybrid ARIMA + LSTM (residual / error-correction).

    Y_t = L_t + N_t
    ARIMA gives the linear forecast  L̂_t  (reused from script 01).
    Residual  e_t = Y_t - L̂_t  carries the nonlinear leftover.
    An LSTM is trained on the residual series to predict  N̂_t.
    Final forecast:  Ŷ_t = L̂_t + N̂_t.

The ARIMA is reused from script 01 via its saved artifacts (no refit):
  - training residuals from arima_train_residuals.csv
  - test linear forecast L̂ from arima_predictions.csv
The very first training residual is the Kalman-filter initialization artifact
(fitted value 0 -> residual == first price) and is dropped.

LSTM architecture, seeds, scaler-on-train-only rule, validation slice, and
one-step-ahead test protocol are identical to Setup 2, applied to residuals.

Artifacts:
  hybrid_predictions.csv   test set: date, actual, arima_pred, lstm_resid, hybrid_pred
  hybrid_info.json         window + test metrics
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

WINDOWS = [60, 30]
UNITS = 50
DROPOUT = 0.2
MAX_EPOCHS = 150
PATIENCE = 12
VAL_FRAC = 0.10
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
        self.lstm = nn.LSTM(input_size=1, hidden_size=units, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(units, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_and_predict(w, scaled_train_resid, scaled_full_resid, n_train_resid, total):
    """Train LSTM on residual windows; one-step-ahead predict the test residuals."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, y = make_windows(scaled_train_resid, w)
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

    # One-step-ahead residual prediction across the test portion, using the TRUE
    # prior residuals as input (mirrors Setup 2's protocol).
    model.eval()
    Xtest = np.array([scaled_full_resid[i - w:i] for i in range(n_train_resid, total)],
                     dtype=np.float32)
    with torch.no_grad():
        pred_scaled = model(torch.tensor(Xtest, device=device).unsqueeze(-1)).cpu().numpy().ravel()
    return np.sqrt(best_val), pred_scaled


# --------------------------------------------------- reuse ARIMA from script 01
with open(str(PRED / "split_info.json")) as f:
    split_info = json.load(f)

train_res = pd.read_csv(str(PRED / "arima_train_residuals.csv"), parse_dates=["date"])
train_res = train_res.iloc[1:].reset_index(drop=True)   # drop Kalman init artifact

arima_test = pd.read_csv(str(PRED / "arima_predictions.csv"), parse_dates=["date"])
actual_test = arima_test["actual"].values
arima_pred = arima_test["arima_pred"].values
test_resid = actual_test - arima_pred                    # true test residuals

train_resid = train_res["residual"].values
full_resid = np.concatenate([train_resid, test_resid])
n_train_resid = len(train_resid)
total = len(full_resid)

print("=" * 70)
print("SETUP 3 — HYBRID ARIMA + LSTM (residual correction)")
print("=" * 70)
print(f"ARIMA reused from script 01 (order {split_info.get('order', 'see arima_info.json')})")
print(f"Train residuals: {n_train_resid}   Test residuals: {len(test_resid)}")

# Scale residuals with scaler fit on TRAIN residuals only.
rscaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_resid = rscaler.fit_transform(train_resid.reshape(-1, 1)).ravel()
scaled_full_resid = rscaler.transform(full_resid.reshape(-1, 1)).ravel()

results = {}
for w in WINDOWS:
    val_rmse, pred_scaled = train_and_predict(
        w, scaled_train_resid, scaled_full_resid, n_train_resid, total)
    lstm_resid = rscaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    hybrid_pred = arima_pred + lstm_resid                # Ŷ = L̂ + N̂
    mae, mape, rmse = metrics(actual_test, hybrid_pred)
    results[w] = {"val_rmse": val_rmse, "lstm_resid": lstm_resid,
                  "hybrid_pred": hybrid_pred, "mae": mae, "mape": mape, "rmse": rmse}
    print(f"\nWindow {w:>2}: val_RMSE(scaled)={val_rmse:.5f}  "
          f"| test MAE={mae:.4f}  MAPE={mape:.4f}%  RMSE={rmse:.4f}")

best_w = min(results, key=lambda k: results[k]["val_rmse"])
best = results[best_w]
print(f"\n==> Selected residual window = {best_w} (lowest validation RMSE)")
print(f"    Hybrid test  MAE={best['mae']:.4f}  MAPE={best['mape']:.4f}%  RMSE={best['rmse']:.4f}")

pd.DataFrame({
    "date": arima_test["date"].values,
    "actual": actual_test,
    "arima_pred": arima_pred,
    "lstm_resid": best["lstm_resid"],
    "hybrid_pred": best["hybrid_pred"],
}).to_csv(str(PRED / "hybrid_predictions.csv"), index=False)

with open(str(PRED / "hybrid_info.json"), "w") as f:
    json.dump({"residual_window": int(best_w),
               "test_MAE": float(best["mae"]), "test_MAPE": float(best["mape"]),
               "test_RMSE": float(best["rmse"])}, f, indent=2)

print("\nSaved: hybrid_predictions.csv, hybrid_info.json")
