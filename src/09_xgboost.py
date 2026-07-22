"""
Experiment 3 — Model A: XGBoost (gradient-boosted trees).

Trees don't understand time, so the series is turned into a supervised table with
lag + rolling features. Crucially the TARGET is the next-day DIFFERENCE
(y_t - y_{t-1}), not the level: trees cannot extrapolate beyond the training range,
and the test period rises above anything seen in training, so a level target would
saturate. Levels are reconstructed by adding the predicted change to the previous
value.

Both forecasting protocols (identical in spirit to Experiments 1 & 2):
  - One-step : features built from the TRUE prior values; predicted change added to
               the true previous value.
  - Multi-step: seeded from the last training values; predicted change added to the
               model's OWN previous prediction and rolled forward — features rebuilt
               from predicted values, never real test values.

Reuses split_info.json (identical 80/20 split). Feature statistics use only past
values (shift >= 1), so there is no look-ahead leak. New filenames — nothing from
earlier experiments is overwritten.

Artifacts:
  xgb_predictions_onestep.csv    test: date, actual, xgb_onestep_pred
  xgb_predictions_multistep.csv  test: date, actual, xgb_multistep_pred
  xgb_info.json                  features, params, one-step + multi-step metrics
"""

import json
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
from xgboost import XGBRegressor

SEED = 42
LAGS = [1, 2, 3, 5, 10, 20]
ROLLS = [5, 10, 20]
FEATURES = ([f"lag_{L}" for L in LAGS]
            + [f"rmean_{W}" for W in ROLLS] + [f"rstd_{W}" for W in ROLLS]
            + ["dow", "month"])
VAL_FRAC = 0.10
PARAMS = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
              random_state=SEED, n_jobs=4)


def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return (float(np.mean(np.abs(a - p))),
            float(np.mean(np.abs((a - p) / a)) * 100),
            float(np.sqrt(np.mean((a - p) ** 2))))


def feature_row(hist, date):
    """Build one feature dict from a history of past LEVELS + the target date."""
    h = np.asarray(hist, float)
    row = {f"lag_{L}": h[-L] for L in LAGS}
    for W in ROLLS:
        w = h[-W:]
        row[f"rmean_{W}"] = w.mean()
        row[f"rstd_{W}"] = w.std(ddof=1)      # pandas rolling.std uses ddof=1
    row["dow"] = date.dayofweek
    row["month"] = date.month
    return row


# ---------------------------------------------------- load + reuse saved split
df = pd.read_csv(DATA / "spsl20_trading_days_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
with open(PRED / "split_info.json") as f:
    split = json.load(f)["split_index"]
n = len(df)
y = df["spsl20_points"].astype(float)
dates = df["date"]

print("=" * 68)
print("EXPERIMENT 3 — MODEL A: XGBoost (predicts daily change, rebuilds level)")
print("=" * 68)
print(f"Train: {split}  Test: {n - split}  ({dates.iloc[split].date()} -> {dates.iloc[-1].date()})")

# ---------------------------------------------------- supervised feature table
feat = pd.DataFrame(index=df.index)
for L in LAGS:
    feat[f"lag_{L}"] = y.shift(L)
for W in ROLLS:
    feat[f"rmean_{W}"] = y.shift(1).rolling(W).mean()
    feat[f"rstd_{W}"] = y.shift(1).rolling(W).std()
feat["dow"] = dates.dt.dayofweek
feat["month"] = dates.dt.month
feat["target"] = y.diff()                      # y_t - y_{t-1}
data = feat.dropna()

train_idx = [i for i in data.index if i < split]
test_idx = [i for i in data.index if i >= split]
assert len(test_idx) == n - split, "test rows must equal the held-out window"

# validation slice carved from the END of train (chronological, no shuffle)
n_val = int(len(train_idx) * VAL_FRAC)
tr_idx, val_idx = train_idx[:-n_val], train_idx[-n_val:]
X_tr, y_tr = data.loc[tr_idx, FEATURES], data.loc[tr_idx, "target"]
X_val, y_val = data.loc[val_idx, FEATURES], data.loc[val_idx, "target"]

model = XGBRegressor(eval_metric="rmse", early_stopping_rounds=30, **PARAMS)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_it = int(getattr(model, "best_iteration", PARAMS["n_estimators"]) or PARAMS["n_estimators"])
print(f"Trained: {len(FEATURES)} features, best_iteration={best_it}")

actual_test = y.iloc[split:].values
test_dates = dates.iloc[split:]

# -------------------------------------------------------------- one-step
# Features already come from true prior values; add predicted change to the TRUE
# previous level (lag_1 holds the actual y_{t-1}).
X_test = data.loc[test_idx, FEATURES]
diff_1s = model.predict(X_test)
pred_1s = X_test["lag_1"].values + diff_1s
mae1, mape1, rmse1 = metrics(actual_test, pred_1s)
print("\n--- One-step ---")
print(f"MAE {mae1:.4f}   MAPE {mape1:.4f}%   RMSE {rmse1:.4f}")

# -------------------------------------------------------------- multi-step
# Seed history with the actual training levels; roll forward on own predictions.
hist = list(y.iloc[:split].values)
pred_ms = []
for t in range(split, n):
    row = feature_row(hist, dates.iloc[t])
    X = pd.DataFrame([row], columns=FEATURES)
    d = float(model.predict(X)[0])
    yhat = hist[-1] + d                         # add change to OWN previous level
    pred_ms.append(yhat)
    hist.append(yhat)
pred_ms = np.array(pred_ms)
mae2, mape2, rmse2 = metrics(actual_test, pred_ms)
print("\n--- Multi-step (recursive) ---")
print(f"MAE {mae2:.4f}   MAPE {mape2:.4f}%   RMSE {rmse2:.4f}")

# ---------------------------------------------------------------- save
pd.DataFrame({"date": test_dates.values, "actual": actual_test,
              "xgb_onestep_pred": pred_1s}).to_csv(
    PRED / "xgb_predictions_onestep.csv", index=False)
pd.DataFrame({"date": test_dates.values, "actual": actual_test,
              "xgb_multistep_pred": pred_ms}).to_csv(
    PRED / "xgb_predictions_multistep.csv", index=False)
with open(PRED / "xgb_info.json", "w") as f:
    json.dump({"features": FEATURES, "params": PARAMS, "best_iteration": best_it,
               "target": "next-day difference (level reconstructed)",
               "onestep": {"MAE": mae1, "MAPE": mape1, "RMSE": rmse1},
               "multistep": {"MAE": mae2, "MAPE": mape2, "RMSE": rmse2}}, f, indent=2)
print("\nSaved: xgb_predictions_onestep.csv, xgb_predictions_multistep.csv, xgb_info.json")
