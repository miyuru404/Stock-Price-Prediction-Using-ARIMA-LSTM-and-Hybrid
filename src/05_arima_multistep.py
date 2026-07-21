"""
Experiment 2 — Setup 1: ARIMA, paper's way (dynamic MULTI-STEP forecast).

Difference from Experiment 1: instead of a one-step-ahead forecast that is fed the
true previous day at each step, this produces a single out-of-sample forecast for
the ENTIRE test horizon in one shot (`get_forecast(steps=len(test))`). No real test
observation is ever fed back. This is the harder, honest test the source paper used,
so the numbers are directly comparable to the paper's Table 4.

Paper recipe: log-transform the series, d = 1, baseline ARIMA(2,1,0); also run
auto_arima and keep the lower-AIC order. Parameters estimated on the training set
only; the log-transform is inverted (exp) before scoring.

Artifacts (new filenames — Experiment 1 files untouched):
  ms_arima_predictions.csv        test: date, actual, ms_arima_pred
  ms_arima_train_residuals.csv    train: date, actual, fitted, residual (price space)
  ms_arima_info.json              order + AIC + test metrics
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
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

warnings.filterwarnings("ignore")

CSV = str(DATA / "spsl20_trading_days_clean.csv")


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return mae, mape, rmse


# ---------------------------------------------------- load + reuse saved split
df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
with open(str(PRED / "split_info.json")) as f:
    split_info = json.load(f)
split = split_info["split_index"]
n = len(df)
assert n == split_info["n"], "CSV length changed since the split was saved"

dates = df["date"]
price = df["spsl20_points"].astype(float).values
# Paper recipe: model the log of the index.
logp = np.log(price)

# Integer index (RangeIndex extends cleanly; DatetimeIndex has irregular gaps).
log_train = pd.Series(logp[:split], index=pd.RangeIndex(split))
price_train = price[:split]
price_test = price[split:]
train_dates, test_dates = dates.iloc[:split], dates.iloc[split:]

print("=" * 70)
print("EXPERIMENT 2 — SETUP 1: ARIMA (multi-step / dynamic, log-transform)")
print("=" * 70)
print(f"Train: {split}  Test: {n - split}  "
      f"({split_info['test_start_date']} -> {split_info['test_end_date']})")

# --------------------------------------------------------------- ADF / d (log)
adf_lvl = adfuller(log_train, autolag="AIC")
adf_d1 = adfuller(log_train.diff().dropna(), autolag="AIC")
print("\n--- ADF on log(price), training set ---")
print(f"Levels         : ADF={adf_lvl[0]:.4f}  p={adf_lvl[1]:.4f}")
print(f"1st difference : ADF={adf_d1[0]:.4f}  p={adf_d1[1]:.4f}")
d = 1  # paper's choice; log-level is non-stationary, 1st difference stationary
print(f"Using d = {d} (paper recipe)")

# --------------------------------------------------------- order selection
baseline_order = (2, 1, 0)
res_base = ARIMA(log_train, order=baseline_order).fit()
print(f"\nBaseline ARIMA{baseline_order}  AIC = {res_base.aic:.2f}")

auto = pm.auto_arima(log_train, start_p=0, start_q=0, max_p=5, max_q=5, d=d,
                     seasonal=False, stepwise=True, suppress_warnings=True,
                     error_action="ignore")
auto_order = auto.order
res_auto = ARIMA(log_train, order=auto_order).fit()
print(f"auto_arima ARIMA{auto_order}  AIC = {res_auto.aic:.2f}")

if res_auto.aic <= res_base.aic:
    order, res = auto_order, res_auto
else:
    order, res = baseline_order, res_base
print(f"==> Selected ARIMA{order} (lower AIC)")

# --------------------------------- single dynamic multi-step forecast (log space)
fc_log = res.get_forecast(steps=len(price_test)).predicted_mean.values
ms_pred = np.exp(fc_log)  # invert log-transform before scoring

mae, mape, rmse = metrics(price_test, ms_pred)
print("\n--- Test-set accuracy (dynamic multi-step) ---")
print(f"MAE  = {mae:.4f}")
print(f"MAPE = {mape:.4f}%")
print(f"RMSE = {rmse:.4f}")

# ------------------------------------------ training residuals in PRICE space
# For the multi-step hybrid (script 07): residuals of THIS log-ARIMA, back in price
# units. Drop day 0 (Kalman initialisation: fitted log = 0 -> exp = 1 -> huge resid).
fitted_price = np.exp(res.fittedvalues.values)
resid_price = price_train - fitted_price

# ---------------------------------------------------------------- save
pd.DataFrame(
    {"date": test_dates.values, "actual": price_test, "ms_arima_pred": ms_pred}
).to_csv(str(PRED / "ms_arima_predictions.csv"), index=False)

pd.DataFrame(
    {"date": train_dates.values, "actual": price_train,
     "fitted": fitted_price, "residual": resid_price}
).iloc[1:].to_csv(str(PRED / "ms_arima_train_residuals.csv"), index=False)  # drop day-0 artifact

with open(str(PRED / "ms_arima_info.json"), "w") as f:
    json.dump({"order": list(order), "d": int(d), "log_transform": True,
               "aic": float(res.aic), "test_MAE": float(mae),
               "test_MAPE": float(mape), "test_RMSE": float(rmse)}, f, indent=2)

print("\nSaved: ms_arima_predictions.csv, ms_arima_train_residuals.csv, ms_arima_info.json")
