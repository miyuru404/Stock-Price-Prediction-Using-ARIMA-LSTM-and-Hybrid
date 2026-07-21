"""
Setup 1 — ARIMA on the S&P SL 20 index.

Loads the cleaned trading-day series, makes an 80/20 chronological split,
selects an ARIMA order (baseline ARIMA(2,1,0) vs. auto_arima), runs residual
diagnostics, and produces a one-step-ahead forecast across the test window.

One-step-ahead (not a single long multi-step forecast) is used so ARIMA is
compared to the LSTM on equal footing: the LSTM also predicts the next day from
the true recent history. Model PARAMETERS are estimated on the training set only
and then held fixed; the state is updated with the actual test observations via
statsmodels' `append(..., refit=False)`. That is NOT walk-forward retraining
(params never change on the test set) and introduces no parameter leakage.

Artifacts written for the later scripts:
  split_info.json            exact split (index + boundary dates)
  arima_info.json            chosen order + AIC
  arima_predictions.csv      test set: date, actual, arima_pred
  arima_train_residuals.csv  train set: date, actual, fitted, residual
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
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
import pmdarima as pm

warnings.filterwarnings("ignore")

CSV = str(DATA / "spsl20_trading_days_clean.csv")
TRAIN_FRAC = 0.80


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return mae, mape, rmse


# ---------------------------------------------------------------- load + split
df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
dates = df["date"]
values = df["spsl20_points"].astype(float)

n = len(values)
split = int(n * TRAIN_FRAC)

# Model on a plain integer index. The DatetimeIndex has irregular trading-day
# gaps (no fixed frequency), which breaks statsmodels' append/extend; an integer
# RangeIndex extends cleanly. Dates are carried alongside only for the outputs.
series = pd.Series(values.values, index=pd.RangeIndex(n))
train, test = series.iloc[:split], series.iloc[split:]
train_dates, test_dates = dates.iloc[:split], dates.iloc[split:]

split_info = {
    "csv": CSV,
    "n": int(n),
    "split_index": int(split),
    "train_frac": TRAIN_FRAC,
    "train_start_date": str(train_dates.iloc[0].date()),
    "train_end_date": str(train_dates.iloc[-1].date()),
    "test_start_date": str(test_dates.iloc[0].date()),
    "test_end_date": str(test_dates.iloc[-1].date()),
}
with open(str(PRED / "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)

print("=" * 70)
print("SETUP 1 — ARIMA")
print("=" * 70)
print(f"Total trading days : {n}")
print(f"Train : {split} days  ({split_info['train_start_date']} -> {split_info['train_end_date']})")
print(f"Test  : {n - split} days  ({split_info['test_start_date']} -> {split_info['test_end_date']})")

# ------------------------------------------------------------------ ADF / d
print("\n--- ADF stationarity test (training set) ---")
adf_lvl = adfuller(train, autolag="AIC")
adf_d1 = adfuller(train.diff().dropna(), autolag="AIC")
print(f"Levels          : ADF stat={adf_lvl[0]:.4f}  p-value={adf_lvl[1]:.4f}")
print(f"1st difference  : ADF stat={adf_d1[0]:.4f}  p-value={adf_d1[1]:.4f}")
# The levels ADF is only marginal (~0.04) and driven by the post-crash recovery
# trend; the price level is not meaningfully stationary. Following standard
# practice for stock-price series (and the source paper), use d=1. Fixing d also
# keeps the ARIMA AIC comparison valid: AIC is only comparable across models
# fit with the same differencing order.
d = 1
print(f"Chosen differencing order d = {d}"
      f"  (levels ADF marginal; 1st difference clearly stationary -> d=1)")

# ---------------------------------------------------------- order selection
print("\n--- Order selection ---")
baseline_order = (2, 1, 0)
res_base = ARIMA(train, order=baseline_order).fit()
print(f"Baseline ARIMA{baseline_order}  AIC = {res_base.aic:.2f}")

auto = pm.auto_arima(
    train, start_p=0, start_q=0, max_p=5, max_q=5, d=d,
    seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore",
)
auto_order = auto.order
res_auto = ARIMA(train, order=auto_order).fit()
print(f"auto_arima ARIMA{auto_order}  AIC = {res_auto.aic:.2f}")

# Keep the lower-AIC model (selection on training info only — no test peeking).
if res_auto.aic <= res_base.aic:
    order, res = auto_order, res_auto
else:
    order, res = baseline_order, res_base
print(f"==> Selected ARIMA{order}  (lower AIC)")

# --------------------------------------------------------- residual diagnostics
print("\n--- Residual diagnostics (training fit) ---")
resid = res.resid.dropna()
lb = acorr_ljungbox(resid, lags=[10], return_df=True)
print(f"Ljung-Box (lag 10)      : stat={lb['lb_stat'].iloc[0]:.4f}  "
      f"p-value={lb['lb_pvalue'].iloc[0]:.4f}  "
      f"({'no' if lb['lb_pvalue'].iloc[0] > 0.05 else 'residual'} autocorrelation)")
jb_stat, jb_p, skew, kurt = jarque_bera(resid)
print(f"Jarque-Bera (normality) : stat={jb_stat:.4f}  p-value={jb_p:.4f}  "
      f"skew={skew:.3f}  kurtosis={kurt:.3f}")
# Heteroskedasticity: regress squared resid on time index (Breusch-Pagan).
exog = np.column_stack([np.ones(len(resid)), np.arange(len(resid))])
bp_stat, bp_p, _, _ = het_breuschpagan(resid, exog)
print(f"Breusch-Pagan (heterosk): stat={bp_stat:.4f}  p-value={bp_p:.4f}  "
      f"({'homoskedastic' if bp_p > 0.05 else 'heteroskedastic'})")

# ------------------------------------------------ one-step-ahead test forecast
# Hold parameters fixed; update state with actual test observations.
res_ext = res.append(test, refit=False)
test_pred = res_ext.predict(start=split, end=n - 1)

mae, mape, rmse = metrics(test.values, test_pred.values)
print("\n--- Test-set accuracy (one-step-ahead) ---")
print(f"MAE  = {mae:.4f}")
print(f"MAPE = {mape:.4f}%")
print(f"RMSE = {rmse:.4f}")

# ------------------------------------------------------------------ save
pd.DataFrame(
    {"date": test_dates.values, "actual": test.values, "arima_pred": test_pred.values}
).to_csv(str(PRED / "arima_predictions.csv"), index=False)

pd.DataFrame(
    {
        "date": train_dates.values,
        "actual": train.values,
        "fitted": res.fittedvalues.values,
        "residual": res.resid.values,
    }
).to_csv(str(PRED / "arima_train_residuals.csv"), index=False)

with open(str(PRED / "arima_info.json"), "w") as f:
    json.dump(
        {"order": list(order), "d": int(d), "aic": float(res.aic),
         "test_MAE": float(mae), "test_MAPE": float(mape), "test_RMSE": float(rmse)},
        f, indent=2,
    )

print("\nSaved: split_info.json, arima_info.json, arima_predictions.csv, "
      "arima_train_residuals.csv")
