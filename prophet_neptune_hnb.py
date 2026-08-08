#!/usr/bin/env python3
"""
Prophet baseline on HNB, with Neptune.ai experiment tracking.

SETUP (one time):
    pip install prophet neptune pandas numpy matplotlib
    1. Create a free account at neptune.ai
    2. Create a project (e.g. "miyuru/cse-fyp")
    3. Get your API token from your profile settings
    4. Set them below, or as environment variables:
         export NEPTUNE_API_TOKEN="your_token_here"
         export NEPTUNE_PROJECT="miyuru/cse-fyp"

RUN:
    python prophet_neptune_hnb.py
"""
import os
import warnings
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)

from prophet import Prophet

# ----------------------------------------------------------------------------
# CONFIG — everything you might tune lives here, and all of it gets logged
# ----------------------------------------------------------------------------
CONFIG = {
    "ticker": "HNB.N0000",
    "data_path": "cleaned_data/HNB_daily_clean.csv",   # <-- adjust path
    "start_date": "2012-01-01",
    "train_frac": 0.85,
    "changepoint_prior_scale": 0.05,   # higher = more flexible trend
    "seasonality_prior_scale": 10.0,
    "weekly_seasonality": True,
    "yearly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": 0.80,            # 80% prediction interval
}

USE_NEPTUNE = True   # set False to run without tracking

# ----------------------------------------------------------------------------
# 1. NEPTUNE INIT
# ----------------------------------------------------------------------------
run = None
if USE_NEPTUNE:
    try:
        import neptune
        run = neptune.init_run(
            project=os.getenv("NEPTUNE_PROJECT", "miyuru/cse-fyp"),
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
            name=f"prophet-{CONFIG['ticker']}",
            tags=["prophet", "baseline", "price-level", CONFIG["ticker"]],
        )
        run["parameters"] = CONFIG
        print("Neptune run started")
    except Exception as e:
        print(f"Neptune unavailable ({e}) — continuing without tracking")
        run = None

# ----------------------------------------------------------------------------
# 2. DATA
# ----------------------------------------------------------------------------
df = pd.read_csv(CONFIG["data_path"], parse_dates=["date"])
df = df[df.date >= CONFIG["start_date"]][["date", "close"]]
df = df.rename(columns={"date": "ds", "close": "y"}).reset_index(drop=True)

cut = int(len(df) * CONFIG["train_frac"])
train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

print(f"rows={len(df)}  train={len(train)}  test={len(test)}")
if run:
    run["data/n_rows"] = len(df)
    run["data/n_train"] = len(train)
    run["data/n_test"] = len(test)
    run["data/train_start"] = str(train.ds.min().date())
    run["data/train_end"] = str(train.ds.max().date())
    run["data/test_start"] = str(test.ds.min().date())
    run["data/test_end"] = str(test.ds.max().date())

# ----------------------------------------------------------------------------
# 3. FIT PROPHET
# ----------------------------------------------------------------------------
m = Prophet(
    changepoint_prior_scale=CONFIG["changepoint_prior_scale"],
    seasonality_prior_scale=CONFIG["seasonality_prior_scale"],
    weekly_seasonality=CONFIG["weekly_seasonality"],
    yearly_seasonality=CONFIG["yearly_seasonality"],
    daily_seasonality=CONFIG["daily_seasonality"],
    interval_width=CONFIG["interval_width"],
)
m.fit(train)

future = m.make_future_dataframe(periods=len(test), freq="D")
fcst = m.predict(future)

pred = fcst.set_index("ds").reindex(test.ds)[["yhat", "yhat_lower", "yhat_upper"]]
res = test.reset_index(drop=True).join(pred.reset_index(drop=True)).dropna()

# ----------------------------------------------------------------------------
# 4. METRICS — Prophet vs NAIVE baseline
# ----------------------------------------------------------------------------
mae = float(np.mean(np.abs(res.y - res.yhat)))
rmse = float(np.sqrt(np.mean((res.y - res.yhat) ** 2)))
mape = float(np.mean(np.abs((res.y - res.yhat) / res.y)) * 100)

full = df.reset_index(drop=True)
idx = full.index[full.ds.isin(res.ds)]
naive = full.y.shift(1).loc[idx].values
act = full.y.loc[idx].values
n_mae = float(np.mean(np.abs(act - naive)))
n_rmse = float(np.sqrt(np.mean((act - naive) ** 2)))
n_mape = float(np.mean(np.abs((act - naive) / act)) * 100)

# DIRECTION accuracy — the metric that matters for this project
r2 = res.copy()
r2["prev"] = r2.y.shift(1)
r2 = r2.dropna()
a_dir = np.sign(r2.y - r2["prev"])
p_dir = np.sign(r2.yhat - r2["prev"])
v = a_dir != 0
dir_acc = float((a_dir[v] == p_dir[v]).mean() * 100)
majority = float(max((a_dir[v] > 0).mean(), (a_dir[v] < 0).mean()) * 100)

# interval calibration
coverage = float(((res.y >= res.yhat_lower) & (res.y <= res.yhat_upper)).mean() * 100)

print(f"\n{'':12}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}")
print(f"{'Prophet':12}{mae:>10.2f}{rmse:>10.2f}{mape:>9.2f}%")
print(f"{'Naive':12}{n_mae:>10.2f}{n_rmse:>10.2f}{n_mape:>9.2f}%")
print(f"\nDirection accuracy : {dir_acc:.1f}%   (majority baseline {majority:.1f}%)")
print(f"{int(CONFIG['interval_width']*100)}% interval coverage: {coverage:.1f}%  (should be ~{int(CONFIG['interval_width']*100)}%)")

if run:
    run["metrics/prophet/mae"] = mae
    run["metrics/prophet/rmse"] = rmse
    run["metrics/prophet/mape"] = mape
    run["metrics/naive/mae"] = n_mae
    run["metrics/naive/rmse"] = n_rmse
    run["metrics/naive/mape"] = n_mape
    run["metrics/prophet/direction_accuracy"] = dir_acc
    run["metrics/baseline/majority_class"] = majority
    run["metrics/prophet/interval_coverage"] = coverage
    run["metrics/prophet_vs_naive_mae_ratio"] = mae / n_mae
    run["conclusion/beats_naive_mae"] = bool(mae < n_mae)
    run["conclusion/beats_majority_direction"] = bool(dir_acc > majority)

# ----------------------------------------------------------------------------
# 5. CHANGEPOINTS — structural breaks Prophet found
# ----------------------------------------------------------------------------
deltas = np.nanmean(m.params["delta"], axis=0)
cps = [str(d.date()) for d, dl in zip(m.changepoints, deltas) if abs(dl) > 0.01]
print(f"\nChangepoints detected: {len(cps)}")
for c in cps:
    print("   ", c)
if run:
    run["changepoints/count"] = len(cps)
    run["changepoints/dates"] = ", ".join(cps)

# ----------------------------------------------------------------------------
# 6. PLOTS -> Neptune
# ----------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig1 = m.plot(fcst)
    plt.title(f"Prophet forecast — {CONFIG['ticker']}")
    fig1.savefig("prophet_forecast.png", dpi=110, bbox_inches="tight")

    fig2 = m.plot_components(fcst)
    fig2.savefig("prophet_components.png", dpi=110, bbox_inches="tight")

    fig3, ax = plt.subplots(figsize=(12, 5))
    ax.plot(res.ds, res.y, label="Actual", lw=1.3)
    ax.plot(res.ds, res.yhat, label="Prophet", lw=1.3)
    ax.fill_between(res.ds, res.yhat_lower, res.yhat_upper, alpha=0.2, label="80% interval")
    ax.set_title(f"{CONFIG['ticker']} — test period: Prophet vs actual")
    ax.legend()
    fig3.savefig("prophet_vs_actual.png", dpi=110, bbox_inches="tight")

    if run:
        run["plots/forecast"].upload("prophet_forecast.png")
        run["plots/components"].upload("prophet_components.png")
        run["plots/vs_actual"].upload("prophet_vs_actual.png")
    print("\nplots saved")
except Exception as e:
    print(f"plotting skipped: {e}")

# ----------------------------------------------------------------------------
# 7. SAVE + CLOSE
# ----------------------------------------------------------------------------
res.to_csv("prophet_forecast_output.csv", index=False)
if run:
    run["artifacts/forecast_csv"].upload("prophet_forecast_output.csv")
    run.stop()
    print("Neptune run closed — check your dashboard")

print("\nDone.")
