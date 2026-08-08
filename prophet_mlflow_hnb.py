#!/usr/bin/env python3
"""
Prophet baseline on CSE stocks, with MLflow experiment tracking.

WHY MLFLOW AND NOT NEPTUNE:
    Neptune.ai was acquired by OpenAI (announced 3 Dec 2025). The hosted
    service was shut down on 5 March 2026 - app, API and data deleted.
    It is no longer possible to sign up. MLflow is the open-source
    equivalent: free, runs locally, no account, no shutdown risk.

SETUP (one time):
    pip install mlflow prophet pandas numpy matplotlib

RUN:
    python prophet_mlflow_hnb.py

VIEW RESULTS:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    then open http://localhost:5000
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
import mlflow

# ---------------------------------------------------------------------------
# CONFIG - change ticker/path and re-run to add another experiment
# ---------------------------------------------------------------------------
CONFIG = {
    "ticker": "HNB.N0000",
    "data_path": "cleaned_data/HNB_daily_clean.csv",   # <-- adjust
    "start_date": "2012-01-01",
    "train_frac": 0.85,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "weekly_seasonality": True,
    "yearly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": 0.80,
}

# MLflow 3.x deprecates the plain file store, so prefer the sqlite backend.
# Falls back to the file store if sqlalchemy is unavailable (mlflow-skinny).
try:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("CSE-FYP-baselines")
    print("MLflow backend: sqlite:///mlflow.db")
except Exception:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CSE-FYP-baselines")
    print("MLflow backend: file:./mlruns  (install full 'mlflow' for sqlite)")

with mlflow.start_run(run_name=f"prophet-{CONFIG['ticker']}"):

    mlflow.set_tags({
        "model": "prophet",
        "ticker": CONFIG["ticker"],
        "task": "price-level-regression",
        "role": "baseline",
    })
    mlflow.log_params(CONFIG)

    # -----------------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------------
    df = pd.read_csv(CONFIG["data_path"], parse_dates=["date"])
    df = df[df.date >= CONFIG["start_date"]][["date", "close"]]
    df = df.rename(columns={"date": "ds", "close": "y"}).reset_index(drop=True)

    cut = int(len(df) * CONFIG["train_frac"])
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    mlflow.log_params({
        "n_rows": len(df),
        "n_train": len(train),
        "n_test": len(test),
        "train_start": str(train.ds.min().date()),
        "train_end": str(train.ds.max().date()),
        "test_start": str(test.ds.min().date()),
        "test_end": str(test.ds.max().date()),
    })
    print(f"rows={len(df)}  train={len(train)}  test={len(test)}")

    # -----------------------------------------------------------------------
    # FIT
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # METRICS: Prophet vs NAIVE baseline
    # -----------------------------------------------------------------------
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

    # DIRECTION accuracy - the metric this project actually uses
    r2 = res.copy()
    r2["prev"] = r2.y.shift(1)
    r2 = r2.dropna()
    a_dir = np.sign(r2.y - r2["prev"])
    p_dir = np.sign(r2.yhat - r2["prev"])
    v = a_dir != 0
    dir_acc = float((a_dir[v] == p_dir[v]).mean() * 100)
    majority = float(max((a_dir[v] > 0).mean(), (a_dir[v] < 0).mean()) * 100)

    # interval calibration - does the 80% band actually contain 80%?
    coverage = float(((res.y >= res.yhat_lower) & (res.y <= res.yhat_upper)).mean() * 100)

    mlflow.log_metrics({
        "prophet_mae": mae,
        "prophet_rmse": rmse,
        "prophet_mape": mape,
        "naive_mae": n_mae,
        "naive_rmse": n_rmse,
        "naive_mape": n_mape,
        "prophet_direction_accuracy": dir_acc,
        "majority_class_baseline": majority,
        "interval_coverage_pct": coverage,
        "prophet_vs_naive_mae_ratio": mae / n_mae,
    })

    print(f"\n{'':12}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}")
    print(f"{'Prophet':12}{mae:>10.2f}{rmse:>10.2f}{mape:>9.2f}%")
    print(f"{'Naive':12}{n_mae:>10.2f}{n_rmse:>10.2f}{n_mape:>9.2f}%")
    print(f"\nDirection accuracy : {dir_acc:.1f}%  (majority baseline {majority:.1f}%)")
    print(f"80% interval coverage: {coverage:.1f}%  (well-calibrated = ~80%)")

    # -----------------------------------------------------------------------
    # CHANGEPOINTS
    # -----------------------------------------------------------------------
    deltas = np.nanmean(m.params["delta"], axis=0)
    cps = [str(d.date()) for d, dl in zip(m.changepoints, deltas) if abs(dl) > 0.01]
    mlflow.log_metric("n_changepoints", len(cps))
    with open("changepoints.txt", "w") as f:
        f.write("\n".join(cps))
    mlflow.log_artifact("changepoints.txt")
    print(f"\nChangepoints detected: {len(cps)}")
    for c in cps:
        print("   ", c)

    # -----------------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        f1 = m.plot(fcst)
        plt.title(f"Prophet forecast - {CONFIG['ticker']}")
        f1.savefig("prophet_forecast.png", dpi=110, bbox_inches="tight")

        f2 = m.plot_components(fcst)
        f2.savefig("prophet_components.png", dpi=110, bbox_inches="tight")

        f3, ax = plt.subplots(figsize=(12, 5))
        ax.plot(res.ds, res.y, label="Actual", lw=1.3)
        ax.plot(res.ds, res.yhat, label="Prophet", lw=1.3)
        ax.fill_between(res.ds, res.yhat_lower, res.yhat_upper, alpha=0.2,
                        label="80% interval")
        ax.set_title(f"{CONFIG['ticker']} - test period: Prophet vs actual")
        ax.legend()
        f3.savefig("prophet_vs_actual.png", dpi=110, bbox_inches="tight")

        for p in ["prophet_forecast.png", "prophet_components.png",
                  "prophet_vs_actual.png"]:
            mlflow.log_artifact(p)
        print("plots logged")
    except Exception as e:
        print(f"plotting skipped: {e}")

    res.to_csv("prophet_forecast_output.csv", index=False)
    mlflow.log_artifact("prophet_forecast_output.csv")

print("\nDone. View results with:")
print("   mlflow ui --backend-store-uri sqlite:///mlflow.db")
