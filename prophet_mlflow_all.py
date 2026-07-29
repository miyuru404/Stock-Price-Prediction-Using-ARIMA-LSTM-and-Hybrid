#!/usr/bin/env python3
"""
Prophet baseline on CSE stocks with MLflow tracking — refactored to loop over tickers.

Usage:
    python prophet_mlflow_all.py                 # run all 10 companies
    python prophet_mlflow_all.py HNB             # run one
    python prophet_mlflow_all.py COMB SAMP JKH   # run several

Each ticker becomes one MLflow run in the "CSE-FYP-baselines" experiment, so they
can be compared side by side in the UI.

VIEW RESULTS:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    then open http://localhost:5000

Prophet is meant to perform badly here — it's a baseline documenting that CSE prices
are unpredictable at the price level. Not tuned to look good.
"""
import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)

from prophet import Prophet
import mlflow

ALL_TICKERS = ["HNB", "COMB", "SAMP", "JKH", "LOLC", "LOFC", "DIAL", "MELS", "CTC", "DIST"]

BASE_CONFIG = {
    "start_date": "2012-01-01",
    "train_frac": 0.85,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "weekly_seasonality": True,
    "yearly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": 0.80,
}

# MLflow backend (sqlite preferred; file store fallback for mlflow-skinny)
try:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("CSE-FYP-baselines")
    print("MLflow backend: sqlite:///mlflow.db")
except Exception:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CSE-FYP-baselines")
    print("MLflow backend: file:./mlruns")


def run_ticker(sym):
    data_path = f"cleaned_data/{sym}_daily_clean.csv"
    if not os.path.exists(data_path):
        print(f"[{sym}] SKIP — {data_path} not found")
        return
    cfg = dict(BASE_CONFIG)
    raw = pd.read_csv(data_path, parse_dates=["date"])
    ticker_label = str(raw["ticker"].iloc[0]) if "ticker" in raw.columns else sym
    cfg["ticker"] = ticker_label
    cfg["data_path"] = data_path

    print(f"\n{'='*60}\n{sym}  ({ticker_label})\n{'='*60}")
    with mlflow.start_run(run_name=f"prophet-{sym}"):
        mlflow.set_tags({"model": "prophet", "ticker": ticker_label,
                         "symbol": sym, "task": "price-level-regression", "role": "baseline"})
        mlflow.log_params(cfg)

        df = raw[raw.date >= cfg["start_date"]][["date", "close"]]
        df = df.rename(columns={"date": "ds", "close": "y"}).reset_index(drop=True)
        cut = int(len(df) * cfg["train_frac"])
        train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
        mlflow.log_params({"n_rows": len(df), "n_train": len(train), "n_test": len(test),
                           "train_start": str(train.ds.min().date()), "train_end": str(train.ds.max().date()),
                           "test_start": str(test.ds.min().date()), "test_end": str(test.ds.max().date())})
        print(f"rows={len(df)}  train={len(train)}  test={len(test)}")

        m = Prophet(changepoint_prior_scale=cfg["changepoint_prior_scale"],
                    seasonality_prior_scale=cfg["seasonality_prior_scale"],
                    weekly_seasonality=cfg["weekly_seasonality"],
                    yearly_seasonality=cfg["yearly_seasonality"],
                    daily_seasonality=cfg["daily_seasonality"],
                    interval_width=cfg["interval_width"])
        m.fit(train)
        future = m.make_future_dataframe(periods=len(test), freq="D")
        fcst = m.predict(future)
        pred = fcst.set_index("ds").reindex(test.ds)[["yhat", "yhat_lower", "yhat_upper"]]
        res = test.reset_index(drop=True).join(pred.reset_index(drop=True)).dropna()

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

        r2 = res.copy(); r2["prev"] = r2.y.shift(1); r2 = r2.dropna()
        a_dir = np.sign(r2.y - r2["prev"]); p_dir = np.sign(r2.yhat - r2["prev"])
        v = a_dir != 0
        dir_acc = float((a_dir[v] == p_dir[v]).mean() * 100)
        majority = float(max((a_dir[v] > 0).mean(), (a_dir[v] < 0).mean()) * 100)
        coverage = float(((res.y >= res.yhat_lower) & (res.y <= res.yhat_upper)).mean() * 100)

        mlflow.log_metrics({"prophet_mae": mae, "prophet_rmse": rmse, "prophet_mape": mape,
                            "naive_mae": n_mae, "naive_rmse": n_rmse, "naive_mape": n_mape,
                            "prophet_direction_accuracy": dir_acc, "majority_class_baseline": majority,
                            "interval_coverage_pct": coverage, "prophet_vs_naive_mae_ratio": mae / n_mae})
        print(f"{'':12}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}")
        print(f"{'Prophet':12}{mae:>10.2f}{rmse:>10.2f}{mape:>9.2f}%")
        print(f"{'Naive':12}{n_mae:>10.2f}{n_rmse:>10.2f}{n_mape:>9.2f}%")
        print(f"Direction accuracy : {dir_acc:.1f}%  (majority baseline {majority:.1f}%)")
        print(f"80% interval coverage: {coverage:.1f}%")

        deltas = np.nanmean(m.params["delta"], axis=0)
        cps = [str(d.date()) for d, dl in zip(m.changepoints, deltas) if abs(dl) > 0.01]
        mlflow.log_metric("n_changepoints", len(cps))
        cp_file = f"{sym}_changepoints.txt"
        with open(cp_file, "w") as f:
            f.write("\n".join(cps))
        mlflow.log_artifact(cp_file)
        print(f"Changepoints detected: {len(cps)}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            f3, ax = plt.subplots(figsize=(12, 5))
            ax.plot(res.ds, res.y, label="Actual", lw=1.3)
            ax.plot(res.ds, res.yhat, label="Prophet", lw=1.3)
            ax.fill_between(res.ds, res.yhat_lower, res.yhat_upper, alpha=0.2, label="80% interval")
            ax.set_title(f"{ticker_label} — test period: Prophet vs actual"); ax.legend()
            png = f"{sym}_prophet_vs_actual.png"
            f3.savefig(png, dpi=110, bbox_inches="tight"); plt.close("all")
            mlflow.log_artifact(png)
        except Exception as e:
            print(f"plotting skipped: {e}")

        out = f"{sym}_prophet_forecast_output.csv"
        res.to_csv(out, index=False); mlflow.log_artifact(out)
    return {"sym": sym, "prophet_mae": mae, "naive_mae": n_mae,
            "dir_acc": dir_acc, "majority": majority, "coverage": coverage, "n_cps": len(cps)}


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] or ALL_TICKERS
    summary = []
    for t in tickers:
        r = run_ticker(t)
        if r:
            summary.append(r)
    if len(summary) > 1:
        print(f"\n{'='*72}\nSUMMARY (all runs)\n{'='*72}")
        s = pd.DataFrame(summary)
        print(s.to_string(index=False))
    print("\nDone. View with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
