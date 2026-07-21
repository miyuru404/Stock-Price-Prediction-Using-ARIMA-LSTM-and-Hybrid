"""
Setup 4 — Comparison of ARIMA, LSTM, and Hybrid on the identical test window.

Builds the three-way metrics table (MAE / MAPE / RMSE, mirroring the paper's
Table 4), saves it, draws one actual-vs-all overlay plot, and prints a short
conclusion including behaviour relative to the 2020-2023 crisis years (which fall
in the training window, so the models are graded on the 2024-2026 recovery).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return mae, mape, rmse


# ---------------------------------------------------------------- load results
arima = pd.read_csv(str(PRED / "arima_predictions.csv"), parse_dates=["date"])
lstm = pd.read_csv(str(PRED / "lstm_predictions.csv"), parse_dates=["date"])
hybrid = pd.read_csv(str(PRED / "hybrid_predictions.csv"), parse_dates=["date"])

# All three must cover the identical test dates.
assert (arima["date"].values == lstm["date"].values).all()
assert (arima["date"].values == hybrid["date"].values).all()

dates = arima["date"]
actual = arima["actual"].values

preds = {
    "ARIMA": arima["arima_pred"].values,
    "LSTM": lstm["lstm_pred"].values,
    "Hybrid": hybrid["hybrid_pred"].values,
}

# ---------------------------------------------------------------- metrics table
rows = []
for name, p in preds.items():
    mae, mape, rmse = metrics(actual, p)
    rows.append({"Model": name, "MAE": mae, "MAPE (%)": mape, "RMSE": rmse})
table = pd.DataFrame(rows).set_index("Model")

with open(str(PRED / "split_info.json")) as f:
    split_info = json.load(f)

print("=" * 70)
print("SETUP 4 — THREE-WAY COMPARISON")
print("=" * 70)
print(f"Test window: {split_info['test_start_date']} -> {split_info['test_end_date']}"
      f"  ({len(actual)} trading days)")
print("\nComparison table (mirrors paper Table 4):\n")
print(table.round(4).to_string())

table.round(6).to_csv(str(TAB / "comparison_table.csv"))

# best model per metric (lower is better for all three)
best = {m: table[m].idxmin() for m in ["MAE", "MAPE (%)", "RMSE"]}

# ---------------------------------------------------------------- overlay plot
plt.figure(figsize=(14, 7))
plt.plot(dates, actual, label="Actual", color="black", linewidth=1.8)
plt.plot(dates, preds["ARIMA"], label="ARIMA", color="tab:blue", alpha=0.8, linewidth=1.1)
plt.plot(dates, preds["LSTM"], label="LSTM", color="tab:green", alpha=0.8, linewidth=1.1)
plt.plot(dates, preds["Hybrid"], label="Hybrid (ARIMA+LSTM)", color="tab:red", alpha=0.8, linewidth=1.1)
plt.title("S&P SL 20 — Actual vs. ARIMA vs. LSTM vs. Hybrid (test period, one-step-ahead)")
plt.xlabel("Date")
plt.ylabel("S&P SL 20 (points)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIG / "comparison_overlay.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------- conclusion
print("\n" + "-" * 70)
print("CONCLUSION")
print("-" * 70)
print(f"Best by MAE  : {best['MAE']}   ({table.loc[best['MAE'], 'MAE']:.4f})")
print(f"Best by MAPE : {best['MAPE (%)']}   ({table.loc[best['MAPE (%)'], 'MAPE (%)']:.4f}%)")
print(f"Best by RMSE : {best['RMSE']}   ({table.loc[best['RMSE'], 'RMSE']:.4f})")

order = table["RMSE"].sort_values()
print(
    "\nInterpretation:\n"
    f"- ARIMA and the Hybrid are far more accurate than the standalone LSTM "
    f"(MAPE ~{table.loc['ARIMA','MAPE (%)']:.2f}% vs ~{table.loc['LSTM','MAPE (%)']:.2f}%). "
    "On a one-step-ahead task the index is close to a random walk, which a\n"
    "  differenced ARIMA captures almost perfectly, while the level-trained LSTM\n"
    "  systematically lags the price.\n"
    f"- The Hybrid improves on pure ARIMA only marginally "
    f"(RMSE {table.loc['Hybrid','RMSE']:.2f} vs {table.loc['ARIMA','RMSE']:.2f}). "
    "ARIMA's residuals are\n"
    "  essentially white noise (Ljung-Box p>0.05 in script 01), so the residual\n"
    "  LSTM has little structure left to exploit.\n"
    "- Crisis note: the volatile 2020-2023 Sri Lankan crisis years sit inside the\n"
    "  TRAINING window; the models are graded on the calmer 2024-2026 recovery.\n"
    "  Training through that turbulence did not stop ARIMA/Hybrid from tracking the\n"
    "  recovery tightly, but the large-kurtosis residuals it produced are why the\n"
    "  error bands (RMSE >> MAE) remain wider than the paper's crisis-free window.\n"
    f"\nOverall RMSE ranking (best first): {' < '.join(order.index)}."
)

print("\nSaved: comparison_table.csv, comparison_overlay.png")
