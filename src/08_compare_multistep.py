"""
Experiment 2 — Setup 4: naive baseline + combined one-step vs multi-step comparison.

Computes the persistence/naive baseline (one-step and multi-step), assembles the
combined table (Experiment 1 one-step numbers beside Experiment 2 multi-step numbers
for Naive / ARIMA / LSTM / Hybrid), draws the multi-step overlay plot, and prints the
conclusion vs. the paper's Table 4.

Reads Experiment 1 metrics from their existing *_info.json files (untouched) and
Experiment 2 metrics from the ms_*_info.json files.

Artifacts:
  naive_baseline_info.json          one-step + multi-step naive metrics
  comparison_multistep.csv          combined table
  comparison_overlay_multistep.png  actual vs all multi-step forecasts
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

CSV = str(DATA / "spsl20_trading_days_clean.csv")


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return mae, mape, rmse


def load(fn):
    with open(fn) as f:
        return json.load(f)


# ---------------------------------------------------- data + split for naive
df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
split_info = load(str(PRED / "split_info.json"))
split, n = split_info["split_index"], split_info["n"]
values = df["spsl20_points"].astype(float).values
actual_test = values[split:]

# One-step naive: prediction for day t = actual value of day t-1.
naive_1 = values[split - 1:n - 1]
n1_mae, n1_mape, n1_rmse = metrics(actual_test, naive_1)

# Multi-step naive (flat): every test day = last training value.
naive_ms = np.full_like(actual_test, values[split - 1])
nm_mae, nm_mape, nm_rmse = metrics(actual_test, naive_ms)

with open(str(PRED / "naive_baseline_info.json"), "w") as f:
    json.dump({
        "one_step_naive": {"MAE": n1_mae, "MAPE": n1_mape, "RMSE": n1_rmse,
                           "rule": "pred[t] = actual[t-1]"},
        "multi_step_naive": {"MAE": nm_mae, "MAPE": nm_mape, "RMSE": nm_rmse,
                             "rule": "pred[t] = last training value (flat)"},
    }, f, indent=2)

# --------------------------------------------------------- gather all metrics
# Experiment 1 (one-step) — from untouched Setup 1-3 info files.
e1 = {"ARIMA": load(str(PRED / "arima_info.json")), "LSTM": load(str(PRED / "lstm_info.json")),
      "Hybrid": load(str(PRED / "hybrid_info.json"))}
# Experiment 2 (multi-step).
e2 = {"ARIMA": load(str(PRED / "ms_arima_info.json")), "LSTM": load(str(PRED / "ms_lstm_info.json")),
      "Hybrid": load(str(PRED / "ms_hybrid_info.json"))}

rows = []
rows.append({"Model": "Naive",
             "OneStep_MAE": n1_mae, "OneStep_MAPE": n1_mape, "OneStep_RMSE": n1_rmse,
             "MultiStep_MAE": nm_mae, "MultiStep_MAPE": nm_mape, "MultiStep_RMSE": nm_rmse})
for m in ["ARIMA", "LSTM", "Hybrid"]:
    rows.append({
        "Model": m,
        "OneStep_MAE": e1[m]["test_MAE"], "OneStep_MAPE": e1[m]["test_MAPE"],
        "OneStep_RMSE": e1[m]["test_RMSE"],
        "MultiStep_MAE": e2[m]["test_MAE"], "MultiStep_MAPE": e2[m]["test_MAPE"],
        "MultiStep_RMSE": e2[m]["test_RMSE"],
    })
table = pd.DataFrame(rows).set_index("Model")
table.round(6).to_csv(str(TAB / "comparison_multistep.csv"))

# ------------------------------------------------------------------ report
pd.set_option("display.width", 200)
print("=" * 78)
print("COMBINED COMPARISON — Experiment 1 (one-step) vs Experiment 2 (multi-step)")
print("=" * 78)
print(f"Test window: {split_info['test_start_date']} -> {split_info['test_end_date']}"
      f"  ({len(actual_test)} trading days)\n")
disp = table.copy()
disp.columns = ["1s_MAE", "1s_MAPE", "1s_RMSE", "ms_MAE", "ms_MAPE", "ms_RMSE"]
print(disp.round(3).to_string())

# ------------------------------------------------------------- overlay plot
arima_ms = pd.read_csv(str(PRED / "ms_arima_predictions.csv"), parse_dates=["date"])
lstm_ms = pd.read_csv(str(PRED / "ms_lstm_predictions.csv"), parse_dates=["date"])
hybrid_ms = pd.read_csv(str(PRED / "ms_hybrid_predictions.csv"), parse_dates=["date"])
dates = arima_ms["date"]

plt.figure(figsize=(14, 7))
plt.plot(dates, actual_test, label="Actual", color="black", linewidth=1.8)
plt.plot(dates, arima_ms["ms_arima_pred"], label="ARIMA (multi-step)", color="tab:blue", linewidth=1.2)
plt.plot(dates, lstm_ms["ms_lstm_pred"], label="LSTM (multi-step)", color="tab:green", linewidth=1.2)
plt.plot(dates, hybrid_ms["ms_hybrid_pred"], label="Hybrid (multi-step)", color="tab:red", linewidth=1.2)
plt.plot(dates, naive_ms, label="Naive (flat, last train value)", color="gray", linestyle="--", linewidth=1.2)
plt.title("S&P SL 20 — Actual vs. multi-step forecasts (Experiment 2, paper-style)")
plt.xlabel("Date")
plt.ylabel("S&P SL 20 (points)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(FIG / "comparison_overlay_multistep.png"), dpi=150)
plt.close()

# ------------------------------------------------------------- conclusion
paper = {"ARIMA": (233.96, 7.04, 269.57), "LSTM": (249.37, 6.96, 269.86)}
print("\n" + "-" * 78)
print("CONCLUSION")
print("-" * 78)
print(
    "1) Vs. the paper's Table 4 (ARIMA 233.96/7.04%/269.57, LSTM 249.37/6.96%/269.86):\n"
    f"   Our MULTI-STEP errors are far larger — ARIMA MAPE {e2['ARIMA']['test_MAPE']:.1f}%, "
    f"LSTM {e2['LSTM']['test_MAPE']:.1f}%, Hybrid {e2['Hybrid']['test_MAPE']:.1f}% — even though\n"
    "   the method matches the paper. The reason is the test window, not the models: the\n"
    "   paper's 2017-2018 test period was roughly flat, so its multi-step forecasts (which\n"
    "   collapse toward a flat line) stayed close to the truth. Our 2024-2026 window is a\n"
    "   strong post-crisis bull run from ~3050 to ~6700, so any flat/converging multi-step\n"
    "   forecast is badly wrong. Same method, much harder horizon.\n"
)
print(
    "2) Do the models beat the naive baseline?\n"
    f"   Multi-step naive (flat) : MAE {nm_mae:.1f}  MAPE {nm_mape:.2f}%  RMSE {nm_rmse:.1f}\n"
    f"   Multi-step ARIMA        : MAE {e2['ARIMA']['test_MAE']:.1f}  MAPE {e2['ARIMA']['test_MAPE']:.2f}%  RMSE {e2['ARIMA']['test_RMSE']:.1f}\n"
    f"   Multi-step LSTM         : MAE {e2['LSTM']['test_MAE']:.1f}  MAPE {e2['LSTM']['test_MAPE']:.2f}%  RMSE {e2['LSTM']['test_RMSE']:.1f}\n"
    f"   Multi-step Hybrid       : MAE {e2['Hybrid']['test_MAE']:.1f}  MAPE {e2['Hybrid']['test_MAPE']:.2f}%  RMSE {e2['Hybrid']['test_RMSE']:.1f}\n"
    "   The multi-step models essentially TIE the naive flat line (ARIMA/Hybrid) or do slightly\n"
    "   worse (LSTM). None adds real multi-step skill over 533 days: in a trending market,\n"
    "   forecasting the whole horizon from a single origin is close to hopeless, and 'flat at the\n"
    "   last value' is a hard reference to beat.\n"
)
print(
    "3) Why are the one-step numbers so much lower?\n"
    f"   One-step naive alone gets MAPE {n1_mape:.2f}% — because tomorrow's index is very close to\n"
    "   today's. Experiment 1's models see the real previous day at every step, so they ride that\n"
    "   persistence and post sub-1% MAPE. That is the EASY test. Experiment 2 cuts the model off\n"
    "   from real test values and makes it roll forward on its own guesses (the HARD, honest test\n"
    "   the paper used). The gap between the two is the difference in difficulty, not a bug — and\n"
    "   one-step numbers should never be quoted as if they were multi-step forecasting skill.\n"
)
print(
    "4) Crisis behaviour (2020-2023):\n"
    "   Those crisis years sit in the TRAINING window. They taught the models a volatile, largely\n"
    "   trendless regime, so the fitted ARIMA carries almost no drift term — which is exactly why\n"
    "   its multi-step forecast flattens and then undershoots the calmer but strongly rising\n"
    "   2024-2026 recovery. Training through the crisis made the long-horizon forecast conservative.\n"
)
print(f"Best multi-step model by RMSE: {table['MultiStep_RMSE'].idxmin()} "
      f"({table['MultiStep_RMSE'].min():.1f}).")
print("\nSaved: naive_baseline_info.json, comparison_multistep.csv, comparison_overlay_multistep.png")
