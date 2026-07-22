"""
Experiment 3 — master comparison across all six models.

Assembles one table with Naive, ARIMA, LSTM, Hybrid, XGBoost, GRU under BOTH
protocols (one-step and multi-step), pulling ARIMA/LSTM/Hybrid/Naive from the
Experiment 1 & 2 result files and adding the two new Experiment 3 rows. Draws a
two-panel overlay (one-step on top, multi-step below) and prints an honest conclusion.

Artifacts:
  comparison_all_models.csv     the master table
  comparison_overlay_all.png    actual vs. all models, both protocols
"""

import json
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
DATA = _ROOT / "data" / "processed"
PRED = _ROOT / "results" / "predictions"
FIG = _ROOT / "results" / "figures"
TAB = _ROOT / "results" / "tables"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(fn):
    with open(PRED / fn) as f:
        return json.load(f)


# ---------------------------------------------------------------- metrics table
naive = load("naive_baseline_info.json")
arima1, arima2 = load("arima_info.json"), load("ms_arima_info.json")
lstm1, lstm2 = load("lstm_info.json"), load("ms_lstm_info.json")
hyb1, hyb2 = load("hybrid_info.json"), load("ms_hybrid_info.json")
xgb = load("xgb_info.json")
gru = load("gru_info.json")["summary_mean_std"]


def row(model, one, multi):
    return {"Model": model,
            "1s_MAE": one[0], "1s_MAPE": one[1], "1s_RMSE": one[2],
            "ms_MAE": multi[0], "ms_MAPE": multi[1], "ms_RMSE": multi[2]}


rows = [
    row("Naive",
        (naive["one_step_naive"]["MAE"], naive["one_step_naive"]["MAPE"], naive["one_step_naive"]["RMSE"]),
        (naive["multi_step_naive"]["MAE"], naive["multi_step_naive"]["MAPE"], naive["multi_step_naive"]["RMSE"])),
    row("ARIMA",
        (arima1["test_MAE"], arima1["test_MAPE"], arima1["test_RMSE"]),
        (arima2["test_MAE"], arima2["test_MAPE"], arima2["test_RMSE"])),
    row("LSTM",
        (lstm1["test_MAE"], lstm1["test_MAPE"], lstm1["test_RMSE"]),
        (lstm2["test_MAE"], lstm2["test_MAPE"], lstm2["test_RMSE"])),
    row("Hybrid",
        (hyb1["test_MAE"], hyb1["test_MAPE"], hyb1["test_RMSE"]),
        (hyb2["test_MAE"], hyb2["test_MAPE"], hyb2["test_RMSE"])),
    row("XGBoost",
        (xgb["onestep"]["MAE"], xgb["onestep"]["MAPE"], xgb["onestep"]["RMSE"]),
        (xgb["multistep"]["MAE"], xgb["multistep"]["MAPE"], xgb["multistep"]["RMSE"])),
    row("GRU",
        (gru["onestep"]["MAE"]["mean"], gru["onestep"]["MAPE"]["mean"], gru["onestep"]["RMSE"]["mean"]),
        (gru["multistep"]["MAE"]["mean"], gru["multistep"]["MAPE"]["mean"], gru["multistep"]["RMSE"]["mean"])),
]
table = pd.DataFrame(rows).set_index("Model")
table.round(4).to_csv(TAB / "comparison_all_models.csv")

print("=" * 84)
print("MASTER COMPARISON — all six models, both protocols (test window 533 days)")
print("=" * 84)
print(table.round(3).to_string())

# ---------------------------------------------------------------- overlay plot
df = pd.read_csv(DATA / "spsl20_trading_days_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
split = json.load(open(PRED / "split_info.json"))["split_index"]
n = len(df)
vals = df["spsl20_points"].astype(float).values

a1 = pd.read_csv(PRED / "arima_predictions.csv", parse_dates=["date"])
dates = a1["date"]
actual = a1["actual"].values
l1 = pd.read_csv(PRED / "lstm_predictions.csv")
h1 = pd.read_csv(PRED / "hybrid_predictions.csv")
x1 = pd.read_csv(PRED / "xgb_predictions_onestep.csv")
g1 = pd.read_csv(PRED / "gru_predictions_onestep.csv")
naive1 = vals[split - 1:n - 1]

a2 = pd.read_csv(PRED / "ms_arima_predictions.csv")
l2 = pd.read_csv(PRED / "ms_lstm_predictions.csv")
h2 = pd.read_csv(PRED / "ms_hybrid_predictions.csv")
x2 = pd.read_csv(PRED / "xgb_predictions_multistep.csv")
g2 = pd.read_csv(PRED / "gru_predictions_multistep.csv")
naive2 = np.full(n - split, vals[split - 1])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

for ax, title, series in [
    (ax1, "One-step-ahead (easy mode — sees the true previous day)", [
        ("ARIMA", a1["arima_pred"].values, "tab:blue"),
        ("LSTM", l1["lstm_pred"].values, "tab:green"),
        ("Hybrid", h1["hybrid_pred"].values, "tab:red"),
        ("XGBoost", x1["xgb_onestep_pred"].values, "tab:purple"),
        ("GRU", g1["gru_onestep_pred"].values, "tab:orange"),
        ("Naive", naive1, "gray"),
    ]),
    (ax2, "Multi-step (hard mode — rolls forward on its own predictions)", [
        ("ARIMA", a2["ms_arima_pred"].values, "tab:blue"),
        ("LSTM", l2["ms_lstm_pred"].values, "tab:green"),
        ("Hybrid", h2["ms_hybrid_pred"].values, "tab:red"),
        ("XGBoost", x2["xgb_multistep_pred"].values, "tab:purple"),
        ("GRU", g2["gru_multistep_pred"].values, "tab:orange"),
        ("Naive", naive2, "gray"),
    ]),
]:
    ax.plot(dates, actual, label="Actual", color="black", linewidth=2.0)
    for name, ser, col in series:
        ls = "--" if name == "Naive" else "-"
        ax.plot(dates, ser, label=name, color=col, linewidth=1.1, alpha=0.85, linestyle=ls)
    ax.set_title(title)
    ax.set_ylabel("S&P SL 20 (points)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)

ax2.set_xlabel("Date")
fig.suptitle("S&P SL 20 — Actual vs. all models (ARIMA, LSTM, Hybrid, XGBoost, GRU, Naive)",
             fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "comparison_overlay_all.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- conclusion
def mape(model, proto):
    return table.loc[model, f"{proto}_MAPE"]


print("\n" + "-" * 84)
print("CONCLUSION")
print("-" * 84)
print(
    f"Does XGBoost (a different ML family) beat ARIMA?  NO.\n"
    f"   One-step  MAPE: XGBoost {mape('XGBoost','1s'):.2f}%  vs  ARIMA {mape('ARIMA','1s'):.2f}%.\n"
    f"   Multi-step MAPE: XGBoost {mape('XGBoost','ms'):.2f}%  vs  ARIMA {mape('ARIMA','ms'):.2f}%.\n"
    f"   XGBoost is decent one-step (better than the LSTM) but worst multi-step: trees cannot\n"
    f"   extrapolate above the training range, so the recursive forecast saturates.\n"
)
print(
    f"Does GRU match or beat LSTM?  ROUGHLY MATCHES.\n"
    f"   One-step  MAPE: GRU {mape('GRU','1s'):.2f}%  vs  LSTM {mape('LSTM','1s'):.2f}%  (within seed noise).\n"
    f"   Multi-step MAPE: GRU {mape('GRU','ms'):.2f}%  vs  LSTM {mape('LSTM','ms'):.2f}%  (GRU slightly better).\n"
    f"   The lighter GRU is on par with the LSTM here — no advantage to the heavier cell.\n"
)
print(
    f"Does either beat the naive baseline?\n"
    f"   One-step  naive MAPE {mape('Naive','1s'):.2f}%: NO ML model beats it "
    f"(XGBoost {mape('XGBoost','1s'):.2f}%, GRU {mape('GRU','1s'):.2f}%); only ARIMA/Hybrid tie it.\n"
    f"   Multi-step naive MAPE {mape('Naive','ms'):.2f}%: NO model beats it either — it is the best\n"
    f"   multi-step score in the whole table.\n"
)
print(
    "Honest framing: one-step numbers flatter every model because tomorrow ~ today. The\n"
    "multi-step column is the one that reflects real forecasting skill, and there NONE of the\n"
    "six models beats a flat line. Adding XGBoost and GRU broadens the comparison but does not\n"
    "change the verdict: this index is not reliably forecastable multiple steps ahead, and the\n"
    "2020-2023 crisis in the training window is a key reason the models carry no usable trend."
)
print("\nSaved: comparison_all_models.csv, comparison_overlay_all.png")
