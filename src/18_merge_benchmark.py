"""
Merge the transformer benchmark with the classical benchmark into one master table.

Reads:
  results/tables/transformer_benchmark_AB.csv   (Informer/Autoformer/PatchTST/TFT + Naive)
  results/tables/classical_benchmark_AB.csv     (Naive/ARIMA/LSTM/Hybrid)

Produces one MAPE table: rows = models, columns = <split> A(whole) / B(60-day),
saved to results/tables/master_transformer_vs_classical.csv and printed.
"""
from pathlib import Path as _Path
import pandas as pd

_ROOT = _Path(__file__).resolve().parents[1]
TAB = _ROOT / "results" / "tables"

tf = pd.read_csv(TAB / "transformer_benchmark_AB.csv")
cl = pd.read_csv(TAB / "classical_benchmark_AB.csv")
both = pd.concat([cl, tf], ignore_index=True).drop_duplicates(
    subset=["split", "model", "protocol"], keep="last")

SPLITS = ["80/20", "50/50", "40/60"]
MODEL_ORDER = ["Naive", "ARIMA", "LSTM", "Hybrid", "Informer", "Autoformer", "PatchTST", "TFT"]
PROTO = {"A_wholewindow": "A_whole", "B_60day": "B_60d"}

# pivot MAPE
piv = both.pivot_table(index="model", columns=["split", "protocol"], values="MAPE")
# order columns split-major, A then B
cols = []
for s in SPLITS:
    for p in ["A_wholewindow", "B_60day"]:
        if (s, p) in piv.columns:
            cols.append((s, p))
piv = piv.reindex(index=[m for m in MODEL_ORDER if m in piv.index], columns=cols)
piv.columns = [f"{s} {PROTO[p]}" for s, p in piv.columns]

piv.round(2).to_csv(TAB / "master_transformer_vs_classical.csv")
print("=" * 88)
print("MASTER COMPARISON — MAPE (%), transformers vs classical, both protocols, 3 splits")
print("  A_whole = forecast the whole test window (recursive)   B_60d = 60-day-ahead rolling")
print("=" * 88)
print(piv.round(2).to_string())
print("\nSaved: results/tables/master_transformer_vs_classical.csv")
