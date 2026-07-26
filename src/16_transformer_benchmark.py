"""
Transformer benchmark — Informer, Autoformer, PatchTST, TFT (+ Naive baseline).

Univariate S&P SL 20, across 80/20, 50/50, 40/60 splits, under TWO protocols:
  A (whole test window, recursive): forecast 60-day chunks, feed own predictions back,
    roll to the end of the test window. Comparable to the ARIMA/LSTM multi-step results.
  B (fixed 60-day horizon, rolling): forecast 60 days ahead from real history at each
    cutoff across the test window. The standard, fair benchmark for these models.

Runs in the .venv-tf environment (neuralforecast). Uses the M4 GPU (MPS). Does not
touch any classical-model outputs.

Artifacts:
  results/tables/transformer_benchmark_AB.csv
  results/predictions/transformer_preds_<split>.csv
"""
import time, warnings, logging, json
from pathlib import Path as _Path
warnings.filterwarnings("ignore")
for _lg in ("lightning", "pytorch_lightning", "lightning.pytorch"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import Informer, Autoformer, PatchTST, TFT

_ROOT = _Path(__file__).resolve().parents[1]
DATA = _ROOT / "data" / "processed"
PRED = _ROOT / "results" / "predictions"
TAB = _ROOT / "results" / "tables"
for _d in (PRED, TAB):
    _d.mkdir(parents=True, exist_ok=True)

H, INPUT, STEPS, SEED = 60, 104, 200, 42
SPLITS = {"80/20": 0.8, "50/50": 0.5, "40/60": 0.4}
# Autoformer dropped from the fill-in run: ~24 min/split on CPU (MPS aborts) and weakest.
# Its already-computed 80/20 & 50/50 rows are preserved from the existing CSV.
MODELS = {"Informer": Informer, "PatchTST": PatchTST, "TFT": TFT}
# Autoformer's FFT autocorrelation aborts on the MPS backend -> run it on CPU.
ACC_MAP = {"Informer": "mps", "PatchTST": "mps", "TFT": "mps", "Autoformer": "cpu"}


def common(mname):
    return dict(h=H, input_size=INPUT, max_steps=STEPS, scaler_type="standard",
                random_seed=SEED, accelerator=ACC_MAP[mname], enable_progress_bar=False,
                logger=False, enable_checkpointing=False)


def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return (float(np.mean(np.abs(a - p))), float(np.mean(np.abs((a - p) / a)) * 100),
            float(np.sqrt(np.mean((a - p) ** 2))))


df = pd.read_csv(DATA / "spsl20_trading_days_clean.csv").sort_values("date").reset_index(drop=True)
prices = df["spsl20_points"].astype(float).values
n = len(prices)
nfdf_full = pd.DataFrame({"unique_id": "spsl20", "ds": np.arange(n), "y": prices})

# Resume: preload any already-computed rows and skip those (split, model) combos.
_exp = TAB / "transformer_benchmark_AB.csv"
if _exp.exists():
    rows = pd.read_csv(_exp).to_dict("records")
    done = {(r["split"], r["model"]) for r in rows}
    print(f"Resuming — {len(done)} (split,model) combos already done.", flush=True)
else:
    rows, done = [], set()

t_start = time.time()
for sname, frac in SPLITS.items():
    split = int(n * frac)
    test = prices[split:]
    tlen = len(test)
    train_df = nfdf_full.iloc[:split].copy()
    cutoffs = list(range(split, n, H))
    print(f"\n===== SPLIT {sname}  train={split} test={tlen} =====", flush=True)

    # -------- Naive baselines
    if (sname, "Naive") not in done:
        naive_A = np.full(tlen, prices[split - 1])             # flat at last train value
        nb, ab = [], []
        for c in cutoffs:
            end = min(c + H, n)
            nb.extend([prices[c - 1]] * (end - c)); ab.extend(prices[c:end])
        for proto, a, p in [("A_wholewindow", test, naive_A), ("B_60day", ab, nb)]:
            mae, mape, rmse = metrics(a, p)
            rows.append({"split": sname, "model": "Naive", "protocol": proto,
                         "MAE": mae, "MAPE": mape, "RMSE": rmse})

    preds_out = {"date": df["date"].values[split:], "actual": test}
    # -------- Transformers (each wrapped so one crash can't kill the run)
    for mname, Cls in MODELS.items():
        if (sname, mname) in done:
            continue
        t0 = time.time()
        try:
            nf = NeuralForecast(models=[Cls(**common(mname))], freq=1)
            nf.fit(df=train_df)

            # A: recursive whole-window
            grown = train_df.copy(); preds = []
            while len(preds) < tlen:
                fc = nf.predict(df=grown)[mname].values
                preds.extend(fc.tolist())
                nd = np.arange(grown["ds"].iloc[-1] + 1, grown["ds"].iloc[-1] + 1 + len(fc))
                grown = pd.concat([grown, pd.DataFrame({"unique_id": "spsl20", "ds": nd, "y": fc})],
                                  ignore_index=True)
            preds_A = np.array(preds[:tlen])

            # B: fixed 60-day rolling from real history
            pB, aB = [], []
            for c in cutoffs:
                end = min(c + H, n); hh = end - c
                fc = nf.predict(df=nfdf_full.iloc[:c])[mname].values[:hh]
                pB.extend(fc.tolist()); aB.extend(prices[c:end])

            mA = metrics(test, preds_A); mB = metrics(aB, pB)
            rows.append({"split": sname, "model": mname, "protocol": "A_wholewindow",
                         "MAE": mA[0], "MAPE": mA[1], "RMSE": mA[2]})
            rows.append({"split": sname, "model": mname, "protocol": "B_60day",
                         "MAE": mB[0], "MAPE": mB[1], "RMSE": mB[2]})
            preds_out[f"{mname}_A"] = preds_A
            print(f"  {mname:<11}[{ACC_MAP[mname]}] {time.time()-t0:5.0f}s  |  "
                  f"A(whole) MAPE {mA[1]:6.2f}%  |  B(60d) MAPE {mB[1]:6.2f}%", flush=True)
        except Exception as e:
            print(f"  {mname:<11}[{ACC_MAP[mname]}] FAILED: {str(e)[:120]}", flush=True)
        pd.DataFrame(rows).to_csv(TAB / "transformer_benchmark_AB.csv", index=False)  # save per model

    pd.DataFrame(preds_out).to_csv(PRED / f"transformer_preds_{sname.replace('/','')}.csv", index=False)

print(f"\nDONE in {(time.time()-t_start)/60:.1f} min. Saved transformer_benchmark_AB.csv", flush=True)
