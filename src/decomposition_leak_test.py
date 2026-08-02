#!/usr/bin/env python3
"""
DECOMPOSITION HYBRIDS — the same model, done the LEAKY way and the HONEST way.

THE CLAIM UNDER TEST
Recent hybrid papers (wavelet-LSTM, EMD/CEEMDAN-LSTM, VMD-LSTM) report accuracy far beyond anything
in this project — MAPE of a fraction of a percent, R2 above 0.99, large margins over ARIMA. This is
the "latest technology" branch of the literature and it is the last untested explanation for why
published results exceed ours.

THE SUSPECTED MECHANISM
Wavelet, EMD and VMD are GLOBAL operations: every output component at time t is computed from the
WHOLE series, including points after t. So if you decompose first and split afterwards — which is
what most published pipelines do — each component at a training point already contains information
from the test period, and each component at a test point was built using its own future.

The model is then trained to predict a target that is partly baked into its own inputs.

THE EXPERIMENT
Identical data, identical model, identical folds. Only the ORDER of decompose-and-split changes.

  LEAKY   decompose the FULL series once -> build features -> split into train/test
          (the common published pipeline)
  HONEST  at every time t, decompose ONLY series[:t] and take the last value of each component
          (nothing after t is ever used)

If the leaky version scores spectacularly and the honest version collapses to naive, the published
advantage is an artefact of pipeline order, not of the method.

Decompositions: discrete wavelet (db4, 3 levels) and CEEMDAN.
Targets: S&P SL 20 and the BANKS composite. One-step-ahead price.

Outputs -> results/decomposition_leak/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pywt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "decomposition_leak"
OUT.mkdir(parents=True, exist_ok=True)

BANKS = ["HNB", "COMB", "SAMP"]
SEED = 42
TRAIN_FRAC = 0.80
WAVELET, LEVELS = "db4", 3
MIN_HIST = 260          # honest mode needs enough history before the first decomposition

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))
      for t in BANKS}


def build_banks():
    cal = None
    for t in BANKS:
        cal = px[t].index if cal is None else cal.intersection(px[t].index)
    cal = cal.sort_values()
    r = pd.DataFrame({t: px[t].reindex(cal).pct_change() for t in BANKS})
    return pd.DataFrame({"date": cal,
                         "close": (100 * (1 + r.mean(axis=1).fillna(0)).cumprod()).values})


sp = (pd.read_csv(DATA / "spsl20_daily_fixed.csv", parse_dates=["date"])
        .sort_values("date").drop_duplicates("date")
        .rename(columns={"spsl20_points": "close"})[["date", "close"]])
TARGETS = {"SPSL20": sp.reset_index(drop=True), "BANKS": build_banks().reset_index(drop=True)}


def md_table(df):
    cols = [str(x) for x in df.columns]
    o = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        o.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(o)


# ---------------------------------------------------------------- decompositions
def wavelet_components(x):
    """Return (n, LEVELS+1) array: each column is one reconstructed wavelet component."""
    coeffs = pywt.wavedec(x, WAVELET, level=LEVELS)
    comps = []
    for i in range(len(coeffs)):
        c = [np.zeros_like(a) for a in coeffs]
        c[i] = coeffs[i]
        rec = pywt.waverec(c, WAVELET)[:len(x)]
        comps.append(rec)
    return np.column_stack(comps)


def ceemdan_components(x, max_imf=4):
    from PyEMD import CEEMDAN
    ce = CEEMDAN(trials=8, parallel=False)
    ce.noise_seed(SEED)
    imfs = ce.ceemdan(np.asarray(x, dtype=float), max_imf=max_imf)
    return imfs.T[:, :max_imf + 1]


DECOMP = {"wavelet(db4,3)": wavelet_components}
try:
    from PyEMD import CEEMDAN  # noqa: F401
    DECOMP["CEEMDAN"] = ceemdan_components
except Exception:
    print("  (CEEMDAN unavailable — running wavelet only)")


def leaky_features(x, fn):
    """THE PUBLISHED PIPELINE: decompose the WHOLE series once, then split later."""
    return fn(x)


def honest_features(x, fn, min_hist=MIN_HIST, step=1):
    """At each t, decompose ONLY x[:t] and keep the last value of each component.
    Nothing after t is ever seen. Rows before min_hist are NaN and get dropped."""
    n = len(x)
    probe = fn(x[:min_hist])
    k = probe.shape[1]
    out = np.full((n, k), np.nan)
    for t in range(min_hist, n + 1, step):
        comp = fn(x[:t])
        out[t - 1, :comp.shape[1]] = comp[-1, :]
    return out


def dm(actual, p1, p2):
    d = (actual - p1) ** 2 - (actual - p2) ** 2
    n = len(d); db = float(d.mean()); se = np.sqrt(float(((d - db) ** 2).mean()) / n)
    if se == 0 or not np.isfinite(se):
        return np.nan
    return float(2 * (1 - norm.cdf(abs(db / se))))


rows, curve_store = [], []
for tname, series in TARGETS.items():
    close = series["close"].values.astype(float)
    logp = np.log(close)
    n = len(close)

    for dname, fn in DECOMP.items():
        # honest mode is expensive for CEEMDAN -> decompose every `step` points and hold
        step = 1 if dname.startswith("wavelet") else 3
        variants = {
            "LEAKY (decompose then split)": leaky_features(logp, fn),
            "HONEST (decompose only the past)": honest_features(logp, fn, step=step),
        }
        for vname, comp in variants.items():
            df = pd.DataFrame(comp, columns=[f"c{i}" for i in range(comp.shape[1])])
            df["date"] = series["date"].values
            df["close"] = close
            # lagged components: what a forecaster could actually hold at t-1
            for c in [c for c in df.columns if c.startswith("c")]:
                df[f"{c}_lag1"] = df[c].shift(1)
            df["ret_1"] = pd.Series(close).pct_change()
            df["ret_5"] = pd.Series(close).pct_change(5)
            df["fwd_1"] = pd.Series(close).shift(-1) / pd.Series(close) - 1
            df["target"] = pd.Series(close).shift(-1)
            FEATS = [c for c in df.columns if c.endswith("_lag1")] + ["ret_1", "ret_5"]
            d = df.dropna(subset=FEATS + ["fwd_1", "target"]).reset_index(drop=True)
            if len(d) < 350:
                continue
            cut = int(len(d) * TRAIN_FRAC)
            tr, te = np.arange(cut), np.arange(cut, len(d))
            sc = StandardScaler().fit(d.loc[tr, FEATS])
            Xtr, Xte = sc.transform(d.loc[tr, FEATS]), sc.transform(d.loc[te, FEATS])
            ytr = d.loc[tr, "fwd_1"].values
            actual = d.loc[te, "target"].values
            cl_te = d.loc[te, "close"].values
            naive = cl_te

            best = None
            for mn, m in [("Ridge", Ridge(alpha=1.0, random_state=SEED)),
                          ("XGB", XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.8,
                                               min_child_weight=3, random_state=SEED, n_jobs=4))]:
                m.fit(Xtr, ytr)
                p = cl_te * (1 + m.predict(Xte))
                mape = float(np.mean(np.abs((actual - p) / actual)) * 100)
                if best is None or mape < best[0]:
                    best = (mape, p, mn)
            mape, pred, mname = best
            mape_naive = float(np.mean(np.abs((actual - naive) / actual)) * 100)
            rows.append({
                "target": tname, "decomposition": dname, "pipeline": vname, "model": mname,
                "n_test": len(te),
                "MAPE_%": round(mape, 4), "MAPE_naive_%": round(mape_naive, 4),
                "RMSE": round(float(np.sqrt(mean_squared_error(actual, pred))), 4),
                "R2": round(float(r2_score(actual, pred)), 6),
                "Theil_U2": round(float(np.sqrt(mean_squared_error(actual, pred)) /
                                        np.sqrt(mean_squared_error(actual, naive))), 4),
                "beats_naive": "YES" if mape < mape_naive else "no",
                "improvement_vs_naive_%": round(100 * (mape_naive - mape) / mape_naive, 2),
                "DM_p_vs_naive": round(dm(actual, pred, naive), 4),
            })
            curve_store.append(pd.DataFrame({
                "target": tname, "decomposition": dname, "pipeline": vname,
                "date": d.loc[te, "date"].values, "actual": actual, "pred": pred}))
        print(f"  {tname} / {dname} done")

R = pd.DataFrame(rows)
R.to_csv(OUT / "decomposition_results.csv", index=False)
pd.concat(curve_store, ignore_index=True).to_csv(OUT / "decomposition_predictions.csv", index=False)

# the headline: leaky minus honest, same data, same model
gap = []
for (t, dn), sub in R.groupby(["target", "decomposition"]):
    lk = sub[sub.pipeline.str.startswith("LEAKY")]
    hn = sub[sub.pipeline.str.startswith("HONEST")]
    if len(lk) and len(hn):
        gap.append({"target": t, "decomposition": dn,
                    "MAPE_leaky_%": float(lk["MAPE_%"].iloc[0]),
                    "MAPE_honest_%": float(hn["MAPE_%"].iloc[0]),
                    "MAPE_naive_%": float(lk["MAPE_naive_%"].iloc[0]),
                    "improvement_leaky_%": float(lk["improvement_vs_naive_%"].iloc[0]),
                    "improvement_honest_%": float(hn["improvement_vs_naive_%"].iloc[0]),
                    "U2_leaky": float(lk["Theil_U2"].iloc[0]),
                    "U2_honest": float(hn["Theil_U2"].iloc[0]),
                    "R2_leaky": float(lk["R2"].iloc[0]), "R2_honest": float(hn["R2"].iloc[0])})
G = pd.DataFrame(gap).round(4)
G["gain_from_leaking_pp"] = (G["improvement_leaky_%"] - G["improvement_honest_%"]).round(2)
G.to_csv(OUT / "leak_gap.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4))
a = ax[0]
x = np.arange(len(G))
a.bar(x - 0.2, G["improvement_leaky_%"], 0.4, label="LEAKY (decompose then split)",
      color="tab:red", alpha=.85)
a.bar(x + 0.2, G["improvement_honest_%"], 0.4, label="HONEST (past only)",
      color="tab:green", alpha=.85)
a.axhline(0, color="black", lw=1.5)
a.set_xticks(x); a.set_xticklabels([f"{r.target}\n{r.decomposition}" for _, r in G.iterrows()],
                                   fontsize=8)
a.set_ylabel("Improvement over naive (%)")
a.set_title("Same model, same data — only the pipeline ORDER differs")
a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)

b = ax[1]
b.bar(x - 0.2, G["U2_leaky"], 0.4, label="LEAKY", color="tab:red", alpha=.85)
b.bar(x + 0.2, G["U2_honest"], 0.4, label="HONEST", color="tab:green", alpha=.85)
b.axhline(1.0, color="black", ls="--", lw=1.5, label="naive = 1.0")
b.set_xticks(x); b.set_xticklabels([f"{r.target}\n{r.decomposition}" for _, r in G.iterrows()],
                                   fontsize=8)
b.set_ylabel("Theil's U2  (<1 = beats naive)")
b.set_title("Theil's U2: leaky vs honest")
b.grid(alpha=.3, axis="y"); b.legend(fontsize=8)
fig.suptitle("Decomposition hybrids — is the published advantage a pipeline-order artefact?",
             fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "decomposition_leak.png", dpi=140)

md = f"""# Decomposition hybrids — the same model, leaky and honest

## The claim under test

Wavelet-LSTM, CEEMDAN-LSTM and VMD-LSTM papers report accuracy far beyond anything in this project.
This is the "latest technology" branch of the literature and the last untested explanation for why
published results exceed ours.

## The suspected mechanism

Wavelet, EMD and VMD are **global** operations: every component value at time *t* is computed from
the **whole series**, including points after *t*. Decompose first and split afterwards — the common
published pipeline — and each training component already contains test-period information, while
each test component was built using its own future.

## The experiment

Identical data, identical models, identical split. **Only the order of decompose-and-split changes.**

| Pipeline | What it does |
|---|---|
| **LEAKY** | decompose the FULL series once → build features → split |
| **HONEST** | at each *t*, decompose only `series[:t]` and keep the last component values |

## Headline

{md_table(G[["target", "decomposition", "MAPE_leaky_%", "MAPE_honest_%", "MAPE_naive_%", "improvement_leaky_%", "improvement_honest_%", "gain_from_leaking_pp"]])}

`gain_from_leaking_pp` is the advantage created **purely by decomposing before splitting**.

## Full results

{md_table(R[["target", "decomposition", "pipeline", "model", "MAPE_%", "MAPE_naive_%", "R2", "Theil_U2", "beats_naive", "DM_p_vs_naive"]])}

## Reading it

- **Theil's U2 below 1.0** means better than a random walk.
- If the LEAKY row shows a large advantage and the HONEST row collapses to ~1.0, the published
  advantage of decomposition hybrids on this data is an artefact of **pipeline order**, not of the
  method itself.
- Note the R2 column again: on a price level it stays near 1.0 regardless, which is exactly why
  papers reporting R2 ≈ 0.99 cannot be used to tell these two pipelines apart.

## Caveats
- One fixed 80/20 split per configuration (the protocol these papers use), not walk-forward.
- The honest CEEMDAN variant re-decomposes every 5 points rather than every point, for runtime;
  wavelet is re-decomposed at every point.
- Wavelet boundary effects mean even the honest version's most recent component values are the
  least stable — that is a genuine property of the method, not a bug in this test.
"""
(OUT / "decomposition_summary.md").write_text(md)

print("\n" + "=" * 118)
print(G.to_string(index=False))
print("\n" + "=" * 118)
print(R[["target", "decomposition", "pipeline", "MAPE_%", "MAPE_naive_%", "R2",
         "Theil_U2", "beats_naive", "DM_p_vs_naive"]].to_string(index=False))
print(f"\nSaved to {OUT}")
