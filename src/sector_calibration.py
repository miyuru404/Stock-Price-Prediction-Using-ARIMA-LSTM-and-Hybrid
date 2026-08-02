#!/usr/bin/env python3
"""
PROBABILITY CALIBRATION — can the SYSTEM honestly say "65% chance the banking sector rises"?

THE PRODUCT QUESTION
The dashboard does not just emit up/down. It shows a CONFIDENCE. So the number it prints has to
mean what it says: of all the days the system said "65%", about 65% should actually have gone up.
A model can be barely better than a coin flip and still be perfectly calibrated — and a
well-calibrated 55% is an honest, usable product. An UNCALIBRATED 85% is a lie to the user.

Accuracy and calibration are different properties, and nothing in this project has tested the second.

WHAT IS MEASURED
  * Brier score          - overall probability accuracy (lower is better); compared with the
                           climatology baseline (always predict the training base rate), which is
                           the probabilistic equivalent of the naive benchmark.
  * Brier skill score    - 1 - Brier/Brier_climatology. Above 0 = better than always saying the
                           base rate. This is the honest headline number.
  * Reliability curve    - predicted probability vs realised frequency, bucketed.
  * ECE / MCE            - expected and maximum calibration error, the standard summary numbers.
  * Resolution           - does the model actually vary its probabilities, or emit the base rate
                           every day? A flat forecaster is perfectly calibrated and useless.

THREE CALIBRATION METHODS, all fitted INSIDE the training window only:
  raw (none) · Platt scaling (sigmoid) · isotonic regression
The calibrator is fitted on a held-out slice of TRAINING data, never on the test fold — fitting it
on test data would be the same class of leak this project has already caught three times.

Targets: BANKS and SECTOR composites (BANKS is the one with real autocorrelation, 0.178).
Horizons: 5, 10, 22 days — the 1/2/4-week horizons the product advertises.

Outputs -> results/sector_calibration/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "sector_calibration"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {"BANKS": ["HNB", "COMB", "SAMP"],
          "SECTOR": ["HNB", "COMB", "SAMP", "LOFC", "LOLC", "LFIN", "CFIN"]}
HORIZONS = [5, 10, 22]                 # 1 week / 2 weeks / ~4 weeks
HORIZON_NAME = {5: "1 week", 10: "2 weeks", 22: "4 weeks"}
SEED = 42
TEST_MONTHS = 6
FIRST_TEST = pd.Timestamp("2017-01-01")
N_BINS = 8

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))
      for t in GROUPS["SECTOR"]}
aspi = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
          .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))


def build_composite(tickers):
    cal = None
    for t in tickers:
        cal = px[t].index if cal is None else cal.intersection(px[t].index)
    cal = cal.sort_values()
    rets = pd.DataFrame({t: px[t].reindex(cal).pct_change() for t in tickers})
    idx = 100 * (1 + rets.mean(axis=1).fillna(0)).cumprod()
    return pd.DataFrame({"date": cal, "close": idx.values}).reset_index(drop=True)


def rsi(s, n=14):
    x = s.diff()
    up = x.clip(lower=0).rolling(n).mean()
    dn = (-x.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def make_features(c):
    cl = c["close"]
    r1 = cl.pct_change()
    f = pd.DataFrame({"date": c["date"], "close": cl})
    f["ret_1"], f["ret_5"], f["ret_10"] = r1, cl.pct_change(5), cl.pct_change(10)
    f["ma5_ratio"] = cl / cl.rolling(5).mean() - 1
    f["ma10_ratio"] = cl / cl.rolling(10).mean() - 1
    f["ma20_ratio"] = cl / cl.rolling(20).mean() - 1
    f["momentum_10"] = cl / cl.shift(10) - 1
    f["vol_10"], f["vol_20"] = r1.rolling(10).std(), r1.rolling(20).std()
    f["rsi_14"] = rsi(cl)
    a = aspi.reindex(c["date"]).reset_index(drop=True)      # no ffill (the ASPI leak fix)
    f["aspi_ret_1"], f["aspi_ret_5"] = a.pct_change(), a.pct_change(5)
    f["rs_vs_aspi_5"] = cl.pct_change(5) - a.pct_change(5)
    for h in HORIZONS:
        f[f"fwd_{h}"] = cl.shift(-h) / cl - 1
    return f


FEATS = ["ret_1", "ret_5", "ret_10", "ma5_ratio", "ma10_ratio", "ma20_ratio",
         "momentum_10", "vol_10", "vol_20", "rsi_14", "aspi_ret_1", "aspi_ret_5", "rs_vs_aspi_5"]


def md_table(df):
    cols = [str(x) for x in df.columns]
    o = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        o.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(o)


def calib_errors(y, p, n_bins=N_BINS):
    """ECE / MCE over equal-width probability bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce, rows = 0.0, 0.0, []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        conf, freq = float(p[m].mean()), float(y[m].mean())
        gap = abs(conf - freq)
        ece += m.mean() * gap
        mce = max(mce, gap)
        rows.append({"bin_low": round(edges[i], 3), "bin_high": round(edges[i + 1], 3),
                     "n": int(m.sum()), "mean_predicted": round(conf, 3),
                     "actual_frequency": round(freq, 3), "gap": round(gap, 3)})
    return float(ece), float(mce), pd.DataFrame(rows)


records, rel_store = [], []
for g, ts in GROUPS.items():
    f = make_features(build_composite(ts))
    for h in HORIZONS:
        d = f[["date", f"fwd_{h}"] + FEATS].dropna().reset_index(drop=True)
        y = (d[f"fwd_{h}"] > 0).astype(int).values

        for t0 in pd.date_range(FIRST_TEST, d.date.max(), freq=f"{TEST_MONTHS}MS"):
            t1 = t0 + pd.DateOffset(months=TEST_MONTHS)
            if t1 > d.date.max():
                break
            purge = t0 - pd.Timedelta(days=int(h * 1.5) + 3)
            tr = np.flatnonzero((d.date <= purge).values)
            te = np.flatnonzero(((d.date >= t0) & (d.date < t1)).values)
            if len(tr) < 500 or len(te) < 50:
                continue
            # inner split of TRAINING data: the calibrator never sees test data
            cut = int(len(tr) * 0.75)
            fit_idx, cal_idx = tr[:cut], tr[cut:]
            if len(cal_idx) < 100 or len(np.unique(y[fit_idx])) < 2:
                continue
            Xtr, Xcal, Xte = d.loc[fit_idx, FEATS], d.loc[cal_idx, FEATS], d.loc[te, FEATS]
            ytr, ycal, yte = y[fit_idx], y[cal_idx], y[te]

            base_rate = float(ytr.mean())
            p_clim = np.full(len(yte), base_rate)          # climatology = probabilistic naive
            bs_clim = brier_score_loss(yte, p_clim)

            for mname, mk in [("Logistic", lambda: make_pipeline(
                                   StandardScaler(),
                                   LogisticRegression(max_iter=2000, random_state=SEED))),
                              ("XGBoost", lambda: XGBClassifier(
                                   n_estimators=200, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                   random_state=SEED, n_jobs=4, eval_metric="logloss"))]:
                variants = {}
                m0 = mk()
                m0.fit(Xtr, ytr)
                variants["raw"] = m0.predict_proba(Xte)[:, 1]
                # NOTE: sklearn >= 1.9 removed cv="prefit"; FrozenEstimator is the replacement.
                # No bare except here — a silently swallowed failure previously meant only the
                # RAW probabilities were ever tested, which is not the experiment.
                for method in ("sigmoid", "isotonic"):
                    cc = CalibratedClassifierCV(FrozenEstimator(m0), method=method)
                    cc.fit(Xcal, ycal)
                    variants["Platt" if method == "sigmoid" else "isotonic"] = \
                        cc.predict_proba(Xte)[:, 1]

                for vname, p in variants.items():
                    p = np.clip(p, 1e-6, 1 - 1e-6)
                    bs = brier_score_loss(yte, p)
                    ece, mce, rel = calib_errors(yte, p)
                    try:
                        auc = roc_auc_score(yte, p)
                    except ValueError:
                        auc = np.nan
                    records.append({
                        "group": g, "horizon_days": h, "model": mname, "calibration": vname,
                        "test_from": f"{t0:%Y-%m}", "n_test": len(yte),
                        "base_rate_%": round(base_rate * 100, 1),
                        "actual_up_%": round(float(yte.mean()) * 100, 1),
                        "accuracy_%": round(accuracy_score(yte, (p > 0.5).astype(int)) * 100, 1),
                        "AUC": round(float(auc), 3) if auc == auc else np.nan,
                        "Brier": round(bs, 4), "Brier_climatology": round(bs_clim, 4),
                        "Brier_skill": round(1 - bs / bs_clim, 4) if bs_clim > 0 else np.nan,
                        "ECE": round(ece, 4), "MCE": round(mce, 4),
                        "prob_spread": round(float(p.max() - p.min()), 3),
                        "prob_std": round(float(p.std()), 4)})
                    rel["group"], rel["horizon_days"] = g, h
                    rel["model"], rel["calibration"] = mname, vname
                    rel_store.append(rel)
    print(f"  {g} done")

R = pd.DataFrame(records)
R.to_csv(OUT / "calibration_all_folds.csv", index=False)
REL = pd.concat(rel_store, ignore_index=True)
REL.to_csv(OUT / "reliability_bins.csv", index=False)

S = (R.groupby(["group", "horizon_days", "model", "calibration"])
     .agg(folds=("Brier", "size"), accuracy_pct=("accuracy_%", "median"),
          AUC=("AUC", "median"), Brier=("Brier", "median"),
          Brier_clim=("Brier_climatology", "median"),
          Brier_skill=("Brier_skill", "median"), ECE=("ECE", "median"), MCE=("MCE", "median"),
          prob_std=("prob_std", "median"),
          folds_skill_positive=("Brier_skill", lambda s: f"{int((s > 0).sum())}/{len(s)}"))
     .reset_index().round(4))
S["USABLE"] = np.where((S.Brier_skill > 0) & (S.ECE < 0.10), "YES", "no")
S.to_csv(OUT / "calibration_summary.csv", index=False)

best = S.sort_values("Brier_skill", ascending=False).head(1).iloc[0]
usable = S[S.USABLE == "YES"]

# ---------------------------------------------------------------- reliability figure
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for i, g in enumerate(GROUPS):
    for j, h in enumerate(HORIZONS):
        a = ax[i][j]
        a.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for cal_, colr in [("raw", "tab:red"), ("Platt", "tab:blue"), ("isotonic", "tab:green")]:
            sub = REL[(REL.group == g) & (REL.horizon_days == h) & (REL.calibration == cal_) &
                      (REL.model == "Logistic")]
            if not len(sub):
                continue
            bins = pd.cut(sub.mean_predicted, np.linspace(0, 1, N_BINS + 1))
            pts = []
            for _, x in sub.groupby(bins, observed=True):
                if x.n.sum() > 0:
                    pts.append((np.average(x.mean_predicted, weights=x.n),
                                np.average(x.actual_frequency, weights=x.n)))
            if pts:
                pts = np.array(sorted(pts))
                a.plot(pts[:, 0], pts[:, 1], "o-", color=colr, label=cal_, ms=4)
        a.set_title(f"{g} — {HORIZON_NAME[h]}", fontsize=10)
        a.set_xlabel("predicted probability"); a.set_ylabel("actual frequency")
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.grid(alpha=.3)
        if i == 0 and j == 0:
            a.legend(fontsize=8)
fig.suptitle("Reliability curves — does '65%' actually mean 65%?  (Logistic)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "reliability_curves.png", dpi=140)

md = f"""# Probability calibration — can the system honestly show a confidence number?

The dashboard does not just say up/down; it shows a **confidence**. That number must mean what it
says: of all the days the system prints "65%", about 65% should actually rise. A model that is
barely better than a coin flip can still be perfectly calibrated — and a well-calibrated 55% is an
honest, usable product. An **uncalibrated 85% is a lie to the user.**

Accuracy and calibration are different properties. Nothing earlier in this project tested the second.

## Method

* Targets: **BANKS** (autocorrelation 0.178, the strongest composite) and **SECTOR** (all 7).
* Horizons: **1 / 2 / 4 weeks** — what the product advertises.
* Calibrators (`raw`, `Platt`, `isotonic`) are fitted on a held-out slice **inside the training
  window**, never on the test fold. Fitting a calibrator on test data would be the same class of
  leak this project has already caught three times.
* Benchmark: **climatology** — always predict the training base rate. This is the probabilistic
  equivalent of the naive baseline.

## Key metrics

| Metric | Meaning | Good |
|---|---|---|
| **Brier skill** | 1 − Brier/Brier_climatology | **> 0** |
| **ECE** | mean gap between stated confidence and reality | **< 0.10** |
| **MCE** | worst bucket's gap | small |
| **prob_std** | how much the forecast actually varies | **not ~0** |

`prob_std` matters: a model that prints the base rate every single day is *perfectly calibrated and
completely useless*. Calibration alone is not enough — the forecast must also move.

## Results (median across walk-forward folds)

{md_table(S[["group", "horizon_days", "model", "calibration", "accuracy_pct", "AUC", "Brier", "Brier_clim", "Brier_skill", "ECE", "prob_std", "folds_skill_positive", "USABLE"]])}

**Configurations that are usable (Brier skill > 0 AND ECE < 0.10): {len(usable)} of {len(S)}.**

Best by Brier skill: **{best.group} / {HORIZON_NAME[best.horizon_days]} / {best.model} / {best.calibration}**
— skill {best.Brier_skill}, ECE {best.ECE}, accuracy {best.accuracy_pct}%.

## Reliability curves

`reliability_curves.png` — the diagonal is perfect calibration. A curve **below** the diagonal means
the system is **overconfident** (says 70%, delivers less); **above** means underconfident.

## What this means for the product

{"At least one configuration is both skilful and calibrated, so the dashboard can show a real confidence number for those settings." if len(usable) else "No configuration is both skilful and well calibrated. The dashboard should NOT display a raw model probability as a confidence figure."}

Whatever the outcome, the honest design is the same: show the **realised track record**
("when we said 60%, it rose 58% of the time, over N predictions") rather than the model's raw
output. That is measurable, cannot be gamed, and is exactly the accuracy-history feature already
planned for the system.

## Caveats
- Walk-forward folds are 6 months; at 22 days the forward windows inside a fold overlap, so the
  effective sample there is smaller than the row count suggests.
- Isotonic regression needs a reasonable calibration slice; with ~500 training rows it can overfit
  and is the more fragile of the two methods.
- Calibration is measured on the composite index, not on individual stocks.
"""
(OUT / "calibration_summary.md").write_text(md)

print("\n" + "=" * 120)
print(S[["group", "horizon_days", "model", "calibration", "accuracy_pct", "AUC",
         "Brier", "Brier_clim", "Brier_skill", "ECE", "prob_std",
         "folds_skill_positive", "USABLE"]].to_string(index=False))
print("=" * 120)
print(f"Usable (skill>0 AND ECE<0.10): {len(usable)}/{len(S)}")
print(f"Saved to {OUT}")
