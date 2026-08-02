#!/usr/bin/env python3
"""
LATE FUSION — one model per data source, combined by a meta-learner.

WHY THIS IS NOT A REPEAT OF THE ABLATIONS
Everything so far was EARLY FUSION: all features concatenated into one table, one model.

    [price..., news..., macro...] -> one model -> prediction

In that design a weak-but-real news signal has to compete against ~30 other columns, and a tree may
simply never allocate a split to it. Late fusion gives each source its own dedicated model:

    price -> model A --.
    news  -> model B --+-> meta-learner -> prediction
    macro -> model C --'

The news model has NOTHING ELSE to use, so if any information exists there, this architecture gives
it the best possible chance to appear. That is a genuinely different hypothesis, not a re-run.

THE META-LEARNER'S WEIGHTS ARE THEMSELVES THE RESULT
If it assigns ~0 weight to news and macro, that is a far more legible confirmation than
"30% feature importance but no accuracy gain", which is what early fusion kept producing.

LEAKAGE GUARD (the part that is easy to get wrong)
The meta-learner MUST be trained on out-of-fold base predictions. Fitting bases and meta-learner on
the same rows lets the meta-learner see predictions the bases have memorised, and the stack leaks.
So each training window is split CHRONOLOGICALLY:
    base-train (first 70%)  -> fit base models
    meta-train (last 30%)   -> bases predict here; meta-learner is fitted on those predictions
    test                    -> never touched by either stage

COMPARED AGAINST
  * each base model alone (price / news / macro)
  * EARLY fusion (all features in one model) - the architecture used everywhere else
  * majority and matched-horizon persistence baselines

Targets: BANKS composite + HNB · horizons 1, 5, 10 · window 2016-2022 (news feed).
Outputs -> results/late_fusion/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import binomtest
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "late_fusion"
OUT.mkdir(parents=True, exist_ok=True)

BANKS = ["HNB", "COMB", "SAMP"]
HORIZONS = [1, 5, 10]
SEED = 42
TEST_MONTHS = 6
FIRST_TEST = pd.Timestamp("2019-01-01")
NEWS_START = pd.Timestamp("2016-01-01")
BASE_FRAC = 0.70            # of the training window; the rest trains the meta-learner

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))
      for t in BANKS}
aspi = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
          .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))
sent = pd.read_csv(DATA / "news_sentiment_daily.csv", parse_dates=["date"]).sort_values("date")
NEWS_END = sent.date.max()

rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True).ffill())
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_spread_3m"] = rt["spread"].diff(3)
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
rt["term_slope"] = rt["tb_12m"] - rt["tb_3m"]
rt["available_from"] = rt["date"] + pd.Timedelta(days=35)
inf = pd.read_csv(DATA / "inflation_monthly.csv", parse_dates=["date"]).sort_values("date")
inf["d_ccpi_yoy_3m"] = inf["ccpi_yoy_pct"].diff(3)
inf["available_from"] = inf["date"] + pd.Timedelta(days=21)
fx = pd.read_csv(DATA / "usd_lkr_daily.csv", parse_dates=["date"]).sort_values("date")

PRICE_F = ["ret_1", "ret_5", "ret_10", "ma5_ratio", "ma10_ratio", "ma20_ratio",
           "momentum_10", "vol_10", "vol_20", "rsi_14", "aspi_ret_1", "aspi_ret_5"]
NEWS_F = ["sent_vader_1", "sent_vader_5", "sent_lm_1", "sent_lm_5", "sent_mom",
          "news_count_5", "neg_share_5"]
MACRO_F = ["d_policy_1m", "d_spread_1m", "d_spread_3m", "d_tb3m_3m", "term_slope",
           "ccpi_yoy_pct", "d_ccpi_yoy_3m", "usd_lkr_ret_5", "usd_lkr_ret_20"]
BLOCKS = {"price": PRICE_F, "news": NEWS_F, "macro": MACRO_F}


def build_banks():
    cal = None
    for t in BANKS:
        cal = px[t].index if cal is None else cal.intersection(px[t].index)
    cal = cal.sort_values()
    r = pd.DataFrame({t: px[t].reindex(cal).pct_change() for t in BANKS})
    return pd.DataFrame({"date": cal,
                         "close": (100 * (1 + r.mean(axis=1).fillna(0)).cumprod()).values})


TARGETS = {"BANKS": build_banks().reset_index(drop=True),
           "HNB": px["HNB"].reset_index().rename(columns={"HNB": "close", "close": "close"})}
TARGETS["HNB"].columns = ["date", "close"]


def rsi(s, n=14):
    x = s.diff()
    up = x.clip(lower=0).rolling(n).mean()
    dn = (-x.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def md_table(df):
    cols = [str(x) for x in df.columns]
    o = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        o.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(o)


def make_features(c):
    d = c[(c.date >= NEWS_START) & (c.date <= NEWS_END)].reset_index(drop=True)
    cl = d["close"].astype(float)
    r1 = cl.pct_change()
    f = pd.DataFrame({"date": d["date"], "close": cl})
    f["ret_1"], f["ret_5"], f["ret_10"] = r1, cl.pct_change(5), cl.pct_change(10)
    f["ma5_ratio"] = cl / cl.rolling(5).mean() - 1
    f["ma10_ratio"] = cl / cl.rolling(10).mean() - 1
    f["ma20_ratio"] = cl / cl.rolling(20).mean() - 1
    f["momentum_10"] = cl / cl.shift(10) - 1
    f["vol_10"], f["vol_20"] = r1.rolling(10).std(), r1.rolling(20).std()
    f["rsi_14"] = rsi(cl)
    a = aspi.reindex(d["date"]).reset_index(drop=True)          # exact calendar, no ffill
    f["aspi_ret_1"], f["aspi_ret_5"] = a.pct_change(), a.pct_change(5)

    j = pd.merge_asof(d[["date"]], sent, on="date", direction="backward",
                      tolerance=pd.Timedelta("4D"))
    sv, sl, nc = j["s_vader"], j["s_lm"], j["n_articles"].fillna(0)
    f["sent_vader_1"], f["sent_lm_1"] = sv.fillna(0).values, sl.fillna(0).values
    f["sent_vader_5"] = sv.rolling(5, min_periods=1).mean().fillna(0).values
    f["sent_lm_5"] = sl.rolling(5, min_periods=1).mean().fillna(0).values
    f["sent_mom"] = (sv.rolling(5, min_periods=1).mean()
                     - sv.rolling(20, min_periods=1).mean()).fillna(0).values
    f["news_count_5"] = nc.rolling(5, min_periods=1).mean().values
    f["neg_share_5"] = j["neg_share_lm"].rolling(5, min_periods=1).mean().fillna(0).values

    jr = pd.merge_asof(d[["date"]], rt[["available_from"] + MACRO_F[:5]]
                       .sort_values("available_from"),
                       left_on="date", right_on="available_from", direction="backward")
    for col in MACRO_F[:5]:
        f[col] = jr[col].values
    ji = pd.merge_asof(d[["date"]], inf[["available_from", "ccpi_yoy_pct", "d_ccpi_yoy_3m"]]
                       .sort_values("available_from"),
                       left_on="date", right_on="available_from", direction="backward")
    f["ccpi_yoy_pct"], f["d_ccpi_yoy_3m"] = ji["ccpi_yoy_pct"].values, ji["d_ccpi_yoy_3m"].values
    jf = pd.merge_asof(d[["date"]], fx[["date", "usd_lkr_ret_5", "usd_lkr_ret_20"]],
                       on="date", direction="backward")
    f["usd_lkr_ret_5"], f["usd_lkr_ret_20"] = jf["usd_lkr_ret_5"].values, jf["usd_lkr_ret_20"].values

    for h in HORIZONS:
        f[f"fwd_{h}"] = cl.shift(-h) / cl - 1
        f[f"past_{h}"] = cl / cl.shift(h) - 1
    return f


def fit_base(X, y, w):
    """One base model per block. XGBoost if the block is wide enough, else Logistic."""
    if X.shape[1] >= 8:
        m = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, random_state=SEED,
                          n_jobs=4, eval_metric="logloss")
        m.fit(X, y, sample_weight=w)
    else:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced",
                                             random_state=SEED))
        m.fit(X, y)
    return m


rows, weight_rows, leaks = [], [], []
ALLF = PRICE_F + NEWS_F + MACRO_F
for tname, series in TARGETS.items():
    f = make_features(series)
    gg = f[list(dict.fromkeys(["ret_1", "fwd_1"] + ALLF))].dropna()
    for col in ALLF:
        if col == "ret_1":
            continue
        now, fut = abs(gg[col].corr(gg["ret_1"])), abs(gg[col].corr(gg["fwd_1"]))
        if fut > now + 0.05 and fut > 0.2:
            leaks.append({"target": tname, "feature": col,
                          "corr_same_day": round(now, 3), "corr_next_day": round(fut, 3)})

    for h in HORIZONS:
        d = f[["date", f"fwd_{h}", f"past_{h}"] + ALLF].dropna().reset_index(drop=True)
        y = (d[f"fwd_{h}"] > 0).astype(int).values
        for t0 in pd.date_range(FIRST_TEST, d.date.max(), freq=f"{TEST_MONTHS}MS"):
            t1 = t0 + pd.DateOffset(months=TEST_MONTHS)
            if t1 > d.date.max():
                break
            purge = t0 - pd.Timedelta(days=int(h * 1.5) + 3)
            tr = np.flatnonzero((d.date <= purge).values)
            te = np.flatnonzero(((d.date >= t0) & (d.date < t1)).values)
            if len(tr) < 500 or len(te) < 40:
                continue
            # CHRONOLOGICAL inner split: bases never see the meta-learner's training rows
            cut = int(len(tr) * BASE_FRAC)
            b_idx, m_idx = tr[:cut], tr[cut:]
            if len(m_idx) < 120 or len(np.unique(y[b_idx])) < 2:
                continue
            yb, ym, yte = y[b_idx], y[m_idx], y[te]
            cb = pd.Series(yb).value_counts()
            wb = pd.Series(yb).map({k: len(yb) / (len(cb) * cb[k]) for k in cb.index}).values

            maj = int(cb.idxmax())
            past = d.loc[te, f"past_{h}"].values
            base_acc = max(accuracy_score(yte, np.full(len(yte), maj)),
                           accuracy_score(yte, (past > 0).astype(int)))

            # ---- base models, one per source ----
            meta_tr, meta_te, solo = {}, {}, {}
            for bname, cols in BLOCKS.items():
                m = fit_base(d.loc[b_idx, cols], yb, wb)
                meta_tr[bname] = m.predict_proba(d.loc[m_idx, cols])[:, 1]
                meta_te[bname] = m.predict_proba(d.loc[te, cols])[:, 1]
                solo[bname] = accuracy_score(yte, (meta_te[bname] > 0.5).astype(int))

            # ---- meta-learner on OUT-OF-FOLD base predictions ----
            Mtr = np.column_stack([meta_tr[b] for b in BLOCKS])
            Mte = np.column_stack([meta_te[b] for b in BLOCKS])
            meta = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    random_state=SEED))
            meta.fit(Mtr, ym)
            p_late = meta.predict_proba(Mte)[:, 1]
            acc_late = accuracy_score(yte, (p_late > 0.5).astype(int))
            coefs = meta.named_steps["logisticregression"].coef_[0]
            absum = np.abs(coefs).sum() + 1e-12
            for bi, bname in enumerate(BLOCKS):
                weight_rows.append({"target": tname, "horizon_days": h, "test_from": f"{t0:%Y-%m}",
                                    "block": bname, "coef": round(float(coefs[bi]), 4),
                                    "abs_weight_share_%": round(float(abs(coefs[bi]) / absum * 100), 1)})

            # ---- EARLY fusion, the architecture used everywhere else ----
            call = pd.Series(y[tr]).value_counts()
            wall = pd.Series(y[tr]).map({k: len(tr) / (len(call) * call[k]) for k in call.index}).values
            e = fit_base(d.loc[tr, ALLF], y[tr], wall)
            acc_early = accuracy_score(yte, e.predict(d.loc[te, ALLF]))

            try:
                auc_late = roc_auc_score(yte, p_late)
            except ValueError:
                auc_late = np.nan
            rec = {"target": tname, "horizon_days": h, "test_from": f"{t0:%Y-%m}",
                   "n_test": len(te), "baseline_%": round(base_acc * 100, 1),
                   "early_fusion_%": round(acc_early * 100, 1),
                   "late_fusion_%": round(acc_late * 100, 1),
                   "AUC_late": round(float(auc_late), 3) if auc_late == auc_late else np.nan,
                   "edge_early_pp": round((acc_early - base_acc) * 100, 1),
                   "edge_late_pp": round((acc_late - base_acc) * 100, 1),
                   "late_minus_early_pp": round((acc_late - acc_early) * 100, 1)}
            for bname in BLOCKS:
                rec[f"solo_{bname}_%"] = round(solo[bname] * 100, 1)
                rec[f"solo_{bname}_edge_pp"] = round((solo[bname] - base_acc) * 100, 1)
            rows.append(rec)
    print(f"  {tname} done")

LK = pd.DataFrame(leaks)
LK.to_csv(OUT / "leak_scan.csv", index=False)
if len(LK):
    print("\n" + "!" * 90)
    print("LEAK DETECTED — ABORTING")
    print(LK.to_string(index=False))
    raise SystemExit(1)
print("\nleak scan: clean")

R = pd.DataFrame(rows)
R.to_csv(OUT / "late_fusion_folds.csv", index=False)
W = pd.DataFrame(weight_rows)
W.to_csv(OUT / "meta_weights.csv", index=False)

S = []
for (t, h), sub in R.groupby(["target", "horizon_days"]):
    row = {"target": t, "horizon_days": h, "folds": len(sub),
           "baseline_%": round(float(sub["baseline_%"].median()), 1)}
    for k, col in [("early", "edge_early_pp"), ("late", "edge_late_pp"),
                   ("solo_price", "solo_price_edge_pp"), ("solo_news", "solo_news_edge_pp"),
                   ("solo_macro", "solo_macro_edge_pp")]:
        v = sub[col].values
        kk = int((v > 0).sum())
        row[f"{k}_edge_pp"] = round(float(np.median(v)), 2)
        row[f"{k}_folds_pos"] = f"{kk}/{len(v)}"
        row[f"{k}_p"] = round(float(binomtest(kk, len(v), 0.5, alternative="greater").pvalue), 4)
    lme = sub["late_minus_early_pp"].values
    kk = int((lme > 0).sum())
    row["late_vs_early_pp"] = round(float(np.median(lme)), 2)
    row["late_beats_early"] = f"{kk}/{len(lme)}"
    row["late_vs_early_p"] = round(float(binomtest(kk, len(lme), 0.5, alternative="greater").pvalue), 4)
    S.append(row)
S = pd.DataFrame(S)
S.to_csv(OUT / "late_fusion_summary.csv", index=False)

WS = (W.groupby(["target", "horizon_days", "block"])
      .agg(median_coef=("coef", "median"),
           median_weight_share_pct=("abs_weight_share_%", "median")).reset_index().round(3))
WS.to_csv(OUT / "meta_weights_summary.csv", index=False)

late_wins = S[(S.late_p < 0.05) & (S.late_edge_pp > 0)]
beat_early = S[(S.late_vs_early_p < 0.05) & (S.late_vs_early_pp > 0)]
news_w = WS[WS.block == "news"].median_weight_share_pct.median()
macro_w = WS[WS.block == "macro"].median_weight_share_pct.median()
price_w = WS[WS.block == "price"].median_weight_share_pct.median()

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
a = ax[0]
x = np.arange(len(S))
for i, (k, lab) in enumerate([("solo_price", "price only"), ("solo_news", "news only"),
                              ("solo_macro", "macro only"), ("early", "EARLY fusion"),
                              ("late", "LATE fusion")]):
    a.bar(x + (i - 2) * 0.16, S[f"{k}_edge_pp"], 0.16, label=lab, alpha=.85)
a.axhline(0, color="black", lw=1)
a.set_xticks(x); a.set_xticklabels([f"{r.target}\n{r.horizon_days}d" for _, r in S.iterrows()],
                                   fontsize=8)
a.set_ylabel("Median edge over baseline (pp)")
a.set_title("One model per source, then combined — does it help?")
a.grid(alpha=.3, axis="y"); a.legend(fontsize=7, ncol=2)

b = ax[1]
piv = WS.pivot_table(index=["target", "horizon_days"], columns="block",
                     values="median_weight_share_pct")
piv = piv[["price", "news", "macro"]]
bot = np.zeros(len(piv))
for blk, colr in [("price", "tab:blue"), ("news", "tab:green"), ("macro", "tab:orange")]:
    b.bar(range(len(piv)), piv[blk], bottom=bot, label=blk, color=colr, alpha=.85)
    bot += piv[blk].values
b.set_xticks(range(len(piv)))
b.set_xticklabels([f"{i[0]}\n{i[1]}d" for i in piv.index], fontsize=8)
b.set_ylabel("Share of meta-learner |weight| (%)")
b.set_title("What does the meta-learner actually trust?")
b.grid(alpha=.3, axis="y"); b.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "late_fusion.png", dpi=140)

md = f"""# Late fusion — one model per data source, combined by a meta-learner

## Why this is a different test

Every earlier phase used **early fusion**: all features concatenated into one table, one model.
A weak-but-real news signal there has to compete with ~30 other columns and may never get a split.

**Late fusion** gives each source its own model:

```
price -> model A --.
news  -> model B --+-> meta-learner -> prediction
macro -> model C --'
```

The news model has nothing else to use. If any information exists in news, this is the architecture
most likely to surface it.

**Leakage guard:** each training window is split chronologically — bases fit on the first {int(BASE_FRAC*100)}%,
the meta-learner fits on base predictions over the remaining {100-int(BASE_FRAC*100)}%. Fitting both on the same rows
would let the meta-learner see predictions the bases had memorised.

**Window:** {NEWS_START:%Y-%m} → {NEWS_END:%Y-%m} (news feed). Compare only within this table.

## Leak scan

CLEAN — no feature tracks the future more than the present.

## Results

{md_table(S[["target", "horizon_days", "baseline_%", "solo_price_edge_pp", "solo_news_edge_pp", "solo_macro_edge_pp", "early_edge_pp", "late_edge_pp", "late_folds_pos", "late_p", "late_vs_early_pp", "late_beats_early"]])}

**Late fusion beats the baseline significantly: {len(late_wins)} of {len(S)}.**
**Late fusion beats early fusion significantly: {len(beat_early)} of {len(S)}.**

## What the meta-learner actually trusts

{md_table(WS)}

Median share of the meta-learner's absolute weight:
**price {price_w:.0f}% · news {news_w:.0f}% · macro {macro_w:.0f}%**

This is the most legible version of the project's central finding. Early fusion kept reporting
"30-60% feature importance for news/macro, but no accuracy gain" — ambiguous. Here a model that is
free to weight the three sources however it likes tells you directly how much it trusts each one.

## Reading it

{"Late fusion beats early fusion — architecture mattered, and the result is worth carrying forward." if len(beat_early) else "Late fusion does NOT beat early fusion. Giving news and macro their own dedicated models does not surface information that concatenation missed — because there is none to surface. Architecture was not the constraint."}

## Caveats
- {len(S) and int(S.folds.max())} folds per target-horizon; power is limited.
- The meta-learner is linear (logistic). A nonlinear meta-learner could in principle find an
  interaction between base predictions, but with 3 inputs and ~150 meta-training rows it would
  overfit more than it learns.
- News sentiment remains market-wide, not sector-specific — a known, still-untested gap.
"""
(OUT / "late_fusion_summary.md").write_text(md)

print("\n" + "=" * 118)
print(S.to_string(index=False))
print("\n" + "=" * 118)
print("META-LEARNER WEIGHTS")
print(WS.to_string(index=False))
print(f"\nweight share — price {price_w:.0f}% | news {news_w:.0f}% | macro {macro_w:.0f}%")
print(f"late fusion beats baseline: {len(late_wins)}/{len(S)} | beats early fusion: {len(beat_early)}/{len(S)}")
print(f"Saved to {OUT}")
