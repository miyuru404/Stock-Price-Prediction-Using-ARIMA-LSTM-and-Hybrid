#!/usr/bin/env python3
"""
RE-TEST OF THE 1-DAY CLAIM — walk-forward across many independent windows.

THE CLAIM UNDER TEST (pre-registered, from src/direction_pooled_and_rules.py):
    "A model trained on all 7 banking/finance stocks pooled beats each stock's own baseline at the
     1-DAY horizon: 6/7 stocks positive, median +2.5 pp, sign-test p = 0.062."
That came from ONE train/test split. p was above 0.05, it was 1 horizon out of 9 tested, and the
sector sweep has already burned us once with a result that did not replicate. So it is re-tested
here properly before it is allowed to be called a finding.

DESIGN
  * Walk-forward: expanding training window, then a fresh 6-month test window, rolled forward.
    Consecutive test windows do NOT overlap, so at a 1-day horizon each fold is a near-independent
    trial. ~20 folds instead of 1.
  * Every fold retrains from scratch on data strictly before the window (with a purge gap).
  * The pooled model is scored PER STOCK against that stock's OWN baseline (majority from its own
    training rows, persistence at the matched horizon) — the fairness lesson from the last run.
  * BOTH models are reported separately. Reporting max(logistic, xgboost) would quietly cherry-pick
    a winner every fold and inflate the result, which is how the original +2.5 pp was computed.
  * h = 5 (1 week) is carried along as a CONTROL. It already failed the fairness check, so it
    should stay flat. If the control lights up too, the whole setup is suspect.

VERDICT RULE, fixed in advance:
  the claim survives only if the per-fold median edge is positive in clearly more than half the
  folds AND a sign test over folds gives p < 0.05.

Outputs -> results/direction/retest_1day/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from scipy.stats import binomtest, wilcoxon
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "direction" / "retest_1day"
OUT.mkdir(parents=True, exist_ok=True)

SECTOR = ["HNB", "COMB", "SAMP", "LOFC", "LOLC", "LFIN", "CFIN"]
HORIZONS = [1, 5]                      # 1 = the claim, 5 = control (expected null)
DEADZONE_1D = 0.005
SEED = 42
TEST_MONTHS = 6                        # length of each walk-forward test window
MIN_TRAIN_YEARS = 3
FIRST_TEST = pd.Timestamp("2016-01-01")

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")) for t in SECTOR}
aspi = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
          .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))

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


def rsi(s, n=14):
    x = s.diff()
    up = x.clip(lower=0).rolling(n).mean()
    dn = (-x.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def md_table(df):
    cols = [str(x) for x in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(out)


def build(t):
    d = px[t].reset_index()
    dates = d["date"]
    c = d["close"].astype(float)
    v = d["volume"].astype(float)
    r1 = c.pct_change()
    f = pd.DataFrame({"date": dates, "ticker": t})
    f["ret_1"], f["ret_5"], f["ret_10"] = r1, c.pct_change(5), c.pct_change(10)
    f["ma5_ratio"] = c / c.rolling(5).mean() - 1
    f["ma10_ratio"] = c / c.rolling(10).mean() - 1
    f["ma20_ratio"] = c / c.rolling(20).mean() - 1
    f["momentum_10"] = c / c.shift(10) - 1
    f["vol_10"], f["vol_20"] = r1.rolling(10).std(), r1.rolling(20).std()
    f["rsi_14"] = rsi(c)
    m_ = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sg = m_.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (m_ - sg) / c
    f["volchg_5"] = v / v.rolling(5).mean() - 1
    a = aspi.reindex(dates).ffill().reset_index(drop=True)
    ar = a.pct_change()
    f["aspi_ret_1"], f["aspi_ret_5"] = ar, a.pct_change(5)
    f["rs_vs_aspi_5"] = c.pct_change(5) - a.pct_change(5)
    f["beta_aspi_60"] = r1.rolling(60).cov(ar) / ar.rolling(60).var()
    peers = [p for p in SECTOR if p != t]
    pp = pd.DataFrame({p: px[p]["close"].astype(float).reindex(dates).ffill().reset_index(drop=True)
                       for p in peers})
    f["peer_ret_1"] = pp.pct_change().mean(axis=1)
    f["peer_ret_5"] = pp.pct_change(5).mean(axis=1)

    def asof(src, cols):
        j = pd.merge_asof(pd.DataFrame({"date": dates}),
                          src[["available_from"] + cols].sort_values("available_from"),
                          left_on="date", right_on="available_from", direction="backward")
        return j[cols]

    for col, s in asof(rt, ["d_policy_1m", "d_spread_1m", "d_spread_3m",
                            "d_tb3m_3m", "term_slope"]).items():
        f[col] = s.values
    for col, s in asof(inf, ["ccpi_yoy_pct", "d_ccpi_yoy_3m"]).items():
        f[col] = s.values
    j = pd.merge_asof(pd.DataFrame({"date": dates}),
                      fx[["date", "usd_lkr_ret_5", "usd_lkr_ret_20"]], on="date", direction="backward")
    f["usd_lkr_ret_5"], f["usd_lkr_ret_20"] = j["usd_lkr_ret_5"].values, j["usd_lkr_ret_20"].values
    for h in HORIZONS:
        f[f"fwd_{h}"] = c.shift(-h) / c - 1
        f[f"past_{h}"] = c / c.shift(h) - 1
    return f


PANEL = pd.concat([build(t) for t in SECTOR], ignore_index=True).sort_values(["date", "ticker"])
FEATS = [c for c in PANEL.columns
         if c not in ["date", "ticker"] + [f"fwd_{h}" for h in HORIZONS]
         + [f"past_{h}" for h in HORIZONS]]

# walk-forward window edges
starts = pd.date_range(FIRST_TEST, PANEL.date.max(), freq=f"{TEST_MONTHS}MS")
folds = [(s, s + pd.DateOffset(months=TEST_MONTHS)) for s in starts
         if s + pd.DateOffset(months=TEST_MONTHS) <= PANEL.date.max()]
print(f"Panel {len(PANEL):,} stock-days · {len(folds)} walk-forward folds "
      f"({TEST_MONTHS}-month test windows, {folds[0][0]:%Y-%m} → {folds[-1][1]:%Y-%m})\n")

rows = []
for h in HORIZONS:
    dz = DEADZONE_1D * np.sqrt(h)
    df = PANEL[["date", "ticker"] + FEATS + [f"fwd_{h}", f"past_{h}"]].dropna().reset_index(drop=True)
    y_all = pd.Series(np.where(df[f"fwd_{h}"] > dz, 2, np.where(df[f"fwd_{h}"] < -dz, 0, 1)))

    for fi, (t0, t1) in enumerate(folds):
        purge = t0 - pd.Timedelta(days=int(max(h, 1) * 1.5) + 3)
        tr = df["date"] <= purge
        te = (df["date"] >= t0) & (df["date"] < t1)
        if tr.sum() < 250 * MIN_TRAIN_YEARS * len(SECTOR) or te.sum() < 300:
            continue
        Xtr, Xte = df.loc[tr, FEATS], df.loc[te, FEATS]
        ytr, yte = y_all[tr.values], y_all[te.values]
        if ytr.nunique() < 2:
            continue

        present = sorted(ytr.unique())
        wmap = {k: len(ytr) / (len(present) * (ytr == k).sum()) for k in present}
        w = ytr.map(wmap).values

        lg = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=SEED))
        lg.fit(Xtr, ytr)
        xc = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=3, random_state=SEED,
                           n_jobs=4, eval_metric="mlogloss")
        xc.fit(Xtr, ytr, sample_weight=w)

        te_df = df.loc[te, ["date", "ticker", f"past_{h}"]].copy()
        te_df["y"] = yte.values
        te_df["p_lg"] = lg.predict(Xte)
        te_df["p_xc"] = xc.predict(Xte)
        tr_df = df.loc[tr, ["ticker"]].copy()
        tr_df["y"] = ytr.values

        for tk, g in te_df.groupby("ticker"):
            own_tr = tr_df[tr_df.ticker == tk]
            if len(own_tr) < 200 or len(g) < 30:
                continue
            own_maj = int(own_tr["y"].value_counts().idxmax())
            b = max(accuracy_score(g["y"], np.full(len(g), own_maj)),
                    accuracy_score(g["y"], np.where(g[f"past_{h}"] > dz, 2,
                                                    np.where(g[f"past_{h}"] < -dz, 0, 1))))
            rows.append({
                "horizon_days": h, "fold": fi, "test_from": f"{t0:%Y-%m}", "test_to": f"{t1:%Y-%m}",
                "ticker": tk, "n_test": len(g), "n_train": int(tr.sum()),
                "acc_logistic_%": round(accuracy_score(g["y"], g["p_lg"]) * 100, 1),
                "acc_xgboost_%": round(accuracy_score(g["y"], g["p_xc"]) * 100, 1),
                "own_baseline_%": round(b * 100, 1),
                "edge_logistic_pp": round((accuracy_score(g["y"], g["p_lg"]) - b) * 100, 1),
                "edge_xgboost_pp": round((accuracy_score(g["y"], g["p_xc"]) - b) * 100, 1),
            })
    print(f"h={h}d done")

R = pd.DataFrame(rows)
R.to_csv(OUT / "retest_per_stock_per_fold.csv", index=False)

# ---- fold-level summary: each fold is one independent trial ----
FOLD = (R.groupby(["horizon_days", "fold", "test_from", "test_to"])
        .agg(median_edge_logistic=("edge_logistic_pp", "median"),
             median_edge_xgboost=("edge_xgboost_pp", "median"),
             stocks_pos_logistic=("edge_logistic_pp", lambda s: int((s > 0).sum())),
             stocks_pos_xgboost=("edge_xgboost_pp", lambda s: int((s > 0).sum())),
             n_stocks=("ticker", "size")).reset_index().round(2))
FOLD.to_csv(OUT / "retest_by_fold.csv", index=False)

summary = []
for h in HORIZONS:
    sub = FOLD[FOLD.horizon_days == h]
    for model, col in [("Logistic", "median_edge_logistic"), ("XGBoost", "median_edge_xgboost")]:
        v = sub[col].values
        k = int((v > 0).sum())
        p = binomtest(k, len(v), 0.5, alternative="greater").pvalue
        try:
            wp = wilcoxon(v, alternative="greater").pvalue
        except ValueError:
            wp = np.nan
        summary.append({
            "horizon": "1 day" if h == 1 else "1 week (control)", "model": model,
            "folds_positive": f"{k}/{len(v)}", "median_of_fold_medians": round(float(np.median(v)), 2),
            "mean_of_fold_medians": round(float(np.mean(v)), 2),
            "sign_test_p": round(float(p), 4),
            "wilcoxon_p": round(float(wp), 4) if wp == wp else "n/a",
            "SURVIVES": "YES" if (k > len(v) / 2 and p < 0.05) else "NO"})
S = pd.DataFrame(summary)
S.to_csv(OUT / "retest_verdict.csv", index=False)

# per-stock across all folds (1 day only)
PS = (R[R.horizon_days == 1].groupby("ticker")
      .agg(folds=("fold", "nunique"),
           median_edge_logistic=("edge_logistic_pp", "median"),
           median_edge_xgboost=("edge_xgboost_pp", "median"),
           pct_folds_pos_xgb=("edge_xgboost_pp", lambda s: round(float((s > 0).mean() * 100), 1)))
      .reset_index().round(2).sort_values("median_edge_xgboost", ascending=False))
PS.to_csv(OUT / "retest_by_stock.csv", index=False)

# ---- plot ----
one = FOLD[FOLD.horizon_days == 1]
five = FOLD[FOLD.horizon_days == 5]
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
a = ax[0]
x = np.arange(len(one))
a.bar(x - 0.2, one.median_edge_logistic, 0.4, label="Logistic", color="tab:blue", alpha=.85)
a.bar(x + 0.2, one.median_edge_xgboost, 0.4, label="XGBoost", color="tab:orange", alpha=.85)
a.axhline(0, color="black", lw=1)
a.axhline(2.5, color="green", ls=":", label="original claim (+2.5 pp)")
a.set_xticks(x); a.set_xticklabels(one.test_from, rotation=90, fontsize=7)
a.set_ylabel("Median per-stock edge (pp)")
a.set_title("1-DAY CLAIM re-tested — each bar is an independent 6-month window")
a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)

b = ax[1]
b.plot(np.arange(len(one)), one.median_edge_xgboost.cumsum() / np.arange(1, len(one) + 1),
       "o-", label="1 day (the claim)", lw=2)
b.plot(np.arange(len(five)), five.median_edge_xgboost.cumsum() / np.arange(1, len(five) + 1),
       "s-", label="1 week (control)", lw=2, color="grey")
b.axhline(0, color="black", ls="--")
b.axhline(2.5, color="green", ls=":", label="original claim (+2.5 pp)")
b.set_xlabel("Folds included (cumulative)"); b.set_ylabel("Running mean of fold medians (pp)")
b.set_title("Does the edge settle above zero as windows accumulate?")
b.grid(alpha=.3); b.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "retest_1day.png", dpi=140)

v1 = S[(S.horizon == "1 day")]
survives = (v1.SURVIVES == "YES").any()

md = f"""# Re-test of the 1-day claim — walk-forward, {len(folds)} independent windows

## The claim being tested (pre-registered)

> *"Pooled over the 7 banking/finance stocks, the model beats each stock's own baseline at the
> 1-day horizon: 6/7 stocks positive, median +2.5 pp, sign-test p = 0.062."*

It came from **one** train/test split, p was **above** 0.05, and it was 1 horizon out of 9 tested.

## Method

- Expanding training window → fresh **{TEST_MONTHS}-month** test window → roll forward.
  **{len(folds)} folds**, non-overlapping test windows, retrained from scratch each time.
- Scored **per stock against that stock's own baseline** (the fairness lesson from the last run).
- **Both models reported separately.** Taking max(Logistic, XGBoost) each fold would cherry-pick a
  winner every time — that optimism is part of how the original +2.5 pp arose.
- **1 week carried as a control.** It already failed; it should stay flat.
- **Verdict rule fixed in advance:** survives only if fold medians are positive in clearly more
  than half the folds AND the sign test gives p < 0.05.

## VERDICT

{md_table(S)}

**The 1-day claim {'SURVIVES' if survives else 'DOES NOT SURVIVE'}.**

## Fold-by-fold (1 day)

{md_table(one[['test_from', 'test_to', 'median_edge_logistic', 'median_edge_xgboost', 'stocks_pos_logistic', 'stocks_pos_xgboost', 'n_stocks']])}

## Per stock, across all folds (1 day)

{md_table(PS)}

## Caveats
- 6-month test windows at a 1-day horizon are effectively independent, but the 7 stocks inside a
  fold are not independent of each other — that is why the fold, not the stock, is the unit of the
  sign test here.
- Walk-forward retrains ~{len(folds)} times per model, so this is a much harder test than the single
  80/20 split every earlier phase used. A result that survives this is worth believing.
"""
(OUT / "retest_summary.md").write_text(md)

print("\n" + "=" * 92)
print(S.to_string(index=False))
print("=" * 92)
print(one[["test_from", "median_edge_logistic", "median_edge_xgboost",
           "stocks_pos_xgboost", "n_stocks"]].to_string(index=False))
print("=" * 92)
print(PS.to_string(index=False))
print(f"\nVERDICT: the 1-day claim {'SURVIVES' if survives else 'DOES NOT SURVIVE'}")
print(f"Saved to {OUT}")
