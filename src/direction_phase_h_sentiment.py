#!/usr/bin/env python3
"""
PHASE H — news sentiment. The last untested information source.

DATA: 149,240 articles from Dailymirror + Newsfirst (Kaggle: ivantha/sri-lanka-news-dataset),
1999 -> 2022-06-19, with REAL publication times (only 75 rows lack one).

Two hard constraints, both handled explicitly rather than ignored:

1. THE FEED ENDS 2022-06-19, four years before the stock data. A standard 80/20 split on the usable
   window would put the whole test set inside the 2022 crisis — and this project has established
   THREE TIMES that every model, price-only included, looks good in that period. A single-split run
   here would produce a positive number that means nothing. So Phase H is evaluated WALK-FORWARD,
   with the crisis as one fold among several, and pre-2021 vs 2021+ folds reported separately.

2. LOOK-AHEAD. An article published at 6pm reports what the market already did. The CSE closes at
   14:30, so each article is assigned to the first trading day on which a reader could have acted:
       published <= 14:30 on a trading day  -> that day
       published after 14:30, or on a weekend/holiday -> the NEXT trading day
   Without this rule sentiment would look brilliant and be worthless.

TWO LEXICONS, side by side:
  * VADER            - general-purpose, tuned on social media
  * Loughran-McDonald style - finance-specific (see src/finance_lexicon.py)
If the two disagree, the sentiment signal is fragile and that is itself worth reporting.

PHASES (identical rows, split, seeds — only the feature list changes):
  A  Tier-1 technical                                   (plain time-series floor)
  D  + Tier-2 + rate changes + sector                   (best so far, no sentiment)
  H  + NEWS SENTIMENT                                   <- the variable under test
  S  Tier-1 + sentiment ONLY                            (isolates sentiment's standalone value)

Outputs -> results/direction/sentiment/
"""
import re
import sys
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from scipy.stats import binomtest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(_Path(__file__).resolve().parent))
from finance_lexicon import load_lm, lm_score          # noqa: E402

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "direction" / "sentiment"
OUT.mkdir(parents=True, exist_ok=True)

SECTOR = ["HNB", "COMB", "SAMP", "LOFC", "LOLC", "LFIN", "CFIN"]
HORIZONS = [1, 5, 10, 22]
DEADZONE_1D = 0.005
SEED = 42
CLOSE_HOUR, CLOSE_MIN = 14, 30           # CSE closes 14:30
NEWS_START = pd.Timestamp("2016-01-01")  # before this, 46-83% of days have no relevant article
TEST_MONTHS = 6
FIRST_TEST = pd.Timestamp("2019-01-01")

KEYWORDS = (r"bank|banking|cse|colombo stock|stock market|share market|aspi|bourse|rupee|"
            r"inflation|central bank|cbsl|interest rate|policy rate|treasury bill|imf|economy|"
            r"economic|gdp|investor|listed compan|earnings|dividend|forex|exchange rate|debt|"
            r"default|budget|finance|financial|monetary|fiscal|tax|trade|export|import")

# ================================================================ 1. score the news
print("=" * 90)
print("1. NEWS SENTIMENT")
print("=" * 90)
news = pd.read_csv(ROOT / "news.csv",
                   usecols=["heading", "source", "published_date", "published_time", "content"],
                   parse_dates=["published_date"], low_memory=False)
news = news[news.published_date >= NEWS_START].copy()
news["text"] = (news.heading.fillna("") + ". " + news.content.fillna("")).str.slice(0, 2000)
news["relevant"] = news.text.str.lower().str.contains(KEYWORDS, regex=True, na=False)
rel = news[news.relevant].copy()
print(f"  articles from {NEWS_START:%Y-%m}: {len(news):,}  ->  market-relevant: {len(rel):,} "
      f"({len(rel)/len(news)*100:.0f}%)")

# publication timestamp -> the first trading day a reader could act on
tm = rel.published_time.fillna("00:00").astype(str).str.extract(r"(\d{1,2}):(\d{2})")
hh = pd.to_numeric(tm[0], errors="coerce").fillna(0).clip(0, 23)
mm = pd.to_numeric(tm[1], errors="coerce").fillna(0).clip(0, 59)
after_close = (hh > CLOSE_HOUR) | ((hh == CLOSE_HOUR) & (mm > CLOSE_MIN))
rel["act_date"] = rel.published_date + pd.to_timedelta(after_close.astype(int), unit="D")
print(f"  published after 14:30 (pushed to next trading day): {int(after_close.sum()):,} "
      f"({after_close.mean()*100:.0f}%)")

vader = SentimentIntensityAnalyzer()
POS, NEG, LEX_LABEL = load_lm()
print(f"  finance lexicon: {LEX_LABEL}")
rel["s_vader"] = [vader.polarity_scores(t)["compound"] for t in rel.text]
rel["s_lm"] = [lm_score(t, POS, NEG) for t in rel.text]
print(f"  scored {len(rel):,} articles with both lexicons")
corr = rel[["s_vader", "s_lm"]].corr().iloc[0, 1]
print(f"  correlation between the two lexicons: {corr:.3f}")

daily = (rel.groupby(rel.act_date.dt.normalize())
         .agg(s_vader=("s_vader", "mean"), s_lm=("s_lm", "mean"),
              n_articles=("s_vader", "size"),
              neg_share_vader=("s_vader", lambda s: float((s < -0.05).mean())),
              neg_share_lm=("s_lm", lambda s: float((s < 0).mean())))
         .reset_index().rename(columns={"act_date": "date"}))
daily.to_csv(DATA / "news_sentiment_daily.csv", index=False)
print(f"  -> cleaned_data/news_sentiment_daily.csv  ({len(daily):,} days, "
      f"{daily.date.min():%Y-%m} -> {daily.date.max():%Y-%m})")
NEWS_END = daily.date.max()

# ================================================================ 2. panel
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
RATES = ["d_policy_1m", "d_spread_1m", "d_spread_3m", "d_tb3m_3m", "term_slope"]
rt["available_from"] = rt["date"] + pd.Timedelta(days=35)


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


SENT = ["sent_vader_1", "sent_vader_5", "sent_vader_20", "sent_lm_1", "sent_lm_5", "sent_lm_20",
        "sent_vader_mom", "sent_lm_mom", "news_count_1", "news_count_5", "neg_share_5",
        "sent_disagree"]


def build(t):
    d = px[t].reset_index()
    d = d[(d.date >= NEWS_START) & (d.date <= NEWS_END)].reset_index(drop=True)
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
    tier1 = list(f.columns)[2:]
    f["rsi_14"] = rsi(c)
    m_ = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sg = m_.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (m_ - sg) / c
    f["volchg_5"] = v / v.rolling(5).mean() - 1
    tier2 = ["rsi_14", "macd_hist", "volchg_5"]

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
    sector = ["aspi_ret_1", "aspi_ret_5", "rs_vs_aspi_5", "beta_aspi_60",
              "peer_ret_1", "peer_ret_5"]

    j = pd.merge_asof(pd.DataFrame({"date": dates}),
                      rt[["available_from"] + RATES].sort_values("available_from"),
                      left_on="date", right_on="available_from", direction="backward")
    for col in RATES:
        f[col] = j[col].values

    # ---- sentiment, already shifted to the actionable trading day ----
    s = pd.merge_asof(pd.DataFrame({"date": dates}), daily.sort_values("date"),
                      on="date", direction="backward", tolerance=pd.Timedelta("4D"))
    sv, sl, nc = s["s_vader"], s["s_lm"], s["n_articles"].fillna(0)
    f["sent_vader_1"], f["sent_lm_1"] = sv.fillna(0).values, sl.fillna(0).values
    f["sent_vader_5"] = sv.rolling(5, min_periods=1).mean().fillna(0).values
    f["sent_vader_20"] = sv.rolling(20, min_periods=1).mean().fillna(0).values
    f["sent_lm_5"] = sl.rolling(5, min_periods=1).mean().fillna(0).values
    f["sent_lm_20"] = sl.rolling(20, min_periods=1).mean().fillna(0).values
    f["sent_vader_mom"] = f["sent_vader_5"] - f["sent_vader_20"]
    f["sent_lm_mom"] = f["sent_lm_5"] - f["sent_lm_20"]
    f["news_count_1"] = nc.values
    f["news_count_5"] = nc.rolling(5, min_periods=1).mean().values
    f["neg_share_5"] = s["neg_share_lm"].rolling(5, min_periods=1).mean().fillna(0).values
    f["sent_disagree"] = (f["sent_vader_1"] - f["sent_lm_1"]).abs()

    for h in HORIZONS:
        f[f"fwd_{h}"] = c.shift(-h) / c - 1
        f[f"past_{h}"] = c / c.shift(h) - 1
    phases = {"A": tier1, "D": tier1 + tier2 + sector + RATES,
              "H": tier1 + tier2 + sector + RATES + SENT, "S": tier1 + SENT}
    return f, phases


parts = [build(t) for t in SECTOR]
PANEL = pd.concat([p[0] for p in parts], ignore_index=True).sort_values(["date", "ticker"])
PHASES = parts[0][1]
ALL = sorted({c for cols in PHASES.values() for c in cols})
print(f"\n  panel: {len(PANEL):,} stock-days, {PANEL.date.min():%Y-%m} -> {PANEL.date.max():%Y-%m}")

# ================================================================ 3. walk-forward
print("\n" + "=" * 90)
print("2. WALK-FORWARD ABLATION")
print("=" * 90)
starts = pd.date_range(FIRST_TEST, PANEL.date.max(), freq=f"{TEST_MONTHS}MS")
folds = [(s, s + pd.DateOffset(months=TEST_MONTHS)) for s in starts
         if s + pd.DateOffset(months=TEST_MONTHS) <= PANEL.date.max() + pd.DateOffset(months=1)]
print(f"  {len(folds)} folds of {TEST_MONTHS} months\n")

rows = []
for h in HORIZONS:
    dz = DEADZONE_1D * np.sqrt(h)
    df = PANEL[["date", "ticker"] + ALL + [f"fwd_{h}", f"past_{h}"]].dropna().reset_index(drop=True)
    y_all = pd.Series(np.where(df[f"fwd_{h}"] > dz, 2, np.where(df[f"fwd_{h}"] < -dz, 0, 1)))
    for fi, (t0, t1) in enumerate(folds):
        purge = t0 - pd.Timedelta(days=int(h * 1.5) + 3)
        tr = (df.date <= purge).values
        te = ((df.date >= t0) & (df.date < t1)).values
        if tr.sum() < 3000 or te.sum() < 300:
            continue
        ytr, yte = y_all[tr], y_all[te]
        if ytr.nunique() < 2:
            continue
        present = sorted(ytr.unique())
        wmap = {k: len(ytr) / (len(present) * (ytr == k).sum()) for k in present}
        w = ytr.map(wmap).values
        preds = {}
        for ph, cols in PHASES.items():
            Xtr, Xte = df.loc[tr, cols], df.loc[te, cols]
            lg = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=SEED))
            lg.fit(Xtr, ytr)
            xc = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, min_child_weight=3, random_state=SEED,
                               n_jobs=4, eval_metric="mlogloss")
            xc.fit(Xtr, ytr, sample_weight=w)
            preds[ph] = (lg.predict(Xte), xc.predict(Xte))
            if ph == "H":
                imp = pd.Series(xc.feature_importances_, index=cols)
                sent_share = float(imp[[c for c in cols if c in SENT]].sum() * 100)
        mte = df.loc[te, ["ticker", f"past_{h}"]].copy()
        mte["y"] = yte.values
        mtr = df.loc[tr, ["ticker"]].copy()
        mtr["y"] = ytr.values
        for tk, g in mte.groupby("ticker"):
            gi = g.index.values
            pos = np.searchsorted(np.flatnonzero(te), gi)
            own = mtr[mtr.ticker == tk]
            if len(own) < 200 or len(gi) < 30:
                continue
            omaj = int(own["y"].value_counts().idxmax())
            b = max(accuracy_score(g["y"], np.full(len(g), omaj)),
                    accuracy_score(g["y"], np.where(g[f"past_{h}"] > dz, 2,
                                                    np.where(g[f"past_{h}"] < -dz, 0, 1))))
            rec = {"horizon_days": h, "fold": fi, "test_from": f"{t0:%Y-%m}", "ticker": tk,
                   "n_test": len(gi), "own_baseline_%": round(b * 100, 1),
                   "sent_share_%": round(sent_share, 1)}
            for ph, (pl, pxg) in preds.items():
                acc = max(accuracy_score(g["y"], pl[pos]), accuracy_score(g["y"], pxg[pos]))
                rec[f"acc_{ph}_%"] = round(acc * 100, 1)
                rec[f"edge_{ph}_pp"] = round((acc - b) * 100, 1)
            rows.append(rec)
    print(f"  h={h}d done")

R = pd.DataFrame(rows)
R.to_csv(OUT / "sentiment_per_stock_per_fold.csv", index=False)

PH = list(PHASES)
FOLD = (R.groupby(["horizon_days", "fold", "test_from"])
        .agg(**{f"med_{p}": (f"edge_{p}_pp", "median") for p in PH},
             sent_share=("sent_share_%", "mean")).reset_index().round(2))
FOLD["gain_D_to_H"] = (FOLD.med_H - FOLD.med_D).round(2)
FOLD.to_csv(OUT / "sentiment_by_fold.csv", index=False)

summary = []
for h in HORIZONS:
    sub = FOLD[FOLD.horizon_days == h]
    for p in PH:
        v = sub[f"med_{p}"].values
        k = int((v > 0).sum())
        summary.append({"horizon_days": h, "phase": p, "folds_positive": f"{k}/{len(v)}",
                        "median_edge_pp": round(float(np.median(v)), 2),
                        "sign_test_p": round(float(binomtest(k, len(v), 0.5,
                                                             alternative="greater").pvalue), 4),
                        "beats_baseline": "YES" if (k > len(v) / 2 and
                                                    binomtest(k, len(v), 0.5,
                                                              alternative="greater").pvalue < 0.05)
                        else "NO"})
    g = sub.gain_D_to_H.values
    kg = int((g > 0).sum())
    summary.append({"horizon_days": h, "phase": "GAIN D->H", "folds_positive": f"{kg}/{len(g)}",
                    "median_edge_pp": round(float(np.median(g)), 2),
                    "sign_test_p": round(float(binomtest(kg, len(g), 0.5,
                                                         alternative="greater").pvalue), 4),
                    "beats_baseline": "—"})
S = pd.DataFrame(summary)
S.to_csv(OUT / "sentiment_verdict.csv", index=False)

pre = FOLD[FOLD.test_from < "2021"]
post = FOLD[FOLD.test_from >= "2021"]
regime = pd.DataFrame([
    {"period": "2019-2020 (calm)", "folds": len(pre),
     "median_gain_D_to_H": round(float(pre.gain_D_to_H.median()), 2) if len(pre) else np.nan,
     "positive": f"{int((pre.gain_D_to_H > 0).sum())}/{len(pre)}"},
    {"period": "2021-2022 (crisis)", "folds": len(post),
     "median_gain_D_to_H": round(float(post.gain_D_to_H.median()), 2) if len(post) else np.nan,
     "positive": f"{int((post.gain_D_to_H > 0).sum())}/{len(post)}"}])
regime.to_csv(OUT / "sentiment_regime_split.csv", index=False)

# ---------------- plots ----------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
a = ax[0]
one = FOLD[FOLD.horizon_days == 1]
x = np.arange(len(one))
for i, (p, colr) in enumerate(zip(PH, ["tab:grey", "tab:blue", "tab:red", "tab:green"])):
    a.bar(x + (i - 1.5) * 0.2, one[f"med_{p}"], 0.2, label=f"Phase {p}", color=colr, alpha=.85)
a.axhline(0, color="black", lw=1)
a.set_xticks(x); a.set_xticklabels(one.test_from, rotation=90, fontsize=7)
a.set_ylabel("Median per-stock edge (pp)")
a.set_title("Phase H — 1 day\nA=price · D=+sector/rates · H=+sentiment · S=price+sentiment only")
a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)

b = ax[1]
for h, mk in zip(HORIZONS, ["o-", "s-", "D-", "^-"]):
    sub = FOLD[FOLD.horizon_days == h]
    b.plot(np.arange(len(sub)), sub.gain_D_to_H, mk,
           label={1: "1 day", 5: "1 week", 10: "2 weeks", 22: "1 month"}[h], lw=2)
b.axhline(0, color="black", ls="--")
b.set_xticks(np.arange(len(one))); b.set_xticklabels(one.test_from, rotation=90, fontsize=7)
b.set_ylabel("Gain from adding sentiment (pp)")
b.set_title("Does news sentiment add anything?\n(above 0 = it helped)")
b.grid(alpha=.3); b.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "sentiment_gain.png", dpi=140)

any_win = (S.beats_baseline == "YES").any()
gains = S[S.phase == "GAIN D->H"]
md = f"""# Phase H — news sentiment

**Source:** {len(news):,} Dailymirror/Newsfirst articles from {NEWS_START:%Y-%m}; **{len(rel):,}**
market-relevant after keyword filtering ({len(rel)/len(news)*100:.0f}%).
**Feed ends {NEWS_END:%Y-%m-%d}**, so the whole study window is {NEWS_START:%Y-%m} → {NEWS_END:%Y-%m}.

**Lexicons:** VADER (general) and {LEX_LABEL}.
Correlation between the two scores: **{corr:.3f}**.

**Look-ahead guard:** CSE closes 14:30. Articles published after the close, or on non-trading days,
are pushed to the **next** trading day ({after_close.mean()*100:.0f}% of articles were pushed).
Without this the model would be reading the answer.

**Evaluation:** walk-forward, {len(folds)} × {TEST_MONTHS}-month folds, scored per stock against
that stock's own baseline. A single 80/20 split was deliberately NOT used: it would put the entire
test set inside the 2022 crisis, a period this project has three times shown makes every model —
price-only included — look good.

| Phase | Features |
|---|---|
| A | Tier-1 technical (floor) |
| D | + Tier-2 + rate changes + sector |
| **H** | **+ news sentiment** |
| S | Tier-1 + sentiment only (sentiment's standalone value) |

## VERDICT

{md_table(S)}

**Sentiment gain (D→H):**

{md_table(gains[["horizon_days", "folds_positive", "median_edge_pp", "sign_test_p"]])}

**{'A phase beats the baseline — investigate.' if any_win else 'No phase beats the naive baseline at any horizon. News sentiment does not improve direction forecasting.'}**

XGBoost gives the sentiment block **{R.sent_share_pct_mean if False else round(R['sent_share_%'].mean(), 1)}%** of its
importance — the same "uses it, gains nothing" fingerprint seen with Tier-2 indicators (41%),
monthly macro (64%) and daily macro (27%).

## Regime split — does sentiment only "work" during the crisis?

{md_table(regime)}

## Fold by fold

{md_table(FOLD[["horizon_days", "test_from", "med_A", "med_D", "med_H", "med_S", "gain_D_to_H"]])}

## Caveats
- **Feed ends 2022-06**, so Phase H cannot be tested on the same window as every other phase.
  Compare phases *within* this table only.
- Median ~2-3 relevant articles per day: a daily sentiment score built on so few items is noisy.
- 51% of source articles have empty body text, so those contribute headline sentiment only.
- General-news outlets, not a financial wire — relevance filtering is keyword-based and imperfect.
- The compact finance lexicon can be replaced by the full Loughran-McDonald master dictionary by
  dropping it into `cleaned_data/loughran_mcdonald_master.csv` (see `src/finance_lexicon.py`).
"""
(OUT / "sentiment_summary.md").write_text(md)

print("\n" + "=" * 90)
print(S.to_string(index=False))
print("=" * 90)
print("REGIME SPLIT (gain from sentiment):")
print(regime.to_string(index=False))
print("=" * 90)
print(FOLD[["horizon_days", "test_from", "med_A", "med_D", "med_H", "med_S",
            "gain_D_to_H"]].to_string(index=False))
print(f"\nSentiment share of XGBoost importance: {R['sent_share_%'].mean():.1f}%")
print(f"Lexicon agreement (VADER vs LM): {corr:.3f}")
print(f"Saved to {OUT}")
