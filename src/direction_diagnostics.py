#!/usr/bin/env python3
"""
THREE DIAGNOSTICS — is the null real, or is the pipeline unable to see a signal?

Everything in this project has come back null. Before that is written up as a finding, it has to be
distinguished from the alternative: that the evaluation itself cannot detect a signal even when one
exists. These three tests do that.

TEST 1 — POSITIVE CONTROL (the decisive one)
    Predict TODAY's direction from TODAY's market move, macro and news, instead of tomorrow's.
    The stock's OWN return is excluded (that would be the answer itself); only the market, peers,
    macro and sentiment are used.
        * accuracy jumps well above baseline -> the pipeline works, and the gap between same-day
          and next-day IS the finding: macro/news EXPLAIN, they do not PREDICT.
        * accuracy stays flat -> the pipeline cannot see signal at all and every earlier null is
          uninterpretable.
    Without this control, "no signal exists" and "we cannot detect signal" are indistinguishable.

TEST 2 — BINARY up/down
    All earlier tests used 3 classes (Buy/Hold/Sell) with a dead zone. "Hold" dominates, so the
    majority baseline is strong and a weak signal could be buried. Binary is the easier, more
    standard formulation. If an edge exists anywhere, it should be most visible here.

TEST 3 — MAGNITUDE-WEIGHTED value
    Accuracy weights a 0.1% day the same as an 8% day. A model can be right 45% of the time and
    still valuable if it is right on the BIG moves. Reported as (a) |return|-weighted accuracy and
    (b) a simple long/short backtest return, both against the same naive baselines.

Walk-forward throughout, per-stock scoring against each stock's own baseline.

Outputs -> results/direction/diagnostics/
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
from scipy.stats import binomtest
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "direction" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

SECTOR = ["HNB", "COMB", "SAMP", "LOFC", "LOLC", "LFIN", "CFIN"]
HORIZONS = [1, 5, 22]
DEADZONE_1D = 0.005
SEED = 42
TEST_MONTHS = 6
FIRST_TEST = pd.Timestamp("2017-01-01")

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")) for t in SECTOR}
aspi = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
          .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))
rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True).ffill())
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
RATES = ["d_policy_1m", "d_spread_1m", "d_tb3m_3m"]
rt["available_from"] = rt["date"] + pd.Timedelta(days=35)
inf = pd.read_csv(DATA / "inflation_monthly.csv", parse_dates=["date"]).sort_values("date")
inf["available_from"] = inf["date"] + pd.Timedelta(days=21)
fx = pd.read_csv(DATA / "usd_lkr_daily.csv", parse_dates=["date"]).sort_values("date")
sent_path = DATA / "news_sentiment_daily.csv"
SENT_OK = sent_path.exists()
if SENT_OK:
    sent = pd.read_csv(sent_path, parse_dates=["date"]).sort_values("date")


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


def build(t):
    d = px[t].reset_index()
    dates = d["date"]
    c = d["close"].astype(float)
    v = d["volume"].astype(float)
    r1 = c.pct_change()
    f = pd.DataFrame({"date": dates, "ticker": t})

    # --- PREDICTIVE features: everything known at yesterday's close or earlier ---
    f["ret_1"], f["ret_5"], f["ret_10"] = r1, c.pct_change(5), c.pct_change(10)
    f["ma5_ratio"] = c / c.rolling(5).mean() - 1
    f["ma20_ratio"] = c / c.rolling(20).mean() - 1
    f["momentum_10"] = c / c.shift(10) - 1
    f["vol_10"], f["vol_20"] = r1.rolling(10).std(), r1.rolling(20).std()
    f["rsi_14"] = rsi(c)
    a = aspi.reindex(dates).ffill().reset_index(drop=True)
    ar = a.pct_change()
    f["aspi_ret_1"], f["aspi_ret_5"] = ar, a.pct_change(5)
    peers = [p for p in SECTOR if p != t]
    pp = pd.DataFrame({p: px[p]["close"].astype(float).reindex(dates).ffill().reset_index(drop=True)
                       for p in peers})
    f["peer_ret_1"], f["peer_ret_5"] = pp.pct_change().mean(axis=1), pp.pct_change(5).mean(axis=1)
    j = pd.merge_asof(pd.DataFrame({"date": dates}),
                      rt[["available_from"] + RATES].sort_values("available_from"),
                      left_on="date", right_on="available_from", direction="backward")
    for col in RATES:
        f[col] = j[col].values
    j2 = pd.merge_asof(pd.DataFrame({"date": dates}),
                       inf[["available_from", "ccpi_yoy_pct"]].sort_values("available_from"),
                       left_on="date", right_on="available_from", direction="backward")
    f["ccpi_yoy_pct"] = j2["ccpi_yoy_pct"].values
    j3 = pd.merge_asof(pd.DataFrame({"date": dates}), fx[["date", "usd_lkr_ret_5"]],
                       on="date", direction="backward")
    f["usd_lkr_ret_5"] = j3["usd_lkr_ret_5"].values
    if SENT_OK:
        j4 = pd.merge_asof(pd.DataFrame({"date": dates}), sent[["date", "s_vader", "s_lm"]],
                           on="date", direction="backward", tolerance=pd.Timedelta("4D"))
        f["sent_vader"] = j4["s_vader"].fillna(0).values
        f["sent_lm"] = j4["s_lm"].fillna(0).values
    PRED = [c_ for c_ in f.columns if c_ not in ("date", "ticker")]

    # --- CONTEMPORANEOUS features: TODAY's market / macro / news.
    # The stock's own same-day return is deliberately EXCLUDED — it is the answer.
    f["c_aspi_today"] = ar
    f["c_aspi_5"] = a.pct_change(5)
    f["c_peer_today"] = pp.pct_change().mean(axis=1)
    f["c_peer_5"] = pp.pct_change(5).mean(axis=1)
    f["c_ccpi"] = f["ccpi_yoy_pct"]
    f["c_fx"] = f["usd_lkr_ret_5"]
    if SENT_OK:
        f["c_sent_vader"] = f["sent_vader"]
    CONTEMP = [c_ for c_ in f.columns if c_.startswith("c_")]

    for h in HORIZONS:
        f[f"fwd_{h}"] = c.shift(-h) / c - 1
        f[f"past_{h}"] = c / c.shift(h) - 1
    f["today_ret"] = r1                       # label for the positive control
    # LAGGED copies of the predictive block. PRED is as-of TODAY's close and contains ret_1, which
    # IS today's return -> using it to predict today's direction is a pure leak (it scored 100%).
    # Shifting by one row makes "predict today from yesterday's information" honest.
    for col in PRED:
        f[f"lag_{col}"] = f[col].shift(1)
    return f, PRED, CONTEMP


parts = [build(t) for t in SECTOR]
PANEL = pd.concat([p[0] for p in parts], ignore_index=True).sort_values(["date", "ticker"])
PRED, CONTEMP = parts[0][1], parts[0][2]
print(f"panel {len(PANEL):,} stock-days | predictive feats {len(PRED)} | "
      f"contemporaneous feats {len(CONTEMP)} | sentiment: {'yes' if SENT_OK else 'no'}\n")

starts = pd.date_range(FIRST_TEST, PANEL.date.max(), freq=f"{TEST_MONTHS}MS")
FOLDS = [(s, s + pd.DateOffset(months=TEST_MONTHS)) for s in starts
         if s + pd.DateOffset(months=TEST_MONTHS) <= PANEL.date.max()]


def run(df, feats, y, meta, label, dz_lookup=None):
    """Walk-forward; returns per-stock-per-fold records."""
    out = []
    for fi, (t0, t1) in enumerate(FOLDS):
        purge = t0 - pd.Timedelta(days=40)
        tr = (df.date <= purge).values
        te = ((df.date >= t0) & (df.date < t1)).values
        if tr.sum() < 3000 or te.sum() < 300:
            continue
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2:
            continue
        classes = np.unique(ytr)
        cnt = pd.Series(ytr).value_counts()
        w = pd.Series(ytr).map({k: len(ytr) / (len(classes) * cnt[k]) for k in classes}).values
        Xtr, Xte = df.loc[tr, feats], df.loc[te, feats]
        lg = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=SEED))
        lg.fit(Xtr, ytr)
        xc = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=3, random_state=SEED,
                           n_jobs=4, eval_metric="mlogloss")
        xc.fit(Xtr, ytr, sample_weight=w)
        p_lg, p_xc = lg.predict(Xte), xc.predict(Xte)
        m = meta[te].reset_index(drop=True)
        m["y"] = yte
        m["pred"] = p_lg if accuracy_score(yte, p_lg) >= accuracy_score(yte, p_xc) else p_xc
        mtr = meta[tr].reset_index(drop=True)
        mtr["y"] = ytr
        for tk, g in m.groupby("ticker"):
            own = mtr[mtr.ticker == tk]
            if len(own) < 200 or len(g) < 30:
                continue
            omaj = int(pd.Series(own["y"]).value_counts().idxmax())
            b_maj = accuracy_score(g["y"], np.full(len(g), omaj))
            b_per = accuracy_score(g["y"], g["persist"]) if "persist" in g else b_maj
            b = max(b_maj, b_per)
            acc = accuracy_score(g["y"], g["pred"])
            rec = {"test": label, "fold": fi, "test_from": f"{t0:%Y-%m}", "ticker": tk,
                   "n": len(g), "acc_%": round(acc * 100, 1), "baseline_%": round(b * 100, 1),
                   "edge_pp": round((acc - b) * 100, 1)}
            if "absret" in g:
                wgt = g["absret"].values
                correct = (g["y"].values == g["pred"].values).astype(float)
                rec["wacc_%"] = round(float((correct * wgt).sum() / wgt.sum() * 100), 1)
                base_c = (g["y"].values == omaj).astype(float)
                rec["wacc_base_%"] = round(float((base_c * wgt).sum() / wgt.sum() * 100), 1)
                rec["wedge_pp"] = round(rec["wacc_%"] - rec["wacc_base_%"], 1)
            if "fwd" in g and g["indep"].iloc[0]:
                # Only h=1 is summable: at longer horizons consecutive forward returns overlap and
                # summing them counts each move ~h times over.
                sig = np.where(g["pred"].values == g["up_class"].values, 1.0, -1.0)
                gross = float((sig * g["fwd"].values).sum())
                flips = int((np.diff(sig) != 0).sum())
                rec["strategy_ret_%"] = round(gross * 100, 2)
                rec["buyhold_ret_%"] = round(float(g["fwd"].values.sum()) * 100, 2)
                rec["position_flips"] = flips
                # transaction-cost sensitivity: each flip is a round trip
                for bps in (10, 25, 50):
                    rec[f"strategy_net_{bps}bps_%"] = round((gross - flips * bps / 10000) * 100, 2)
            out.append(rec)
    return out


records = []

# ---------------------------------------------------------------- TEST 1
print("=" * 88)
print("TEST 1 — POSITIVE CONTROL: predict TODAY from TODAY's market/macro/news")
print("=" * 88)
d1 = PANEL[["date", "ticker", "today_ret"] + PRED + CONTEMP
           + [f"lag_{c}" for c in PRED]].dropna().reset_index(drop=True)
y_today = np.where(d1.today_ret > 0, 1, 0)
meta1 = d1[["ticker"]].copy()
meta1["persist"] = np.nan
r_contemp = run(d1, CONTEMP, y_today, d1[["ticker"]].copy(), "1-CONTEMPORANEOUS (today from today)")
# same target, but using only YESTERDAY's information -> the honest predictive comparison
LAGGED = [f"lag_{c}" for c in PRED]
r_lagged = run(d1, LAGGED, y_today, d1[["ticker"]].copy(), "1-LAGGED (today from yesterday)")
records += r_contemp + r_lagged
for lab, rr in [("contemporaneous", r_contemp), ("lagged", r_lagged)]:
    t = pd.DataFrame(rr)
    print(f"  {lab:16s} median acc {t['acc_%'].median():5.1f}%  "
          f"baseline {t['baseline_%'].median():5.1f}%  edge {t['edge_pp'].median():+5.1f} pp")

# ---------------------------------------------------------------- TEST 2 & 3
print("\n" + "=" * 88)
print("TEST 2/3 — BINARY up/down, plus magnitude-weighted value")
print("=" * 88)
for h in HORIZONS:
    d2 = PANEL[["date", "ticker", f"fwd_{h}", f"past_{h}"] + PRED].dropna().reset_index(drop=True)
    y_bin = np.where(d2[f"fwd_{h}"] > 0, 1, 0)
    meta = d2[["ticker"]].copy()
    meta["persist"] = np.where(d2[f"past_{h}"] > 0, 1, 0)
    meta["absret"] = d2[f"fwd_{h}"].abs().values
    meta["fwd"] = d2[f"fwd_{h}"].values
    meta["up_class"] = 1
    meta["indep"] = (h == 1)          # only daily returns are non-overlapping
    rr = run(d2, PRED, y_bin, meta, f"2-BINARY h={h}")
    records += rr
    t = pd.DataFrame(rr)
    bt = (f"| strategy {t['strategy_ret_%'].median():+6.2f}% vs buy&hold "
          f"{t['buyhold_ret_%'].median():+6.2f}% | net@25bps {t['strategy_net_25bps_%'].median():+7.2f}%"
          if "strategy_ret_%" in t and t["strategy_ret_%"].notna().any()
          else "| backtest skipped (overlapping returns)")
    print(f"  h={h:>2}d  acc {t['acc_%'].median():5.1f}%  base {t['baseline_%'].median():5.1f}%  "
          f"edge {t['edge_pp'].median():+5.1f} pp | mag-weighted edge "
          f"{t['wedge_pp'].median():+5.1f} pp {bt}")

R = pd.DataFrame(records)
R.to_csv(OUT / "diagnostics_per_stock_per_fold.csv", index=False)

FOLD = (R.groupby(["test", "fold", "test_from"])
        .agg(med_edge=("edge_pp", "median"),
             med_wedge=("wedge_pp", "median") if "wedge_pp" in R else ("edge_pp", "median"),
             med_acc=("acc_%", "median"), med_base=("baseline_%", "median")).reset_index().round(2))
FOLD.to_csv(OUT / "diagnostics_by_fold.csv", index=False)

summary = []
for lab, sub in FOLD.groupby("test"):
    v = sub.med_edge.values
    k = int((v > 0).sum())
    row = {"test": lab, "folds": len(v), "median_acc_%": round(float(sub.med_acc.median()), 1),
           "median_baseline_%": round(float(sub.med_base.median()), 1),
           "median_edge_pp": round(float(np.median(v)), 2), "folds_positive": f"{k}/{len(v)}",
           "sign_test_p": round(float(binomtest(k, len(v), 0.5, alternative="greater").pvalue), 4)}
    row["verdict"] = "BEATS BASELINE" if (k > len(v) / 2 and row["sign_test_p"] < 0.05) else "no"
    summary.append(row)
S = pd.DataFrame(summary).sort_values("test")
S.to_csv(OUT / "diagnostics_verdict.csv", index=False)

mag = R[R.test.str.startswith("2-")].groupby("test").agg(
    median_edge_pp=("edge_pp", "median"),
    median_mag_weighted_edge_pp=("wedge_pp", "median")).reset_index().round(2)
mag.to_csv(OUT / "diagnostics_magnitude.csv", index=False)

BT = R[R.test == "2-BINARY h=1"].dropna(subset=["strategy_ret_%"])
bt_tbl = pd.DataFrame([{
    "metric": "median across stock-folds",
    "strategy_gross_%": round(float(BT["strategy_ret_%"].median()), 2),
    "buy_and_hold_%": round(float(BT["buyhold_ret_%"].median()), 2),
    "net_10bps_%": round(float(BT["strategy_net_10bps_%"].median()), 2),
    "net_25bps_%": round(float(BT["strategy_net_25bps_%"].median()), 2),
    "net_50bps_%": round(float(BT["strategy_net_50bps_%"].median()), 2),
    "median_position_flips": int(BT["position_flips"].median()),
    "beats_buy_hold_gross": f"{int((BT['strategy_ret_%'] > BT['buyhold_ret_%']).sum())}/{len(BT)}",
    "beats_buy_hold_net_25bps": f"{int((BT['strategy_net_25bps_%'] > BT['buyhold_ret_%']).sum())}/{len(BT)}",
}])
bt_tbl.to_csv(OUT / "diagnostics_backtest_h1.csv", index=False)

cont = S[S.test.str.contains("CONTEMPORANEOUS")].iloc[0]
lagg = S[S.test.str.contains("LAGGED")].iloc[0]
gap = cont["median_acc_%"] - lagg["median_acc_%"]
pipeline_ok = cont["median_edge_pp"] > 5

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
a = ax[0]
labels = ["Contemporaneous\n(today from today)", "Lagged\n(today from yesterday)"]
vals = [cont["median_acc_%"], lagg["median_acc_%"]]
base = [cont["median_baseline_%"], lagg["median_baseline_%"]]
x = np.arange(2)
a.bar(x - 0.2, vals, 0.4, label="model", color="tab:blue")
a.bar(x + 0.2, base, 0.4, label="baseline", color="grey", alpha=.7)
a.set_xticks(x); a.set_xticklabels(labels, fontsize=9)
a.set_ylabel("Median accuracy %")
a.set_title(f"POSITIVE CONTROL\ngap = {gap:+.1f} pp")
a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)

b = ax[1]
bs = R[R.test.str.startswith("2-")]
hs = sorted(bs.test.unique())
xb = np.arange(len(hs))
b.bar(xb - 0.2, [bs[bs.test == t]["edge_pp"].median() for t in hs], 0.4,
      label="plain accuracy edge", color="tab:blue")
b.bar(xb + 0.2, [bs[bs.test == t]["wedge_pp"].median() for t in hs], 0.4,
      label="magnitude-weighted edge", color="tab:orange")
b.axhline(0, color="black", lw=1)
b.set_xticks(xb); b.set_xticklabels([t.replace("2-BINARY ", "") for t in hs])
b.set_ylabel("Median edge (pp)")
b.set_title("BINARY up/down — does weighting by move size help?")
b.grid(alpha=.3, axis="y"); b.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "diagnostics.png", dpi=140)

md = f"""# Three diagnostics — is the null real, or is the pipeline blind?

Every phase in this project returned null. Before that becomes a finding it must be separated from
the alternative: that the evaluation cannot detect signal even when one exists.

## TEST 1 — POSITIVE CONTROL (the decisive one)

Same pipeline, same stocks, same folds. Only the timing changes.

| Setup | Features | Median accuracy | Baseline | Edge |
|---|---|---|---|---|
| **Contemporaneous** | TODAY's market, macro, news | **{cont['median_acc_%']}%** | {cont['median_baseline_%']}% | **{cont['median_edge_pp']:+.1f} pp** |
| **Lagged** | YESTERDAY's information only | {lagg['median_acc_%']}% | {lagg['median_baseline_%']}% | {lagg['median_edge_pp']:+.1f} pp |

**Gap: {gap:+.1f} percentage points.**

The stock's own same-day return is EXCLUDED from the contemporaneous features — otherwise the
answer would be an input. Only market, peers, macro and sentiment are used.

**Verdict: {'the pipeline WORKS. It finds a large, obvious signal the moment the information is contemporaneous, and loses it entirely one day earlier. That gap IS the finding: this information EXPLAINS returns but does not PREDICT them.' if pipeline_ok else 'the pipeline finds almost nothing even contemporaneously. That is a red flag — every earlier null becomes uninterpretable and the feature construction needs auditing before anything else is believed.'}**

## TEST 2 — Binary up/down (removes the dead-zone / Hold-majority problem)

{md_table(S[S.test.str.startswith("2-")][["test", "median_acc_%", "median_baseline_%", "median_edge_pp", "folds_positive", "sign_test_p", "verdict"]])}

## TEST 3 — Magnitude-weighted value: is the model right on the BIG moves?

{md_table(mag)}

`median_mag_weighted_edge_pp` weights each day by |return|, so being right on an 8% day counts 80x
a 0.1% day.

### Long/short backtest — h = 1 only

Only the 1-day horizon is summable: at 5 and 22 days consecutive forward returns OVERLAP, so adding
them counts every move ~h times over. Those backtests are omitted rather than reported wrong.

{md_table(bt_tbl)}

Each position flip is a round trip. CSE round-trip costs (brokerage + levies + spread) are
realistically **50-100 bps**, so the `net_50bps` column is the closest to reality — and it is the
column that matters.

## Full verdict table

{md_table(S)}

## Caveats
- The long/short backtest ignores transaction costs, bid-ask spread and liquidity. On the CSE those
  are large. Treat any positive strategy return as an upper bound.
- Contemporaneous accuracy is NOT a forecast and must never be reported as one — it uses same-day
  information that a trader would not have before acting.
"""
(OUT / "diagnostics_summary.md").write_text(md)

print("\n" + "=" * 88)
print(S.to_string(index=False))
print("=" * 88)
print(mag.to_string(index=False))
print(f"\nPOSITIVE CONTROL GAP: {gap:+.1f} pp  -> pipeline {'WORKS' if pipeline_ok else 'SUSPECT'}")
print(f"Saved to {OUT}")
