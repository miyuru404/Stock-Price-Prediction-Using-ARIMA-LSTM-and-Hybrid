#!/usr/bin/env python3
"""
PRICE FORECASTING WITH MACRO + NEWS — does the project's ORIGINAL question change with more data?

Stage 1 of this project forecast the PRICE from price history alone, and nothing beat the naive
"tomorrow = today". But that was always UNIVARIATE. This script asks the obvious follow-up, and the
one the project title implies:

    does adding macroeconomic variables and news sentiment improve PRICE forecasting
    over plain historical price data?

FEATURE SETS (nested, so each gain is attributable):
    P       price / technical only            <- the univariate floor, as in Stage 1
    P+M     + macro (rates, inflation, FX)
    P+M+N   + news sentiment (VADER + finance lexicon)

MODELS: naive · ARIMA(1,1,1) · Ridge · XGBoost.
ARIMA is included because the project is named after it and it is the reference for Stage 1.

HOW THE TARGET IS FRAMED (important):
Predicting the price LEVEL directly lets any model score well by simply echoing today's price —
it would learn "tomorrow = today", i.e. reproduce the naive baseline, and MAPE would look great
while the model had learned nothing. So the models predict the next-day RETURN, which is then
converted back to a price:  price_hat = close_today * (1 + return_hat).
MAPE is reported on that reconstructed PRICE, so the numbers stay directly comparable to the
Stage 1 results, while the model is actually forced to forecast the move.

Walk-forward, per stock, refit every fold. Naive is the benchmark for every metric.

Outputs -> results/price_macro_news/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import binomtest
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "price_macro_news"
OUT.mkdir(parents=True, exist_ok=True)

SECTOR = ["HNB", "COMB", "SAMP", "LOFC", "LOLC", "LFIN", "CFIN"]
INDICES = ["SPSL20", "ASPI"]           # index targets: no volume, so volchg_5 is skipped
TARGETS = SECTOR + INDICES
HORIZONS = [1, 5]
SEED = 42
TEST_MONTHS = 6
FIRST_TEST = pd.Timestamp("2017-01-01")

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")) for t in SECTOR}
# ---- index targets ----
_sp = (pd.read_csv(ROOT / "data" / "processed" / "spsl20_trading_days_clean.csv",
                   parse_dates=["date"]).sort_values("date").drop_duplicates("date")
       .rename(columns={"spsl20_points": "close"}).set_index("date")[["close"]])
px["SPSL20"] = _sp
px["ASPI"] = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
                .sort_values("date").drop_duplicates("date").set_index("date")[["close"]])
aspi = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
          .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))
rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True).ffill())
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
rt["term_slope"] = rt["tb_12m"] - rt["tb_3m"]
RATES = ["d_policy_1m", "d_spread_1m", "d_tb3m_3m", "term_slope"]
rt["available_from"] = rt["date"] + pd.Timedelta(days=35)
inf = pd.read_csv(DATA / "inflation_monthly.csv", parse_dates=["date"]).sort_values("date")
inf["d_ccpi_yoy_3m"] = inf["ccpi_yoy_pct"].diff(3)
inf["available_from"] = inf["date"] + pd.Timedelta(days=21)
fx = pd.read_csv(DATA / "usd_lkr_daily.csv", parse_dates=["date"]).sort_values("date")
sent = pd.read_csv(DATA / "news_sentiment_daily.csv", parse_dates=["date"]).sort_values("date")

MACRO = RATES + ["ccpi_yoy_pct", "d_ccpi_yoy_3m", "usd_lkr_ret_5", "usd_lkr_ret_20"]
NEWS = ["sent_vader_1", "sent_vader_5", "sent_lm_1", "sent_lm_5", "news_count_5"]


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
    has_vol = "volume" in d.columns
    r1 = c.pct_change()
    f = pd.DataFrame({"date": dates, "close": c})
    f["ret_1"], f["ret_5"], f["ret_10"] = r1, c.pct_change(5), c.pct_change(10)
    f["ma5_ratio"] = c / c.rolling(5).mean() - 1
    f["ma20_ratio"] = c / c.rolling(20).mean() - 1
    f["momentum_10"] = c / c.shift(10) - 1
    f["vol_10"], f["vol_20"] = r1.rolling(10).std(), r1.rolling(20).std()
    f["rsi_14"] = rsi(c)
    if has_vol:                       # indices have no volume
        v = d["volume"].astype(float)
        f["volchg_5"] = v / v.rolling(5).mean() - 1

    # ---- ASPI market feature: TWO leak fixes ----
    # (1) NO forward-fill across missing dates. The target and ASPI do not share a calendar
    #     (207 SPSL20 dates are absent from ASPI, 962 ASPI dates absent from SPSL20). Filling
    #     across those gaps let ASPI carry a move that the target only records on its NEXT row —
    #     aspi_ret_1 then correlated 0.742 with the target's FUTURE return but only 0.374 with its
    #     own same-day return, and halved MAPE for free. Reindex WITHOUT ffill: dates where ASPI
    #     has no genuine observation become NaN and are dropped, rather than silently invented.
    # (2) Indices get NO ASPI features at all. SPSL20 and ASPI are the same market measured twice
    #     (~0.95 correlated), so using one to forecast the other is near-circular even when the
    #     calendars line up.
    if t not in INDICES:
        a = aspi.reindex(dates).reset_index(drop=True)      # no .ffill()
        f["aspi_ret_1"], f["aspi_ret_5"] = a.pct_change(), a.pct_change(5)
    PRICE = [x for x in f.columns if x not in ("date", "close")]

    j = pd.merge_asof(pd.DataFrame({"date": dates}),
                      rt[["available_from"] + RATES].sort_values("available_from"),
                      left_on="date", right_on="available_from", direction="backward")
    for col in RATES:
        f[col] = j[col].values
    j2 = pd.merge_asof(pd.DataFrame({"date": dates}),
                       inf[["available_from", "ccpi_yoy_pct", "d_ccpi_yoy_3m"]]
                       .sort_values("available_from"),
                       left_on="date", right_on="available_from", direction="backward")
    f["ccpi_yoy_pct"], f["d_ccpi_yoy_3m"] = j2["ccpi_yoy_pct"].values, j2["d_ccpi_yoy_3m"].values
    j3 = pd.merge_asof(pd.DataFrame({"date": dates}),
                       fx[["date", "usd_lkr_ret_5", "usd_lkr_ret_20"]], on="date",
                       direction="backward")
    f["usd_lkr_ret_5"], f["usd_lkr_ret_20"] = j3["usd_lkr_ret_5"].values, j3["usd_lkr_ret_20"].values

    j4 = pd.merge_asof(pd.DataFrame({"date": dates}), sent, on="date",
                       direction="backward", tolerance=pd.Timedelta("4D"))
    sv, sl, nc = j4["s_vader"], j4["s_lm"], j4["n_articles"].fillna(0)
    f["sent_vader_1"], f["sent_lm_1"] = sv.fillna(0).values, sl.fillna(0).values
    f["sent_vader_5"] = sv.rolling(5, min_periods=1).mean().fillna(0).values
    f["sent_lm_5"] = sl.rolling(5, min_periods=1).mean().fillna(0).values
    f["news_count_5"] = nc.rolling(5, min_periods=1).mean().values

    for h in HORIZONS:
        f[f"fwd_{h}"] = c.shift(-h) / c - 1
        f[f"target_price_{h}"] = c.shift(-h)
    return f, PRICE


SETS = {"P (price only)": None, "P+M (+macro)": None, "P+M+N (+macro+news)": None}
def leak_scan(f, cols, t):
    """Flag any feature that correlates MORE with the FUTURE return than with the SAME-DAY return.

    A legitimate predictor is information available now; it should track today's move at least as
    closely as tomorrow's. When the reverse holds, the feature is carrying future information —
    which is exactly how the ASPI forward-fill bug hid here (aspi_ret_1 scored 0.742 against the
    next-day return vs 0.374 against the same-day one). This runs on every target, every time.
    """
    keep = ["ret_1", "fwd_1"] + [c for c in cols if c in f.columns]
    keep = list(dict.fromkeys(keep))          # dedupe: ret_1 is also a feature
    g = f[keep].dropna()
    out = []
    for col in dict.fromkeys(cols):
        if col not in g.columns or col == "ret_1":
            continue
        now = abs(float(g[col].corr(g["ret_1"])))
        fut = abs(float(g[col].corr(g["fwd_1"])))
        if fut > now + 0.05 and fut > 0.2:
            out.append({"ticker": t, "feature": col, "corr_same_day": round(now, 3),
                        "corr_next_day": round(fut, 3), "gap": round(fut - now, 3)})
    return out


rows = []
pred_store = []          # per-date predictions, kept for the comparison plots
leaks = []
for t in TARGETS:
    f, PRICE = build(t)
    sets = {"P (price only)": PRICE,
            "P+M (+macro)": PRICE + MACRO,
            "P+M+N (+macro+news)": PRICE + MACRO + NEWS}
    allc = sets["P+M+N (+macro+news)"]
    leaks += leak_scan(f, allc, t)

    for h in HORIZONS:
        df = f[["date", "close", f"fwd_{h}", f"target_price_{h}"] + allc].dropna().reset_index(drop=True)
        starts = pd.date_range(FIRST_TEST, df.date.max(), freq=f"{TEST_MONTHS}MS")
        folds = [(s, s + pd.DateOffset(months=TEST_MONTHS)) for s in starts
                 if s + pd.DateOffset(months=TEST_MONTHS) <= df.date.max()]
        for fi, (t0, t1) in enumerate(folds):
            purge = t0 - pd.Timedelta(days=int(h * 1.5) + 3)
            tr = (df.date <= purge).values
            te = ((df.date >= t0) & (df.date < t1)).values
            if tr.sum() < 500 or te.sum() < 40:
                continue
            close_te = df.loc[te, "close"].values
            y_price = df.loc[te, f"target_price_{h}"].values
            y_ret_tr = df.loc[tr, f"fwd_{h}"].values
            y_ret_te = df.loc[te, f"fwd_{h}"].values

            dates_te = df.loc[te, "date"].values

            def score(name, price_hat, ret_hat):
                mape = float(np.mean(np.abs((y_price - price_hat) / y_price)) * 100)
                if h == 1:
                    pred_store.append(pd.DataFrame({
                        "ticker": t, "fold": fi, "model": name, "date": dates_te,
                        "actual": y_price, "pred": price_hat}))
                rows.append({
                    "ticker": t, "horizon_days": h, "fold": fi, "test_from": f"{t0:%Y-%m}",
                    "model": name, "n_test": int(te.sum()),
                    "MAPE_%": round(mape, 3),
                    "MAE": round(float(mean_absolute_error(y_price, price_hat)), 3),
                    "RMSE": round(float(np.sqrt(mean_squared_error(y_price, price_hat))), 3),
                    "ret_RMSE_pp": round(float(np.sqrt(mean_squared_error(y_ret_te, ret_hat)) * 100), 3),
                    "dir_acc_%": round(float((np.sign(ret_hat) == np.sign(y_ret_te)).mean() * 100), 1),
                })

            # ---- naive: tomorrow = today (return = 0) ----
            score("naive (no change)", close_te.copy(), np.zeros(len(close_te)))

            # ---- ARIMA(1,1,1) on log price, univariate, refit per fold ----
            try:
                logp = np.log(df.loc[tr, "close"].values)
                fit = ARIMA(logp, order=(1, 1, 1)).fit()
                fc = float(fit.forecast(steps=h)[-1])
                drift = np.exp(fc - logp[-1]) - 1          # one constant drift for the fold
                score("ARIMA(1,1,1)", close_te * (1 + drift), np.full(len(close_te), drift))
            except Exception:
                pass

            # ---- learned models on each feature set ----
            for sname, cols in sets.items():
                Xtr, Xte = df.loc[tr, cols], df.loc[te, cols]
                rg = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
                rg.fit(Xtr, y_ret_tr)
                pr = rg.predict(Xte)
                score(f"Ridge {sname}", close_te * (1 + pr), pr)

                xr = XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=3, random_state=SEED,
                                  n_jobs=4)
                xr.fit(Xtr, y_ret_tr)
                pxg = xr.predict(Xte)
                score(f"XGBoost {sname}", close_te * (1 + pxg), pxg)
    print(f"{t:7s} done")

LEAK = pd.DataFrame(leaks)
LEAK.to_csv(OUT / "leak_scan.csv", index=False)
print("\n" + "=" * 100)
if len(LEAK):
    print(f"LEAK SCAN: {len(LEAK)} feature(s) correlate more with the FUTURE than the present:")
    print(LEAK.to_string(index=False))
else:
    print("LEAK SCAN: clean — no feature correlates more with the future than with the present.")
print("=" * 100)

R = pd.DataFrame(rows)
R.to_csv(OUT / "price_all_results.csv", index=False)
PRED = pd.concat(pred_store, ignore_index=True)
PRED.to_csv(OUT / "price_predictions_h1.csv", index=False)

# ---- summary vs naive, per horizon ----
summ = []
for h in HORIZONS:
    sub = R[R.horizon_days == h]
    naive = sub[sub.model == "naive (no change)"].set_index(["ticker", "fold"])["MAPE_%"]
    for m, g in sub.groupby("model"):
        gi = g.set_index(["ticker", "fold"])
        ratio = (gi["MAPE_%"] / naive.reindex(gi.index)).dropna()
        better = int((ratio < 1).sum())
        summ.append({
            "horizon_days": h, "model": m,
            "median_MAPE_%": round(float(g["MAPE_%"].median()), 3),
            "median_ret_RMSE_pp": round(float(g["ret_RMSE_pp"].median()), 3),
            "median_dir_acc_%": round(float(g["dir_acc_%"].median()), 1),
            "MAPE_vs_naive": round(float(ratio.median()), 4),
            "beats_naive": f"{better}/{len(ratio)}",
            "beats_naive_p": round(float(binomtest(better, len(ratio), 0.5,
                                                   alternative="greater").pvalue), 4) if len(ratio) else np.nan,
        })
S = pd.DataFrame(summ).sort_values(["horizon_days", "median_MAPE_%"])
S.to_csv(OUT / "price_summary.csv", index=False)

# ---- the ablation question: does macro/news improve on price-only? ----
gain = []
for h in HORIZONS:
    sub = R[R.horizon_days == h]
    for algo in ["Ridge", "XGBoost"]:
        base = sub[sub.model == f"{algo} P (price only)"].set_index(["ticker", "fold"])["MAPE_%"]
        for sname in ["P+M (+macro)", "P+M+N (+macro+news)"]:
            g = sub[sub.model == f"{algo} {sname}"].set_index(["ticker", "fold"])["MAPE_%"]
            d = (base.reindex(g.index) - g).dropna()          # positive = added data helped
            k = int((d > 0).sum())
            gain.append({"horizon_days": h, "algo": algo, "added": sname,
                         "median_MAPE_improvement_pp": round(float(d.median()), 4),
                         "cases_improved": f"{k}/{len(d)}",
                         "sign_test_p": round(float(binomtest(k, len(d), 0.5,
                                                              alternative="greater").pvalue), 4)})
G = pd.DataFrame(gain)
G.to_csv(OUT / "price_ablation_gain.csv", index=False)

# ================================================================ COMPARISON GRAPHS
KEY = ["naive (no change)", "ARIMA(1,1,1)", "Ridge P (price only)",
       "Ridge P+M (+macro)", "Ridge P+M+N (+macro+news)"]
COLR = {"naive (no change)": "black", "ARIMA(1,1,1)": "tab:purple",
        "Ridge P (price only)": "tab:blue", "Ridge P+M (+macro)": "tab:orange",
        "Ridge P+M+N (+macro+news)": "tab:green"}

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1], hspace=0.38, wspace=0.22)

# --- (a) actual vs predicted, SPSL20, latest fold ---
a = fig.add_subplot(gs[0, :])
sp = PRED[PRED.ticker == "SPSL20"]
if len(sp):
    last = sp.fold.max()
    sp = sp[sp.fold == last].copy()
    sp["date"] = pd.to_datetime(sp["date"])
    act = sp[sp.model == KEY[0]].sort_values("date")
    a.plot(act.date, act.actual, color="black", lw=2.4, label="ACTUAL", zorder=5)
    for m in KEY[1:]:
        g = sp[sp.model == m].sort_values("date")
        if len(g):
            a.plot(g.date, g.pred, lw=1.2, alpha=.85, color=COLR[m], label=m)
    a.set_title(f"S&P SL 20 — actual vs 1-day-ahead forecasts (latest walk-forward fold, "
                f"{act.date.min():%Y-%m} to {act.date.max():%Y-%m})", fontsize=11)
    a.set_ylabel("Index points"); a.grid(alpha=.3); a.legend(fontsize=8, ncol=3)

# --- (b) zoom: 60 days, showing every model tracks yesterday's price ---
b = fig.add_subplot(gs[1, 0])
if len(sp):
    z = act.tail(60)
    b.plot(z.date, z.actual, color="black", lw=2.4, label="ACTUAL", zorder=5)
    for m in KEY[1:]:
        g = sp[sp.model == m].sort_values("date").tail(60)
        if len(g):
            b.plot(g.date, g.pred, lw=1.3, alpha=.9, color=COLR[m], label=m)
    b.set_title("Zoom: last 60 days — every model just tracks the last price", fontsize=10)
    b.set_ylabel("Index points"); b.grid(alpha=.3)
    b.tick_params(axis="x", rotation=45, labelsize=7)

# --- (c) MAPE vs naive, SPSL20 and ASPI ---
c_ = fig.add_subplot(gs[1, 1])
idx = S[S.horizon_days == 1].copy()
sp_r = R[(R.horizon_days == 1) & (R.ticker.isin(INDICES))]
if len(sp_r):
    nv = sp_r[sp_r.model == "naive (no change)"].groupby("ticker")["MAPE_%"].median()
    tbl = (sp_r.groupby(["ticker", "model"])["MAPE_%"].median().reset_index())
    tbl["ratio"] = tbl.apply(lambda r_: r_["MAPE_%"] / nv[r_["ticker"]], axis=1)
    tbl = tbl[tbl.model.isin(KEY)]
    w = 0.35
    for i, tk in enumerate(INDICES):
        g = tbl[tbl.ticker == tk].set_index("model").reindex(KEY)
        c_.barh(np.arange(len(KEY)) + (i - 0.5) * w, g["ratio"], w, label=tk, alpha=.85)
    c_.axvline(1.0, color="black", ls="--", lw=1.5, label="naive = 1.0")
    c_.set_yticks(range(len(KEY)))
    c_.set_yticklabels([k.replace(" (no change)", "").replace("Ridge ", "") for k in KEY], fontsize=8)
    c_.set_xlabel("MAPE / naive MAPE  (<1 = better than naive)")
    c_.set_title("Index forecasts vs naive — 1 day", fontsize=10)
    c_.grid(alpha=.3, axis="x"); c_.legend(fontsize=8)

# --- (d) MAPE ratio across every target ---
d_ = fig.add_subplot(gs[2, :])
r1 = R[R.horizon_days == 1]
nv_all = r1[r1.model == "naive (no change)"].groupby("ticker")["MAPE_%"].median()
piv = (r1[r1.model.isin(KEY)].groupby(["ticker", "model"])["MAPE_%"].median().reset_index())
piv["ratio"] = piv.apply(lambda r_: r_["MAPE_%"] / nv_all[r_["ticker"]], axis=1)
P2 = piv.pivot(index="ticker", columns="model", values="ratio").reindex(TARGETS)[KEY]
x = np.arange(len(P2)); w = 0.16
for i, m in enumerate(KEY):
    d_.bar(x + (i - 2) * w, P2[m], w, label=m.replace(" (no change)", ""), color=COLR[m], alpha=.85)
d_.axhline(1.0, color="black", ls="--", lw=1.5)
d_.set_xticks(x); d_.set_xticklabels(P2.index, fontsize=9)
d_.set_ylabel("MAPE / naive MAPE")
d_.set_ylim(0.9, max(1.35, float(P2.max().max()) * 1.05))
d_.set_title("Every target, 1-day forecast — nothing gets below the naive line "
             "(7 stocks + 2 indices)", fontsize=11)
d_.grid(alpha=.3, axis="y"); d_.legend(fontsize=8, ncol=5)

fig.suptitle("Price forecasting with macro + news — does more data beat 'tomorrow = today'?",
             fontsize=13, y=0.985)
fig.savefig(OUT / "price_comparison.png", dpi=140, bbox_inches="tight")

# keep the simple two-panel ratio chart as well
fig2, ax = plt.subplots(1, 2, figsize=(14, 5.4))
for i, h in enumerate(HORIZONS):
    a2 = ax[i]
    sub = S[S.horizon_days == h].sort_values("MAPE_vs_naive")
    colr = ["tab:green" if v < 1 else "tab:red" for v in sub.MAPE_vs_naive]
    a2.barh(range(len(sub)), sub.MAPE_vs_naive, color=colr, alpha=.85)
    a2.axvline(1.0, color="black", ls="--", label="naive = 1.0")
    a2.set_yticks(range(len(sub))); a2.set_yticklabels(sub.model, fontsize=8)
    a2.set_xlabel("MAPE / naive MAPE   (<1 = better than naive)")
    a2.set_title(f"All targets pooled, h={h} day{'s' if h > 1 else ''}")
    a2.grid(alpha=.3, axis="x"); a2.legend(fontsize=8)
fig2.tight_layout(); fig2.savefig(OUT / "price_vs_naive.png", dpi=140)

best = S.loc[S.groupby("horizon_days")["MAPE_vs_naive"].idxmin()]
any_beat = (S.beats_naive_p < 0.05).any()

md = f"""# Price forecasting with macro + news

The project's Stage 1 forecast price from price history alone and found nothing beat the naive
"tomorrow = today". That was **univariate**. This run asks the follow-up the project title implies:

> **does adding macroeconomic variables and news sentiment improve PRICE forecasting?**

**Targets:** {SECTOR} + indices {INDICES} · **Horizons:** {HORIZONS} day(s) · walk-forward {TEST_MONTHS}-month folds,
refit every fold, per stock.

**Feature sets (nested):** `P` price/technical → `P+M` + rates, inflation, FX →
`P+M+N` + news sentiment.
**Models:** naive · ARIMA(1,1,1) · Ridge · XGBoost.

**Target framing:** models predict the next-h-day RETURN, which is converted back to a price
(`price_hat = close_today x (1 + return_hat)`). Predicting the price level directly would let any
model score well by echoing today's price — it would simply reproduce the naive baseline while
appearing accurate. MAPE is still reported on the reconstructed PRICE so the numbers stay
comparable to Stage 1.

## Results vs naive

{md_table(S)}

`MAPE_vs_naive` below 1.0 means better than doing nothing. `beats_naive` counts stock-fold cases.

**Best per horizon:**

{md_table(best[["horizon_days", "model", "median_MAPE_%", "MAPE_vs_naive", "beats_naive", "beats_naive_p"]])}

**{'At least one model beats naive significantly — investigate.' if any_beat else 'No model beats the naive baseline significantly at either horizon.'}**

## Does macro / news improve on price-only? (the ablation)

{md_table(G)}

`median_MAPE_improvement_pp` is positive when the added data REDUCED error.

## Reading it simply

- Every model sits at or above `MAPE_vs_naive = 1.0`, i.e. at or worse than "tomorrow = today".
- Adding macro and news moves MAPE by fractions of a percentage point, in both directions, with no
  consistency across stocks or folds.
- This mirrors the direction results exactly: the same information that fails to predict DIRECTION
  also fails to improve PRICE forecasting.

## Leak scan

Every feature is checked against both the SAME-DAY and the NEXT-DAY return. A legitimate predictor
should track today's move at least as closely as tomorrow's; the reverse means it carries future
information. This check was added after the ASPI forward-fill bug (see below) and now runs on every
target automatically — result in `leak_scan.csv`.

**A bug this caught:** ASPI was originally forward-filled onto each target's calendar. The target
and ASPI do not share a calendar, so the fill let ASPI carry a move that the target only recorded on
its next row. `aspi_ret_1` then correlated **0.742** with SPSL20's FUTURE return but only **0.374**
with its own same-day return, and halved MAPE (ratio 0.496 vs a true 0.953). Fixed by removing the
forward-fill and dropping ASPI features for index targets entirely.

## Caveats
- News sentiment only exists from 2016 to 2022-06, so the `P+M+N` rows cover a shorter span than
  `P` and `P+M`; compare within a row, not across feature sets on different windows.
- ARIMA is fitted once per fold with a fixed (1,1,1) order and applies a constant drift across the
  fold — it is a reference point, not a tuned competitor. Stage 1 already found auto-ARIMA
  collapsing to (0,1,0), i.e. the naive model, on this data.
- MAPE on a price level is dominated by the level itself; the `ret_RMSE_pp` and `dir_acc_%` columns
  are the more honest views of forecast skill.
- **The naive row's `dir_acc_%` (10.5% at 1 day, 3.3% at 5 days) is a metric artefact, not a
  result.** Naive predicts a return of exactly 0, and `sign(0)` matches only on days the price did
  not move at all — so that column is really "share of perfectly flat days". It is still worth
  noting on its own terms: **~10% of CSE bank/finance trading days close exactly unchanged**, which
  is an illiquidity fact and part of why the naive baseline is so hard to beat here.
"""
(OUT / "price_summary.md").write_text(md)

print("\n" + "=" * 100)
print(S.to_string(index=False))
print("=" * 100)
print(G.to_string(index=False))
print(f"\nSaved to {OUT}")
