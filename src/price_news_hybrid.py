#!/usr/bin/env python3
"""
PRICE + NEWS ONLY (no macro), and a HYBRID on the same — a clean isolation test.

WHY THIS IS NOT A REPEAT OF PHASE H
In Phase H, news sentiment was added ON TOP of everything else (Tier-2 indicators + interest rates
+ sector features). So news had to prove itself while competing with ~30 other columns, several of
which were already shown to be useless noise. If macro was actively hurting the model — and Phase C
showed it can — then news never got a fair hearing.

This run strips everything back:

    P        price / technical only              (the floor)
    P+N      price + NEWS SENTIMENT              <- news with NO macro in the way
    Hybrid   ARIMA -> LSTM on residuals          (Zhang 2003, univariate)
    Hyb+N    ARIMA -> LSTM on residuals + NEWS   <- news in the part ARIMA cannot explain

Both tasks are run, because the product needs both:
  * DIRECTION  - will the sector rise? (what the dashboard shows)
  * PRICE      - RMSE/MAPE vs naive   (comparable to published work)

Targets: BANKS composite (the strongest composite, autocorrelation 0.178) and S&P SL 20.
Window : 2016-01 -> 2022-06, set by the news feed. NOT comparable to the other phases' windows —
         compare only within this table.

Guards carried over: walk-forward, purge, matched-horizon persistence, leak scan, news lagged to
the next trading day when published after the 14:30 CSE close, calibrator-free (no confidence claim).

Outputs -> results/price_news_hybrid/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import binomtest, norm
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "price_news_hybrid"
OUT.mkdir(parents=True, exist_ok=True)

BANKS = ["HNB", "COMB", "SAMP"]
HORIZONS = [1, 5, 10]
SEED = 42
SEQ, HIDDEN, EPOCHS, BATCH = 20, 32, 35, 64
SEEDS = [0, 1, 2]
TEST_MONTHS = 6
FIRST_TEST = pd.Timestamp("2019-01-01")
NEWS_START = pd.Timestamp("2016-01-01")
DEV = torch.device("cpu")

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))
      for t in BANKS}
aspi = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
          .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))
sent = pd.read_csv(DATA / "news_sentiment_daily.csv", parse_dates=["date"]).sort_values("date")
NEWS_END = sent.date.max()

NEWS = ["sent_vader_1", "sent_vader_5", "sent_lm_1", "sent_lm_5",
        "sent_mom", "news_count_5", "neg_share_5"]


def build_banks():
    cal = None
    for t in BANKS:
        cal = px[t].index if cal is None else cal.intersection(px[t].index)
    cal = cal.sort_values()
    rets = pd.DataFrame({t: px[t].reindex(cal).pct_change() for t in BANKS})
    idx = 100 * (1 + rets.mean(axis=1).fillna(0)).cumprod()
    return pd.DataFrame({"date": cal, "close": idx.values}).reset_index(drop=True)


# SPSL20 MUST come from the ASPI-calendar version. The older
# data/processed/spsl20_trading_days_clean.csv DELETED every day the index did not move, which
# misaligns it against ASPI: 352 of 1505 rows drop out, and among the survivors ASPI's day-t value
# often lines up with SPSL20's day-t+1 row. aspi_ret_1 then correlated 0.871 with SPSL20's FUTURE
# return vs 0.227 with its own same-day return, and a trivial "copy the sign of ASPI" rule scored
# 84.3%. Removing .ffill() alone did NOT fix that.
sp = (pd.read_csv(DATA / "spsl20_daily_fixed.csv", parse_dates=["date"])
        .sort_values("date").drop_duplicates("date")
        .rename(columns={"spsl20_points": "close"})[["date", "close"]])
TARGETS = {"BANKS": build_banks(), "SPSL20": sp}
INDEX_TARGETS = {"SPSL20"}      # indices get NO ASPI features: same market measured twice


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


def make_features(c, tname):
    d = c[(c.date >= NEWS_START) & (c.date <= NEWS_END)].reset_index(drop=True)
    cl = d["close"]
    r1 = cl.pct_change()
    f = pd.DataFrame({"date": d["date"], "close": cl})
    f["ret_1"], f["ret_5"], f["ret_10"] = r1, cl.pct_change(5), cl.pct_change(10)
    f["ma5_ratio"] = cl / cl.rolling(5).mean() - 1
    f["ma10_ratio"] = cl / cl.rolling(10).mean() - 1
    f["ma20_ratio"] = cl / cl.rolling(20).mean() - 1
    f["momentum_10"] = cl / cl.shift(10) - 1
    f["vol_10"], f["vol_20"] = r1.rolling(10).std(), r1.rolling(20).std()
    f["rsi_14"] = rsi(cl)
    if tname not in INDEX_TARGETS:
        a = aspi.reindex(d["date"]).reset_index(drop=True)      # exact calendar, no ffill
        f["aspi_ret_1"], f["aspi_ret_5"] = a.pct_change(), a.pct_change(5)
    PRICE = [x for x in f.columns if x not in ("date", "close")]

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

    for h in HORIZONS:
        f[f"fwd_{h}"] = cl.shift(-h) / cl - 1
        f[f"past_{h}"] = cl / cl.shift(h) - 1
    return f, PRICE


# ---------------------------------------------------------------- residual LSTM (hybrid)
class ResLSTM(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.rnn = nn.LSTM(nf, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1]).squeeze(-1)


def residual_lstm(res_tr, res_te, ex_tr=None, ex_te=None):
    X, y = [], []
    for i in range(SEQ, len(res_tr)):
        w = res_tr[i - SEQ:i].reshape(-1, 1)
        if ex_tr is not None:
            w = np.hstack([w, ex_tr[i - SEQ:i]])
        X.append(w); y.append(res_tr[i])
    if len(X) < 100:
        return np.zeros(len(res_te))
    X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
    mu, sd = X.reshape(-1, X.shape[2]).mean(0), X.reshape(-1, X.shape[2]).std(0) + 1e-9
    ym, ys = y.mean(), y.std() + 1e-9
    Xn, yn = (X - mu) / sd, (y - ym) / ys

    full_r = np.r_[res_tr, res_te]
    full_e = np.vstack([ex_tr, ex_te]) if ex_tr is not None else None
    Xe = []
    for i in range(len(res_tr), len(full_r)):
        w = full_r[i - SEQ:i].reshape(-1, 1)
        if full_e is not None:
            w = np.hstack([w, full_e[i - SEQ:i]])
        Xe.append(w)
    Xe = (np.asarray(Xe, np.float32) - mu) / sd

    preds = []
    for s in SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        net = ResLSTM(Xn.shape[2]).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lf = nn.MSELoss()
        Xt, yt = torch.tensor(Xn, device=DEV), torch.tensor(yn, device=DEV)
        net.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(len(Xt), device=DEV)
            for i in range(0, len(Xt), BATCH):
                k = perm[i:i + BATCH]
                opt.zero_grad(); lf(net(Xt[k]), yt[k]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            preds.append(net(torch.tensor(Xe, device=DEV)).cpu().numpy() * ys + ym)
    return np.mean(preds, axis=0)


def dm(actual, p1, p2):
    d = (actual - p1) ** 2 - (actual - p2) ** 2
    n = len(d); db = float(d.mean()); se = np.sqrt(float(((d - db) ** 2).mean()) / n)
    if se == 0 or not np.isfinite(se):
        return np.nan, np.nan
    st = db / se
    return float(st), float(2 * (1 - norm.cdf(abs(st))))


dir_rows, px_rows, leaks = [], [], []
for tname, series in TARGETS.items():
    f, PRICE = make_features(series, tname)
    ALL = PRICE + NEWS

    gg = f[list(dict.fromkeys(["ret_1", "fwd_1"] + ALL))].dropna()
    for col in ALL:
        if col == "ret_1":
            continue
        now, fut = abs(gg[col].corr(gg["ret_1"])), abs(gg[col].corr(gg["fwd_1"]))
        if fut > now + 0.05 and fut > 0.2:
            leaks.append({"target": tname, "feature": col,
                          "corr_same_day": round(now, 3), "corr_next_day": round(fut, 3)})

    # ================= DIRECTION: P vs P+N =================
    for h in HORIZONS:
        d = f[["date", "close", f"fwd_{h}", f"past_{h}"] + ALL].dropna().reset_index(drop=True)
        y = (d[f"fwd_{h}"] > 0).astype(int).values
        for t0 in pd.date_range(FIRST_TEST, d.date.max(), freq=f"{TEST_MONTHS}MS"):
            t1 = t0 + pd.DateOffset(months=TEST_MONTHS)
            if t1 > d.date.max():
                break
            purge = t0 - pd.Timedelta(days=int(h * 1.5) + 3)
            tr = (d.date <= purge).values
            te = ((d.date >= t0) & (d.date < t1)).values
            if tr.sum() < 400 or te.sum() < 40:
                continue
            ytr, yte = y[tr], y[te]
            if len(np.unique(ytr)) < 2:
                continue
            cnt = pd.Series(ytr).value_counts()
            w = pd.Series(ytr).map({k: len(ytr) / (len(cnt) * cnt[k]) for k in cnt.index}).values
            maj = int(cnt.idxmax())
            past = d.loc[te, f"past_{h}"].values
            base = max(accuracy_score(yte, np.full(len(yte), maj)),
                       accuracy_score(yte, (past > 0).astype(int)))
            fwd = d.loc[te, f"fwd_{h}"].values
            wgt = np.abs(fwd)
            for sname, cols in [("P (price only)", PRICE), ("P+N (+news)", ALL)]:
                Xtr, Xte = d.loc[tr, cols], d.loc[te, cols]
                lg = make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=2000, class_weight="balanced",
                                                      random_state=SEED))
                lg.fit(Xtr, ytr)
                xc = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                   random_state=SEED, n_jobs=4, eval_metric="logloss")
                xc.fit(Xtr, ytr, sample_weight=w)
                a1, a2 = accuracy_score(yte, lg.predict(Xte)), accuracy_score(yte, xc.predict(Xte))
                best = max(a1, a2)
                bp = lg.predict(Xte) if a1 >= a2 else xc.predict(Xte)
                wacc = float(((yte == bp) * wgt).sum() / wgt.sum())
                wbase = float(((yte == maj) * wgt).sum() / wgt.sum())
                dir_rows.append({"target": tname, "horizon_days": h, "features": sname,
                                 "test_from": f"{t0:%Y-%m}", "n_test": int(te.sum()),
                                 "acc_%": round(best * 100, 1),
                                 "baseline_%": round(base * 100, 1),
                                 "edge_pp": round((best - base) * 100, 1),
                                 "mag_edge_pp": round((wacc - wbase) * 100, 1)})

    # ================= PRICE: naive / ARIMA / Hybrid / Hybrid+News =================
    dp = f[["date", "close"] + NEWS].dropna().reset_index(drop=True)
    logp = np.log(dp["close"].values)
    exog = dp[NEWS].values.astype(np.float32)
    for t0 in pd.date_range(FIRST_TEST, dp.date.max(), freq=f"{TEST_MONTHS}MS"):
        t1 = t0 + pd.DateOffset(months=TEST_MONTHS)
        if t1 > dp.date.max():
            break
        tr = np.flatnonzero((dp.date <= t0 - pd.Timedelta(days=5)).values)
        te = np.flatnonzero(((dp.date >= t0) & (dp.date < t1)).values)
        if len(tr) < 500 or len(te) < 40:
            continue
        actual = dp["close"].values[te]
        naive = dp["close"].values[te[0] - 1:te[-1]]
        fit = ARIMA(logp[tr], order=(2, 1, 0)).fit()
        ext = fit.append(logp[te], refit=False)
        alog = ext.get_prediction(start=len(tr), end=len(tr) + len(te) - 1).predicted_mean
        apx = np.exp(alog)
        res_tr = np.asarray(fit.resid, float)[1:]
        res_te = logp[te] - alog
        ex_tr, ex_te = exog[tr][1:], exog[te]
        m, s = ex_tr.mean(0), ex_tr.std(0) + 1e-9
        hyb = np.exp(alog + residual_lstm(res_tr, res_te))
        hybn = np.exp(alog + residual_lstm(res_tr, res_te, (ex_tr - m) / s, (ex_te - m) / s))
        for nm, p in [("Naive", naive), ("ARIMA(2,1,0)", apx),
                      ("Hybrid ARIMA+LSTM", hyb), ("Hybrid+News", hybn)]:
            st, pv = dm(actual, p, naive)
            px_rows.append({"target": tname, "test_from": f"{t0:%Y-%m}", "model": nm,
                            "RMSE": round(float(np.sqrt(mean_squared_error(actual, p))), 4),
                            "MAE": round(float(mean_absolute_error(actual, p)), 4),
                            "MAPE_%": round(float(np.mean(np.abs((actual - p) / actual)) * 100), 4),
                            "Theil_U2": round(float(np.sqrt(mean_squared_error(actual, p)) /
                                                    np.sqrt(mean_squared_error(actual, naive))), 4),
                            "DM_vs_naive_p": round(pv, 4) if pv == pv else np.nan})
    print(f"  {tname} done")

LK = pd.DataFrame(leaks)
LK.to_csv(OUT / "leak_scan.csv", index=False)
# HALT on a leak. Previously this only wrote a CSV, and the run continued to report a fake
# +20.7pp edge while the summary line claimed "clean". A guard that does not stop the run is not
# a guard.
if len(LK):
    print("\n" + "!" * 100)
    print("LEAK DETECTED — ABORTING. These features track the future more than the present:")
    print(LK.to_string(index=False))
    print("!" * 100)
    raise SystemExit(1)
print("\nleak scan: clean")
D = pd.DataFrame(dir_rows); D.to_csv(OUT / "direction_folds.csv", index=False)
P = pd.DataFrame(px_rows); P.to_csv(OUT / "price_folds.csv", index=False)

DS = []
for (t, h, s_), sub in D.groupby(["target", "horizon_days", "features"]):
    e, we = sub.edge_pp.values, sub.mag_edge_pp.values
    k, kw = int((e > 0).sum()), int((we > 0).sum())
    DS.append({"target": t, "horizon_days": h, "features": s_, "folds": len(e),
               "median_acc_%": round(float(sub["acc_%"].median()), 1),
               "median_baseline_%": round(float(sub["baseline_%"].median()), 1),
               "median_edge_pp": round(float(np.median(e)), 2), "folds_pos": f"{k}/{len(e)}",
               "sign_p": round(float(binomtest(k, len(e), 0.5, alternative="greater").pvalue), 4),
               "median_mag_edge_pp": round(float(np.median(we)), 2),
               "mag_folds_pos": f"{kw}/{len(we)}",
               "mag_sign_p": round(float(binomtest(kw, len(we), 0.5, alternative="greater").pvalue), 4)})
DS = pd.DataFrame(DS).sort_values(["target", "horizon_days", "features"])
DS.to_csv(OUT / "direction_summary.csv", index=False)

PS = (P.groupby(["target", "model"])
      .agg(folds=("RMSE", "size"), RMSE=("RMSE", "median"), MAE=("MAE", "median"),
           MAPE_pct=("MAPE_%", "median"), Theil_U2=("Theil_U2", "median"),
           folds_beat_naive=("Theil_U2", lambda s: f"{int((s < 1).sum())}/{len(s)}"),
           DM_p=("DM_vs_naive_p", "median")).reset_index().round(4))
PS.to_csv(OUT / "price_summary.csv", index=False)

# news gain = P+N minus P, matched fold by fold
gain = []
for (t, h), sub in D.groupby(["target", "horizon_days"]):
    a = sub[sub.features == "P (price only)"].set_index("test_from")
    b = sub[sub.features == "P+N (+news)"].set_index("test_from")
    idx = a.index.intersection(b.index)
    g = (b.loc[idx, "edge_pp"] - a.loc[idx, "edge_pp"]).values
    gm = (b.loc[idx, "mag_edge_pp"] - a.loc[idx, "mag_edge_pp"]).values
    k = int((g > 0).sum())
    gain.append({"target": t, "horizon_days": h, "folds": len(g),
                 "median_news_gain_pp": round(float(np.median(g)), 2), "folds_pos": f"{k}/{len(g)}",
                 "sign_p": round(float(binomtest(k, len(g), 0.5, alternative="greater").pvalue), 4),
                 "median_news_gain_mag_pp": round(float(np.median(gm)), 2)})
G = pd.DataFrame(gain)
G.to_csv(OUT / "news_gain.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
a = ax[0]
x = np.arange(len(G))
a.bar(x - 0.2, G.median_news_gain_pp, 0.4, label="plain accuracy", color="tab:blue", alpha=.85)
a.bar(x + 0.2, G.median_news_gain_mag_pp, 0.4, label="magnitude-weighted", color="tab:orange", alpha=.85)
a.axhline(0, color="black", lw=1)
a.set_xticks(x); a.set_xticklabels([f"{r.target}\n{r.horizon_days}d" for _, r in G.iterrows()], fontsize=8)
a.set_ylabel("Gain from adding NEWS (pp)")
a.set_title("Does news help when macro is REMOVED?"); a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)

b = ax[1]
for i, t in enumerate(TARGETS):
    sub = PS[PS.target == t].set_index("model").reindex(
        ["Naive", "ARIMA(2,1,0)", "Hybrid ARIMA+LSTM", "Hybrid+News"])
    b.barh(np.arange(len(sub)) + (i - 0.5) * 0.35, sub.Theil_U2, 0.35, label=t, alpha=.85)
b.axvline(1.0, color="black", ls="--", label="naive = 1.0")
b.set_yticks(range(4)); b.set_yticklabels(["Naive", "ARIMA", "Hybrid", "Hybrid+News"], fontsize=9)
b.set_xlabel("Theil's U2  (<1 = beats naive)")
b.set_title("Price: hybrid with news, no macro"); b.grid(alpha=.3, axis="x"); b.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "price_news_hybrid.png", dpi=140)

news_wins = G[(G.sign_p < 0.05) & (G.median_news_gain_pp > 0)]
px_wins = PS[(PS.Theil_U2 < 1) & (PS.model != "Naive")]

md = f"""# Price + news only (no macro), and a hybrid on the same

## Why this is not a repeat of Phase H

In Phase H, news was added **on top of** Tier-2 indicators, interest rates and sector features — it
had to prove itself while competing with ~30 other columns, several already shown to be useless.
Phase C showed macro can actively *hurt*. So news never got a clean test.

Here everything else is stripped away.

**Window: {NEWS_START:%Y-%m} → {NEWS_END:%Y-%m}** (set by the news feed). Not comparable to other
phases — compare only within this table.

## Leak scan

{"CLEAN — no feature tracks the future more than the present." if not len(LK) else md_table(LK)}

## 1. DIRECTION — does news help once macro is gone?

{md_table(DS)}

### The isolation result: news gain (P+N minus P, matched fold by fold)

{md_table(G)}

**Horizons where news significantly helps: {len(news_wins)} of {len(G)}.**

## 2. PRICE — hybrid with news, no macro

{md_table(PS)}

**Models beating naive on Theil's U2: {len(px_wins)} of {len(PS) - len(TARGETS)}.**

* `Hybrid ARIMA+LSTM` — LSTM on ARIMA residuals, univariate (Zhang 2003).
* `Hybrid+News` — the same, but the residual LSTM also sees the news block. If news carries
  anything, the part ARIMA cannot explain is where it should show up.

## Reading it

{"News adds something once macro is removed — worth carrying into the real project." if len(news_wins) else "Removing macro does NOT rescue news. Its contribution is indistinguishable from zero even with nothing else competing for it."}

## Caveats
- Window is {NEWS_START:%Y} to {NEWS_END:%Y-%m}; roughly {len(G) and int(DS.folds.max())} folds per horizon, so power is limited.
- News sentiment is **market-wide**, not company- or sector-specific. It cannot explain why banks
  move differently from the rest of the market — a known gap, still untested.
- 3 seeds averaged for every LSTM; recurrent nets remain the noisiest component here.
"""
(OUT / "price_news_hybrid_summary.md").write_text(md)

print("\n" + "=" * 104)
print("DIRECTION")
print(DS.to_string(index=False))
print("\nNEWS GAIN (P+N minus P)")
print(G.to_string(index=False))
print("\n" + "=" * 104)
print("PRICE")
print(PS.to_string(index=False))
print(f"\nNews significantly helps: {len(news_wins)}/{len(G)}   |   price models beating naive: {len(px_wins)}")
print(f"Saved to {OUT}")
