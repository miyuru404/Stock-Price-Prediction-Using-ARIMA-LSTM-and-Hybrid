#!/usr/bin/env python3
"""
REPRODUCE, THEN BREAK IT — where exactly does the published "macro improves accuracy" gain go?

THE QUESTION
Papers in this area report that adding macroeconomic variables improves forecasting accuracy.
Every test in this project says it does not. Rather than argue, this script REPRODUCES the
paper-style setup and then adds our guards ONE AT A TIME, recording where the improvement dies.

Two possible outcomes, both useful:
  * the paper-style setup does NOT reproduce a gain -> our pipeline is the problem, and we learn that
  * it DOES reproduce, then evaporates at a specific guard -> we have identified precisely which
    methodological shortcut manufactures the published result

THE LADDER (each level adds ONE guard, nothing else changes)

  L0  PAPER STYLE     single 80/20 split · macro aligned by REFERENCE DATE (no publication lag)
                      · scaler fitted on the FULL series (train+test) · price-level target
                      · reported as MAPE / R2 / "% improvement over price-only"
  L1  + train-only scaling      the scaler no longer sees the test set
  L2  + publication lag         macro lagged to when it was actually PUBLISHED (21-65 days)
  L3  + walk-forward            many rolling test windows instead of one split
  L4  + naive baseline          the comparison most papers never include

TRACKED QUANTITY
At every level: MAPE of price-only vs price+macro, and the "improvement" a paper would report:
    improvement_% = 100 * (MAPE_price_only - MAPE_price_macro) / MAPE_price_only
A positive number is the headline claim. We watch it die.

Targets: S&P SL 20 (the reference paper's series) and the BANKS composite.
Outputs -> results/reproduce_then_break/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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
OUT = ROOT / "results" / "reproduce_then_break"
OUT.mkdir(parents=True, exist_ok=True)

BANKS = ["HNB", "COMB", "SAMP"]
SEED = 42
TRAIN_FRAC = 0.80
TEST_MONTHS = 12
FIRST_TEST = pd.Timestamp("2019-01-01")

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))
      for t in BANKS}
aspi = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
          .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))

rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True).ffill())
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
rt["term_slope"] = rt["tb_12m"] - rt["tb_3m"]
RATE_C = ["policy_rate", "spread", "tb_3m", "d_policy_1m", "d_spread_1m", "d_tb3m_3m", "term_slope"]
inf = pd.read_csv(DATA / "inflation_monthly.csv", parse_dates=["date"]).sort_values("date")
INF_C = ["ccpi_index_2021base", "ccpi_yoy_pct", "ccpi_mom_pct"]
fx = pd.read_csv(DATA / "usd_lkr_daily.csv", parse_dates=["date"]).sort_values("date")
FX_C = ["usd_lkr", "usd_lkr_ret_5"]

# Publication lags. L0/L1 use 0 (the paper-style shortcut); L2 onward uses the real ones.
LAG_REAL = {"rates": 35, "cpi": 21}
MACRO = RATE_C + INF_C + FX_C


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
TARGETS = {"SPSL20": sp, "BANKS": build_banks()}
INDEX_TARGETS = {"SPSL20"}


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


def make_features(c, tname, lagged):
    """lagged=False reproduces the common paper shortcut: macro aligned by its REFERENCE date,
    i.e. March CPI available on 1 March, when in reality it is published weeks later."""
    d = c.reset_index(drop=True)
    cl = d["close"].astype(float)
    r1 = cl.pct_change()
    f = pd.DataFrame({"date": d["date"], "close": cl})
    f["ret_1"], f["ret_5"], f["ret_10"] = r1, cl.pct_change(5), cl.pct_change(10)
    f["ma5_ratio"] = cl / cl.rolling(5).mean() - 1
    f["ma20_ratio"] = cl / cl.rolling(20).mean() - 1
    f["momentum_10"] = cl / cl.shift(10) - 1
    f["vol_10"], f["vol_20"] = r1.rolling(10).std(), r1.rolling(20).std()
    f["rsi_14"] = rsi(cl)
    if tname not in INDEX_TARGETS:
        a = aspi.reindex(d["date"]).reset_index(drop=True)
        f["aspi_ret_1"], f["aspi_ret_5"] = a.pct_change(), a.pct_change(5)
    PRICE = [x for x in f.columns if x not in ("date", "close")]

    rt2, inf2 = rt.copy(), inf.copy()
    rt2["avail"] = rt2["date"] + pd.Timedelta(days=LAG_REAL["rates"] if lagged else 0)
    inf2["avail"] = inf2["date"] + pd.Timedelta(days=LAG_REAL["cpi"] if lagged else 0)
    j = pd.merge_asof(d[["date"]], rt2[["avail"] + RATE_C].sort_values("avail"),
                      left_on="date", right_on="avail", direction="backward")
    for col in RATE_C:
        f[col] = j[col].values
    j2 = pd.merge_asof(d[["date"]], inf2[["avail"] + INF_C].sort_values("avail"),
                       left_on="date", right_on="avail", direction="backward")
    for col in INF_C:
        f[col] = j2[col].values
    j3 = pd.merge_asof(d[["date"]], fx[["date"] + FX_C], on="date", direction="backward")
    for col in FX_C:
        f[col] = j3[col].values

    f["fwd_1"] = cl.shift(-1) / cl - 1
    f["target_price"] = cl.shift(-1)
    return f.dropna().reset_index(drop=True), PRICE


def fit_predict(Xtr, ytr, Xte, scaler_full=None):
    """scaler_full: a scaler ALREADY fitted on train+test (the paper-style leak) or None."""
    if scaler_full is not None:
        Xtr_s, Xte_s = scaler_full.transform(Xtr), scaler_full.transform(Xte)
    else:
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    rg = Ridge(alpha=1.0, random_state=SEED).fit(Xtr_s, ytr)
    xg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=3, random_state=SEED,
                      n_jobs=4).fit(Xtr_s, ytr)
    return rg.predict(Xte_s), xg.predict(Xte_s)


def dm(actual, p1, p2):
    d = (actual - p1) ** 2 - (actual - p2) ** 2
    n = len(d); db = float(d.mean()); se = np.sqrt(float(((d - db) ** 2).mean()) / n)
    if se == 0 or not np.isfinite(se):
        return np.nan
    return float(2 * (1 - norm.cdf(abs(db / se))))


LEVELS = [
    ("L0 PAPER STYLE", dict(lagged=False, full_scaler=True, walk=False)),
    ("L1 + train-only scaling", dict(lagged=False, full_scaler=False, walk=False)),
    ("L2 + publication lag", dict(lagged=True, full_scaler=False, walk=False)),
    ("L3 + walk-forward", dict(lagged=True, full_scaler=False, walk=True)),
]

rows = []
for tname, series in TARGETS.items():
    for lname, cfg in LEVELS:
        f, PRICE = make_features(series, tname, cfg["lagged"])
        sets = {"price only": PRICE, "price + macro": PRICE + MACRO}
        windows = []
        if cfg["walk"]:
            for t0 in pd.date_range(FIRST_TEST, f.date.max(), freq=f"{TEST_MONTHS}MS"):
                t1 = t0 + pd.DateOffset(months=TEST_MONTHS)
                if t1 > f.date.max():
                    break
                tr = np.flatnonzero((f.date <= t0 - pd.Timedelta(days=5)).values)
                te = np.flatnonzero(((f.date >= t0) & (f.date < t1)).values)
                if len(tr) > 500 and len(te) > 60:
                    windows.append((tr, te))
        else:
            cut = int(len(f) * TRAIN_FRAC)
            windows.append((np.arange(cut), np.arange(cut, len(f))))

        for wi, (tr, te) in enumerate(windows):
            actual = f.loc[te, "target_price"].values
            close = f.loc[te, "close"].values
            naive = close
            res = {}
            for sname, cols in sets.items():
                full_sc = StandardScaler().fit(f.loc[np.r_[tr, te], cols]) \
                    if cfg["full_scaler"] else None
                pr, pxg = fit_predict(f.loc[tr, cols], f.loc[tr, "fwd_1"].values,
                                      f.loc[te, cols], full_sc)
                # pick the better of the two, as papers routinely report their best model
                cand = {}
                for mn, p in [("Ridge", pr), ("XGB", pxg)]:
                    price_hat = close * (1 + p)
                    cand[mn] = (float(np.mean(np.abs((actual - price_hat) / actual)) * 100),
                                price_hat)
                bestm = min(cand, key=lambda k: cand[k][0])
                res[sname] = {"MAPE": cand[bestm][0], "pred": cand[bestm][1], "model": bestm}

            mape_p, mape_m = res["price only"]["MAPE"], res["price + macro"]["MAPE"]
            mape_naive = float(np.mean(np.abs((actual - naive) / actual)) * 100)
            rows.append({
                "target": tname, "level": lname, "window": wi, "n_test": len(te),
                "MAPE_price_only": round(mape_p, 4), "MAPE_price_macro": round(mape_m, 4),
                "MAPE_naive": round(mape_naive, 4),
                "improvement_%": round(100 * (mape_p - mape_m) / mape_p, 3),
                "R2_price_macro": round(float(r2_score(actual, res["price + macro"]["pred"])), 5),
                "macro_vs_naive_ratio": round(mape_m / mape_naive, 4),
                "DM_p_macro_vs_naive": round(dm(actual, res["price + macro"]["pred"], naive), 4),
            })
    print(f"  {tname} done")

R = pd.DataFrame(rows)
R.to_csv(OUT / "ladder_all.csv", index=False)

S = (R.groupby(["target", "level"])
     .agg(windows=("window", "size"),
          MAPE_price_only=("MAPE_price_only", "median"),
          MAPE_price_macro=("MAPE_price_macro", "median"),
          MAPE_naive=("MAPE_naive", "median"),
          improvement_pct=("improvement_%", "median"),
          R2=("R2_price_macro", "median"),
          macro_vs_naive=("macro_vs_naive_ratio", "median"),
          DM_p=("DM_p_macro_vs_naive", "median")).reset_index().round(4))
order = {n: i for i, (n, _) in enumerate(LEVELS)}
S["_o"] = S.level.map(order)
S = S.sort_values(["target", "_o"]).drop(columns="_o")
S.to_csv(OUT / "ladder_summary.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4))
a = ax[0]
for t in TARGETS:
    sub = S[S.target == t]
    a.plot(range(len(sub)), sub.improvement_pct, "o-", lw=2, label=t)
a.axhline(0, color="black", ls="--", lw=1.5)
a.set_xticks(range(len(LEVELS)))
a.set_xticklabels([n for n, _ in LEVELS], rotation=20, ha="right", fontsize=8)
a.set_ylabel('"macro improves accuracy" (%)')
a.set_title("The reported improvement, guard by guard")
a.grid(alpha=.3); a.legend(fontsize=9)

b = ax[1]
w = 0.35
for i, t in enumerate(TARGETS):
    sub = S[S.target == t]
    b.bar(np.arange(len(sub)) + (i - 0.5) * w, sub.macro_vs_naive, w, label=t, alpha=.85)
b.axhline(1.0, color="black", ls="--", lw=1.5, label="naive = 1.0")
b.set_xticks(range(len(LEVELS)))
b.set_xticklabels([n for n, _ in LEVELS], rotation=20, ha="right", fontsize=8)
b.set_ylabel("MAPE(price+macro) / MAPE(naive)")
b.set_title("L4: the comparison papers omit")
b.grid(alpha=.3, axis="y"); b.legend(fontsize=8)
fig.suptitle("Reproduce, then break it — where does the macro gain go?", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "ladder.png", dpi=140)

l0 = S[S.level == "L0 PAPER STYLE"]
l3 = S[S.level == "L3 + walk-forward"]
reproduced = (l0.improvement_pct > 0).any()

md = f"""# Reproduce, then break it — where does the "macro improves accuracy" gain go?

Papers report that adding macroeconomic variables improves forecasting. Every test in this project
says it does not. Rather than argue, this reproduces the **paper-style setup** and then adds our
guards **one at a time**, recording where the improvement dies.

## The ladder

| Level | What changes |
|---|---|
| **L0 PAPER STYLE** | single 80/20 split · macro aligned by **reference date (no publication lag)** · scaler fitted on **train+test** · price-level target |
| **L1** | + scaler fitted on **training data only** |
| **L2** | + macro lagged to when it was actually **published** (rates 35d, CPI 21d) |
| **L3** | + **walk-forward** instead of one split |
| **L4** | + the **naive baseline**, which most papers never include (the `macro_vs_naive` column) |

`improvement_%` is the headline a paper would report:
`100 x (MAPE_price_only - MAPE_price_macro) / MAPE_price_only`. Positive = "macro helped".

## Results

{md_table(S)}

**Did the paper-style setup reproduce a gain? {"YES" if reproduced else "NO"}.**

## Reading it

- **`improvement_%`** is the claim. Follow it down the ladder.
- **`macro_vs_naive`** is L4: below 1.0 means the model actually beats "tomorrow = today".
  This column is the one most papers never compute — and it is the one that matters.
- **`R2`** stays around 0.99 at every level, for every configuration. That is the trap: on a price
  LEVEL, R2 near 1.0 is what a random walk scores. It is not evidence of anything.

## Caveats
- The models here are Ridge and XGBoost (best of the two reported, as papers routinely report their
  best). ARIMA/LSTM/hybrid variants are covered in `hybrid_arima_lstm_macro_news.py`.
- L0's full-sample scaler is a mild leak — it leaks distributional information, not the target.
  It is included because it is extremely common in published code.
- "No publication lag" at L0/L1 is the honest reconstruction of a widespread practice: aligning
  month-end macro to month-end prices means the model sees a figure weeks before it existed.
"""
(OUT / "ladder_summary.md").write_text(md)

print("\n" + "=" * 118)
print(S.to_string(index=False))
print(f"\nSaved to {OUT}")
