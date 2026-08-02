#!/usr/bin/env python3
"""
IN-SAMPLE FITTING — the last untested shortcut, plus a control that settles it.

THE SHORTCUT
Cointegration / VECM / ARDL studies fit a model on the FULL sample and report how well it explains
the data, together with coefficient significance. That is a legitimate way to describe a
RELATIONSHIP. It is routinely read, however, as evidence of forecasting ability — and this project
has found no forecasting ability at all. So: does macro "improve accuracy" in-sample even though it
does nothing out-of-sample?

THE CONTROL THAT MAKES IT UNARGUABLE
Alongside the real macro block, the same models are given a block of PURE RANDOM NUMBERS — same
count, same shape, generated from a fixed seed and containing no information whatsoever.

If random noise improves in-sample fit by roughly as much as macro does, then in-sample improvement
is a property of ADDING COLUMNS, not of the columns meaning anything. That is the whole argument,
and it cannot be waved away.

ALSO REPORTED
  * adjusted R2 - penalises parameter count, which raw R2 does not
  * OLS t-statistics on the macro coefficients, in-sample - i.e. exactly the "macro is significant"
    table these papers publish
  * the same model's OUT-OF-SAMPLE skill vs a naive random walk

Targets: S&P SL 20 and the BANKS composite. Next-day return.
Outputs -> results/in_sample_fitting/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "in_sample_fitting"
OUT.mkdir(parents=True, exist_ok=True)

BANKS = ["HNB", "COMB", "SAMP"]
SEED = 42
TRAIN_FRAC = 0.80
rng = np.random.default_rng(SEED)

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))
      for t in BANKS}

rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True).ffill())
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
rt["term_slope"] = rt["tb_12m"] - rt["tb_3m"]
RATE_C = ["policy_rate", "spread", "tb_3m", "d_policy_1m", "d_spread_1m", "d_tb3m_3m", "term_slope"]
rt["avail"] = rt["date"] + pd.Timedelta(days=35)
inf = pd.read_csv(DATA / "inflation_monthly.csv", parse_dates=["date"]).sort_values("date")
INF_C = ["ccpi_index_2021base", "ccpi_yoy_pct", "ccpi_mom_pct"]
inf["avail"] = inf["date"] + pd.Timedelta(days=21)
fx = pd.read_csv(DATA / "usd_lkr_daily.csv", parse_dates=["date"]).sort_values("date")
FX_C = ["usd_lkr", "usd_lkr_ret_5"]
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
TARGETS = {"SPSL20": sp.reset_index(drop=True), "BANKS": build_banks().reset_index(drop=True)}


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
    PRICE = [x for x in f.columns if x not in ("date", "close")]

    j = pd.merge_asof(d[["date"]], rt[["avail"] + RATE_C].sort_values("avail"),
                      left_on="date", right_on="avail", direction="backward")
    for col in RATE_C:
        f[col] = j[col].values
    j2 = pd.merge_asof(d[["date"]], inf[["avail"] + INF_C].sort_values("avail"),
                       left_on="date", right_on="avail", direction="backward")
    for col in INF_C:
        f[col] = j2[col].values
    j3 = pd.merge_asof(d[["date"]], fx[["date"] + FX_C], on="date", direction="backward")
    for col in FX_C:
        f[col] = j3[col].values

    # THE CONTROL: same number of columns as MACRO, pure noise, zero information.
    NOISE = []
    for i in range(len(MACRO)):
        nm = f"noise_{i}"
        f[nm] = rng.standard_normal(len(f))
        NOISE.append(nm)

    f["fwd_1"] = cl.shift(-1) / cl - 1
    f["target_price"] = cl.shift(-1)
    return f.dropna().reset_index(drop=True), PRICE, NOISE


def adj_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)


rows, ols_rows = [], []
for tname, series in TARGETS.items():
    f, PRICE, NOISE = make_features(series)
    SETS = {"price only": PRICE,
            "price + MACRO": PRICE + MACRO,
            "price + RANDOM NOISE (control)": PRICE + NOISE}
    n = len(f)
    cut = int(n * TRAIN_FRAC)

    for sname, cols in SETS.items():
        X = f[cols].values
        y = f["fwd_1"].values
        close = f["close"].values
        actual = f["target_price"].values

        # ---------- IN-SAMPLE: fit on everything, score on the same rows ----------
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        for mn, m in [("Ridge", Ridge(alpha=1.0, random_state=SEED)),
                      ("XGBoost", XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.8,
                                               min_child_weight=3, random_state=SEED, n_jobs=4))]:
            m.fit(Xs, y)
            p_in = m.predict(Xs)
            price_in = close * (1 + p_in)
            r2_ret = float(r2_score(y, p_in))
            rows.append({
                "target": tname, "features": sname, "model": mn, "evaluation": "IN-SAMPLE",
                "n": n, "k": len(cols),
                "R2_return": round(r2_ret, 5),
                "adjR2_return": round(adj_r2(r2_ret, n, len(cols)), 5),
                "R2_price": round(float(r2_score(actual, price_in)), 6),
                "MAPE_%": round(float(np.mean(np.abs((actual - price_in) / actual)) * 100), 4),
                "vs_naive_ratio": np.nan})

        # ---------- OUT-OF-SAMPLE: honest split ----------
        sc2 = StandardScaler().fit(X[:cut])
        Xtr, Xte = sc2.transform(X[:cut]), sc2.transform(X[cut:])
        ytr = y[:cut]
        a_te, c_te = actual[cut:], close[cut:]
        naive = c_te
        mape_naive = float(np.mean(np.abs((a_te - naive) / a_te)) * 100)
        for mn, m in [("Ridge", Ridge(alpha=1.0, random_state=SEED)),
                      ("XGBoost", XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.8,
                                               min_child_weight=3, random_state=SEED, n_jobs=4))]:
            m.fit(Xtr, ytr)
            p_out = m.predict(Xte)
            price_out = c_te * (1 + p_out)
            r2_ret = float(r2_score(y[cut:], p_out))
            mape = float(np.mean(np.abs((a_te - price_out) / a_te)) * 100)
            rows.append({
                "target": tname, "features": sname, "model": mn, "evaluation": "OUT-OF-SAMPLE",
                "n": len(a_te), "k": len(cols),
                "R2_return": round(r2_ret, 5),
                "adjR2_return": round(adj_r2(r2_ret, len(a_te), len(cols)), 5),
                "R2_price": round(float(r2_score(a_te, price_out)), 6),
                "MAPE_%": round(mape, 4),
                "vs_naive_ratio": round(mape / mape_naive, 4)})

    # ---------- the "macro is significant" table these papers publish ----------
    Xo = sm.add_constant(f[PRICE + MACRO])
    res = sm.OLS(f["fwd_1"], Xo).fit()
    for v in MACRO:
        ols_rows.append({"target": tname, "variable": v,
                         "coef": round(float(res.params[v]), 6),
                         "t_stat": round(float(res.tvalues[v]), 3),
                         "p_value": round(float(res.pvalues[v]), 4),
                         "significant_5%": "YES" if res.pvalues[v] < 0.05 else "no"})
    Xn = sm.add_constant(f[PRICE + NOISE])
    resn = sm.OLS(f["fwd_1"], Xn).fit()
    n_sig_noise = int((resn.pvalues[NOISE] < 0.05).sum())
    n_sig_macro = int((res.pvalues[MACRO] < 0.05).sum())
    print(f"  {tname}: in-sample OLS — macro significant {n_sig_macro}/{len(MACRO)}, "
          f"random noise significant {n_sig_noise}/{len(NOISE)}")

R = pd.DataFrame(rows)
R.to_csv(OUT / "in_sample_results.csv", index=False)
O = pd.DataFrame(ols_rows)
O.to_csv(OUT / "ols_significance.csv", index=False)

# gain from adding a block, in-sample vs out-of-sample
gain = []
for (t, mn, ev), sub in R.groupby(["target", "model", "evaluation"]):
    base = sub[sub.features == "price only"]
    if not len(base):
        continue
    b_r2, b_mape = float(base.R2_return.iloc[0]), float(base["MAPE_%"].iloc[0])
    for blk in ["price + MACRO", "price + RANDOM NOISE (control)"]:
        s2 = sub[sub.features == blk]
        if not len(s2):
            continue
        gain.append({"target": t, "model": mn, "evaluation": ev, "block": blk,
                     "R2_gain": round(float(s2.R2_return.iloc[0]) - b_r2, 5),
                     "adjR2_gain": round(float(s2.adjR2_return.iloc[0])
                                         - float(base.adjR2_return.iloc[0]), 5),
                     "MAPE_improvement_%": round(100 * (b_mape - float(s2["MAPE_%"].iloc[0]))
                                                 / b_mape, 3)})
G = pd.DataFrame(gain)
G.to_csv(OUT / "block_gain.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4))
for i, ev in enumerate(["IN-SAMPLE", "OUT-OF-SAMPLE"]):
    a = ax[i]
    sub = G[G.evaluation == ev]
    lbl = [f"{r.target}\n{r.model}" for _, r in sub[sub.block == "price + MACRO"].iterrows()]
    x = np.arange(len(lbl))
    a.bar(x - 0.2, sub[sub.block == "price + MACRO"].R2_gain.values, 0.4,
          label="MACRO", color="tab:blue", alpha=.85)
    a.bar(x + 0.2, sub[sub.block == "price + RANDOM NOISE (control)"].R2_gain.values, 0.4,
          label="RANDOM NOISE", color="tab:red", alpha=.85)
    a.axhline(0, color="black", lw=1)
    a.set_xticks(x); a.set_xticklabels(lbl, fontsize=8)
    a.set_ylabel("R² gain over price-only")
    a.set_title(f"{ev}\nDoes macro beat pure noise?")
    a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)
fig.suptitle("In-sample fitting — macro vs a block of random numbers", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "in_sample.png", dpi=140)

IS = G[G.evaluation == "IN-SAMPLE"]
OS = G[G.evaluation == "OUT-OF-SAMPLE"]
macro_is = IS[IS.block == "price + MACRO"].R2_gain.median()
noise_is = IS[IS.block == "price + RANDOM NOISE (control)"].R2_gain.median()
macro_os = OS[OS.block == "price + MACRO"].R2_gain.median()

md = f"""# In-sample fitting — and a control made of random numbers

## The shortcut

Cointegration / VECM / ARDL studies fit on the **full sample** and report explanatory power plus
coefficient significance. That legitimately describes a **relationship**. It is routinely read as
evidence of **forecasting** ability — which is a different claim.

## The control

Alongside the real macro block, the same models get a block of **pure random numbers** — same
column count, fixed seed, zero information.

**If noise improves in-sample fit as much as macro does, in-sample improvement is a property of
adding columns, not of the columns meaning anything.**

## Results

{md_table(R[["target", "features", "model", "evaluation", "n", "k", "R2_return", "adjR2_return", "R2_price", "MAPE_%", "vs_naive_ratio"]])}

## Gain from adding a block

{md_table(G)}

Median R² gain over price-only:

| | MACRO | RANDOM NOISE |
|---|---|---|
| **IN-SAMPLE** | {macro_is:+.5f} | {noise_is:+.5f} |
| **OUT-OF-SAMPLE** | {macro_os:+.5f} | — |

## The "macro is significant" table

This is the output these papers publish — in-sample OLS of next-day return on price + macro:

{md_table(O)}

## Reading it

- **`R2_price` stays around 0.99 everywhere**, in-sample and out, for every feature set including
  pure noise. On a price level it is not a measure of skill.
- **`adjR2_return`** penalises parameter count. Compare it with raw `R2_return` to see how much of
  the in-sample "improvement" is just extra columns.
- **`vs_naive_ratio`** (out-of-sample only): below 1.0 means better than a random walk.

## Caveats
- Single 80/20 split for the out-of-sample side, matching the protocol these papers use.
- The OLS table uses next-day RETURN as the dependent variable. A VECM on levels would show far
  larger t-statistics still — non-stationary levels inflate significance, which is precisely why
  cointegration methods exist and precisely why their output is not a forecast.
"""
(OUT / "in_sample_summary.md").write_text(md)

print("\n" + "=" * 118)
print(R[["target", "features", "model", "evaluation", "k", "R2_return", "adjR2_return",
         "R2_price", "MAPE_%", "vs_naive_ratio"]].to_string(index=False))
print("\n" + "=" * 118)
print(G.to_string(index=False))
print(f"\nmedian R2 gain IN-SAMPLE — macro {macro_is:+.5f} vs random noise {noise_is:+.5f}")
print(f"median R2 gain OUT-OF-SAMPLE — macro {macro_os:+.5f}")
print(f"Saved to {OUT}")
