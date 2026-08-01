#!/usr/bin/env python3
"""
PHASE E — daily macro. Does macro work when it moves EVERY DAY instead of once a month?

Phase C failed with monthly CBSL rates. The obvious suspect was frequency, not content: a monthly
series is a flat line for ~20 trading days, so it cannot say anything about a 1-day or 1-week move.
This script retests macro at DAILY frequency, where it actually varies.

Phases (each adds to the one before):
  A   Tier-1 technical                                        (the floor)
  D   + Tier-2 + monthly macro CHANGES + sector               (best so far)
  E   + DAILY USD/LKR returns and volatility                  <- the real test
  E2  + DAILY global factors (oil, US 10Y, DXY) + CPI changes

Run across ALL 11 stocks at once, because the sector sweep showed single-stock results mislead.
Same protocol as every earlier phase: direct per-horizon models, 80/20 chronological split with an
h-bar purge gap, train-only scaling, peers exclude the target, macro is fed as CHANGES not levels,
and monthly series are publication-lagged.

Money supply is deliberately EXCLUDED: the CBSL export stops at 2024-08, which is inside the test
window, so including it would delete most of the test set. See cleaned_data/_macro_quality_report.csv.

Outputs -> results/direction/daily_macro/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import binomtest
from xgboost import XGBClassifier, XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "direction" / "daily_macro"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 5, 10, 15, 22, 44, 66, 132, 252]
HORIZON_NAME = {1: "1 day", 5: "1 week", 10: "2 weeks", 15: "3 weeks", 22: "1 month",
                44: "2 months", 66: "3 months", 132: "6 months", 252: "1 year"}
DEADZONE_1D = 0.005
TRAIN_FRAC = 0.80
SEED = 42
RATE_LAG_DAYS = 35        # CBSL monthly rates: published during the following month
CPI_LAG_DAYS = 21         # CCPI is released around the end of the reference month

GROUPS = {"bank": ["HNB", "COMB", "SAMP"],
          "finance": ["LOFC", "LOLC", "LFIN", "CFIN"],
          "control": ["JKH", "DIAL", "CTC", "DIST"]}
TARGETS = [(t, g) for g, ts in GROUPS.items() for t in ts]

# ================================================================ shared inputs
px_all = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
                .sort_values("date").drop_duplicates("date").set_index("date"))
          for _g, ts in GROUPS.items() for t in ts}
aspi_raw = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
              .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))

# ---- monthly rates (changes only) ----
rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True).ffill())
rt["term_slope"] = rt["tb_12m"] - rt["tb_3m"]
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_spread_3m"] = rt["spread"].diff(3)
rt["d_tb3m_1m"] = rt["tb_3m"].diff()
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
rt["d_awlr_3m"] = rt["awlr"].diff(3)
MACRO_M = ["term_slope", "d_policy_1m", "d_spread_1m", "d_spread_3m",
           "d_tb3m_1m", "d_tb3m_3m", "d_awlr_3m"]
rt["available_from"] = rt["date"] + pd.Timedelta(days=RATE_LAG_DAYS)

# ---- monthly inflation (changes only) ----
inf = pd.read_csv(DATA / "inflation_monthly.csv", parse_dates=["date"]).sort_values("date")
inf["d_ccpi_yoy_3m"] = inf["ccpi_yoy_pct"].diff(3)
CPI_M = ["ccpi_mom_pct", "ccpi_yoy_pct", "d_ccpi_yoy_3m"]
inf["available_from"] = inf["date"] + pd.Timedelta(days=CPI_LAG_DAYS)

# ---- DAILY USD/LKR ----
fx = pd.read_csv(DATA / "usd_lkr_daily.csv", parse_dates=["date"]).sort_values("date")
FX_D = ["usd_lkr_ret_1", "usd_lkr_ret_5", "usd_lkr_ret_20", "usd_lkr_vol_20"]

# ---- DAILY global factors ----
gl = (pd.read_csv(ROOT / "data" / "raw" / "cse_indices_macro_clean.csv", parse_dates=["date"])
        .sort_values("date").ffill())
gl["oil_ret_1"] = gl["crude_oil_usd"].pct_change()
gl["oil_ret_5"] = gl["crude_oil_usd"].pct_change(5)
gl["oil_ret_20"] = gl["crude_oil_usd"].pct_change(20)
gl["us10y_d_1"] = gl["us_10y_yield"].diff()          # a yield -> difference, not return
gl["us10y_d_5"] = gl["us_10y_yield"].diff(5)
gl["us10y_d_20"] = gl["us_10y_yield"].diff(20)
gl["dxy_ret_1"] = gl["dxy_index"].pct_change()
gl["dxy_ret_5"] = gl["dxy_index"].pct_change(5)
gl["dxy_ret_20"] = gl["dxy_index"].pct_change(20)
GLOBAL_D = ["oil_ret_1", "oil_ret_5", "oil_ret_20", "us10y_d_1", "us10y_d_5", "us10y_d_20",
            "dxy_ret_1", "dxy_ret_5", "dxy_ret_20"]


def rsi(s, n=14):
    d_ = s.diff()
    up = d_.clip(lower=0).rolling(n).mean()
    dn = (-d_.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def deadzone(h):
    return DEADZONE_1D * np.sqrt(h)


def to_class(r, dz):
    return np.where(r > dz, 2, np.where(r < -dz, 0, 1))


def md_table(df):
    cols = [str(x) for x in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(out)


def build(target, group):
    d = px_all[target].reset_index()
    dates = d["date"]
    c = d["close"].astype(float)
    v = d["volume"].astype(float)
    r1 = c.pct_change()
    f = pd.DataFrame(index=d.index)

    f["ret_1"], f["ret_5"], f["ret_10"] = r1, c.pct_change(5), c.pct_change(10)
    f["ma5_ratio"] = c / c.rolling(5).mean() - 1
    f["ma10_ratio"] = c / c.rolling(10).mean() - 1
    f["ma20_ratio"] = c / c.rolling(20).mean() - 1
    f["momentum_10"] = c / c.shift(10) - 1
    f["vol_10"], f["vol_20"] = r1.rolling(10).std(), r1.rolling(20).std()
    tier1 = list(f.columns)

    f["rsi_14"] = rsi(c)
    m_ = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sg = m_.ewm(span=9, adjust=False).mean()
    f["macd"], f["macd_signal"], f["macd_hist"] = m_ / c, sg / c, (m_ - sg) / c
    f["volchg_5"] = v / v.rolling(5).mean() - 1
    f["volchg_20"] = v / v.rolling(20).mean() - 1
    f.loc[:34, ["macd", "macd_signal", "macd_hist"]] = np.nan

    def asof(src, cols):
        j = pd.merge_asof(pd.DataFrame({"date": dates}),
                          src[["available_from"] + cols].sort_values("available_from"),
                          left_on="date", right_on="available_from", direction="backward")
        return j[cols]

    for col, s in asof(rt, MACRO_M).items():
        f[col] = s.values

    def daily_join(src, cols):
        j = pd.merge_asof(pd.DataFrame({"date": dates}), src[["date"] + cols],
                          on="date", direction="backward")
        return j[cols]

    for col, s in daily_join(fx, FX_D).items():
        f[col] = s.values
    for col, s in daily_join(gl, GLOBAL_D).items():
        f[col] = s.values
    for col, s in asof(inf, CPI_M).items():
        f[col] = s.values

    def align(s):
        return s.reindex(dates).ffill().reset_index(drop=True)

    aspi = align(aspi_raw)
    ar1 = aspi.pct_change()
    own = [t for t in GROUPS[group] if t != target]
    oth = [t for g2, ts in GROUPS.items() if g2 != group for t in ts]
    op = pd.DataFrame({t: align(px_all[t]["close"].astype(float)) for t in own})
    tp = pd.DataFrame({t: align(px_all[t]["close"].astype(float)) for t in oth})

    f["aspi_ret_1"], f["aspi_ret_5"] = ar1, aspi.pct_change(5)
    f["aspi_ret_10"] = aspi.pct_change(10)
    f["aspi_ma20_ratio"] = aspi / aspi.rolling(20).mean() - 1
    f["aspi_vol_20"] = ar1.rolling(20).std()
    f["rs_vs_aspi_1"] = r1 - ar1
    f["rs_vs_aspi_5"] = c.pct_change(5) - aspi.pct_change(5)
    f["rs_vs_aspi_20"] = c.pct_change(20) - aspi.pct_change(20)
    f["peer_ret_1"] = op.pct_change().mean(axis=1)
    f["peer_ret_5"] = op.pct_change(5).mean(axis=1)
    f["rs_vs_peers_5"] = c.pct_change(5) - op.pct_change(5).mean(axis=1)
    f["other_ret_1"] = tp.pct_change().mean(axis=1)
    f["other_ret_5"] = tp.pct_change(5).mean(axis=1)
    f["peer_minus_other_5"] = op.pct_change(5).mean(axis=1) - tp.pct_change(5).mean(axis=1)
    f["corr_aspi_60"] = r1.rolling(60).corr(ar1)
    f["beta_aspi_60"] = r1.rolling(60).cov(ar1) / ar1.rolling(60).var()
    sector = [x for x in f.columns
              if x not in tier1 + MACRO_M + FX_D + GLOBAL_D + CPI_M
              and x not in ["rsi_14", "macd", "macd_signal", "macd_hist", "volchg_5", "volchg_20"]]
    tier2 = ["rsi_14", "macd", "macd_signal", "macd_hist", "volchg_5", "volchg_20"]

    phases = {
        "A": tier1,
        "D": tier1 + tier2 + MACRO_M + sector,
        "E": tier1 + tier2 + MACRO_M + sector + FX_D,
        "E2": tier1 + tier2 + MACRO_M + sector + FX_D + GLOBAL_D + CPI_M,
    }
    return f, c, phases, phases["E2"]


rows = []
for target, group in TARGETS:
    f, c, phases, allcols = build(target, group)
    for h in HORIZONS:
        dz = deadzone(h)
        df = f.copy()
        df["fwd"] = c.shift(-h) / c - 1
        df["past_h"] = c / c.shift(h) - 1
        df = df.dropna(subset=allcols + ["fwd", "past_h"]).reset_index(drop=True)
        if len(df) < 400:
            continue

        y_dir = pd.Series(to_class(df["fwd"].values, dz))
        y_ret = df["fwd"] * 100.0
        n = len(df)
        split = int(n * TRAIN_FRAC)
        tr, te = slice(0, max(split - h, 50)), slice(split, n)
        n_te = n - split

        ydir_tr, ydir_te = y_dir.iloc[tr], y_dir.iloc[te]
        yret_tr, yret_te = y_ret.iloc[tr], y_ret.iloc[te]

        majority = int(ydir_tr.value_counts().idxmax())
        base = max(accuracy_score(ydir_te, np.full(n_te, majority)),
                   accuracy_score(ydir_te, to_class(df["past_h"].iloc[te].values, dz)))
        rmse_mean = float(np.sqrt(mean_squared_error(yret_te, np.full(n_te, yret_tr.mean()))))
        up = float((yret_te > 0).mean())
        dacc_null = max(up, 1 - up)

        present = sorted(ydir_tr.unique())
        wmap = {k: len(ydir_tr) / (len(present) * (ydir_tr == k).sum()) for k in present}
        w = ydir_tr.map(wmap).values

        for ph, cols in phases.items():
            Xtr, Xte = df[cols].iloc[tr], df[cols].iloc[te]
            lg = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=SEED))
            lg.fit(Xtr, ydir_tr)
            a1 = accuracy_score(ydir_te, lg.predict(Xte))
            xc = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, min_child_weight=3, random_state=SEED,
                               n_jobs=4, eval_metric="mlogloss")
            xc.fit(Xtr, ydir_tr, sample_weight=w)
            a2 = accuracy_score(ydir_te, xc.predict(Xte))
            best = max(a1, a2)

            rg = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
            rg.fit(Xtr, yret_tr)
            pr = rg.predict(Xte)
            xr = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=3, random_state=SEED, n_jobs=4)
            xr.fit(Xtr, yret_tr)
            px_ = xr.predict(Xte)

            def dacc(p):
                return float((np.sign(p) == np.sign(yret_te.values)).mean())

            ratio = min(np.sqrt(mean_squared_error(yret_te, pr)),
                        np.sqrt(mean_squared_error(yret_te, px_))) / rmse_mean

            s = pd.Series(xc.feature_importances_, index=cols)
            rows.append({
                "ticker": target, "group": group, "phase": ph, "n_features": len(cols),
                "horizon_days": h, "horizon": HORIZON_NAME[h], "n_train": Xtr.shape[0],
                "n_test": n_te, "acc_%": round(best * 100, 1), "baseline_%": round(base * 100, 1),
                "edge_pp": round((best - base) * 100, 1), "beats_baseline": best > base,
                "ret_ratio": round(ratio, 3),
                "sign_edge_pp": round((max(dacc(pr), dacc(px_)) - dacc_null) * 100, 1),
                "fx_share_%": round(s[[x for x in cols if x in FX_D]].sum() * 100, 1),
                "global_share_%": round(s[[x for x in cols if x in GLOBAL_D]].sum() * 100, 1),
                "cpi_share_%": round(s[[x for x in cols if x in CPI_M]].sum() * 100, 1),
            })
    print(f"{target:5s} ({group:7s}) done")

R = pd.DataFrame(rows)
R.to_csv(OUT / "daily_macro_all_results.csv", index=False)

P = {p: R[R.phase == p].set_index(["ticker", "horizon_days"]) for p in ["A", "D", "E", "E2"]}
G = pd.DataFrame({
    "gain_D_to_E_pp": (P["E"]["acc_%"] - P["D"]["acc_%"]).round(1),
    "gain_E_to_E2_pp": (P["E2"]["acc_%"] - P["E"]["acc_%"]).round(1),
    "gain_D_to_E2_pp": (P["E2"]["acc_%"] - P["D"]["acc_%"]).round(1),
}).reset_index().merge(
    R[R.phase == "E2"][["ticker", "group", "horizon_days", "horizon", "edge_pp",
                        "beats_baseline", "ret_ratio", "sign_edge_pp",
                        "fx_share_%", "global_share_%", "cpi_share_%"]],
    on=["ticker", "horizon_days"])
G.to_csv(OUT / "daily_macro_gain.csv", index=False)

by_h = (R.groupby(["phase", "horizon_days", "horizon"])
        .agg(mean_edge=("edge_pp", "mean"), median_edge=("edge_pp", "median"),
             n_beat=("beats_baseline", "sum"), n=("edge_pp", "size")).reset_index().round(1))
by_h.to_csv(OUT / "daily_macro_by_horizon.csv", index=False)

sig = []
for p in ["A", "D", "E", "E2"]:
    for h in HORIZONS:
        e = R[(R.phase == p) & (R.horizon_days == h)].edge_pp.values
        if len(e) == 0:
            continue
        k = int((e > 0).sum())
        sig.append({"phase": p, "horizon": HORIZON_NAME[h], "horizon_days": h,
                    "positive": f"{k}/{len(e)}", "median_edge_pp": round(float(np.median(e)), 1),
                    "sign_test_p": round(binomtest(k, len(e), 0.5, alternative="greater").pvalue, 3)})
SIG = pd.DataFrame(sig)
SIG.to_csv(OUT / "daily_macro_significance.csv", index=False)

wins = {p: int(R[(R.phase == p) & R.beats_baseline].shape[0]) for p in ["A", "D", "E", "E2"]}
n_cells = len(P["A"])
mean_de = G.gain_D_to_E_pp.mean()
mean_ee2 = G.gain_E_to_E2_pp.mean()
pos_de = int((G.gain_D_to_E_pp > 0).sum())
fx_share = R[R.phase == "E2"]["fx_share_%"].mean()
gl_share = R[R.phase == "E2"]["global_share_%"].mean()
cpi_share = R[R.phase == "E2"]["cpi_share_%"].mean()
best_sig = SIG[SIG.phase == "E2"].nsmallest(1, "sign_test_p").iloc[0]
sig_hits = SIG[SIG.sign_test_p < 0.05]

# ================================================================ plots
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.5))
a = ax[0]
for p, mk in zip(["A", "D", "E", "E2"], ["o-", "s-", "D-", "*-"]):
    m = R[R.phase == p].groupby("horizon_days").edge_pp.median()
    a.plot(m.index, m.values, mk, label=f"Phase {p}", lw=2)
a.axhline(0, color="black", ls="--", label="baseline (must be above)")
a.set_xscale("log"); a.set_xticks(HORIZONS); a.set_xticklabels(HORIZONS)
a.set_xlabel("Horizon (trading days, log scale)")
a.set_ylabel("Median edge over baseline (pp), 11 stocks")
a.set_title("Does daily macro help?\nA=price · D=+sector · E=+daily FX · E2=+global+CPI")
a.grid(alpha=.3); a.legend(fontsize=8)

b = ax[1]
x = np.arange(len(HORIZONS))
gm = G.groupby("horizon_days")[["gain_D_to_E_pp", "gain_E_to_E2_pp"]].mean()
b.bar(x - 0.2, gm["gain_D_to_E_pp"], 0.4, label="D→E  (+daily FX)", color="tab:blue", alpha=.85)
b.bar(x + 0.2, gm["gain_E_to_E2_pp"], 0.4, label="E→E2 (+global, CPI)", color="tab:orange", alpha=.85)
b.axhline(0, color="black", lw=1)
b.set_xticks(x); b.set_xticklabels([HORIZON_NAME[h] for h in HORIZONS], rotation=45, ha="right")
b.set_ylabel("Mean accuracy gain (pp)")
b.set_title("Gain from daily macro, averaged over 11 stocks")
b.grid(alpha=.3, axis="y"); b.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "daily_macro_gain.png", dpi=140)

md = f"""# Phase E — daily macro (does frequency fix what Phase C got wrong?)

**Stocks:** all {len(TARGETS)} (banks/finance/control) · **Horizons:** {HORIZONS} · **Cells:** {n_cells} per phase
**Window:** set by the global data (2015-01 → 2026-04); every phase uses identical rows.

| Phase | Adds | Features |
|---|---|---|
| A | Tier-1 technical | 9 |
| D | + Tier-2 + monthly rate Δ + sector | {R[R.phase == 'D'].n_features.iloc[0]} |
| E | **+ daily USD/LKR** returns & vol | {R[R.phase == 'E'].n_features.iloc[0]} |
| E2 | + daily oil / US 10Y / DXY + CPI Δ | {R[R.phase == 'E2'].n_features.iloc[0]} |

Money supply is excluded on purpose — the CBSL export ends 2024-08, inside the test window.

## BOTTOM LINE (caveman)

- **Daily FX gain (D→E): {mean_de:+.1f} pp**, positive in {pos_de} of {len(G)} cells.
- **Global + CPI gain (E→E2): {mean_ee2:+.1f} pp.**
- **Cells beating the baseline:** A {wins['A']}/{n_cells} · D {wins['D']}/{n_cells} ·
  E {wins['E']}/{n_cells} · **E2 {wins['E2']}/{n_cells}**.
- Significance: **{len(sig_hits)} of {len(SIG)} phase×horizon combinations reach p < 0.05.**
  Best for E2: {best_sig.horizon}, {best_sig.positive} stocks positive, p = {best_sig.sign_test_p}.
- The model does *use* the new data — FX gets {fx_share:.0f}% of XGBoost's importance,
  global {gl_share:.0f}%, CPI {cpi_share:.0f}% — and still gains nothing.
- **Verdict: {'daily macro helps — investigate further' if len(sig_hits) > 0 and mean_de > 1 else 'frequency was NOT the problem. Macro fails daily exactly as it failed monthly. The information is not there.'}**

## Median edge by phase and horizon

{md_table(by_h.pivot(index="horizon", columns="phase", values="median_edge").reindex([HORIZON_NAME[h] for h in HORIZONS]).reset_index())}

## Significance screen (sign test across the 11 stocks)

{md_table(SIG[SIG.phase == "E2"][["horizon", "positive", "median_edge_pp", "sign_test_p"]])}

## Caveats
- Window starts 2015 (global data), so these numbers are not directly comparable to the earlier
  2012-start ablation — compare phases *within* this table only.
- Stocks share the market factor, so the p-values are optimistic already.
- Monthly CPI is still flat within a month; only FX, oil, US 10Y and DXY truly vary daily.

## Next
The untested sources left are all **event-timed or textual**: foreign investor daily net flows,
dividend/XD dates, earnings dates + EPS, and news sentiment. Also worth more than any of them:
switch the target from direction to **volatility**, which is known to be predictable and needs no
new data.
"""
(OUT / "daily_macro_summary.md").write_text(md)

print("\n" + "=" * 92)
print(by_h.pivot(index="horizon", columns="phase", values="median_edge")
      .reindex([HORIZON_NAME[h] for h in HORIZONS]).to_string())
print("=" * 92)
print(SIG[SIG.phase == "E2"].to_string(index=False))
print("=" * 92)
print(f"Cells beating baseline: " + " · ".join(f"{p} {wins[p]}/{n_cells}" for p in ["A", "D", "E", "E2"]))
print(f"Gain D->E  (daily FX)      : {mean_de:+.1f} pp  (positive in {pos_de}/{len(G)} cells)")
print(f"Gain E->E2 (global + CPI)  : {mean_ee2:+.1f} pp")
print(f"Importance given to new data: FX {fx_share:.0f}%  global {gl_share:.0f}%  CPI {cpi_share:.0f}%")
print(f"Phase x horizon cells with p<0.05: {len(sig_hits)}/{len(SIG)}")
print(f"Saved to {OUT}")
