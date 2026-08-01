#!/usr/bin/env python3
"""
PHASE F — money supply, and the FULL classic macro set.

Money supply was excluded from Phase E because the CBSL export stops at 2024-08, which sits inside
the usual 2024-2026 test window. Dropping a core macro variable for that reason was the wrong call:
the fix is to MOVE THE WINDOW, not to skip the variable. This script cuts the whole sample at
2024-08-31 and re-runs the ablation there, so M1/M2/M2b/M4 get a fair test.

Bonus: the resulting test window (~2021-08 -> 2024-08) CONTAINS the 2022 Sri Lankan crisis, which
is a harder and more interesting test than the calmer 2024-2026 window used earlier.

Phases (identical rows, split, seeds — only the feature list changes):
  A   Tier-1 technical                                          (the plain time-series floor)
  D   + Tier-2 + monthly rate changes + sector                  (best so far)
  M   + MONEY SUPPLY changes (M1, M2, M2b, M4)                  <- the variable under test
  F   + daily USD/LKR + CPI changes                             = the full classic macro set

"Full classic macro set" = money supply, interest rates, exchange rate, inflation — i.e. four of
the five variables in the Naik & Padhi (2012) reference paper. The fifth, industrial production,
is not yet collected for Sri Lanka.

Publication lag (important): the CBSL file itself states "Update frequency: Two months after the
end of Month", so money supply is lagged 65 days. Rates 35 days, CPI 21 days.

Reserve Money is excluded: its series ends 2024-02, six months earlier than the rest.

Outputs -> results/direction/money_supply/
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
OUT = ROOT / "results" / "direction" / "money_supply"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 5, 10, 15, 22, 44, 66, 132, 252]
HORIZON_NAME = {1: "1 day", 5: "1 week", 10: "2 weeks", 15: "3 weeks", 22: "1 month",
                44: "2 months", 66: "3 months", 132: "6 months", 252: "1 year"}
DEADZONE_1D = 0.005
TRAIN_FRAC = 0.80
SEED = 42
SAMPLE_END = pd.Timestamp("2024-08-31")     # where the money supply data genuinely ends
MS_LAG_DAYS = 65                            # "two months after the end of month" (CBSL's own note)
RATE_LAG_DAYS = 35
CPI_LAG_DAYS = 21

GROUPS = {"bank": ["HNB", "COMB", "SAMP"],
          "finance": ["LOFC", "LOLC", "LFIN", "CFIN"],
          "control": ["JKH", "DIAL", "CTC", "DIST"]}
TARGETS = [(t, g) for g, ts in GROUPS.items() for t in ts]

px_all = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
                .sort_values("date").drop_duplicates("date").set_index("date"))
          for _g, ts in GROUPS.items() for t in ts}
aspi_raw = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
              .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))

# ---- monthly rates ----
rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True).ffill())
rt["term_slope"] = rt["tb_12m"] - rt["tb_3m"]
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_spread_3m"] = rt["spread"].diff(3)
rt["d_tb3m_1m"] = rt["tb_3m"].diff()
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
rt["d_awlr_3m"] = rt["awlr"].diff(3)
RATES = ["term_slope", "d_policy_1m", "d_spread_1m", "d_spread_3m",
         "d_tb3m_1m", "d_tb3m_3m", "d_awlr_3m"]
rt["available_from"] = rt["date"] + pd.Timedelta(days=RATE_LAG_DAYS)

# ---- MONEY SUPPLY (the variable under test) — changes only, per the Phase C lesson ----
ms = pd.read_csv(DATA / "money_supply_monthly.csv", parse_dates=["date"]).sort_values("date")
ms["d_m2_yoy_3m"] = ms["m2_yoy_pct"].diff(3)          # is money growth accelerating?
ms["d_m1_yoy_3m"] = ms["m1_yoy_pct"].diff(3)
ms["m1_minus_m2_yoy"] = ms["m1_yoy_pct"] - ms["m2_yoy_pct"]   # narrow vs broad divergence
MONEY = ["m1_mom_pct", "m1_yoy_pct", "m2_mom_pct", "m2_yoy_pct", "m2b_yoy_pct", "m4_yoy_pct",
         "d_m2_yoy_3m", "d_m1_yoy_3m", "m1_minus_m2_yoy"]
ms["available_from"] = ms["date"] + pd.Timedelta(days=MS_LAG_DAYS)

# ---- inflation ----
inf = pd.read_csv(DATA / "inflation_monthly.csv", parse_dates=["date"]).sort_values("date")
inf["d_ccpi_yoy_3m"] = inf["ccpi_yoy_pct"].diff(3)
CPI = ["ccpi_mom_pct", "ccpi_yoy_pct", "d_ccpi_yoy_3m"]
inf["available_from"] = inf["date"] + pd.Timedelta(days=CPI_LAG_DAYS)

# ---- daily FX ----
fx = pd.read_csv(DATA / "usd_lkr_daily.csv", parse_dates=["date"]).sort_values("date")
FX = ["usd_lkr_ret_1", "usd_lkr_ret_5", "usd_lkr_ret_20", "usd_lkr_vol_20"]


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
    d = d[d["date"] <= SAMPLE_END].reset_index(drop=True)      # <-- the moved window
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
    tier2 = ["rsi_14", "macd", "macd_signal", "macd_hist", "volchg_5", "volchg_20"]

    def asof(src, cols):
        j = pd.merge_asof(pd.DataFrame({"date": dates}),
                          src[["available_from"] + cols].sort_values("available_from"),
                          left_on="date", right_on="available_from", direction="backward")
        return j[cols]

    for col, s in asof(rt, RATES).items():
        f[col] = s.values
    for col, s in asof(ms, MONEY).items():
        f[col] = s.values
    for col, s in asof(inf, CPI).items():
        f[col] = s.values
    j = pd.merge_asof(pd.DataFrame({"date": dates}), fx[["date"] + FX], on="date",
                      direction="backward")
    for col in FX:
        f[col] = j[col].values

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
    sector = [x for x in f.columns if x not in tier1 + tier2 + RATES + MONEY + CPI + FX]

    phases = {
        "A": tier1,
        "D": tier1 + tier2 + RATES + sector,
        "M": tier1 + tier2 + RATES + sector + MONEY,
        "F": tier1 + tier2 + RATES + sector + MONEY + CPI + FX,
    }
    return f, c, dates, phases, phases["F"]


rows = []
for target, group in TARGETS:
    f, c, dates, phases, allcols = build(target, group)
    for h in HORIZONS:
        dz = deadzone(h)
        df = f.copy()
        df["fwd"] = c.shift(-h) / c - 1
        df["past_h"] = c / c.shift(h) - 1
        df["date"] = dates
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
        dnull = max(up, 1 - up)

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

            s = pd.Series(xc.feature_importances_, index=cols)
            rows.append({
                "ticker": target, "group": group, "phase": ph, "n_features": len(cols),
                "horizon_days": h, "horizon": HORIZON_NAME[h],
                "test_from": str(df["date"].iloc[split].date()),
                "test_to": str(df["date"].iloc[-1].date()),
                "n_train": Xtr.shape[0], "n_test": n_te,
                "acc_%": round(best * 100, 1), "baseline_%": round(base * 100, 1),
                "edge_pp": round((best - base) * 100, 1), "beats_baseline": best > base,
                "ret_ratio": round(min(np.sqrt(mean_squared_error(yret_te, pr)),
                                       np.sqrt(mean_squared_error(yret_te, px_))) / rmse_mean, 3),
                "sign_edge_pp": round((max(dacc(pr), dacc(px_)) - dnull) * 100, 1),
                "money_share_%": round(s[[x for x in cols if x in MONEY]].sum() * 100, 1),
            })
    print(f"{target:5s} ({group:7s}) done")

R = pd.DataFrame(rows)
R.to_csv(OUT / "money_all_results.csv", index=False)

P = {p: R[R.phase == p].set_index(["ticker", "horizon_days"]) for p in ["A", "D", "M", "F"]}
G = pd.DataFrame({
    "gain_D_to_M_pp": (P["M"]["acc_%"] - P["D"]["acc_%"]).round(1),
    "gain_M_to_F_pp": (P["F"]["acc_%"] - P["M"]["acc_%"]).round(1),
    "gain_A_to_F_pp": (P["F"]["acc_%"] - P["A"]["acc_%"]).round(1),
}).reset_index().merge(
    R[R.phase == "M"][["ticker", "group", "horizon_days", "horizon", "edge_pp",
                       "beats_baseline", "money_share_%"]], on=["ticker", "horizon_days"])
G.to_csv(OUT / "money_gain.csv", index=False)

by_h = (R.groupby(["phase", "horizon_days", "horizon"])
        .agg(median_edge=("edge_pp", "median"), n_beat=("beats_baseline", "sum"),
             n=("edge_pp", "size")).reset_index().round(1))
by_h.to_csv(OUT / "money_by_horizon.csv", index=False)

sig = []
for p in ["A", "D", "M", "F"]:
    for h in HORIZONS:
        e = R[(R.phase == p) & (R.horizon_days == h)].edge_pp.values
        if len(e) == 0:
            continue
        k = int((e > 0).sum())
        sig.append({"phase": p, "horizon": HORIZON_NAME[h], "positive": f"{k}/{len(e)}",
                    "median_edge_pp": round(float(np.median(e)), 1),
                    "sign_test_p": round(binomtest(k, len(e), 0.5, alternative="greater").pvalue, 3)})
SIG = pd.DataFrame(sig)
SIG.to_csv(OUT / "money_significance.csv", index=False)

wins = {p: int(R[(R.phase == p) & R.beats_baseline].shape[0]) for p in ["A", "D", "M", "F"]}
n_cells = len(P["A"])
mean_dm, mean_mf = G.gain_D_to_M_pp.mean(), G.gain_M_to_F_pp.mean()
pos_dm = int((G.gain_D_to_M_pp > 0).sum())
money_share = R[R.phase == "M"]["money_share_%"].mean()
sig_hits = SIG[SIG.sign_test_p < 0.05]
tw = R[R.phase == "M"].iloc[0]
# CRITICAL CHECK: if a "significant" cell also shows up in Phase A (price only), then macro is
# NOT the cause — it is a property of this test window.
sig_in_A = SIG[(SIG.phase == "A") & (SIG.sign_test_p < 0.05)]
expected_by_chance = round(len(SIG) * 0.05, 1)
a1d = SIG[(SIG.phase == "A") & (SIG.horizon == "1 day")].iloc[0]
m1d = SIG[(SIG.phase == "M") & (SIG.horizon == "1 day")].iloc[0]

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.5))
a = ax[0]
for p, mk in zip(["A", "D", "M", "F"], ["o-", "s-", "D-", "*-"]):
    m = R[R.phase == p].groupby("horizon_days").edge_pp.median()
    a.plot(m.index, m.values, mk, label=f"Phase {p}", lw=2)
a.axhline(0, color="black", ls="--", label="baseline (must be above)")
a.set_xscale("log"); a.set_xticks(HORIZONS); a.set_xticklabels(HORIZONS)
a.set_xlabel("Horizon (trading days, log scale)")
a.set_ylabel("Median edge over baseline (pp), 11 stocks")
a.set_title("Money supply test — window moved to 2024-08\nA=price · D=+sector · M=+money · F=full macro")
a.grid(alpha=.3); a.legend(fontsize=8)

b = ax[1]
x = np.arange(len(HORIZONS))
gm = G.groupby("horizon_days")[["gain_D_to_M_pp", "gain_M_to_F_pp"]].mean()
b.bar(x - 0.2, gm["gain_D_to_M_pp"], 0.4, label="D→M (+money supply)", color="tab:purple", alpha=.85)
b.bar(x + 0.2, gm["gain_M_to_F_pp"], 0.4, label="M→F (+FX, CPI)", color="tab:orange", alpha=.85)
b.axhline(0, color="black", lw=1)
b.set_xticks(x); b.set_xticklabels([HORIZON_NAME[h] for h in HORIZONS], rotation=45, ha="right")
b.set_ylabel("Mean accuracy gain (pp)")
b.set_title("Gain from money supply, averaged over 11 stocks")
b.grid(alpha=.3, axis="y"); b.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "money_gain.png", dpi=140)

md = f"""# Phase F — money supply and the full classic macro set

**Window MOVED:** sample cut at **{SAMPLE_END:%Y-%m-%d}**, where the CBSL money supply export
genuinely ends. Test window is therefore roughly **{tw.test_from} → {tw.test_to}**, which
**contains the 2022 Sri Lankan crisis** — a harder test than the calm 2024-2026 window.

| Phase | Adds | Features |
|---|---|---|
| A | Tier-1 technical (the plain time-series floor) | {R[R.phase=='A'].n_features.iloc[0]} |
| D | + Tier-2 + monthly rate Δ + sector | {R[R.phase=='D'].n_features.iloc[0]} |
| **M** | **+ money supply Δ (M1, M2, M2b, M4)** | {R[R.phase=='M'].n_features.iloc[0]} |
| F | + USD/LKR + CPI Δ = **full classic macro set** | {R[R.phase=='F'].n_features.iloc[0]} |

Money supply lagged **{MS_LAG_DAYS} days** — the CBSL file itself says "Update frequency: Two
months after the end of Month". Reserve Money dropped (series ends 2024-02, six months short).
All macro fed as **changes**, never levels (Phase C lesson).

## BOTTOM LINE (caveman)

- **Money supply gain (D→M): {mean_dm:+.1f} pp**, positive in {pos_dm} of {len(G)} cells.
- **Full macro set gain (M→F): {mean_mf:+.1f} pp.**
- **Cells beating the baseline:** A {wins['A']}/{n_cells} · D {wins['D']}/{n_cells} ·
  **M {wins['M']}/{n_cells}** · F {wins['F']}/{n_cells}.
- Significance: **{len(sig_hits)} of {len(SIG)} phase×horizon combinations reach p < 0.05.**
- XGBoost gives money supply **{money_share:.0f}%** of its importance.
- **Verdict: {'money supply adds real predictive power — investigate' if len(sig_hits) > 0 and mean_dm > 1 else 'money supply adds nothing, even on a crisis-era test window. The full classic macro set (money + rates + FX + inflation) does not beat plain price history at any horizon.'}**

## ⚠ The 1-day cell — read this before getting excited

{len(sig_hits)} cells reach p < 0.05 (chance alone would give ~{expected_by_chance} out of {len(SIG)}). Both are at 1 day:

| Phase | What it contains | Stocks positive | Median edge | p |
|---|---|---|---|---|
| **A** | **price only, NO macro** | {a1d.positive} | **{a1d.median_edge_pp:+.1f} pp** | **{a1d.sign_test_p}** |
| M | + money supply + everything else | {m1d.positive} | {m1d.median_edge_pp:+.1f} pp | {m1d.sign_test_p} |

**The effect is strongest in the model with NO macro in it at all.** Adding money supply makes it
*smaller* ({a1d.median_edge_pp:+.1f} → {m1d.median_edge_pp:+.1f} pp). So this is not a macro result —
it is a property of the **crisis test window**, where 1-day moves became more persistent and the
naive baselines got easier to beat. Note the overall win counts jumped from ~17-27/99 on the calm
2024-2026 window to {wins['A']}-{wins['M']}/99 here, for price-only features too.

Honest reading: **the window, not the variable.** And 2 hits out of {len(SIG)} tests is what luck
produces anyway.

## Median edge by phase and horizon

{md_table(by_h.pivot(index="horizon", columns="phase", values="median_edge").reindex([HORIZON_NAME[h] for h in HORIZONS]).reset_index())}

## Significance screen (sign test across the 11 stocks)

{md_table(SIG[SIG.phase == "M"][["horizon", "positive", "median_edge_pp", "sign_test_p"]])}

## Why this test matters for the write-up

This closes the macro chapter properly. The full classic macro set is now tested:

| Reference-paper variable (Naik & Padhi 2012) | Tested here? |
|---|---|
| Money supply (M3 / here M1, M2, M2b, M4) | ✅ this run |
| Treasury bill / policy rate | ✅ Phase C |
| Exchange rate | ✅ Phase E (daily) + here |
| Inflation (WPI / here CCPI) | ✅ Phase E + here |
| Industrial production | ❌ **not collected — the one remaining gap** |

## Caveats
- Different window from the earlier phases, so compare phases *within* this table only.
- Stocks share the market factor, so p-values are already optimistic.
- Money supply is monthly AND lagged two months, so at a 1-day horizon it is nearly a constant.
  Its fair test is the long horizons — where it also fails.
"""
(OUT / "money_summary.md").write_text(md)

print("\n" + "=" * 92)
print(f"TEST WINDOW: {tw.test_from} -> {tw.test_to}   (contains the 2022 crisis)")
print("=" * 92)
print(by_h.pivot(index="horizon", columns="phase", values="median_edge")
      .reindex([HORIZON_NAME[h] for h in HORIZONS]).to_string())
print("=" * 92)
print(SIG[SIG.phase == "M"].to_string(index=False))
print("=" * 92)
print("Cells beating baseline: " + " · ".join(f"{p} {wins[p]}/{n_cells}" for p in ["A", "D", "M", "F"]))
print(f"Gain D->M (money supply)  : {mean_dm:+.1f} pp  (positive in {pos_dm}/{len(G)} cells)")
print(f"Gain M->F (+FX, CPI)      : {mean_mf:+.1f} pp")
print(f"Money supply share of XGB importance: {money_share:.0f}%")
print(f"Phase x horizon cells with p<0.05: {len(sig_hits)}/{len(SIG)}")
print(f"Saved to {OUT}")
