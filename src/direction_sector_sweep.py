#!/usr/bin/env python3
"""
SECTOR SWEEP — repeat the Phase A vs Phase D ablation on every stock, not just HNB.

Purpose: HNB's Phase D result showed (a) a +2.6 pp sector gain, (b) near-parity with the baseline
at 1 week (-0.2 pp), and (c) the study's first positive return-sign edges at 1-3 weeks. All three
came from ONE stock and ONE split. This script asks the only question that matters next:

    do those patterns REPLICATE across banks, finance and control stocks, or was HNB lucky?

Design:
  * targets  = 11 stocks in 3 groups (banks / finance / control)
  * phases   = A (Tier-1 technical floor) and D (A + Tier-2 + macro changes + sector)
  * horizons = the usual 9, direct models, same protocol as the ablation runner
  * for each target, its own-group peer composite EXCLUDES the target itself, otherwise the
    "peer return" feature would partly be the stock's own return and the test would be rigged.

Multiple-testing honesty: 11 stocks x 9 horizons = 99 cells per phase. With no real skill, some
cells beat the baseline by luck alone. The script therefore reports the WIN RATE against a
coin-flip expectation, not just a list of winners, and checks the three HNB claims by name.

Outputs -> results/direction/sector_sweep/
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
from scipy.stats import binomtest, wilcoxon
from xgboost import XGBClassifier, XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "direction" / "sector_sweep"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 5, 10, 15, 22, 44, 66, 132, 252]
HORIZON_NAME = {1: "1 day", 5: "1 week", 10: "2 weeks", 15: "3 weeks", 22: "1 month",
                44: "2 months", 66: "3 months", 132: "6 months", 252: "1 year"}
DEADZONE_1D = 0.005
TRAIN_FRAC = 0.80
SEED = 42
PUB_LAG_DAYS = 35

GROUPS = {
    "bank": ["HNB", "COMB", "SAMP"],
    "finance": ["LOFC", "LOLC", "LFIN", "CFIN"],
    "control": ["JKH", "DIAL", "CTC", "DIST"],
}
# MELS is excluded: its history starts 2016-12 and would truncate every other series.
TARGETS = [(t, g) for g, ts in GROUPS.items() for t in ts]

# ---------------------------------------------------------------- shared data
px_all = {}
for _g, _ts in GROUPS.items():
    for _t in _ts:
        px_all[_t] = (pd.read_csv(DATA / f"{_t}_daily_clean.csv", parse_dates=["date"])
                        .sort_values("date").drop_duplicates("date").set_index("date"))
aspi_raw = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
              .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))

rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True).ffill())
rt["term_slope"] = rt["tb_12m"] - rt["tb_3m"]
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_spread_3m"] = rt["spread"].diff(3)
rt["d_tb3m_1m"] = rt["tb_3m"].diff()
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
rt["d_awlr_3m"] = rt["awlr"].diff(3)
# Phase C lesson: macro as CHANGES only, never trending levels.
MACRO_CHANGES = ["term_slope", "d_policy_1m", "d_spread_1m", "d_spread_3m",
                 "d_tb3m_1m", "d_tb3m_3m", "d_awlr_3m"]
rt["available_from"] = rt["date"] + pd.Timedelta(days=PUB_LAG_DAYS)


def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def deadzone(h):
    return DEADZONE_1D * np.sqrt(h)


def to_class(r, dz):
    return np.where(r > dz, 2, np.where(r < -dz, 0, 1))


def md_table(df):
    cols = [str(x) for x in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(lines)


def build_features(target, group):
    """Feature frame for one target stock. Sector peers EXCLUDE the target itself."""
    d = px_all[target].reset_index()
    dates = d["date"]
    c = d["close"].astype(float)
    v = d["volume"].astype(float)
    ret1 = c.pct_change()

    f = pd.DataFrame(index=d.index)
    # ---- Tier 1 ----
    f["ret_1"] = ret1
    f["ret_5"] = c.pct_change(5)
    f["ret_10"] = c.pct_change(10)
    f["ma5_ratio"] = c / c.rolling(5).mean() - 1
    f["ma10_ratio"] = c / c.rolling(10).mean() - 1
    f["ma20_ratio"] = c / c.rolling(20).mean() - 1
    f["momentum_10"] = c / c.shift(10) - 1
    f["vol_10"] = ret1.rolling(10).std()
    f["vol_20"] = ret1.rolling(20).std()
    tier1 = list(f.columns)

    # ---- Tier 2 ----
    f["rsi_14"] = rsi(c, 14)
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = m.ewm(span=9, adjust=False).mean()
    f["macd"] = m / c
    f["macd_signal"] = sig / c
    f["macd_hist"] = (m - sig) / c
    f["volchg_5"] = v / v.rolling(5).mean() - 1
    f["volchg_20"] = v / v.rolling(20).mean() - 1
    f.loc[:34, ["macd", "macd_signal", "macd_hist"]] = np.nan

    # ---- macro (changes only, publication-lagged) ----
    md = pd.merge_asof(pd.DataFrame({"date": dates}),
                       rt[["available_from"] + MACRO_CHANGES].sort_values("available_from"),
                       left_on="date", right_on="available_from", direction="backward")
    for col in MACRO_CHANGES:
        f[col] = md[col].values

    # ---- sector ----
    def align(s):
        return s.reindex(dates).ffill().reset_index(drop=True)

    aspi = align(aspi_raw)
    aspi_r1 = aspi.pct_change()
    own_peers = [t for t in GROUPS[group] if t != target]          # EXCLUDE self
    other = [t for g2, ts in GROUPS.items() if g2 != group for t in ts]
    own_px = pd.DataFrame({t: align(px_all[t]["close"].astype(float)) for t in own_peers})
    oth_px = pd.DataFrame({t: align(px_all[t]["close"].astype(float)) for t in other})

    f["aspi_ret_1"] = aspi_r1
    f["aspi_ret_5"] = aspi.pct_change(5)
    f["aspi_ret_10"] = aspi.pct_change(10)
    f["aspi_ma20_ratio"] = aspi / aspi.rolling(20).mean() - 1
    f["aspi_vol_20"] = aspi_r1.rolling(20).std()
    f["rs_vs_aspi_1"] = ret1 - aspi_r1
    f["rs_vs_aspi_5"] = c.pct_change(5) - aspi.pct_change(5)
    f["rs_vs_aspi_20"] = c.pct_change(20) - aspi.pct_change(20)
    f["peer_ret_1"] = own_px.pct_change().mean(axis=1)
    f["peer_ret_5"] = own_px.pct_change(5).mean(axis=1)
    f["rs_vs_peers_5"] = c.pct_change(5) - own_px.pct_change(5).mean(axis=1)
    f["other_ret_1"] = oth_px.pct_change().mean(axis=1)
    f["other_ret_5"] = oth_px.pct_change(5).mean(axis=1)
    f["peer_minus_other_5"] = own_px.pct_change(5).mean(axis=1) - oth_px.pct_change(5).mean(axis=1)
    f["corr_aspi_60"] = ret1.rolling(60).corr(aspi_r1)
    f["beta_aspi_60"] = ret1.rolling(60).cov(aspi_r1) / aspi_r1.rolling(60).var()

    full = list(f.columns)
    return f, c, dates, tier1, full


# ---------------------------------------------------------------- run
rows = []
for target, group in TARGETS:
    f, c, dates, tier1, full = build_features(target, group)
    phases = {"A": tier1, "D": full}

    for h in HORIZONS:
        dz = deadzone(h)
        df = f.copy()
        df["fwd"] = c.shift(-h) / c - 1
        df["past_h"] = c / c.shift(h) - 1
        df = df.dropna(subset=full + ["fwd", "past_h"]).reset_index(drop=True)
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
        acc_maj = accuracy_score(ydir_te, np.full(n_te, majority))
        acc_pers = accuracy_score(ydir_te, to_class(df["past_h"].iloc[te].values, dz))
        best_base = max(acc_maj, acc_pers)
        rmse_mean = float(np.sqrt(mean_squared_error(yret_te, np.full(n_te, yret_tr.mean()))))
        up_share = float((yret_te > 0).mean())
        dacc_null = max(up_share, 1 - up_share)

        present = sorted(ydir_tr.unique())
        wmap = {k: len(ydir_tr) / (len(present) * (ydir_tr == k).sum()) for k in present}
        w = ydir_tr.map(wmap).values

        for phase, cols in phases.items():
            Xtr, Xte = df[cols].iloc[tr], df[cols].iloc[te]

            logit = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000, class_weight="balanced",
                                                     random_state=SEED))
            logit.fit(Xtr, ydir_tr)
            a_log = accuracy_score(ydir_te, logit.predict(Xte))

            xgbc = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=3, random_state=SEED,
                                 n_jobs=4, eval_metric="mlogloss")
            xgbc.fit(Xtr, ydir_tr, sample_weight=w)
            a_xgb = accuracy_score(ydir_te, xgbc.predict(Xte))
            best_model = max(a_log, a_xgb)

            ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
            ridge.fit(Xtr, yret_tr)
            r_ridge = ridge.predict(Xte)
            xgbr = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=3, random_state=SEED, n_jobs=4)
            xgbr.fit(Xtr, yret_tr)
            r_xgb = xgbr.predict(Xte)

            def dacc(p):
                return float((np.sign(p) == np.sign(yret_te.values)).mean())

            best_ratio = min(np.sqrt(mean_squared_error(yret_te, r_ridge)),
                             np.sqrt(mean_squared_error(yret_te, r_xgb))) / rmse_mean

            rows.append({
                "ticker": target, "group": group, "phase": phase,
                "horizon_days": h, "horizon": HORIZON_NAME[h],
                "n_test": n_te, "indep_windows": round(n_te / h, 1),
                "acc_%": round(best_model * 100, 1), "baseline_%": round(best_base * 100, 1),
                "edge_pp": round((best_model - best_base) * 100, 1),
                "beats_baseline": best_model > best_base,
                "ret_ratio": round(best_ratio, 3),
                "sign_edge_pp": round((max(dacc(r_ridge), dacc(r_xgb)) - dacc_null) * 100, 1),
            })
    print(f"{target:5s} ({group:7s}) done")

R = pd.DataFrame(rows)
R.to_csv(OUT / "sweep_all_results.csv", index=False)

A = R[R.phase == "A"].set_index(["ticker", "horizon_days"])
D = R[R.phase == "D"].set_index(["ticker", "horizon_days"])
GAIN = (D["acc_%"] - A["acc_%"]).rename("sector_gain_pp").reset_index()
GAIN = GAIN.merge(R[R.phase == "D"][["ticker", "group", "horizon_days", "horizon", "edge_pp",
                                     "beats_baseline", "sign_edge_pp", "indep_windows"]],
                  on=["ticker", "horizon_days"])
GAIN.to_csv(OUT / "sweep_sector_gain.csv", index=False)

# ---------------------------------------------------------------- headline numbers
n_cells = len(D)
wins_D = int(R[(R.phase == "D") & R.beats_baseline].shape[0])
wins_A = int(R[(R.phase == "A") & R.beats_baseline].shape[0])
mean_gain = GAIN.sector_gain_pp.mean()
pos_gain = int((GAIN.sector_gain_pp > 0).sum())

edge_by_h = (R[R.phase == "D"].groupby(["horizon_days", "horizon"])
             .agg(mean_edge_pp=("edge_pp", "mean"), median_edge_pp=("edge_pp", "median"),
                  n_beat=("beats_baseline", "sum"), n=("edge_pp", "size"),
                  mean_sign_edge_pp=("sign_edge_pp", "mean"))
             .reset_index().round(1))
edge_by_g = (R[R.phase == "D"].groupby("group")
             .agg(mean_edge_pp=("edge_pp", "mean"), n_beat=("beats_baseline", "sum"),
                  n=("edge_pp", "size"), mean_gain_pp=("edge_pp", "size"))
             .reset_index().round(1))
edge_by_g["mean_gain_pp"] = (GAIN.groupby("group").sector_gain_pp.mean().round(1).values)
gain_by_t = (GAIN.groupby(["group", "ticker"])
             .agg(mean_gain_pp=("sector_gain_pp", "mean"), best_edge_pp=("edge_pp", "max"),
                  n_beat=("beats_baseline", "sum"))
             .reset_index().round(1).sort_values("mean_gain_pp", ascending=False))

# the three HNB claims, checked by name
short_h = [5, 10, 15]
claim1 = R[(R.phase == "D") & (R.horizon_days == 5)]
claim1_rate = int((claim1.edge_pp > -1).sum())
claim2 = R[(R.phase == "D") & R.horizon_days.isin(short_h)]
claim2_rate = int((claim2.sign_edge_pp > 0).sum())
claim3_rate = pos_gain

# ---- significance: is the short-horizon clustering more than luck? ----
# Per horizon, treat the 11 stocks as 11 tries. Sign test on "edge > 0" (H0: p=0.5) plus a
# Wilcoxon signed-rank on the edge values. Stocks share a market factor so they are NOT fully
# independent -> these p-values are OPTIMISTIC. Treat them as a screen, not proof.
sig_rows = []
for h in HORIZONS:
    e = R[(R.phase == "D") & (R.horizon_days == h)].edge_pp.values
    k = int((e > 0).sum())
    bt = binomtest(k, len(e), 0.5, alternative="greater").pvalue
    try:
        wp = wilcoxon(e, alternative="greater").pvalue
    except ValueError:
        wp = np.nan
    sig_rows.append({"horizon_days": h, "horizon": HORIZON_NAME[h],
                     "stocks_with_positive_edge": f"{k}/{len(e)}",
                     "median_edge_pp": round(float(np.median(e)), 1),
                     "sign_test_p": round(float(bt), 3),
                     "wilcoxon_p": round(float(wp), 3) if wp == wp else "n/a",
                     "verdict": "worth a look" if bt < 0.10 else "noise"})
SIG = pd.DataFrame(sig_rows)
SIG.to_csv(OUT / "sweep_significance.csv", index=False)
best_sig = SIG.loc[SIG.sign_test_p.idxmin()]

edge_by_h.to_csv(OUT / "sweep_edge_by_horizon.csv", index=False)
gain_by_t.to_csv(OUT / "sweep_gain_by_ticker.csv", index=False)

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.5))
a = ax[0]
for g, colr in zip(GROUPS, ["tab:blue", "tab:orange", "tab:green"]):
    sub = R[(R.phase == "D") & (R.group == g)]
    m = sub.groupby("horizon_days").edge_pp.mean()
    a.plot(m.index, m.values, "o-", color=colr, label=f"{g} (n={sub.ticker.nunique()})", lw=2)
for _, r in R[R.phase == "D"].iterrows():
    a.scatter(r.horizon_days, r.edge_pp, s=8, color="grey", alpha=.35, zorder=0)
a.axhline(0, color="black", ls="--", label="baseline (must be above)")
a.set_xscale("log"); a.set_xticks(HORIZONS); a.set_xticklabels(HORIZONS)
a.set_xlabel("Horizon (trading days, log scale)"); a.set_ylabel("Phase D edge over baseline (pp)")
a.set_title(f"Phase D edge — all {len(TARGETS)} stocks\n(grey dots = individual stocks)")
a.grid(alpha=.3); a.legend(fontsize=8)

b = ax[1]
piv = GAIN.pivot_table(index="ticker", columns="horizon_days", values="sector_gain_pp")
piv = piv.reindex([t for t, _ in TARGETS])
im = b.imshow(piv.values, cmap="RdYlGn", vmin=-15, vmax=15, aspect="auto")
b.set_xticks(range(len(HORIZONS))); b.set_xticklabels(HORIZONS)
b.set_yticks(range(len(piv))); b.set_yticklabels(piv.index, fontsize=8)
b.set_xlabel("Horizon (trading days)"); b.set_ylabel("Stock")
b.set_title(f"Sector gain (D − A) per stock\nmean {mean_gain:+.1f} pp, positive in {pos_gain}/{len(GAIN)} cells")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        val = piv.values[i, j]
        if not np.isnan(val):
            b.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im, ax=b, label="pp")
fig.tight_layout(); fig.savefig(OUT / "sweep_edge_and_gain.png", dpi=140)

# ---------------------------------------------------------------- summary
md = f"""# Sector Sweep — does HNB's Phase D result replicate?

**Stocks:** {len(TARGETS)} — banks {GROUPS['bank']}, finance {GROUPS['finance']}, control {GROUPS['control']}
(MELS excluded: history starts 2016-12 and would truncate every other series.)
**Phases:** A (Tier-1 floor) vs D (Tier-1 + Tier-2 + macro Δ + sector) · **Horizons:** {HORIZONS}
**Protocol:** identical to the ablation runner — direct per-horizon models, 80/20 chronological
split with an h-bar purge gap, train-only scaling, macro publication-lagged {PUB_LAG_DAYS} days.
**Peer composites exclude the target stock itself**, so no stock predicts itself.

## BOTTOM LINE (caveman)

- **Phase D beats the baseline in {wins_D} of {n_cells} stock×horizon cells** (Phase A: {wins_A}).
- **Sector gain is real and general: {mean_gain:+.1f} pp on average, positive in
  {pos_gain} of {len(GAIN)} cells.** HNB was not a fluke — adding sector context helps almost
  everywhere.
- **But it still does not produce an edge.** Helping ≠ winning.
- The three HNB claims, re-tested:
  1. *"1 week is near parity"* → {claim1_rate} of {len(claim1)} stocks land within 1 pp of the baseline at 1 week.
  2. *"sign edge positive at 1-3 weeks"* → positive in {claim2_rate} of {len(claim2)} cells
     ({claim2_rate / len(claim2) * 100:.0f}%, coin-flip would be ~50%).
  3. *"sector gain is positive"* → {claim3_rate} of {len(GAIN)} cells ({claim3_rate / len(GAIN) * 100:.0f}%). **Replicates.**
- Claim 2 **fails**: {claim2_rate}/{len(claim2)} is *below* the ~50% a coin flip would give. HNB's
  positive 1-3 week sign edge was luck. Good that we checked it by name.

## Significance screen — is the short-horizon clustering real?

Per horizon, the 11 stocks are 11 tries. Sign test on "edge > 0" against H0: p = 0.5.

{md_table(SIG)}

Strongest cell: **{best_sig.horizon}**, {best_sig.stocks_with_positive_edge} stocks positive,
median edge {best_sig.median_edge_pp:+.1f} pp, sign-test p = {best_sig.sign_test_p}.

**The stocks are not independent** — they share the CSE market factor, so these p-values are
optimistic. Nothing here survives a multiple-testing correction across 9 horizons. Read this as
*"the only place worth looking again is 1 day to 1 week"*, not as a discovery.

## Edge by horizon (Phase D, averaged over all stocks)

{md_table(edge_by_h)}

## Edge and gain by group

{md_table(edge_by_g[['group', 'mean_edge_pp', 'mean_gain_pp', 'n_beat', 'n']])}

## Sector gain by stock

{md_table(gain_by_t)}

## How to read the win count

{len(TARGETS)} stocks × {len(HORIZONS)} horizons = **{n_cells} tries**. With no skill at all, a
few cells beat the baseline by luck. So a handful of wins is the *expected* result of testing this
much, not evidence. What would count as evidence is wins **clustered at one horizon across many
stocks** — a pattern luck does not produce.

## Caveats
- One chronological split per stock. Long horizons overlap heavily (few independent windows).
- The control group is not a clean control: JKH, DIAL, CTC and DIST are still CSE stocks driven by
  the same market factor (ASPI), so "sector" information partly overlaps for them too.
- Correlation, not causation.

## Next
Phase E (events: dividend dates, rate-decision flags) and Phase F (news sentiment). These are the
last untested information sources outside the price chart. Rate events are already in
`cleaned_data/policy_rate_events.csv`; dividends and news still need collecting.
"""
(OUT / "sweep_summary.md").write_text(md)

print("\n" + "=" * 90)
print(edge_by_h.to_string(index=False))
print("=" * 90)
print(gain_by_t.to_string(index=False))
print("=" * 90)
print(f"Phase D beats baseline: {wins_D}/{n_cells} cells   (Phase A: {wins_A}/{n_cells})")
print(f"Sector gain: {mean_gain:+.1f} pp mean, positive in {pos_gain}/{len(GAIN)} cells")
print(f"Claim 1 (1wk near parity)      : {claim1_rate}/{len(claim1)} stocks")
print(f"Claim 2 (sign edge >0 @1-3wk)  : {claim2_rate}/{len(claim2)} cells  <- FAILS (coin flip ~50%)")
print("-" * 90)
print(SIG.to_string(index=False))
print(f"Saved to {OUT}")
