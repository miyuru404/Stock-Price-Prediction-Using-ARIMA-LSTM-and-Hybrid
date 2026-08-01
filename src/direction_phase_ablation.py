#!/usr/bin/env python3
"""
Phase ABLATION runner — Phase A (Tier-1 technical) vs Phase B (Tier-1 + Tier-2 technical).

Same stock, same horizons, same split, same seeds, SAME ROWS. The ONLY thing that changes is
the feature set. So the accuracy difference (B - A) is the honest GAIN from adding Tier-2.

  Phase A features: recent returns, MA ratios, momentum, volatility        (9 features)
  Phase B features: A + RSI(14), MACD (line/signal/hist), volume change    (+6 features)

Fairness note: every feature (both phases) is computed FIRST, then rows with any NaN are
dropped once. MACD needs a 35-bar warmup, so Phase A here starts ~15 rows later than the
standalone Phase A run -> its numbers move a hair. That is the price of a fair A/B, and the
Phase A column re-reported here is the one to compare B against.

Honesty guards (same as Phase A): chronological 80/20, h-bar purge gap, train-only scaling,
independent-window count reported, and the two extra nulls (train-mean drift for return RMSE,
always-guess-winning-side for sign accuracy).

Outputs -> results/direction/phase_ablation/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "direction" / "phase_ablation"
OUT.mkdir(parents=True, exist_ok=True)

TICKER = "HNB"
HORIZONS = [1, 5, 10, 15, 22, 44, 66, 132, 252]
HORIZON_NAME = {1: "1 day", 5: "1 week", 10: "2 weeks", 15: "3 weeks", 22: "1 month",
                44: "2 months", 66: "3 months", 132: "6 months", 252: "1 year"}
DEADZONE_1D = 0.005
CLASSES = ["Sell", "Hold", "Buy"]
TRAIN_FRAC = 0.80
SEED = 42

# ---------------------------------------------------------------- features
d = (pd.read_csv(DATA / f"{TICKER}_daily_clean.csv", parse_dates=["date"])
       .sort_values("date").reset_index(drop=True))
c = d["close"].astype(float)
v = d["volume"].astype(float)
ret1 = c.pct_change()


def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


feat = pd.DataFrame(index=d.index)
# ---- TIER 1 (Phase A) ----
feat["ret_1"] = ret1
feat["ret_5"] = c.pct_change(5)
feat["ret_10"] = c.pct_change(10)
feat["ma5_ratio"] = c / c.rolling(5).mean() - 1
feat["ma10_ratio"] = c / c.rolling(10).mean() - 1
feat["ma20_ratio"] = c / c.rolling(20).mean() - 1
feat["momentum_10"] = c / c.shift(10) - 1
feat["vol_10"] = ret1.rolling(10).std()
feat["vol_20"] = ret1.rolling(20).std()
TIER1 = list(feat.columns)

# ---- TIER 2 (added in Phase B) ----
feat["rsi_14"] = rsi(c, 14)
_macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
_signal = _macd.ewm(span=9, adjust=False).mean()
feat["macd"] = _macd / c                       # divided by price -> scale-free across years
feat["macd_signal"] = _signal / c
feat["macd_hist"] = (_macd - _signal) / c
feat["volchg_5"] = v / v.rolling(5).mean() - 1
feat["volchg_20"] = v / v.rolling(20).mean() - 1
TIER2 = [x for x in feat.columns if x not in TIER1]

# MACD's ewm never produces NaN, so force the real 35-bar warmup to be dropped.
feat.loc[:34, ["macd", "macd_signal", "macd_hist"]] = np.nan

PHASES = {"A (Tier-1)": TIER1, "B (Tier-1+2)": TIER1 + TIER2}
ALL_FEATURES = TIER1 + TIER2


def deadzone(h):
    return DEADZONE_1D * np.sqrt(h)


def to_class(r, dz):
    return np.where(r > dz, 2, np.where(r < -dz, 0, 1))


def md_table(df, index=False):
    t = df.reset_index() if index else df
    cols = [str(x) for x in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(lines)


dir_rows, ret_rows, imp_rows = [], [], []

for h in HORIZONS:
    dz = deadzone(h)
    df = feat.copy()
    df["fwd"] = c.shift(-h) / c - 1
    df["past_h"] = c / c.shift(h) - 1
    df["date"] = d["date"]
    # ONE dropna over ALL features -> both phases see identical rows.
    df = df.dropna(subset=ALL_FEATURES + ["fwd", "past_h"]).reset_index(drop=True)

    y_dir = pd.Series(to_class(df["fwd"].values, dz), index=df.index)
    y_ret = df["fwd"] * 100.0

    n = len(df)
    split = int(n * TRAIN_FRAC)
    tr = slice(0, max(split - h, 50))          # purge h bars of label overlap
    te = slice(split, n)
    n_te = n - split

    ydir_tr, ydir_te = y_dir.iloc[tr], y_dir.iloc[te]
    yret_tr, yret_te = y_ret.iloc[tr], y_ret.iloc[te]

    # ---- baselines (identical for both phases) ----
    majority = int(ydir_tr.value_counts().idxmax())
    acc_maj = accuracy_score(ydir_te, np.full(n_te, majority))
    acc_pers = accuracy_score(ydir_te, to_class(df["past_h"].iloc[te].values, dz))
    best_base = max(acc_maj, acc_pers)

    r_naive = np.zeros(n_te)
    r_mean = np.full(n_te, yret_tr.mean())
    rmse_naive = float(np.sqrt(mean_squared_error(yret_te, r_naive)))
    rmse_mean = float(np.sqrt(mean_squared_error(yret_te, r_mean)))
    up_share = float((yret_te > 0).mean())
    dacc_null = max(up_share, 1 - up_share)

    for phase, cols in PHASES.items():
        Xtr, Xte = df[cols].iloc[tr], df[cols].iloc[te]

        # ---------- direction ----------
        present = sorted(ydir_tr.unique())
        wmap = {k: len(ydir_tr) / (len(present) * (ydir_tr == k).sum()) for k in present}
        w = ydir_tr.map(wmap).values

        logit = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced",
                                                 random_state=SEED))
        logit.fit(Xtr, ydir_tr)
        p_logit = logit.predict(Xte)

        xgbc = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=3, random_state=SEED,
                             n_jobs=4, eval_metric="mlogloss")
        xgbc.fit(Xtr, ydir_tr, sample_weight=w)
        p_xgb = xgbc.predict(Xte)

        a_log, a_xgb = accuracy_score(ydir_te, p_logit), accuracy_score(ydir_te, p_xgb)
        best_model = max(a_log, a_xgb)

        dir_rows.append({
            "horizon_days": h, "horizon": HORIZON_NAME[h], "phase": phase,
            "n_features": len(cols), "n_train": Xtr.shape[0], "n_test": n_te,
            "indep_test_windows": round(n_te / h, 1),
            "acc_logistic_%": round(a_log * 100, 1), "acc_xgboost_%": round(a_xgb * 100, 1),
            "acc_majority_%": round(acc_maj * 100, 1), "acc_persistence_%": round(acc_pers * 100, 1),
            "best_model_%": round(best_model * 100, 1), "best_baseline_%": round(best_base * 100, 1),
            "edge_pp": round((best_model - best_base) * 100, 1),
            "beats_baseline": "Yes" if best_model > best_base else "No",
            "f1_macro_xgb": round(f1_score(ydir_te, p_xgb, average="macro", zero_division=0), 3),
        })

        # ---------- return % ----------
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
        ridge.fit(Xtr, yret_tr)
        r_ridge = ridge.predict(Xte)
        xgbr = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=3, random_state=SEED, n_jobs=4)
        xgbr.fit(Xtr, yret_tr)
        r_xgb = xgbr.predict(Xte)

        def rmse(p):
            return float(np.sqrt(mean_squared_error(yret_te, p)))

        def dacc(p):
            return float((np.sign(p) == np.sign(yret_te.values)).mean())

        ret_rows.append({
            "horizon_days": h, "horizon": HORIZON_NAME[h], "phase": phase, "n_test": n_te,
            "rmse_ridge": round(rmse(r_ridge), 2), "rmse_xgb": round(rmse(r_xgb), 2),
            "rmse_naive0": round(rmse_naive, 2), "rmse_trainmean": round(rmse_mean, 2),
            "mae_ridge": round(mean_absolute_error(yret_te, r_ridge), 2),
            "rmse_ratio_ridge_vs_trainmean": round(rmse(r_ridge) / rmse_mean, 3),
            "rmse_ratio_xgb_vs_trainmean": round(rmse(r_xgb) / rmse_mean, 3),
            "dir_acc_ridge_%": round(dacc(r_ridge) * 100, 1),
            "dir_acc_xgb_%": round(dacc(r_xgb) * 100, 1),
            "dir_acc_alwaysguess_%": round(dacc_null * 100, 1),
            "dir_edge_pp": round((max(dacc(r_ridge), dacc(r_xgb)) - dacc_null) * 100, 1),
            "beats_trainmean": "Yes" if min(rmse(r_xgb), rmse(r_ridge)) < rmse_mean else "No",
            "beats_sign_null": "Yes" if max(dacc(r_ridge), dacc(r_xgb)) > dacc_null else "No",
        })

        if phase.startswith("B"):
            s = pd.Series(xgbc.feature_importances_, index=cols)
            imp_rows.append({"horizon_days": h, "horizon": HORIZON_NAME[h],
                             "tier2_share_of_importance_%": round(s[TIER2].sum() * 100, 1),
                             "top_feature": s.idxmax(),
                             "top_feature_is_tier2": s.idxmax() in TIER2})

    ra = [r for r in dir_rows if r["horizon_days"] == h and r["phase"].startswith("A")][0]
    rb = [r for r in dir_rows if r["horizon_days"] == h and r["phase"].startswith("B")][0]
    print(f"h={h:>3}d  A best {ra['best_model_%']:5.1f}%  ->  B best {rb['best_model_%']:5.1f}%  "
          f"GAIN {rb['best_model_%']-ra['best_model_%']:+5.1f} pp   "
          f"(baseline {ra['best_baseline_%']:5.1f}%  edge A {ra['edge_pp']:+5.1f} -> B {rb['edge_pp']:+5.1f})")

DIR = pd.DataFrame(dir_rows)
RET = pd.DataFrame(ret_rows)
IMP = pd.DataFrame(imp_rows)

# ---------------------------------------------------------------- gain table
pa = DIR[DIR.phase.str.startswith("A")].set_index("horizon_days")
pb = DIR[DIR.phase.str.startswith("B")].set_index("horizon_days")
ra_ = RET[RET.phase.str.startswith("A")].set_index("horizon_days")
rb_ = RET[RET.phase.str.startswith("B")].set_index("horizon_days")

GAIN = pd.DataFrame({
    "horizon": pa["horizon"],
    "indep_test_windows": pa["indep_test_windows"],
    "baseline_%": pa["best_baseline_%"],
    "A_best_%": pa["best_model_%"], "B_best_%": pb["best_model_%"],
    "gain_pp": (pb["best_model_%"] - pa["best_model_%"]).round(1),
    "A_edge_pp": pa["edge_pp"], "B_edge_pp": pb["edge_pp"],
    "B_beats_baseline": pb["beats_baseline"],
    "A_ret_ratio": ra_["rmse_ratio_ridge_vs_trainmean"],
    "B_ret_ratio": rb_["rmse_ratio_ridge_vs_trainmean"],
    "ret_ratio_gain": (rb_["rmse_ratio_ridge_vs_trainmean"] - ra_["rmse_ratio_ridge_vs_trainmean"]).round(3),
    "A_sign_edge_pp": ra_["dir_edge_pp"], "B_sign_edge_pp": rb_["dir_edge_pp"],
}).reset_index()

DIR.to_csv(OUT / "ablation_direction_table.csv", index=False)
RET.to_csv(OUT / "ablation_return_table.csv", index=False)
GAIN.to_csv(OUT / "ablation_gain_table.csv", index=False)
IMP.to_csv(OUT / "ablation_tier2_importance.csv", index=False)

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
a = ax[0]
a.plot(GAIN.horizon_days, GAIN["A_best_%"], "o-", label="Phase A (Tier-1)", lw=2)
a.plot(GAIN.horizon_days, GAIN["B_best_%"], "s-", label="Phase B (Tier-1+2)", lw=2)
a.plot(GAIN.horizon_days, GAIN["baseline_%"], "^--", color="black", label="best baseline")
a.set_xscale("log"); a.set_xticks(HORIZONS); a.set_xticklabels(HORIZONS)
a.set_xlabel("Horizon (trading days, log scale)"); a.set_ylabel("Accuracy %")
a.set_title(f"{TICKER} direction — Phase A vs Phase B\n(both must clear the black line)")
a.grid(alpha=.3); a.legend(fontsize=8)

g = ax[1]
g.bar(range(len(GAIN)), GAIN["gain_pp"],
      color=["green" if x > 0 else "red" for x in GAIN["gain_pp"]], alpha=.7)
g.axhline(0, color="black", lw=1)
g.set_xticks(range(len(GAIN))); g.set_xticklabels(GAIN["horizon"], rotation=45, ha="right")
g.set_ylabel("Phase B − Phase A (percentage points)")
g.set_title("Gain from adding Tier-2 (RSI, MACD, volume)\n(above 0 = Tier-2 helped)")
g.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "ablation_gain.png", dpi=140)

fig2, ax2 = plt.subplots(figsize=(7.5, 5))
ax2.plot(GAIN.horizon_days, GAIN["A_edge_pp"], "o-", label="Phase A edge", lw=2)
ax2.plot(GAIN.horizon_days, GAIN["B_edge_pp"], "s-", label="Phase B edge", lw=2)
ax2.axhline(0, color="black", ls="--", label="baseline (must be above)")
ax2.set_xscale("log"); ax2.set_xticks(HORIZONS); ax2.set_xticklabels(HORIZONS)
ax2.set_xlabel("Horizon (trading days, log scale)")
ax2.set_ylabel("Model − baseline (pp)")
ax2.set_title(f"{TICKER} — edge over baseline, A vs B\n(the only number that matters)")
ax2.grid(alpha=.3); ax2.legend(fontsize=8)
fig2.tight_layout(); fig2.savefig(OUT / "ablation_edge.png", dpi=140)

# ---------------------------------------------------------------- summary
b_wins = GAIN[GAIN.B_beats_baseline == "Yes"]
pos_gain = GAIN[GAIN.gain_pp > 0]
mean_gain = GAIN.gain_pp.mean()
tier2_share = IMP["tier2_share_of_importance_%"].mean()

md = f"""# Phase Ablation — A (Tier-1) vs B (Tier-1 + Tier-2)

**Stock:** {TICKER} daily · **Horizons:** {HORIZONS} trading days · **Method:** DIRECT (model per horizon)
**Phase A:** {len(TIER1)} features — {', '.join(TIER1)}
**Phase B:** {len(TIER1) + len(TIER2)} features — A + {', '.join(TIER2)}
**Identical** rows, split (80/20 chrono + h-bar purge), seeds and baselines in both phases.
Only the feature set changes, so `gain_pp` is the clean effect of Tier-2.

## BOTTOM LINE (caveman)

- **Tier-2 gain: {mean_gain:+.1f} pp on average.** Positive at {len(pos_gain)} of {len(GAIN)} horizons.
- **Horizons where Phase B beats the baseline: {len(b_wins)} of {len(GAIN)}.**
- XGBoost spends **{tier2_share:.0f}%** of its importance on the Tier-2 features — it *uses* them,
  it just does not get *paid* for using them.
- Verdict: {'Tier-2 gives a real, repeatable lift.' if len(b_wins) > 0 and mean_gain > 1 else 'RSI, MACD and volume add essentially nothing. Noise-level wiggle, no horizon crosses the baseline. Technical indicators are exhausted — the missing information is not in the price chart.'}

## The gain table (the finding)

{md_table(GAIN[['horizon', 'baseline_%', 'A_best_%', 'B_best_%', 'gain_pp', 'A_edge_pp', 'B_edge_pp', 'B_beats_baseline']])}

`gain_pp` = Phase B best model − Phase A best model. `edge_pp` = model − best baseline.
**Edge is what counts; gain only matters if it pushes edge above 0.**

## Full direction detail (both phases)

{md_table(DIR[['horizon', 'phase', 'n_features', 'acc_logistic_%', 'acc_xgboost_%', 'acc_majority_%', 'acc_persistence_%', 'edge_pp', 'beats_baseline']])}

## Return % — did Tier-2 help there?

{md_table(GAIN[['horizon', 'A_ret_ratio', 'B_ret_ratio', 'ret_ratio_gain', 'A_sign_edge_pp', 'B_sign_edge_pp']])}

`ret_ratio` = model RMSE ÷ train-mean-drift RMSE. **Below 1.0 = features helped.**
`ret_ratio_gain` **negative = Tier-2 improved it.**
`sign_edge_pp` = sign accuracy − "always guess the winning side". **Above 0 = real.**

## Where XGBoost put its attention in Phase B

{md_table(IMP[['horizon', 'tier2_share_of_importance_%', 'top_feature', 'top_feature_is_tier2']])}

## Caveats
- Phase A numbers here differ slightly from `results/direction/multi_horizon/` because MACD's
  35-bar warmup removes ~15 extra early rows from BOTH phases. This table is the fair A/B.
- Long horizons overlap: at 252 days only ~{GAIN.iloc[-1]['indep_test_windows']} independent
  windows exist. No significance claims there.
- A {mean_gain:+.1f} pp average shift is inside noise for this sample size. The honest reading is
  "no effect", not "Tier-2 actively hurts".
- One stock, one split. Correlation, not causation.

## Next
Phase C — macro (interest rates / spread, then inflation, FX, M2). Rates data is already in
`cleaned_data/interest_rates_monthly.csv`. Monthly data suits the LONG horizons (22d+), which is
exactly where technicals are weakest.
"""
(OUT / "ablation_summary.md").write_text(md)

print("\n" + "=" * 96)
print(GAIN[["horizon", "baseline_%", "A_best_%", "B_best_%", "gain_pp",
            "A_edge_pp", "B_edge_pp", "B_beats_baseline"]].to_string(index=False))
print("=" * 96)
print(GAIN[["horizon", "A_ret_ratio", "B_ret_ratio", "ret_ratio_gain",
            "A_sign_edge_pp", "B_sign_edge_pp"]].to_string(index=False))
print(f"\nMean Tier-2 gain           : {mean_gain:+.1f} pp  (positive at {len(pos_gain)}/{len(GAIN)} horizons)")
print(f"Phase B beats baseline at  : {len(b_wins)}/{len(GAIN)} horizons")
print(f"Tier-2 share of XGB importance: {tier2_share:.0f}%")
print(f"Saved to {OUT}")
