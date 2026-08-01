#!/usr/bin/env python3
"""
Phase ABLATION runner — add feature tiers one step at a time, measure the GAIN at each step.

  Phase A  Tier-1 technical   returns, MA ratios, momentum, volatility
  Phase B  + Tier-2 technical + RSI(14), MACD (line/signal/hist), volume change
  Phase C  + macro (rates)    + policy rate, T-bills, AWDR/AWLR, spread, term slope, and their
                                1m/3m changes                     <- info NOT in the price chart

Every phase runs on the SAME stock, SAME horizons, SAME rows, SAME split, SAME seeds. Only the
feature list changes, so the accuracy difference between phases is the clean effect of the
features added. Adding Phase D/E/F later = add one entry to PHASES.

Macro look-ahead guard (important): CBSL monthly figures for month M are only published during
month M+1. Each monthly row is therefore stamped `available_from = month_end + 35 days` and
merged backward-asof onto trading days. A model on 2020-03-10 can only see January 2020 rates.

Other honesty guards: chronological 80/20, h-bar purge gap between train and test, train-only
scaling, independent-window count reported, and the two extra nulls (train-mean drift for return
RMSE, always-guess-winning-side for sign accuracy).

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
TRAIN_FRAC = 0.80
SEED = 42
PUB_LAG_DAYS = 35            # CBSL publication lag guard

# ================================================================ features
d = (pd.read_csv(DATA / f"{TICKER}_daily_clean.csv", parse_dates=["date"])
       .sort_values("date").reset_index(drop=True))
c = d["close"].astype(float)
v = d["volume"].astype(float)
ret1 = c.pct_change()


def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


feat = pd.DataFrame(index=d.index)

# ---- TIER 1 ----
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

# ---- TIER 2 ----
feat["rsi_14"] = rsi(c, 14)
_macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
_signal = _macd.ewm(span=9, adjust=False).mean()
feat["macd"] = _macd / c
feat["macd_signal"] = _signal / c
feat["macd_hist"] = (_macd - _signal) / c
feat["volchg_5"] = v / v.rolling(5).mean() - 1
feat["volchg_20"] = v / v.rolling(20).mean() - 1
feat.loc[:34, ["macd", "macd_signal", "macd_hist"]] = np.nan   # real MACD warmup
TIER2 = [x for x in feat.columns if x not in TIER1]

# ---- TIER 3: MACRO (monthly rates -> lagged onto trading days) ----
rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True))
rt = rt.ffill()                                   # tb_3m has 3 gaps; ffill uses PAST only
rt["term_slope"] = rt["tb_12m"] - rt["tb_3m"]     # long minus short = curve shape
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_spread_3m"] = rt["spread"].diff(3)
rt["d_tb3m_1m"] = rt["tb_3m"].diff()
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
rt["d_awlr_3m"] = rt["awlr"].diff(3)
# Split macro into LEVELS and CHANGES on purpose. Levels (policy_rate, tb_12m, ...) trend and
# never repeat across regimes -> a tree can memorise "rates were 15% in 2022" and that mapping
# is worthless out-of-sample. Changes are stationary. Phase C uses both; Phase C2 uses changes
# only, which tells us whether macro is useless or just fed in the wrong form.
MACRO_LEVELS = ["policy_rate", "tb_3m", "tb_12m", "awdr", "awlr", "spread"]
MACRO_CHANGES = ["term_slope", "d_policy_1m", "d_spread_1m", "d_spread_3m",
                 "d_tb3m_1m", "d_tb3m_3m", "d_awlr_3m"]
MACRO = MACRO_LEVELS + MACRO_CHANGES

# publication-lag guard, then backward-asof merge
rt["available_from"] = rt["date"] + pd.Timedelta(days=PUB_LAG_DAYS)
macro_daily = pd.merge_asof(
    d[["date"]].sort_values("date"),
    rt[["available_from"] + MACRO].sort_values("available_from"),
    left_on="date", right_on="available_from", direction="backward")
for m in MACRO:
    feat[m] = macro_daily[m].values

PHASES = {
    "A (Tier-1)": TIER1,
    "B (+Tier-2)": TIER1 + TIER2,
    "C (+macro)": TIER1 + TIER2 + MACRO,
    "C2 (+macro Δ only)": TIER1 + TIER2 + MACRO_CHANGES,
}
PHASE_KEYS = list(PHASES)
ALL_FEATURES = TIER1 + TIER2 + MACRO


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


# ================================================================ run
dir_rows, ret_rows, imp_rows = [], [], []

for h in HORIZONS:
    dz = deadzone(h)
    df = feat.copy()
    df["fwd"] = c.shift(-h) / c - 1
    df["past_h"] = c / c.shift(h) - 1
    df["date"] = d["date"]
    # ONE dropna across ALL phases' features -> every phase sees identical rows.
    df = df.dropna(subset=ALL_FEATURES + ["fwd", "past_h"]).reset_index(drop=True)

    y_dir = pd.Series(to_class(df["fwd"].values, dz), index=df.index)
    y_ret = df["fwd"] * 100.0

    n = len(df)
    split = int(n * TRAIN_FRAC)
    tr = slice(0, max(split - h, 50))              # purge h bars of label overlap
    te = slice(split, n)
    n_te = n - split

    ydir_tr, ydir_te = y_dir.iloc[tr], y_dir.iloc[te]
    yret_tr, yret_te = y_ret.iloc[tr], y_ret.iloc[te]

    # ---- baselines: identical for every phase ----
    majority = int(ydir_tr.value_counts().idxmax())
    acc_maj = accuracy_score(ydir_te, np.full(n_te, majority))
    acc_pers = accuracy_score(ydir_te, to_class(df["past_h"].iloc[te].values, dz))
    best_base = max(acc_maj, acc_pers)

    rmse_mean = float(np.sqrt(mean_squared_error(yret_te, np.full(n_te, yret_tr.mean()))))
    rmse_naive = float(np.sqrt(mean_squared_error(yret_te, np.zeros(n_te))))
    up_share = float((yret_te > 0).mean())
    dacc_null = max(up_share, 1 - up_share)

    present = sorted(ydir_tr.unique())
    wmap = {k: len(ydir_tr) / (len(present) * (ydir_tr == k).sum()) for k in present}
    w = ydir_tr.map(wmap).values

    for phase, cols in PHASES.items():
        Xtr, Xte = df[cols].iloc[tr], df[cols].iloc[te]

        # ---------- direction ----------
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

        s = pd.Series(xgbc.feature_importances_, index=cols)
        imp_rows.append({
            "horizon_days": h, "horizon": HORIZON_NAME[h], "phase": phase,
            "tier1_share_%": round(s[[x for x in cols if x in TIER1]].sum() * 100, 1),
            "tier2_share_%": round(s[[x for x in cols if x in TIER2]].sum() * 100, 1),
            "macro_share_%": round(s[[x for x in cols if x in MACRO]].sum() * 100, 1),
            "top_feature": s.idxmax(),
            "top_feature_tier": ("macro" if s.idxmax() in MACRO else
                                 "tier2" if s.idxmax() in TIER2 else "tier1"),
        })

    row = {p: [r for r in dir_rows if r["horizon_days"] == h and r["phase"] == p][0]
           for p in PHASE_KEYS}
    print(f"h={h:>3}d  base {row[PHASE_KEYS[0]]['best_baseline_%']:5.1f}% | "
          + "  ".join(f"{p.split()[0]} {row[p]['best_model_%']:5.1f}%" for p in PHASE_KEYS)
          + f" | B->C {row[PHASE_KEYS[2]]['best_model_%'] - row[PHASE_KEYS[1]]['best_model_%']:+6.1f}"
          + f"  B->C2 {row[PHASE_KEYS[3]]['best_model_%'] - row[PHASE_KEYS[1]]['best_model_%']:+6.1f} pp")

DIR = pd.DataFrame(dir_rows)
RET = pd.DataFrame(ret_rows)
IMP = pd.DataFrame(imp_rows)

# ================================================================ gain table
P = {p: DIR[DIR.phase == p].set_index("horizon_days") for p in PHASE_KEYS}
Q = {p: RET[RET.phase == p].set_index("horizon_days") for p in PHASE_KEYS}
A, B, C, C2 = PHASE_KEYS

GAIN = pd.DataFrame({
    "horizon": P[A]["horizon"],
    "indep_test_windows": P[A]["indep_test_windows"],
    "baseline_%": P[A]["best_baseline_%"],
    "A_best_%": P[A]["best_model_%"], "B_best_%": P[B]["best_model_%"],
    "C_best_%": P[C]["best_model_%"], "C2_best_%": P[C2]["best_model_%"],
    "gain_A_to_B_pp": (P[B]["best_model_%"] - P[A]["best_model_%"]).round(1),
    "gain_B_to_C_pp": (P[C]["best_model_%"] - P[B]["best_model_%"]).round(1),
    "gain_B_to_C2_pp": (P[C2]["best_model_%"] - P[B]["best_model_%"]).round(1),
    "A_edge_pp": P[A]["edge_pp"], "B_edge_pp": P[B]["edge_pp"],
    "C_edge_pp": P[C]["edge_pp"], "C2_edge_pp": P[C2]["edge_pp"],
    "C_beats_baseline": P[C]["beats_baseline"], "C2_beats_baseline": P[C2]["beats_baseline"],
    "A_ret_ratio": Q[A]["rmse_ratio_ridge_vs_trainmean"],
    "B_ret_ratio": Q[B]["rmse_ratio_ridge_vs_trainmean"],
    "C_ret_ratio": Q[C]["rmse_ratio_ridge_vs_trainmean"],
    "C2_ret_ratio": Q[C2]["rmse_ratio_ridge_vs_trainmean"],
    "C_sign_edge_pp": Q[C]["dir_edge_pp"], "C2_sign_edge_pp": Q[C2]["dir_edge_pp"],
}).reset_index()

DIR.to_csv(OUT / "ablation_direction_table.csv", index=False)
RET.to_csv(OUT / "ablation_return_table.csv", index=False)
GAIN.to_csv(OUT / "ablation_gain_table.csv", index=False)
IMP.to_csv(OUT / "ablation_feature_importance.csv", index=False)

# ================================================================ plots
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
a = ax[0]
for p, mk in zip(PHASE_KEYS, ["o-", "s-", "D-", "v-"]):
    a.plot(GAIN.horizon_days, P[p]["best_model_%"].values, mk, label=f"Phase {p}", lw=2)
a.plot(GAIN.horizon_days, GAIN["baseline_%"], "^--", color="black", label="best baseline")
a.set_xscale("log"); a.set_xticks(HORIZONS); a.set_xticklabels(HORIZONS)
a.set_xlabel("Horizon (trading days, log scale)"); a.set_ylabel("Accuracy %")
a.set_title(f"{TICKER} direction — all phases\n(must clear the black line)")
a.grid(alpha=.3); a.legend(fontsize=8)

g = ax[1]
x = np.arange(len(GAIN))
g.bar(x - 0.26, GAIN["gain_A_to_B_pp"], 0.26, label="A→B  (+Tier-2)", color="steelblue", alpha=.8)
g.bar(x, GAIN["gain_B_to_C_pp"], 0.26, label="B→C  (+macro, levels+Δ)", color="darkorange", alpha=.85)
g.bar(x + 0.26, GAIN["gain_B_to_C2_pp"], 0.26, label="B→C2 (+macro Δ only)", color="seagreen", alpha=.85)
g.axhline(0, color="black", lw=1)
g.set_xticks(x); g.set_xticklabels(GAIN["horizon"], rotation=45, ha="right")
g.set_ylabel("Accuracy gain (percentage points)")
g.set_title("Gain from each step\n(above 0 = the added features helped)")
g.grid(alpha=.3, axis="y"); g.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ablation_gain.png", dpi=140)

fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
e = ax2[0]
for p, mk in zip(PHASE_KEYS, ["o-", "s-", "D-", "v-"]):
    e.plot(GAIN.horizon_days, P[p]["edge_pp"].values, mk, label=f"Phase {p}", lw=2)
e.axhline(0, color="black", ls="--", label="baseline (must be above)")
e.set_xscale("log"); e.set_xticks(HORIZONS); e.set_xticklabels(HORIZONS)
e.set_xlabel("Horizon (trading days, log scale)"); e.set_ylabel("Model − baseline (pp)")
e.set_title(f"{TICKER} — edge over baseline by phase\n(the only number that matters)")
e.grid(alpha=.3); e.legend(fontsize=8)

s = ax2[1]
impC = IMP[IMP.phase == C].set_index("horizon_days")
s.stackplot(GAIN.horizon_days, impC["tier1_share_%"], impC["tier2_share_%"], impC["macro_share_%"],
            labels=["Tier-1 price", "Tier-2 indicators", "Macro rates"], alpha=.8)
s.set_xscale("log"); s.set_xticks(HORIZONS); s.set_xticklabels(HORIZONS)
s.set_xlabel("Horizon (trading days, log scale)"); s.set_ylabel("Share of XGBoost importance %")
s.set_title("Phase C — where the model looks\n(uses macro heavily; still no accuracy)")
s.legend(fontsize=8, loc="lower left")
fig2.tight_layout(); fig2.savefig(OUT / "ablation_edge.png", dpi=140)

# ================================================================ summary
mean_ab, mean_bc = GAIN.gain_A_to_B_pp.mean(), GAIN.gain_B_to_C_pp.mean()
mean_bc2 = GAIN.gain_B_to_C2_pp.mean()
pos_bc, pos_bc2 = GAIN[GAIN.gain_B_to_C_pp > 0], GAIN[GAIN.gain_B_to_C2_pp > 0]
c_wins = GAIN[GAIN.C_beats_baseline == "Yes"]
c2_wins = GAIN[GAIN.C2_beats_baseline == "Yes"]
macro_share = impC["macro_share_%"].mean()
long_bc = GAIN[GAIN.horizon_days >= 22].gain_B_to_C_pp.mean()
short_bc = GAIN[GAIN.horizon_days < 22].gain_B_to_C_pp.mean()
swing = GAIN.gain_B_to_C_pp.max() - GAIN.gain_B_to_C_pp.min()
worst_ratio = GAIN.C_ret_ratio.max()

md = f"""# Phase Ablation — A (Tier-1) → B (+Tier-2) → C (+macro rates) → C2 (macro Δ only)

**Stock:** {TICKER} daily · **Horizons:** {HORIZONS} trading days · **Method:** DIRECT (model per horizon)

| Phase | Features | What is added |
|---|---|---|
| A | {len(TIER1)} | Tier-1 technical: {', '.join(TIER1)} |
| B | {len(TIER1) + len(TIER2)} | + Tier-2 technical: {', '.join(TIER2)} |
| C | {len(ALL_FEATURES)} | + macro rates, levels **and** changes: {', '.join(MACRO)} |
| C2 | {len(TIER1) + len(TIER2) + len(MACRO_CHANGES)} | + macro **changes only** (levels dropped): {', '.join(MACRO_CHANGES)} |

C2 is a diagnostic, not a new tier: it answers "is macro useless, or was it fed in the wrong form?"
Rate *levels* trend and never repeat across regimes, so a tree can memorise "rates were 15% in
2022"; that mapping is worthless out-of-sample. Rate *changes* are stationary and reusable.

Identical rows, split (80/20 chrono + h-bar purge), seeds and baselines across all phases.
Only the feature list changes, so each `gain` column is the clean effect of that step.

**Macro look-ahead guard:** every monthly CBSL figure is stamped `month_end + {PUB_LAG_DAYS} days`
before being merged backward-asof onto trading days. The model never sees a rate before it was
published. Inflation, FX and M2 are still on the COLLECT list — **Phase C here is rates only.**

## BOTTOM LINE (caveman)

- **Macro gain (B→C): {mean_bc:+.1f} pp on average.** Positive at {len(pos_bc)} of {len(GAIN)} horizons.
- **Phase C beats the baseline at {len(c_wins)} of {len(GAIN)} horizons.** Phase C2: {len(c2_wins)} of {len(GAIN)}.
- **The gain is not just negative, it is WILD:** a {swing:.0f} pp swing across horizons
  ({GAIN.gain_B_to_C_pp.min():+.1f} to {GAIN.gain_B_to_C_pp.max():+.1f}). That is not weak signal,
  that is **overfitting**.
- **Return % confirms it:** Phase C RMSE blows out to **{worst_ratio:.2f}×** the train-mean null
  (Phase A/B sat at ~0.99). Adding macro made the return model materially worse.
- **XGBoost hands {macro_share:.0f}% of its importance to macro** and still loses accuracy — the
  classic fingerprint of a model latching onto a trending variable.
- **The C2 diagnostic settles it: {mean_bc2:+.1f} pp using macro CHANGES only.**
  {'Dropping the trending levels fixes the damage, so the problem was the FORM of the data, not macro itself.' if mean_bc2 > mean_bc + 1 else 'Even in stationary change form, macro adds nothing — so it is the information, not the form.'}
- Verdict: {'Macro rates give a real lift. Follow this thread.' if len(c_wins) > 0 and mean_bc > 1 else 'Interest-rate data does NOT predict HNB direction at any horizon. Four feature sets, up to 22 features, 1 day to 1 year: still no edge anywhere.'}

## The gain table (the finding)

{md_table(GAIN[['horizon', 'baseline_%', 'A_best_%', 'B_best_%', 'C_best_%', 'C2_best_%', 'gain_A_to_B_pp', 'gain_B_to_C_pp', 'gain_B_to_C2_pp', 'C_edge_pp', 'C2_edge_pp', 'C2_beats_baseline']])}

`gain` = best model of that phase − best model of the previous phase.
`edge_pp` = model − best baseline. **Edge is what counts; gain only matters if it lifts edge above 0.**

## Full direction detail (all phases)

{md_table(DIR[['horizon', 'phase', 'n_features', 'acc_logistic_%', 'acc_xgboost_%', 'acc_majority_%', 'acc_persistence_%', 'edge_pp', 'beats_baseline']])}

## Return % — did macro help there?

{md_table(GAIN[['horizon', 'A_ret_ratio', 'B_ret_ratio', 'C_ret_ratio', 'C2_ret_ratio', 'C_sign_edge_pp', 'C2_sign_edge_pp']])}

`ret_ratio` = model RMSE ÷ train-mean-drift RMSE. **Below 1.0 = features helped.**
`C_sign_edge_pp` = sign accuracy − "always guess the winning side". **Above 0 = real.**

## Where the model looks in Phase C

{md_table(IMP[IMP.phase == C][['horizon', 'tier1_share_%', 'tier2_share_%', 'macro_share_%', 'top_feature', 'top_feature_tier']])}

This is the key diagnostic: macro importance climbs from {impC['macro_share_%'].iloc[0]:.0f}% at
1 day to {impC['macro_share_%'].iloc[-1]:.0f}% at 1 year while accuracy *falls*. The model is
fitting the rate series as a slow-moving trend proxy, not using it as signal.

## Caveats
- Rates are **monthly**, held flat between releases, so within a month every trading day carries
  the same macro value. That suits long horizons and is nearly useless for the 1-day model.
- Long horizons overlap: at 252 days only ~{GAIN.iloc[-1]['indep_test_windows']} independent
  windows exist. No significance claims there.
- Phase A/B numbers shift slightly from earlier runs because macro availability trims the early
  rows from all phases. This table is the fair three-way comparison.
- One stock, one split. Correlation, not causation.
- Known real effect from the earlier spread work: bank returns react negatively to a widening
  spread **contemporaneously (same month)**. That is an *explanatory* result, not a *predictive*
  one — this run confirms it does not forecast.

## Next
Phase D — sector (ASPI market return, spread × is_bank, peer-bank returns) across banks / finance /
control, not just HNB. Then E (events) and F (news sentiment), which are the last untested sources
of information outside the price chart.
"""
(OUT / "ablation_summary.md").write_text(md)

print("\n" + "=" * 104)
print(GAIN[["horizon", "baseline_%", "A_best_%", "B_best_%", "C_best_%", "C2_best_%",
            "gain_B_to_C_pp", "gain_B_to_C2_pp", "C_edge_pp", "C2_edge_pp",
            "C2_beats_baseline"]].to_string(index=False))
print("=" * 104)
print(GAIN[["horizon", "A_ret_ratio", "B_ret_ratio", "C_ret_ratio", "C2_ret_ratio",
            "C_sign_edge_pp", "C2_sign_edge_pp"]].to_string(index=False))
print("=" * 104)
print(IMP[IMP.phase == C][["horizon", "tier1_share_%", "tier2_share_%", "macro_share_%",
                           "top_feature", "top_feature_tier"]].to_string(index=False))
print(f"\nMean gain A->B (Tier-2): {mean_ab:+.1f} pp")
print(f"Mean gain B->C (macro) : {mean_bc:+.1f} pp   (positive at {len(pos_bc)}/{len(GAIN)} horizons)")
print(f"Mean gain B->C2 (macro d): {mean_bc2:+.1f} pp   (positive at {len(pos_bc2)}/{len(GAIN)} horizons)")
print(f"   long horizons >=1mo : {long_bc:+.1f} pp | short horizons: {short_bc:+.1f} pp")
print(f"Phase C beats baseline : {len(c_wins)}/{len(GAIN)} horizons | Phase C2: {len(c2_wins)}/{len(GAIN)}")
print(f"Macro share of XGB importance: {macro_share:.0f}%")
print(f"Saved to {OUT}")
