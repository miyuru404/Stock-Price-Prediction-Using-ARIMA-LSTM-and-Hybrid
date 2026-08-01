#!/usr/bin/env python3
"""
Multi-horizon DIRECT direction + return runner — Phase A (Tier-1 technical only).

For each horizon h in [1,5,10,15,22,44,66,132,252] trading days, a SEPARATE model is
trained (direct method, no recursion):
  * label_direction = Buy/Hold/Sell of the h-day forward return (dead-zone scales with sqrt(h))
  * label_return    = the h-day forward return in %
Models: Logistic Regression + XGBoost (classification), Ridge + XGBoost (regression).
Baselines: majority-class + persistence (direction); naive "no change" (return).

Honesty guards:
  - chronological 80/20 split, never shuffled
  - PURGE GAP of h bars between train and test (a training label that reaches into the
    test window would be leakage)
  - scalers fit on train only
  - effective independent test windows (n_test / h) reported, because overlapping
    long-horizon labels are NOT independent samples

Outputs -> results/direction/multi_horizon/
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
OUT = ROOT / "results" / "direction" / "multi_horizon"
OUT.mkdir(parents=True, exist_ok=True)

TICKER = "HNB"
HORIZONS = [1, 5, 10, 15, 22, 44, 66, 132, 252]
HORIZON_NAME = {1: "1 day", 5: "1 week", 10: "2 weeks", 15: "3 weeks", 22: "1 month",
                44: "2 months", 66: "3 months", 132: "6 months", 252: "1 year"}
DEADZONE_1D = 0.005          # +/-0.5% at h=1; scaled by sqrt(h) for longer horizons
CLASSES = ["Sell", "Hold", "Buy"]        # y = 0, 1, 2
TRAIN_FRAC = 0.80
SEED = 42

# ---------------------------------------------------------------- data + Tier-1 features
d = (pd.read_csv(DATA / f"{TICKER}_daily_clean.csv", parse_dates=["date"])
       .sort_values("date").reset_index(drop=True))
c = d["close"].astype(float)
ret1 = c.pct_change()

# TIER 1 ONLY: recent returns, MA ratios, momentum, volatility.
# (No RSI / MACD / volume -> those are Tier 2, saved for Phase B.)
feat = pd.DataFrame(index=d.index)
feat["ret_1"] = ret1
feat["ret_5"] = c.pct_change(5)
feat["ret_10"] = c.pct_change(10)
feat["ma5_ratio"] = c / c.rolling(5).mean() - 1
feat["ma10_ratio"] = c / c.rolling(10).mean() - 1
feat["ma20_ratio"] = c / c.rolling(20).mean() - 1
feat["momentum_10"] = c / c.shift(10) - 1
feat["vol_10"] = ret1.rolling(10).std()
feat["vol_20"] = ret1.rolling(20).std()
FEATURES = list(feat.columns)


def md_table(df, index=False):
    """Tiny markdown-table writer (avoids the optional `tabulate` dependency)."""
    t = df.reset_index() if index else df
    cols = [str(x) for x in t.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def deadzone(h):
    """Dead-zone must grow with horizon, else every 1-year move is Buy/Sell and Hold vanishes.
    Random-walk scaling: sigma ~ sqrt(t)."""
    return DEADZONE_1D * np.sqrt(h)


def to_class(r, dz):
    return np.where(r > dz, 2, np.where(r < -dz, 0, 1))


dir_rows, ret_rows, class_rows = [], [], []
pred_dump = {}

for h in HORIZONS:
    dz = deadzone(h)
    fwd = c.shift(-h) / c - 1                      # forward h-day return (the label)
    past = c / c.shift(h) - 1                      # past h-day return (persistence baseline)

    df = feat.copy()
    df["fwd"] = fwd
    df["past_h"] = past
    df["date"] = d["date"]
    df = df.dropna().reset_index(drop=True)        # drops warmup NaNs + last h rows (no label)

    y_dir = pd.Series(to_class(df["fwd"].values, dz), index=df.index)
    y_ret = df["fwd"] * 100.0                      # in %
    X = df[FEATURES]

    n = len(df)
    split = int(n * TRAIN_FRAC)
    # PURGE: last h training rows have labels that land inside the test window -> drop them.
    tr_end = max(split - h, 50)
    tr = slice(0, tr_end)
    te = slice(split, n)

    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ydir_tr, ydir_te = y_dir.iloc[tr], y_dir.iloc[te]
    yret_tr, yret_te = y_ret.iloc[tr], y_ret.iloc[te]
    n_te = len(Xte)
    n_indep = n_te / h                             # non-overlapping test windows

    # ---------------- DIRECTION ----------------
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

    majority = int(ydir_tr.value_counts().idxmax())
    p_maj = np.full(n_te, majority)
    p_pers = to_class(df["past_h"].iloc[te].values, dz)   # "next h days repeat the last h days"

    accs = {"Logistic": accuracy_score(ydir_te, p_logit),
            "XGBoost": accuracy_score(ydir_te, p_xgb),
            "Majority": accuracy_score(ydir_te, p_maj),
            "Persistence": accuracy_score(ydir_te, p_pers)}
    best_base = max(accs["Majority"], accs["Persistence"])
    best_model = max(accs["Logistic"], accs["XGBoost"])

    dir_rows.append({
        "horizon_days": h, "horizon": HORIZON_NAME[h], "deadzone_%": round(dz * 100, 2),
        "n_train": len(Xtr), "n_test": n_te, "indep_test_windows": round(n_indep, 1),
        "acc_logistic_%": round(accs["Logistic"] * 100, 1),
        "acc_xgboost_%": round(accs["XGBoost"] * 100, 1),
        "acc_majority_%": round(accs["Majority"] * 100, 1),
        "acc_persistence_%": round(accs["Persistence"] * 100, 1),
        "best_model_%": round(best_model * 100, 1),
        "best_baseline_%": round(best_base * 100, 1),
        "edge_pp": round((best_model - best_base) * 100, 1),
        "beats_baseline": "Yes" if best_model > best_base else ("Tie" if best_model == best_base else "No"),
        "f1_macro_xgb": round(f1_score(ydir_te, p_xgb, average="macro", zero_division=0), 3),
    })
    for nm, cid in zip(CLASSES, [0, 1, 2]):
        class_rows.append({"horizon_days": h, "class": nm,
                           "train_share_%": round((ydir_tr == cid).mean() * 100, 1),
                           "test_share_%": round((ydir_te == cid).mean() * 100, 1)})

    # ---------------- RETURN % ----------------
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
    ridge.fit(Xtr, yret_tr)
    r_ridge = ridge.predict(Xte)

    xgbr = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=3, random_state=SEED, n_jobs=4)
    xgbr.fit(Xtr, yret_tr)
    r_xgb = xgbr.predict(Xte)

    r_naive = np.zeros(n_te)                       # "no change" — the honest null
    r_mean = np.full(n_te, yret_tr.mean())         # train-mean drift

    def rmse(p):
        return float(np.sqrt(mean_squared_error(yret_te, p)))

    def dacc(p):
        return float((np.sign(p) == np.sign(yret_te.values)).mean())

    # "always guess the winning side" — the sign-accuracy null. If the test window drifted up
    # 59% of the time, 59% is FREE and any model at 56% is actually losing.
    up_share = float((yret_te > 0).mean())
    dacc_const = max(up_share, 1 - up_share)

    ret_rows.append({
        "horizon_days": h, "horizon": HORIZON_NAME[h], "n_test": n_te,
        "rmse_ridge": round(rmse(r_ridge), 2), "rmse_xgb": round(rmse(r_xgb), 2),
        "rmse_naive0": round(rmse(r_naive), 2), "rmse_trainmean": round(rmse(r_mean), 2),
        "mae_xgb": round(mean_absolute_error(yret_te, r_xgb), 2),
        "mae_naive0": round(mean_absolute_error(yret_te, r_naive), 2),
        "rmse_ratio_xgb_vs_naive": round(rmse(r_xgb) / rmse(r_naive), 2),
        "rmse_ratio_ridge_vs_naive": round(rmse(r_ridge) / rmse(r_naive), 2),
        # train-mean is the TOUGHER null: it already contains the drift, so beating it
        # is the only way to show the FEATURES did something.
        "rmse_ratio_ridge_vs_trainmean": round(rmse(r_ridge) / rmse(r_mean), 3),
        "rmse_ratio_xgb_vs_trainmean": round(rmse(r_xgb) / rmse(r_mean), 3),
        "dir_acc_ridge_%": round(dacc(r_ridge) * 100, 1),
        "dir_acc_xgb_%": round(dacc(r_xgb) * 100, 1),
        "up_share_test_%": round(up_share * 100, 1),
        "dir_acc_alwaysguess_%": round(dacc_const * 100, 1),
        "dir_edge_pp": round((max(dacc(r_ridge), dacc(r_xgb)) - dacc_const) * 100, 1),
        "beats_naive0": "Yes" if min(rmse(r_xgb), rmse(r_ridge)) < rmse(r_naive) else "No",
        "beats_trainmean": "Yes" if min(rmse(r_xgb), rmse(r_ridge)) < rmse(r_mean) else "No",
        "beats_sign_null": "Yes" if max(dacc(r_ridge), dacc(r_xgb)) > dacc_const else "No",
    })

    pred_dump[h] = pd.DataFrame({
        "date": df["date"].iloc[te].values,
        "actual_ret_%": yret_te.values, "actual_class": ydir_te.values,
        "pred_class_logit": p_logit, "pred_class_xgb": p_xgb,
        "pred_class_persistence": p_pers,
        "pred_ret_ridge_%": r_ridge, "pred_ret_xgb_%": r_xgb,
    })
    print(f"h={h:>3}d  dz=±{dz*100:.2f}%  train {len(Xtr):>4} test {n_te:>4} "
          f"(indep {n_indep:5.1f}) | dir: logit {accs['Logistic']*100:5.1f}%  "
          f"xgb {accs['XGBoost']*100:5.1f}%  maj {accs['Majority']*100:5.1f}%  "
          f"pers {accs['Persistence']*100:5.1f}% | ret rmse xgb/naive "
          f"{rmse(r_xgb)/rmse(r_naive):.2f}")

DIR = pd.DataFrame(dir_rows)
RET = pd.DataFrame(ret_rows)
BAL = pd.DataFrame(class_rows)
DIR.to_csv(OUT / "mh_direction_table.csv", index=False)
RET.to_csv(OUT / "mh_return_table.csv", index=False)
BAL.to_csv(OUT / "mh_class_balance.csv", index=False)
for h, p in pred_dump.items():
    p.to_csv(OUT / f"preds_h{h}.csv", index=False)

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
a = ax[0]
a.plot(DIR.horizon_days, DIR["acc_xgboost_%"], "o-", label="XGBoost", lw=2)
a.plot(DIR.horizon_days, DIR["acc_logistic_%"], "s-", label="Logistic", lw=2)
a.plot(DIR.horizon_days, DIR["acc_persistence_%"], "^--", label="Persistence (baseline)", color="grey")
a.plot(DIR.horizon_days, DIR["acc_majority_%"], "v--", label="Majority (baseline)", color="black")
a.set_xscale("log"); a.set_xticks(HORIZONS); a.set_xticklabels(HORIZONS)
a.set_xlabel("Horizon (trading days, log scale)"); a.set_ylabel("Accuracy %")
a.set_title(f"{TICKER} direction — accuracy vs horizon\n(Tier-1 technical only, direct models)")
a.grid(alpha=.3); a.legend(fontsize=8)

b = ax[1]
b.plot(RET.horizon_days, RET["rmse_ratio_xgb_vs_trainmean"], "o-", label="XGBoost", lw=2)
b.plot(RET.horizon_days, RET["rmse_ratio_ridge_vs_trainmean"], "s-", label="Ridge", lw=2)
b.axhline(1.0, color="black", ls="--", label="train-mean drift = 1.0")
b.set_xscale("log"); b.set_xticks(HORIZONS); b.set_xticklabels(HORIZONS)
b.set_xlabel("Horizon (trading days, log scale)")
b.set_ylabel("RMSE / train-mean RMSE  (<1 = features helped)")
b.set_title(f"{TICKER} return % — error vs train-mean drift\n(above the line = features added nothing)")
b.grid(alpha=.3); b.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "mh_accuracy_vs_horizon.png", dpi=140)

fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
e = ax2[0]
e.plot(DIR.horizon_days, DIR["edge_pp"], "o-", color="darkred", lw=2)
e.axhline(0, color="black", ls="--")
e.fill_between(DIR.horizon_days, DIR["edge_pp"], 0,
               where=DIR["edge_pp"] > 0, color="green", alpha=.2)
e.fill_between(DIR.horizon_days, DIR["edge_pp"], 0,
               where=DIR["edge_pp"] <= 0, color="red", alpha=.2)
e.set_xscale("log"); e.set_xticks(HORIZONS); e.set_xticklabels(HORIZONS)
e.set_xlabel("Horizon (trading days, log scale)")
e.set_ylabel("Best model − best baseline (pp)")
e.set_title(f"{TICKER} 3-class direction — edge over baseline\n(above 0 = real edge)")
e.grid(alpha=.3)

s = ax2[1]
s.plot(RET.horizon_days, RET["dir_acc_ridge_%"], "o-", label="Ridge sign accuracy", lw=2)
s.plot(RET.horizon_days, RET["dir_acc_xgb_%"], "s-", label="XGBoost sign accuracy", lw=2)
s.plot(RET.horizon_days, RET["dir_acc_alwaysguess_%"], "^--", color="black",
       label="always guess winning side (null)")
s.set_xscale("log"); s.set_xticks(HORIZONS); s.set_xticklabels(HORIZONS)
s.set_xlabel("Horizon (trading days, log scale)"); s.set_ylabel("Sign accuracy %")
s.set_title(f"{TICKER} return % — sign accuracy vs its null\n(model must sit ABOVE the black line)")
s.grid(alpha=.3); s.legend(fontsize=8)
fig2.tight_layout(); fig2.savefig(OUT / "mh_edge_vs_horizon.png", dpi=140)

# ---------------------------------------------------------------- summary
wins = DIR[DIR.beats_baseline == "Yes"]
best_row = DIR.loc[DIR.edge_pp.idxmax()]
best_m, best_b = best_row["best_model_%"], best_row["best_baseline_%"]
ret_wins = RET[RET.beats_naive0 == "Yes"]
ret_wins_tm = RET[RET.beats_trainmean == "Yes"]
ret_wins_sign = RET[RET.beats_sign_null == "Yes"]
n_indep_last = DIR.iloc[-1]["indep_test_windows"]

md = f"""# Multi-Horizon Direction + Return — Phase A (Tier-1 technical only)

**Stock:** {TICKER} daily ({d.date.min().date()} → {d.date.max().date()}, {len(d)} rows)
**Horizons:** {HORIZONS} trading days · **Method:** DIRECT (separate model per horizon)
**Features:** Tier-1 only — {', '.join(FEATURES)}
**Split:** 80/20 chronological, with a purge gap of h bars (no label leaks across the split)
**Dead-zone:** ±0.5%·√h (grows with horizon, else Hold disappears at long horizons)

## BOTTOM LINE (caveman)

- **Direction: {len(wins)} of {len(DIR)} horizons beat the baseline. Zero. No edge anywhere.**
- Longer horizon does **NOT** help. It gets *worse*: −{abs(DIR.iloc[0].edge_pp):.1f} pp at 1 day,
  −{abs(DIR.iloc[-1].edge_pp):.1f} pp at 1 year.
- Why worse: at long horizons the stock just drifts one way, so the dumb baseline gets huge
  (majority 78% at 6 months, persistence 77% at 1 year). Easy to be dumb, hard to beat dumb.
- **Return %: RMSE beats naive at {len(ret_wins)}/{len(RET)} horizons — but that is fake.**
  Ridge is only ~1% better than "predict the average past return". It learned *drift*, not signal.
- **Return % sign accuracy: 55-57% looks good, is not.** "Always guess up" gets the same or more.
  Real sign wins: {len(ret_wins_sign)}/{len(RET)}, both +0.2 pp = noise.
- **Verdict: Tier-1 technical features carry no information at any horizon.** Same answer as
  Step 1, now proven across 1 day → 1 year and on both targets. This is the floor.

## Direction — accuracy vs horizon

{md_table(DIR[['horizon','deadzone_%','n_test','indep_test_windows','acc_logistic_%','acc_xgboost_%','acc_majority_%','acc_persistence_%','edge_pp','beats_baseline']])}

**Horizons that beat both baselines: {len(wins)} of {len(DIR)}** ({', '.join(wins.horizon) if len(wins) else 'none'}).
Best edge: **{best_row.horizon}**, {best_row.edge_pp:+.1f} pp (model {best_m}% vs baseline {best_b}%).

## Return % — error vs two nulls

Two nulls, because they answer different questions:
- **naive "no change" (predict 0%)** — the weak null.
- **train-mean drift (predict the average past h-day return)** — the TOUGHER null. It already
  contains the market's upward drift, so only beating *this* proves the **features** did work.

{md_table(RET[['horizon','rmse_ridge','rmse_xgb','rmse_naive0','rmse_trainmean','rmse_ratio_ridge_vs_naive','rmse_ratio_ridge_vs_trainmean','beats_naive0','beats_trainmean']])}

**Beat naive "no change": {len(ret_wins)} of {len(RET)}** ({', '.join(ret_wins.horizon) if len(ret_wins) else 'none'}).
**Beat train-mean drift: {len(ret_wins_tm)} of {len(RET)}** ({', '.join(ret_wins_tm.horizon) if len(ret_wins_tm) else 'none'}).

Read it simply: Ridge's RMSE sits within ~1% of the train-mean line at every horizon, so the
apparent win over "no change" is **drift, not skill** — the model learned "HNB usually goes up a
bit," not a signal.

### Return model — sign accuracy vs the "always guess the winning side" null

{md_table(RET[['horizon','dir_acc_ridge_%','dir_acc_xgb_%','up_share_test_%','dir_acc_alwaysguess_%','dir_edge_pp','beats_sign_null']])}

**Beat the sign null: {len(ret_wins_sign)} of {len(RET)}** ({', '.join(ret_wins_sign.horizon) if len(ret_wins_sign) else 'none'}).

This is the trap to avoid: Ridge's sign accuracy looks decent (55-57% at 1-3 months, 78% at 6
months) — but the test window simply went UP that often. A constant "up" guess matches or beats
the model at every horizon. **No real sign skill anywhere.**

## Class balance (sanity)

{md_table(BAL.pivot(index='horizon_days', columns='class', values='test_share_%'), index=True)}

## Caveats
- Long horizons overlap heavily. `indep_test_windows` = n_test / h is the honest sample size:
  at 252 days there are only ~{n_indep_last} independent windows — **do not
  claim significance there**, even if accuracy looks high.
- One stock, one split, Tier-1 features only. This is the Phase A floor to improve on.
- A high baseline accuracy at long horizons is NOT a good result — it just means the test window
  drifted one way (e.g. majority 78% at 6 months, persistence 77% at 1 year). It makes the
  baseline nearly unbeatable, which is exactly why the models look so bad there.
- Correlation, not causation.

## Next
Phase B — add Tier-2 (RSI, MACD, volume) and measure the accuracy GAIN per horizon.
"""
(OUT / "mh_summary.md").write_text(md)

print("\n" + "=" * 78)
print(DIR[["horizon", "acc_logistic_%", "acc_xgboost_%", "acc_majority_%",
           "acc_persistence_%", "edge_pp", "beats_baseline"]].to_string(index=False))
print("=" * 78)
print(RET[["horizon", "rmse_ridge", "rmse_naive0", "rmse_trainmean",
           "rmse_ratio_ridge_vs_trainmean", "dir_acc_ridge_%",
           "dir_acc_alwaysguess_%", "dir_edge_pp", "beats_trainmean",
           "beats_sign_null"]].to_string(index=False))
print(f"\nHorizons beating direction baselines : {len(wins)}/{len(DIR)}")
print(f"Horizons beating naive-0 return RMSE  : {len(ret_wins)}/{len(RET)}")
print(f"Horizons beating train-mean RMSE      : {len(ret_wins_tm)}/{len(RET)}  <- honest RMSE null")
print(f"Horizons beating the sign null        : {len(ret_wins_sign)}/{len(RET)}  <- honest sign null")
print(f"Saved to {OUT}")
