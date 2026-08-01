#!/usr/bin/env python3
"""
THREE TESTS IN ONE — answering "can the model just learn 'if X goes up, the stock goes up'?"

PART 1 — CO-MOVEMENT TABLE (no machine learning at all)
    For every indicator and every horizon, simply COUNT:
        P(stock up over the next h days | the indicator just went UP)
        P(stock up over the next h days | the indicator just went DOWN)
    The gap between those two is the indicator's entire directional value. A gap near 0 means the
    indicator says nothing. Pooled over the banking + finance sector, with a two-proportion z-test.
    This is the most readable possible answer to the question and it uses no model.

PART 2 — POOLED PANEL MODEL
    Every earlier test trained ONE MODEL PER STOCK (~2,700 rows each). Here all 7 bank/finance
    stocks are stacked into a single training set (~19,000 rows) and one model is trained per
    horizon. If a pattern is real but weak, more data is the honest way to find it.
    CRITICAL: the train/test split is by DATE, not by row, so no stock's future can leak into
    another stock's past.

PART 3 — READABLE RULES
    Fit a depth-3 decision tree on the pooled training data, print each branch as plain English,
    and score every rule on the UNSEEN test set. Shows what the model actually believes, and
    whether those beliefs hold up out of sample.

Outputs -> results/direction/pooled_rules/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm, binomtest
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "direction" / "pooled_rules"
OUT.mkdir(parents=True, exist_ok=True)

SECTOR = ["HNB", "COMB", "SAMP", "LOFC", "LOLC", "LFIN", "CFIN"]     # banking + finance
HORIZONS = [1, 5, 10, 15, 22, 44, 66, 132, 252]
HORIZON_NAME = {1: "1 day", 5: "1 week", 10: "2 weeks", 15: "3 weeks", 22: "1 month",
                44: "2 months", 66: "3 months", 132: "6 months", 252: "1 year"}
DEADZONE_1D = 0.005
TRAIN_FRAC = 0.80
SEED = 42

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")) for t in SECTOR}
aspi = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
          .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))

rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True).ffill())
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_spread_3m"] = rt["spread"].diff(3)
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
rt["term_slope"] = rt["tb_12m"] - rt["tb_3m"]
rt["available_from"] = rt["date"] + pd.Timedelta(days=35)

inf = pd.read_csv(DATA / "inflation_monthly.csv", parse_dates=["date"]).sort_values("date")
inf["d_ccpi_yoy_3m"] = inf["ccpi_yoy_pct"].diff(3)
inf["available_from"] = inf["date"] + pd.Timedelta(days=21)

ms = pd.read_csv(DATA / "money_supply_monthly.csv", parse_dates=["date"]).sort_values("date")
ms["d_m2_yoy_3m"] = ms["m2_yoy_pct"].diff(3)
ms["available_from"] = ms["date"] + pd.Timedelta(days=65)

ip = pd.read_csv(DATA / "industrial_production_monthly.csv", parse_dates=["date"]).sort_values("date")
ip["available_from"] = ip["date"] + pd.Timedelta(days=50)

fx = pd.read_csv(DATA / "usd_lkr_daily.csv", parse_dates=["date"]).sort_values("date")


def rsi(s, n=14):
    x = s.diff()
    up = x.clip(lower=0).rolling(n).mean()
    dn = (-x.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def md_table(df):
    cols = [str(x) for x in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(out)


def build(t):
    """One stock -> features + all indicators, publication-lagged."""
    d = px[t].reset_index()
    dates = d["date"]
    c = d["close"].astype(float)
    v = d["volume"].astype(float)
    r1 = c.pct_change()
    f = pd.DataFrame({"date": dates, "ticker": t, "close": c})

    f["ret_1"], f["ret_5"], f["ret_10"] = r1, c.pct_change(5), c.pct_change(10)
    f["ma5_ratio"] = c / c.rolling(5).mean() - 1
    f["ma10_ratio"] = c / c.rolling(10).mean() - 1
    f["ma20_ratio"] = c / c.rolling(20).mean() - 1
    f["momentum_10"] = c / c.shift(10) - 1
    f["vol_10"], f["vol_20"] = r1.rolling(10).std(), r1.rolling(20).std()
    f["rsi_14"] = rsi(c)
    m_ = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sg = m_.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (m_ - sg) / c
    f["volchg_5"] = v / v.rolling(5).mean() - 1

    a = aspi.reindex(dates).ffill().reset_index(drop=True)
    ar = a.pct_change()
    f["aspi_ret_1"], f["aspi_ret_5"] = ar, a.pct_change(5)
    f["rs_vs_aspi_5"] = c.pct_change(5) - a.pct_change(5)
    f["beta_aspi_60"] = r1.rolling(60).cov(ar) / ar.rolling(60).var()
    peers = [p for p in SECTOR if p != t]
    pp = pd.DataFrame({p: px[p]["close"].astype(float).reindex(dates).ffill().reset_index(drop=True)
                       for p in peers})
    f["peer_ret_1"] = pp.pct_change().mean(axis=1)
    f["peer_ret_5"] = pp.pct_change(5).mean(axis=1)

    def asof(src, cols):
        j = pd.merge_asof(pd.DataFrame({"date": dates}),
                          src[["available_from"] + cols].sort_values("available_from"),
                          left_on="date", right_on="available_from", direction="backward")
        return j[cols]

    for col, s in asof(rt, ["d_policy_1m", "d_spread_1m", "d_spread_3m",
                            "d_tb3m_3m", "term_slope"]).items():
        f[col] = s.values
    for col, s in asof(inf, ["ccpi_yoy_pct", "d_ccpi_yoy_3m"]).items():
        f[col] = s.values
    for col, s in asof(ms, ["m2_yoy_pct", "d_m2_yoy_3m"]).items():
        f[col] = s.values
    for col, s in asof(ip, ["iip_yoy_pct", "d_iip_yoy_3m"]).items():
        f[col] = s.values
    j = pd.merge_asof(pd.DataFrame({"date": dates}),
                      fx[["date", "usd_lkr_ret_5", "usd_lkr_ret_20"]], on="date",
                      direction="backward")
    f["usd_lkr_ret_5"], f["usd_lkr_ret_20"] = j["usd_lkr_ret_5"].values, j["usd_lkr_ret_20"].values

    for h in HORIZONS:
        f[f"fwd_{h}"] = c.shift(-h) / c - 1
        f[f"past_{h}"] = c / c.shift(h) - 1      # matched-horizon persistence baseline
    return f


PANEL = pd.concat([build(t) for t in SECTOR], ignore_index=True).sort_values(["date", "ticker"])
print(f"Panel built: {len(PANEL):,} stock-days, {PANEL.ticker.nunique()} stocks, "
      f"{PANEL.date.min():%Y-%m} -> {PANEL.date.max():%Y-%m}\n")

# ===================================================================== PART 1
print("=" * 92)
print("PART 1 — CO-MOVEMENT: when an indicator goes UP, does the stock go up after?")
print("=" * 92)

INDICATORS = {
    "ASPI (market) 1d": "aspi_ret_1",
    "ASPI (market) 5d": "aspi_ret_5",
    "Peer banks 5d": "peer_ret_5",
    "Own momentum 10d": "momentum_10",
    "RSI(14) above 50": "rsi_14",
    "MACD histogram": "macd_hist",
    "Volume vs 5d avg": "volchg_5",
    "Policy rate change": "d_policy_1m",
    "Lending-deposit spread Δ": "d_spread_1m",
    "T-bill 3m change (3m)": "d_tb3m_3m",
    "USD/LKR 5d": "usd_lkr_ret_5",
    "USD/LKR 20d": "usd_lkr_ret_20",
    "Inflation YoY Δ (3m)": "d_ccpi_yoy_3m",
    "Money supply M2 YoY Δ": "d_m2_yoy_3m",
    "Industrial production YoY": "iip_yoy_pct",
}

rows = []
for label, col in INDICATORS.items():
    thresh = 50.0 if col == "rsi_14" else 0.0
    for h in HORIZONS:
        sub = PANEL[[col, f"fwd_{h}"]].dropna()
        if len(sub) < 500:
            continue
        up_ind = sub[col] > thresh
        stock_up = sub[f"fwd_{h}"] > 0
        n1, n2 = int(up_ind.sum()), int((~up_ind).sum())
        if min(n1, n2) < 100:
            continue
        p1 = float(stock_up[up_ind].mean())            # P(stock up | indicator up)
        p2 = float(stock_up[~up_ind].mean())           # P(stock up | indicator down)
        lift = (p1 - p2) * 100
        # two-proportion z-test (overlapping windows inflate significance -> see caveats)
        pp_ = (stock_up[up_ind].sum() + stock_up[~up_ind].sum()) / (n1 + n2)
        se = np.sqrt(pp_ * (1 - pp_) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se if se > 0 else 0.0
        rows.append({"indicator": label, "horizon": HORIZON_NAME[h], "horizon_days": h,
                     "n_up": n1, "n_down": n2,
                     "indep_windows": round((n1 + n2) / (h * len(SECTOR)), 1),
                     "P_up_given_ind_up_%": round(p1 * 100, 1),
                     "P_up_given_ind_down_%": round(p2 * 100, 1),
                     "lift_pp": round(lift, 1), "abs_lift": abs(round(lift, 1)),
                     "z": round(float(z), 2),
                     "p_value": round(float(2 * (1 - norm.cdf(abs(z)))), 4)})

CO = pd.DataFrame(rows)
CO.to_csv(OUT / "comovement_table.csv", index=False)

piv = CO.pivot(index="indicator", columns="horizon_days", values="lift_pp")
piv = piv.reindex(list(INDICATORS))[[h for h in HORIZONS if h in piv.columns]]
piv.columns = [HORIZON_NAME[h] for h in piv.columns]
print(piv.to_string())
print("\n(lift = P(stock up | indicator up) − P(stock up | indicator down), in percentage points)")

strong = CO[(CO.abs_lift >= 5) & (CO.p_value < 0.01)].sort_values("abs_lift", ascending=False)
print(f"\nCells with |lift| >= 5 pp AND p < 0.01: {len(strong)} of {len(CO)}")
if len(strong):
    print(strong.head(12)[["indicator", "horizon", "P_up_given_ind_up_%",
                           "P_up_given_ind_down_%", "lift_pp", "p_value"]].to_string(index=False))

# ===================================================================== PART 2
print("\n" + "=" * 92)
print("PART 2 — POOLED PANEL MODEL (all 7 bank/finance stocks in ONE training set)")
print("=" * 92)

FEATS = ["ret_1", "ret_5", "ret_10", "ma5_ratio", "ma10_ratio", "ma20_ratio", "momentum_10",
         "vol_10", "vol_20", "rsi_14", "macd_hist", "volchg_5",
         "aspi_ret_1", "aspi_ret_5", "rs_vs_aspi_5", "beta_aspi_60", "peer_ret_1", "peer_ret_5",
         "d_policy_1m", "d_spread_1m", "d_spread_3m", "d_tb3m_3m", "term_slope",
         "ccpi_yoy_pct", "d_ccpi_yoy_3m", "usd_lkr_ret_5", "usd_lkr_ret_20"]

pooled_rows = []
fair_rows = []
tree_store = {}
for h in HORIZONS:
    dz = DEADZONE_1D * np.sqrt(h)
    df = PANEL[["date", "ticker"] + FEATS + [f"fwd_{h}", f"past_{h}"]].dropna().reset_index(drop=True)
    if len(df) < 2000:
        continue
    y = pd.Series(np.where(df[f"fwd_{h}"] > dz, 2, np.where(df[f"fwd_{h}"] < -dz, 0, 1)))

    # SPLIT BY DATE (not by row) so no stock's future leaks into another stock's past,
    # then purge h trading days around the boundary.
    cut = df["date"].quantile(TRAIN_FRAC)
    purge = cut - pd.Timedelta(days=int(h * 1.5))
    tr = df["date"] <= purge
    te = df["date"] > cut

    Xtr, Xte = df.loc[tr, FEATS], df.loc[te, FEATS]
    ytr, yte = y[tr.values], y[te.values]
    if len(Xte) < 200 or ytr.nunique() < 2:
        continue

    maj = int(ytr.value_counts().idxmax())
    acc_maj = accuracy_score(yte, np.full(len(yte), maj))
    # Persistence must use the SAME horizon as the label ("the next h days repeat the last h
    # days"). Using a fixed 5-day lookback here would weaken the baseline and fake an edge.
    past = df.loc[te, f"past_{h}"]
    acc_pers = accuracy_score(yte, np.where(past > dz, 2, np.where(past < -dz, 0, 1)))
    base = max(acc_maj, acc_pers)

    present = sorted(ytr.unique())
    wmap = {k: len(ytr) / (len(present) * (ytr == k).sum()) for k in present}
    w = ytr.map(wmap).values

    lg = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
    lg.fit(Xtr, ytr)
    a_lg = accuracy_score(yte, lg.predict(Xte))

    xc = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, min_child_weight=3, random_state=SEED,
                       n_jobs=4, eval_metric="mlogloss")
    xc.fit(Xtr, ytr, sample_weight=w)
    a_xc = accuracy_score(yte, xc.predict(Xte))
    best = max(a_lg, a_xc)

    # ---- FAIRNESS CHECK -------------------------------------------------------------
    # The pooled baseline is computed on all stocks at once. Pooling stocks with different
    # class balances can WEAKEN the majority baseline and fake an edge. So also score the
    # pooled model PER STOCK against that stock's OWN baseline on the same test window,
    # then sign-test across the 7 stocks (the discipline the sector sweep taught us).
    te_df = df.loc[te, ["date", "ticker", f"past_{h}"]].copy()
    te_df["y"] = yte.values
    te_df["pred"] = (xc if a_xc >= a_lg else lg).predict(Xte)
    tr_df = df.loc[tr, ["ticker"]].copy()
    tr_df["y"] = ytr.values
    per_stock = []
    for tk, g in te_df.groupby("ticker"):
        own_tr = tr_df[tr_df.ticker == tk]
        if len(own_tr) == 0 or len(g) < 30:
            continue
        own_maj = int(own_tr["y"].value_counts().idxmax())
        b_maj = accuracy_score(g["y"], np.full(len(g), own_maj))
        b_per = accuracy_score(g["y"], np.where(g[f"past_{h}"] > dz, 2,
                                                np.where(g[f"past_{h}"] < -dz, 0, 1)))
        own_base = max(b_maj, b_per)
        acc_tk = accuracy_score(g["y"], g["pred"])
        per_stock.append({"horizon_days": h, "horizon": HORIZON_NAME[h], "ticker": tk,
                          "n": len(g), "pooled_model_acc_%": round(acc_tk * 100, 1),
                          "own_baseline_%": round(own_base * 100, 1),
                          "edge_pp": round((acc_tk - own_base) * 100, 1)})
    fair_rows.extend(per_stock)

    tree_store[h] = (Xtr, ytr, Xte, yte, w, base)
    pooled_rows.append({
        "horizon_days": h, "horizon": HORIZON_NAME[h],
        "n_train": len(Xtr), "n_test": len(Xte),
        "train_to": f"{purge:%Y-%m-%d}", "test_from": f"{df.loc[te, 'date'].min():%Y-%m-%d}",
        "acc_logistic_%": round(a_lg * 100, 1), "acc_xgboost_%": round(a_xc * 100, 1),
        "acc_majority_%": round(acc_maj * 100, 1), "acc_persistence_%": round(acc_pers * 100, 1),
        "best_%": round(best * 100, 1), "baseline_%": round(base * 100, 1),
        "edge_pp": round((best - base) * 100, 1), "beats_baseline": best > base,
    })
    print(f"h={h:>3}d  train {len(Xtr):>6,} rows  test {len(Xte):>5,}  "
          f"logit {a_lg*100:5.1f}%  xgb {a_xc*100:5.1f}%  base {base*100:5.1f}%  "
          f"edge {(best-base)*100:+5.1f} pp")

POOL = pd.DataFrame(pooled_rows)
POOL.to_csv(OUT / "pooled_model_results.csv", index=False)

FAIR = pd.DataFrame(fair_rows)
FAIR.to_csv(OUT / "pooled_vs_own_baseline.csv", index=False)
fair_sig = []
for h in sorted(FAIR.horizon_days.unique()):
    e = FAIR[FAIR.horizon_days == h].edge_pp.values
    k = int((e > 0).sum())
    fair_sig.append({"horizon": HORIZON_NAME[h], "horizon_days": h,
                     "stocks_positive": f"{k}/{len(e)}",
                     "median_edge_pp": round(float(np.median(e)), 1),
                     "sign_test_p": round(binomtest(k, len(e), 0.5, alternative="greater").pvalue, 3)})
FAIRSIG = pd.DataFrame(fair_sig)
FAIRSIG.to_csv(OUT / "pooled_fairness_significance.csv", index=False)
print("\n  FAIRNESS CHECK — pooled model scored against each stock's OWN baseline:")
print(FAIRSIG.to_string(index=False))

# ===================================================================== PART 3
print("\n" + "=" * 92)
print("PART 3 — READABLE RULES (depth-3 tree on the pooled data, scored on UNSEEN test data)")
print("=" * 92)

CLASSES = {0: "Sell", 1: "Hold", 2: "Buy"}
rule_rows = []
for h in [1, 5, 22]:
    if h not in tree_store:
        continue
    Xtr, ytr, Xte, yte, w, base = tree_store[h]
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=SEED,
                                class_weight="balanced")
    dt.fit(Xtr, ytr, sample_weight=w)
    tree = dt.tree_
    feat_names = list(Xtr.columns)
    leaf_of_test = dt.apply(Xte)

    def walk(node, conds):
        if tree.children_left[node] == -1:
            pred = int(np.argmax(tree.value[node][0]))
            mask = leaf_of_test == node
            n = int(mask.sum())
            if n >= 50:
                hit = float((yte.values[mask] == pred).mean())
                rule_rows.append({
                    "horizon": HORIZON_NAME[h], "horizon_days": h,
                    "rule": " AND ".join(conds) if conds else "(always)",
                    "says": CLASSES[pred], "fired_n": n,
                    "correct_%": round(hit * 100, 1),
                    "vs_baseline_pp": round((hit - base) * 100, 1)})
            return
        f = feat_names[tree.feature[node]]
        thr = tree.threshold[node]
        walk(tree.children_left[node], conds + [f"{f} <= {thr:.4f}"])
        walk(tree.children_right[node], conds + [f"{f} > {thr:.4f}"])

    walk(0, [])

RULES = pd.DataFrame(rule_rows).sort_values(["horizon_days", "correct_%"], ascending=[True, False])
RULES.to_csv(OUT / "readable_rules.csv", index=False)
for h in RULES.horizon_days.unique():
    sub = RULES[RULES.horizon_days == h]
    print(f"\n--- {HORIZON_NAME[h]} (baseline {tree_store[h][5]*100:.1f}%) ---")
    print(sub[["rule", "says", "fired_n", "correct_%", "vs_baseline_pp"]].to_string(index=False))

# ===================================================================== plots
fig, ax = plt.subplots(1, 2, figsize=(14.5, 6))
a = ax[0]
im = a.imshow(piv.values, cmap="RdBu_r", vmin=-12, vmax=12, aspect="auto")
a.set_xticks(range(piv.shape[1])); a.set_xticklabels(piv.columns, rotation=45, ha="right")
a.set_yticks(range(piv.shape[0])); a.set_yticklabels(piv.index, fontsize=8)
a.set_title("PART 1 — co-movement lift (pp)\nP(up | indicator up) − P(up | indicator down)")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if not np.isnan(v):
            a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im, ax=a, label="pp")

b = ax[1]
b.plot(POOL.horizon_days, POOL["best_%"], "o-", label="pooled model", lw=2)
b.plot(POOL.horizon_days, POOL["baseline_%"], "^--", color="black", label="baseline")
b.set_xscale("log"); b.set_xticks(HORIZONS); b.set_xticklabels(HORIZONS)
b.set_xlabel("Horizon (trading days, log scale)"); b.set_ylabel("Accuracy %")
b.set_title(f"PART 2 — pooled panel model\n{POOL.n_train.max():,} training rows "
            f"(vs ~2,700 per-stock)")
b.grid(alpha=.3); b.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "pooled_and_comovement.png", dpi=140)

# ===================================================================== summary
pool_wins = int(POOL.beats_baseline.sum())
best_ind = CO.loc[CO.abs_lift.idxmax()]
median_abs_lift = CO.abs_lift.median()
rule_best = RULES.loc[RULES.vs_baseline_pp.idxmax()] if len(RULES) else None

md = f"""# Pooled panel, co-movement and readable rules

**Sector:** {SECTOR} (banking + finance) · **Panel:** {len(PANEL):,} stock-days,
{PANEL.date.min():%Y-%m} → {PANEL.date.max():%Y-%m} · **Horizons:** {HORIZONS}

---

## PART 1 — Co-movement: "if the indicator goes up, does the stock go up?"

No model. Pure counting, pooled over all 7 stocks.
`lift` = P(stock up | indicator up) − P(stock up | indicator down), in percentage points.
**Lift near 0 = the indicator tells you nothing about direction.**

{md_table(piv.round(1).reset_index())}

- **Median |lift| across all {len(CO)} cells: {median_abs_lift:.1f} pp.**
- Cells with |lift| ≥ 5 pp AND p < 0.01: **{len(strong)} of {len(CO)}**.
- Largest single lift: **{best_ind.indicator}** at {best_ind.horizon},
  {best_ind['P_up_given_ind_up_%']}% vs {best_ind['P_up_given_ind_down_%']}%
  (**{best_ind.lift_pp:+.1f} pp**, p = {best_ind.p_value}).

⚠ These p-values are **too generous**: overlapping forward windows mean the rows are not
independent (a 252-day window shares 251 days with the next one). Treat large-lift long-horizon
cells with suspicion — the effective sample there is tiny.

---

## PART 2 — Pooled panel model ({POOL.n_train.max():,} training rows vs ~2,700 per stock)

Split by **date**, not by row, so no stock's future leaks into another's past.

{md_table(POOL[['horizon', 'n_train', 'n_test', 'acc_logistic_%', 'acc_xgboost_%', 'acc_majority_%', 'acc_persistence_%', 'edge_pp', 'beats_baseline']])}

**Horizons where the pooled model beats the pooled baseline: {pool_wins} of {len(POOL)}.**

### ⚠ Fairness check — the pooled baseline is not a fair benchmark

Pooling stocks with different class balances can WEAKEN the majority baseline and manufacture an
edge out of nothing. So the pooled model was also scored **per stock, against that stock's own
baseline**, on the same test window, then sign-tested across the 7 stocks:

{md_table(FAIRSIG)}

{md_table(FAIR.pivot(index="ticker", columns="horizon", values="edge_pp").reindex(columns=[HORIZON_NAME[h] for h in HORIZONS if HORIZON_NAME[h] in set(FAIR.horizon)]).reset_index())}

**What this changes:**
- The pooled table's **+2.7 pp at 1 week was an artifact** — against each stock's own baseline it
  is **{FAIRSIG[FAIRSIG.horizon == '1 week'].median_edge_pp.iloc[0]:+.1f} pp** ({FAIRSIG[FAIRSIG.horizon == '1 week'].stocks_positive.iloc[0]} stocks).
- **1 day survives the fair test**: {FAIRSIG[FAIRSIG.horizon == '1 day'].stocks_positive.iloc[0]} stocks positive,
  median **{FAIRSIG[FAIRSIG.horizon == '1 day'].median_edge_pp.iloc[0]:+.1f} pp**,
  p = **{FAIRSIG[FAIRSIG.horizon == '1 day'].sign_test_p.iloc[0]}**.
  That is the **strongest result in the entire project** — but it is still not significant at 0.05,
  it is 1 horizon out of 9 tested, and it comes from a single test window. **Not a finding yet.**

7× more training data therefore did {'produce a significant edge' if (FAIRSIG.sign_test_p < 0.05).any() else 'NOT produce a significant edge. The per-stock models were not starved of data.'}

---

## PART 3 — What rules did the model actually learn?

Depth-3 decision tree on the pooled training data, every branch scored on **unseen** test data.

{md_table(RULES[['horizon', 'rule', 'says', 'fired_n', 'correct_%', 'vs_baseline_pp']])}

`vs_baseline_pp` is the honest column: how much better than the naive guess that rule is.
{f"Best rule: **{rule_best['rule']}** → says {rule_best['says']}, correct {rule_best['correct_%']}% ({rule_best['vs_baseline_pp']:+.1f} pp vs baseline)." if rule_best is not None else ""}

---

## Caveats
- Overlapping forward windows inflate significance at long horizons; the 1-day and 1-week rows are
  the only ones with a decent effective sample.
- Pooling assumes the 7 stocks share one pattern. If each stock behaved differently, pooling would
  blur them — but the per-stock runs already failed too, so that is not the explanation here.
- Correlation, not causation.
"""
(OUT / "pooled_rules_summary.md").write_text(md)

print("\n" + "=" * 92)
print(f"PART 1: median |lift| {median_abs_lift:.1f} pp | strong cells {len(strong)}/{len(CO)}")
print(f"PART 2: pooled model beats baseline at {pool_wins}/{len(POOL)} horizons "
      f"(max train rows {POOL.n_train.max():,})")
print(f"PART 3: {len(RULES)} rules extracted and scored out-of-sample")
print(f"Saved to {OUT}")
