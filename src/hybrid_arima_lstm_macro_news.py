#!/usr/bin/env python3
"""
HYBRID ARIMA + LSTM, with and without macro/news — and metrics that are COMPARABLE TO PUBLISHED WORK.

This script closes two gaps at once.

GAP 1 — THE HYBRID ARCHITECTURE (Zhang, 2003)
    The project is named after it, but the hybrid had only ever been run univariate on price in
    Stage 1. The architecture decomposes the series into a linear and a nonlinear part:

        Y_t = L_t + N_t
        ARIMA  -> L_hat   (linear structure)
        e_t = Y_t - L_hat (what ARIMA could not explain)
        LSTM(e) -> N_hat  (nonlinear structure in the residual)
        FINAL   = L_hat + N_hat

    Two variants are tested:
      * Hybrid          - residual LSTM sees only past residuals            (classic Zhang 2003)
      * Hybrid+MN       - residual LSTM ALSO sees macro + news              (this project's extension)
    The second is the actual contribution: if macro/news carry information, the natural place for
    it is the part ARIMA provably cannot explain.

GAP 2 — COMPARABILITY WITH PUBLISHED PAPERS
    Papers in this area report a fixed train/test split and RMSE / MAE / MAPE / R2, and claim
    superiority with the DIEBOLD-MARIANO test. This project had moved to walk-forward evaluation
    with baseline-relative "edge", which is more rigorous but NOT directly comparable to a
    published table. So both are produced:

      * FIXED 80/20 SPLIT  -> the table a reader can put beside a paper's Table 4
      * WALK-FORWARD       -> this project's stricter standard
      * DIEBOLD-MARIANO    -> the standard test of whether one forecast beats another,
                              with Newey-West (HAC) correction, vs naive AND vs ARIMA
      * THEIL'S U2         -> classic naive-relative statistic (U2 < 1 means better than naive)

    Reporting both is the honest move: it shows the paper-style numbers AND shows what happens to
    them under a stricter protocol.

Targets: S&P SL 20 (the source paper's series) and HNB.
Outputs -> results/hybrid_comparable/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "hybrid_comparable"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["SPSL20", "HNB"]
ARIMA_ORDER = (2, 1, 0)        # the source paper's choice; (1,1,1) checked as a robustness order
SEQ = 20                       # residual window for the LSTM
HIDDEN, EPOCHS, BATCH = 32, 40, 64
SEEDS = [0, 1, 2]
TRAIN_FRAC = 0.80
TEST_MONTHS = 12
FIRST_TEST = pd.Timestamp("2019-01-01")
DEV = torch.device("cpu")

# ---------------------------------------------------------------- data
sp = (pd.read_csv(ROOT / "data" / "processed" / "spsl20_trading_days_clean.csv",
                  parse_dates=["date"]).sort_values("date").drop_duplicates("date")
      .rename(columns={"spsl20_points": "close"})[["date", "close"]])
hnb = (pd.read_csv(DATA / "HNB_daily_clean.csv", parse_dates=["date"])
       .sort_values("date").drop_duplicates("date")[["date", "close"]])
SERIES = {"SPSL20": sp, "HNB": hnb}

rt = (pd.read_csv(DATA / "interest_rates_monthly.csv", parse_dates=["date"])
        .sort_values("date").reset_index(drop=True).ffill())
rt["d_policy_1m"] = rt["policy_rate"].diff()
rt["d_spread_1m"] = rt["spread"].diff()
rt["d_tb3m_3m"] = rt["tb_3m"].diff(3)
rt["available_from"] = rt["date"] + pd.Timedelta(days=35)
inf = pd.read_csv(DATA / "inflation_monthly.csv", parse_dates=["date"]).sort_values("date")
inf["available_from"] = inf["date"] + pd.Timedelta(days=21)
fx = pd.read_csv(DATA / "usd_lkr_daily.csv", parse_dates=["date"]).sort_values("date")
sent = pd.read_csv(DATA / "news_sentiment_daily.csv", parse_dates=["date"]).sort_values("date")
EXOG = ["d_policy_1m", "d_spread_1m", "d_tb3m_3m", "ccpi_yoy_pct",
        "usd_lkr_ret_5", "s_vader", "s_lm"]


def with_exog(df):
    d = df.copy()
    j = pd.merge_asof(d[["date"]], rt[["available_from", "d_policy_1m", "d_spread_1m",
                                       "d_tb3m_3m"]].sort_values("available_from"),
                      left_on="date", right_on="available_from", direction="backward")
    for c in ["d_policy_1m", "d_spread_1m", "d_tb3m_3m"]:
        d[c] = j[c].values
    j2 = pd.merge_asof(d[["date"]], inf[["available_from", "ccpi_yoy_pct"]]
                       .sort_values("available_from"),
                       left_on="date", right_on="available_from", direction="backward")
    d["ccpi_yoy_pct"] = j2["ccpi_yoy_pct"].values
    j3 = pd.merge_asof(d[["date"]], fx[["date", "usd_lkr_ret_5"]], on="date", direction="backward")
    d["usd_lkr_ret_5"] = j3["usd_lkr_ret_5"].values
    j4 = pd.merge_asof(d[["date"]], sent[["date", "s_vader", "s_lm"]], on="date",
                       direction="backward", tolerance=pd.Timedelta("4D"))
    d["s_vader"], d["s_lm"] = j4["s_vader"].fillna(0).values, j4["s_lm"].fillna(0).values
    return d.dropna().reset_index(drop=True)


# ---------------------------------------------------------------- metrics
def theil_u2(actual, pred, naive):
    return float(np.sqrt(mean_squared_error(actual, pred)) /
                 np.sqrt(mean_squared_error(actual, naive)))


def diebold_mariano(actual, p1, p2, h=1):
    """DM test on squared-error loss. H0: equal accuracy. Negative stat => p1 is BETTER than p2.
    Newey-West (HAC) variance with h-1 lags, as the standard test prescribes."""
    d = (actual - p1) ** 2 - (actual - p2) ** 2
    n = len(d)
    dbar = float(d.mean())
    g0 = float(((d - dbar) ** 2).mean())
    var = g0
    for k in range(1, h):
        gk = float(((d[k:] - dbar) * (d[:-k] - dbar)).mean())
        var += 2 * (1 - k / h) * gk
    se = np.sqrt(var / n)
    if se == 0 or not np.isfinite(se):
        return np.nan, np.nan
    stat = dbar / se
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


def score_all(name, target, split, actual, pred, naive):
    stat_n, p_n = diebold_mariano(actual, pred, naive)
    return {"target": target, "split": split, "model": name,
            "RMSE": round(float(np.sqrt(mean_squared_error(actual, pred))), 4),
            "MAE": round(float(mean_absolute_error(actual, pred)), 4),
            "MAPE_%": round(float(np.mean(np.abs((actual - pred) / actual)) * 100), 4),
            "R2": round(float(r2_score(actual, pred)), 4),
            "Theil_U2": round(theil_u2(actual, pred, naive), 4),
            "dir_acc_%": round(float((np.sign(np.diff(np.r_[actual[0], pred])) ==
                                      np.sign(np.diff(np.r_[actual[0], actual]))).mean() * 100), 1),
            "DM_vs_naive": round(stat_n, 3) if stat_n == stat_n else np.nan,
            "DM_p_vs_naive": round(p_n, 4) if p_n == p_n else np.nan}


# ---------------------------------------------------------------- LSTM on residuals
class ResLSTM(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.rnn = nn.LSTM(nf, HIDDEN, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(self.drop(o[:, -1])).squeeze(-1)


def make_seq(resid, exog=None):
    """Windows of past residuals (optionally stacked with exogenous inputs) -> next residual."""
    X, y = [], []
    for i in range(SEQ, len(resid)):
        w = resid[i - SEQ:i].reshape(-1, 1)
        if exog is not None:
            w = np.hstack([w, exog[i - SEQ:i]])
        X.append(w)
        y.append(resid[i])
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)) if X else (None, None)


def fit_residual_lstm(res_tr, res_te, ex_tr=None, ex_te=None):
    """Train on training residuals, predict the test residual path. Averaged over seeds."""
    Xtr, ytr = make_seq(res_tr, ex_tr)
    if Xtr is None or len(Xtr) < 100:
        return np.zeros(len(res_te))
    mu, sd = Xtr.reshape(-1, Xtr.shape[2]).mean(0), Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-9
    ys, yss = ytr.mean(), ytr.std() + 1e-9
    Xtr_n = (Xtr - mu) / sd
    ytr_n = (ytr - ys) / yss

    # test windows: each prediction uses only residuals available BEFORE that point
    full_res = np.r_[res_tr, res_te]
    full_ex = np.vstack([ex_tr, ex_te]) if ex_tr is not None else None
    Xte = []
    for i in range(len(res_tr), len(full_res)):
        w = full_res[i - SEQ:i].reshape(-1, 1)
        if full_ex is not None:
            w = np.hstack([w, full_ex[i - SEQ:i]])
        Xte.append(w)
    Xte = (np.asarray(Xte, dtype=np.float32) - mu) / sd

    preds = []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        net = ResLSTM(Xtr_n.shape[2]).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lf = nn.MSELoss()
        Xt = torch.tensor(Xtr_n, device=DEV)
        yt = torch.tensor(ytr_n, device=DEV)
        net.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(len(Xt), device=DEV)
            for i in range(0, len(Xt), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                lf(net(Xt[idx]), yt[idx]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            preds.append(net(torch.tensor(Xte, device=DEV)).cpu().numpy() * yss + ys)
    return np.mean(preds, axis=0)


# ---------------------------------------------------------------- one evaluation window
def evaluate(target, d, tr_idx, te_idx, split_label, store=None):
    logp = np.log(d["close"].values)
    tr_log, te_log = logp[tr_idx], logp[te_idx]
    actual = d["close"].values[te_idx]
    naive = d["close"].values[te_idx[0] - 1:te_idx[-1]]        # yesterday's close
    rows = []

    rows.append(score_all("Naive (random walk)", target, split_label, actual, naive, naive))

    # ---- ARIMA, genuine rolling one-step-ahead (state updated, parameters frozen) ----
    fit = ARIMA(tr_log, order=ARIMA_ORDER).fit()
    ext = fit.append(te_log, refit=False)
    arima_log = ext.get_prediction(start=len(tr_log),
                                   end=len(tr_log) + len(te_log) - 1).predicted_mean
    arima_px = np.exp(arima_log)
    rows.append(score_all(f"ARIMA{ARIMA_ORDER}", target, split_label, actual, arima_px, naive))

    res_tr = np.asarray(fit.resid, dtype=float)[1:]            # drop the differencing warm-up
    res_te = te_log - arima_log

    # ---- LSTM alone, on the log-price series (no ARIMA) ----
    lstm_only = fit_residual_lstm(np.diff(tr_log), np.diff(np.r_[tr_log[-1], te_log]))
    lstm_px = np.exp(np.r_[tr_log[-1], te_log[:-1]] + lstm_only)
    rows.append(score_all("LSTM (univariate)", target, split_label, actual, lstm_px, naive))

    # ---- HYBRID: ARIMA + LSTM on ARIMA residuals ----
    n_hat = fit_residual_lstm(res_tr, res_te)
    rows.append(score_all("Hybrid ARIMA+LSTM", target, split_label,
                          actual, np.exp(arima_log + n_hat), naive))

    # ---- HYBRID + macro/news in the residual model ----
    ex = d[EXOG].values.astype(np.float32)
    ex_tr_full, ex_te = ex[tr_idx], ex[te_idx]
    ex_tr = ex_tr_full[1:]                                     # align with res_tr
    m, s = ex_tr.mean(0), ex_tr.std(0) + 1e-9
    n_hat_mn = fit_residual_lstm(res_tr, res_te, (ex_tr - m) / s, (ex_te - m) / s)
    hyb_mn = np.exp(arima_log + n_hat_mn)
    rows.append(score_all("Hybrid+MacroNews", target, split_label, actual, hyb_mn, naive))

    # DM of every model against ARIMA as well (papers usually claim superiority over ARIMA)
    preds = {"Naive (random walk)": naive, f"ARIMA{ARIMA_ORDER}": arima_px,
             "LSTM (univariate)": lstm_px, "Hybrid ARIMA+LSTM": np.exp(arima_log + n_hat),
             "Hybrid+MacroNews": hyb_mn}
    for r in rows:
        st, pv = diebold_mariano(actual, preds[r["model"]], arima_px)
        r["DM_vs_ARIMA"] = round(st, 3) if st == st else np.nan
        r["DM_p_vs_ARIMA"] = round(pv, 4) if pv == pv else np.nan
    if store is not None:
        for k, v in preds.items():
            store.append(pd.DataFrame({"target": target, "split": split_label, "model": k,
                                       "date": d["date"].values[te_idx],
                                       "actual": actual, "pred": v}))
    return rows


records, pred_store = [], []
for target in TARGETS:
    d = with_exog(SERIES[target])
    n = len(d)

    # ---------- FIXED 80/20 SPLIT (the paper-comparable table) ----------
    cut = int(n * TRAIN_FRAC)
    records += evaluate(target, d, np.arange(cut), np.arange(cut, n), "fixed 80/20", pred_store)
    print(f"{target}: fixed split done ({cut} train / {n - cut} test)")

    # ---------- WALK-FORWARD (this project's stricter standard) ----------
    starts = pd.date_range(FIRST_TEST, d.date.max(), freq=f"{TEST_MONTHS}MS")
    for fi, t0 in enumerate(starts):
        t1 = t0 + pd.DateOffset(months=TEST_MONTHS)
        if t1 > d.date.max():
            break
        tr = np.flatnonzero((d.date <= t0 - pd.Timedelta(days=5)).values)
        te = np.flatnonzero(((d.date >= t0) & (d.date < t1)).values)
        if len(tr) < 600 or len(te) < 60:
            continue
        records += evaluate(target, d, tr, te, f"WF {t0:%Y}")
    print(f"{target}: walk-forward done")

R = pd.DataFrame(records)
R.to_csv(OUT / "hybrid_all_results.csv", index=False)
PRED = pd.concat(pred_store, ignore_index=True)
PRED.to_csv(OUT / "hybrid_predictions_fixed_split.csv", index=False)

FIXED = R[R.split == "fixed 80/20"].copy()
WF = (R[R.split.str.startswith("WF")]
      .groupby(["target", "model"])
      .agg(RMSE=("RMSE", "median"), MAE=("MAE", "median"), MAPE_pct=("MAPE_%", "median"),
           R2=("R2", "median"), Theil_U2=("Theil_U2", "median"),
           dir_acc_pct=("dir_acc_%", "median"),
           folds_better_than_naive=("Theil_U2", lambda s: f"{int((s < 1).sum())}/{len(s)}"))
      .reset_index().round(4))
FIXED.to_csv(OUT / "hybrid_fixed_split_table.csv", index=False)
WF.to_csv(OUT / "hybrid_walkforward_table.csv", index=False)


def md_table(df):
    cols = [str(x) for x in df.columns]
    o = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        o.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(o)


# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
for i, target in enumerate(TARGETS):
    p = PRED[(PRED.target == target) & (PRED.split == "fixed 80/20")]
    a = ax[i][0]
    act = p[p.model == "Naive (random walk)"].sort_values("date")
    a.plot(act.date, act.actual, color="black", lw=2.2, label="ACTUAL", zorder=5)
    for m, colr in [(f"ARIMA{ARIMA_ORDER}", "tab:purple"), ("Hybrid ARIMA+LSTM", "tab:red"),
                    ("Hybrid+MacroNews", "tab:green")]:
        g = p[p.model == m].sort_values("date")
        if len(g):
            a.plot(g.date, g.pred, lw=1.1, alpha=.85, color=colr, label=m)
    a.set_title(f"{target} — fixed 80/20 test period"); a.grid(alpha=.3)
    a.legend(fontsize=7); a.tick_params(axis="x", rotation=30, labelsize=7)

    b = ax[i][1]
    sub = FIXED[FIXED.target == target].set_index("model")
    order = ["Naive (random walk)", f"ARIMA{ARIMA_ORDER}", "LSTM (univariate)",
             "Hybrid ARIMA+LSTM", "Hybrid+MacroNews"]
    sub = sub.reindex([o_ for o_ in order if o_ in sub.index])
    colr = ["grey" if v >= 1 else "tab:green" for v in sub["Theil_U2"]]
    b.barh(range(len(sub)), sub["Theil_U2"], color=colr, alpha=.85)
    b.axvline(1.0, color="black", ls="--", label="naive = 1.0")
    b.set_yticks(range(len(sub))); b.set_yticklabels(sub.index, fontsize=8)
    b.set_xlabel("Theil's U2  (<1 = beats naive)")
    b.set_title(f"{target} — Theil's U2, fixed split"); b.grid(alpha=.3, axis="x")
    b.legend(fontsize=8)
fig.suptitle("Hybrid ARIMA+LSTM with macro & news — paper-comparable evaluation", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "hybrid_comparison.png", dpi=140)

wins = FIXED[(FIXED.Theil_U2 < 1) & (FIXED.model != "Naive (random walk)")]
dm_sig = FIXED[(FIXED.DM_p_vs_naive < 0.05) & (FIXED.model != "Naive (random walk)")]

md = f"""# Hybrid ARIMA + LSTM with macro & news — and a paper-comparable evaluation

## Why this run exists

1. **The hybrid architecture.** The project is named after it, but the hybrid had only been run
   univariate on price in Stage 1. Here it is implemented properly and extended with macro + news.
2. **Comparability.** Published papers report a fixed train/test split with RMSE / MAE / MAPE / R2
   and claim superiority using the **Diebold-Mariano** test. This project had moved to
   walk-forward evaluation with baseline-relative "edge" — stricter, but not directly comparable to
   a published table. Both are now produced.

## The architecture (Zhang, 2003)

```
Y_t = L_t + N_t
  ARIMA{ARIMA_ORDER}  -> L_hat        linear structure
  e_t = Y_t - L_hat                   what ARIMA could not explain
  LSTM(e_t)           -> N_hat        nonlinear structure in the residual
  FINAL = L_hat + N_hat
```

* **Hybrid ARIMA+LSTM** — residual LSTM sees only past residuals (classic).
* **Hybrid+MacroNews** — residual LSTM also sees {EXOG}.
  This is the project's extension: if macro and news carry information, the natural place for it is
  precisely the part ARIMA provably cannot explain.

LSTM: {SEQ}-step residual window, {HIDDEN} hidden units, {EPOCHS} epochs, averaged over {len(SEEDS)} seeds.
ARIMA forecasts are genuine rolling **one-step-ahead** (state updated each step, parameters frozen).

## TABLE 1 — Fixed 80/20 split (this is the table to place beside a paper)

{md_table(FIXED[["target", "model", "RMSE", "MAE", "MAPE_%", "R2", "Theil_U2", "dir_acc_%", "DM_vs_naive", "DM_p_vs_naive", "DM_p_vs_ARIMA"]])}

**How to read it**
* **Theil's U2 < 1** means better than the naive random walk. **U2 >= 1 means worse.**
* **Diebold-Mariano**: H0 is equal accuracy. A *negative* DM statistic means the model has lower
  loss than the comparison; the p-value says whether that difference is significant.
  Newey-West (HAC) variance, squared-error loss.

Models with Theil's U2 < 1: **{len(wins)}**.
Models significantly better than naive at p < 0.05 (DM): **{len(dm_sig)}**.

## TABLE 2 — Walk-forward (this project's stricter standard)

{md_table(WF)}

`folds_better_than_naive` counts folds with Theil's U2 < 1. A model that wins on the fixed split
but only ~half the folds was benefiting from that particular test window.

## Why report both

A single fixed split is what the literature uses, and it is what makes this work comparable. It is
also the weakest protocol in this project: earlier runs showed the win rate roughly DOUBLES purely
by moving the test window into the 2022 crisis. Publishing the fixed-split table alone would
overstate the result; publishing only walk-forward would be incomparable to prior work. Both is the
defensible answer, and the gap between them is itself a finding.

## Caveats
- ARIMA order is fixed at {ARIMA_ORDER} (the source paper's choice) rather than re-selected per fold;
  Stage 1 found auto-ARIMA collapsing to (0,1,0), i.e. the naive model, on this data.
- The DM test assumes the loss differential is covariance-stationary. At one-step horizons with
  daily data that is reasonable; it would need more care at long horizons.
- News sentiment only exists from 2016 to 2022-06, so `Hybrid+MacroNews` is estimated on a shorter
  sample than the other models where the split extends beyond that.
"""
(OUT / "hybrid_summary.md").write_text(md)

print("\n" + "=" * 110)
print("TABLE 1 — FIXED 80/20 SPLIT (paper-comparable)")
print(FIXED[["target", "model", "RMSE", "MAE", "MAPE_%", "R2", "Theil_U2", "dir_acc_%",
             "DM_vs_naive", "DM_p_vs_naive"]].to_string(index=False))
print("\n" + "=" * 110)
print("TABLE 2 — WALK-FORWARD")
print(WF.to_string(index=False))
print(f"\nModels with Theil U2 < 1 on the fixed split: {len(wins)}")
print(f"Models significantly better than naive (DM p<0.05): {len(dm_sig)}")
print(f"Saved to {OUT}")
