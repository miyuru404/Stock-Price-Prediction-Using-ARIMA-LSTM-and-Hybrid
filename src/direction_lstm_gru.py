#!/usr/bin/env python3
"""
LSTM / GRU FOR DIRECTION — the last untested model class.

WHY THIS IS A DIFFERENT TEST, not just another classifier:
Every model so far reads a FLAT ROW of summary numbers (ret_5, vol_20, rsi_14, ...). Those
summaries throw away the ORDER of what happened. A recurrent net reads the raw SEQUENCE of the
last N days, so it can in principle react to shape — "three down days then a spike" — that a
summary feature cannot express. That is a genuinely different hypothesis about where a signal
might hide, and nothing in this project has tested it for the DIRECTION target.

(The project is also named "... Using ARIMA, LSTM and Hybrid", and LSTM had only ever been used
on the PRICE target here. This closes that gap.)

DESIGN — deliberately the same discipline as every earlier phase:
  * POOLED over the 7 banking/finance stocks (maximum data for a data-hungry model class).
  * Input = last SEQ_LEN days x daily-varying features only. Monthly macro is excluded ON PURPOSE:
    inside a 30-day window it is a flat line, so it gives a sequence model nothing.
  * WALK-FORWARD over several non-overlapping test windows, not one split — the single-window trap
    already produced (and then destroyed) the 1-day claim.
  * Scored PER STOCK against that stock's OWN baseline (majority + matched-horizon persistence).
  * MULTIPLE SEEDS per model, averaged. Recurrent nets swing with seed, and this project has
    already been fooled once by seed noise (the 50/50 split "LSTM win").
  * Logistic Regression is trained on the FLATTENED same sequences, so the comparison isolates
    "does sequence structure help?" rather than "is one library better than another?".

Outputs -> results/direction/lstm_gru/
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "direction" / "lstm_gru"
OUT.mkdir(parents=True, exist_ok=True)

SECTOR = ["HNB", "COMB", "SAMP", "LOFC", "LOLC", "LFIN", "CFIN"]
HORIZONS = [1, 5, 22]
SEQ_LEN = 30
DEADZONE_1D = 0.005
SEEDS = [0, 1, 2]
EPOCHS = 25
BATCH = 256
HIDDEN = 48
TEST_MONTHS = 12
FIRST_TEST = pd.Timestamp("2019-01-01")
DEV = torch.device("cpu")          # tiny model; CPU is faster than MPS transfer overhead here

px = {t: (pd.read_csv(DATA / f"{t}_daily_clean.csv", parse_dates=["date"])
            .sort_values("date").drop_duplicates("date").set_index("date")) for t in SECTOR}
aspi = (pd.read_csv(DATA / "ASPI_daily_clean.csv", parse_dates=["date"])
          .sort_values("date").drop_duplicates("date").set_index("date")["close"].astype(float))


def md_table(df):
    cols = [str(x) for x in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- per-day sequence features
SEQ_FEATS = ["ret_1", "vol_chg", "hl_range", "aspi_ret_1", "peer_ret_1", "rel_aspi"]


def build(t):
    d = px[t].reset_index()
    dates = d["date"]
    c = d["close"].astype(float)
    v = d["volume"].astype(float)
    r1 = c.pct_change()
    a = aspi.reindex(dates).ffill().reset_index(drop=True)
    ar = a.pct_change()
    peers = [p for p in SECTOR if p != t]
    pp = pd.DataFrame({p: px[p]["close"].astype(float).reindex(dates).ffill().reset_index(drop=True)
                       for p in peers})
    f = pd.DataFrame({"date": dates, "ticker": t})
    f["ret_1"] = r1
    f["vol_chg"] = (v / v.rolling(5).mean() - 1).clip(-3, 3)
    f["hl_range"] = ((d["high"].astype(float) - d["low"].astype(float)) / c).clip(0, 0.5)
    f["aspi_ret_1"] = ar
    f["peer_ret_1"] = pp.pct_change().mean(axis=1)
    f["rel_aspi"] = r1 - ar
    for h in HORIZONS:
        f[f"fwd_{h}"] = c.shift(-h) / c - 1
        f[f"past_{h}"] = c / c.shift(h) - 1
    return f


PANEL = pd.concat([build(t) for t in SECTOR], ignore_index=True)


def make_sequences(h):
    """Build (N, SEQ_LEN, F) windows per stock, never crossing stock boundaries."""
    Xs, ys, meta = [], [], []
    dz = DEADZONE_1D * np.sqrt(h)
    for t in SECTOR:
        g = PANEL[PANEL.ticker == t].reset_index(drop=True)
        cols = g[SEQ_FEATS].values
        fwd = g[f"fwd_{h}"].values
        past = g[f"past_{h}"].values
        dates = g["date"].values
        ok = ~np.isnan(cols).any(axis=1)
        for i in range(SEQ_LEN, len(g)):
            if not ok[i - SEQ_LEN:i].all() or np.isnan(fwd[i - 1]) or np.isnan(past[i - 1]):
                continue
            Xs.append(cols[i - SEQ_LEN:i])
            ys.append(2 if fwd[i - 1] > dz else (0 if fwd[i - 1] < -dz else 1))
            meta.append((t, dates[i - 1], past[i - 1]))
    X = np.asarray(Xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    M = pd.DataFrame(meta, columns=["ticker", "date", "past_h"])
    M["date"] = pd.to_datetime(M["date"])
    return X, y, M


class Net(nn.Module):
    def __init__(self, kind, nf):
        super().__init__()
        rnn = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = rnn(nf, HIDDEN, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(HIDDEN, 3)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(self.drop(o[:, -1]))


def train_predict(kind, Xtr, ytr, Xte, seed, cw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(kind, Xtr.shape[2]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEV))
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(ytr, device=DEV)
    n = len(Xt)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out = []
        Xe = torch.tensor(Xte, device=DEV)
        for i in range(0, len(Xe), 1024):
            out.append(net(Xe[i:i + 1024]).argmax(1).cpu().numpy())
    return np.concatenate(out)


rows = []
for h in HORIZONS:
    X, y, M = make_sequences(h)
    dz = DEADZONE_1D * np.sqrt(h)
    starts = pd.date_range(FIRST_TEST, M.date.max(), freq=f"{TEST_MONTHS}MS")
    folds = [(s, s + pd.DateOffset(months=TEST_MONTHS)) for s in starts
             if s + pd.DateOffset(months=TEST_MONTHS) <= M.date.max()]
    print(f"\nh={h}d  sequences {len(X):,}  folds {len(folds)}")

    for fi, (t0, t1) in enumerate(folds):
        purge = t0 - pd.Timedelta(days=int(h * 1.5) + SEQ_LEN + 3)
        tr = (M.date <= purge).values
        te = ((M.date >= t0) & (M.date < t1)).values
        if tr.sum() < 3000 or te.sum() < 300:
            continue
        # scale on TRAIN only, per feature, over all timesteps
        mu = X[tr].reshape(-1, X.shape[2]).mean(0)
        sd = X[tr].reshape(-1, X.shape[2]).std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 3:
            continue
        cnt = np.bincount(ytr, minlength=3).astype(float)
        cw = np.where(cnt > 0, len(ytr) / (3 * np.maximum(cnt, 1)), 0.0)

        preds = {}
        for kind in ["LSTM", "GRU"]:
            seed_preds = np.stack([train_predict(kind, Xtr, ytr, Xte, s, cw) for s in SEEDS])
            # majority vote across seeds = the honest "averaged" model
            preds[kind] = np.apply_along_axis(lambda v: np.bincount(v, minlength=3).argmax(),
                                              0, seed_preds)
        # Logistic on the FLATTENED identical sequences -> isolates the value of sequence structure
        lg = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0))
        lg.fit(Xtr.reshape(len(Xtr), -1), ytr)
        preds["Logistic(flat)"] = lg.predict(Xte.reshape(len(Xte), -1))

        mte = M[te].reset_index(drop=True)
        mtr = M[tr].reset_index(drop=True)
        mtr["y"] = ytr
        for tk, g in mte.groupby("ticker"):
            gi = g.index.values
            own_tr = mtr[mtr.ticker == tk]
            if len(own_tr) < 200 or len(gi) < 30:
                continue
            own_maj = int(pd.Series(own_tr["y"]).value_counts().idxmax())
            yy = yte[gi]
            b = max(accuracy_score(yy, np.full(len(yy), own_maj)),
                    accuracy_score(yy, np.where(g["past_h"] > dz, 2,
                                                np.where(g["past_h"] < -dz, 0, 1))))
            rec = {"horizon_days": h, "fold": fi, "test_from": f"{t0:%Y-%m}",
                   "ticker": tk, "n_test": len(gi), "own_baseline_%": round(b * 100, 1)}
            for k, p in preds.items():
                acc = accuracy_score(yy, p[gi])
                rec[f"acc_{k}_%"] = round(acc * 100, 1)
                rec[f"edge_{k}_pp"] = round((acc - b) * 100, 1)
            rows.append(rec)
        print(f"  fold {fi} {t0:%Y-%m}  train {tr.sum():,} test {te.sum():,}  "
              + "  ".join(f"{k} {accuracy_score(yte, p)*100:.1f}%" for k, p in preds.items()))

R = pd.DataFrame(rows)
R.to_csv(OUT / "lstm_gru_per_stock_per_fold.csv", index=False)

MODELS = ["LSTM", "GRU", "Logistic(flat)"]
FOLD = (R.groupby(["horizon_days", "fold", "test_from"])
        .agg(**{f"med_{m}": (f"edge_{m}_pp", "median") for m in MODELS}).reset_index().round(2))
FOLD.to_csv(OUT / "lstm_gru_by_fold.csv", index=False)

summary = []
for h in HORIZONS:
    sub = FOLD[FOLD.horizon_days == h]
    for m in MODELS:
        v = sub[f"med_{m}"].values
        k = int((v > 0).sum())
        p = binomtest(k, len(v), 0.5, alternative="greater").pvalue if len(v) else 1.0
        summary.append({"horizon": {1: "1 day", 5: "1 week", 22: "1 month"}[h], "model": m,
                        "folds_positive": f"{k}/{len(v)}",
                        "median_of_fold_medians_pp": round(float(np.median(v)), 2) if len(v) else np.nan,
                        "sign_test_p": round(float(p), 4),
                        "beats_baseline": "YES" if (k > len(v) / 2 and p < 0.05) else "NO"})
S = pd.DataFrame(summary)
S.to_csv(OUT / "lstm_gru_verdict.csv", index=False)

fig, ax = plt.subplots(1, len(HORIZONS), figsize=(5 * len(HORIZONS), 4.8), squeeze=False)
for i, h in enumerate(HORIZONS):
    a = ax[0][i]
    sub = FOLD[FOLD.horizon_days == h]
    x = np.arange(len(sub))
    for j, (m, colr) in enumerate(zip(MODELS, ["tab:red", "tab:purple", "tab:blue"])):
        a.bar(x + (j - 1) * 0.27, sub[f"med_{m}"], 0.27, label=m, color=colr, alpha=.85)
    a.axhline(0, color="black", lw=1)
    a.set_xticks(x); a.set_xticklabels(sub.test_from, rotation=90, fontsize=7)
    a.set_title(f"{ {1:'1 day',5:'1 week',22:'1 month'}[h] } — median per-stock edge")
    a.set_ylabel("pp vs own baseline"); a.grid(alpha=.3, axis="y")
    if i == 0:
        a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "lstm_gru_edge.png", dpi=140)

any_win = (S.beats_baseline == "YES").any()
md = f"""# LSTM / GRU for direction — the last untested model class

**Why this is a different test:** every earlier model read a FLAT ROW of summary features
(`ret_5`, `vol_20`, `rsi_14`, …), which throws away the ORDER of recent days. A recurrent net reads
the raw **sequence** of the last **{SEQ_LEN} days**, so it can react to shape that a summary cannot
express. Different hypothesis, not just a different algorithm.

**Setup:** pooled over {SECTOR}; sequence features {SEQ_FEATS};
{len(SEEDS)} seeds per model (majority vote — recurrent nets swing with seed and this project has
been fooled by seed noise before); walk-forward {TEST_MONTHS}-month test windows; scored per stock
against that stock's own baseline. Monthly macro deliberately excluded — inside a {SEQ_LEN}-day
window it is a flat line and tells a sequence model nothing.

**Logistic(flat)** is the control: the SAME sequences, flattened into one long row. If the recurrent
nets beat it, sequence structure genuinely matters. If not, the ordering carries nothing.

## VERDICT

{md_table(S)}

**{'A recurrent model beats the baseline somewhere — investigate.' if any_win else 'No recurrent model beats the naive baseline at any horizon. Sequence structure adds nothing.'}**

## Fold by fold (median per-stock edge, pp)

{md_table(FOLD)}

## Reading it simply

- If **LSTM/GRU ≈ Logistic(flat)**, the order of the last {SEQ_LEN} days carries no extra
  information — the summary features were already enough (and they were already useless).
- If **LSTM/GRU < Logistic(flat)**, the extra flexibility is just fitting noise, which is the same
  story as XGBoost losing to Logistic throughout this project.

## Caveats
- ~{len(SEEDS)} seeds averaged, but recurrent nets remain the noisiest models here.
- Pooled sequences from 7 correlated stocks are not independent samples.
- {SEQ_LEN}-day window chosen a priori; a longer window was not searched, because searching window
  lengths until one works is exactly how false findings are manufactured.
"""
(OUT / "lstm_gru_summary.md").write_text(md)

print("\n" + "=" * 88)
print(S.to_string(index=False))
print("=" * 88)
print(FOLD.to_string(index=False))
print(f"\nSaved to {OUT}")
