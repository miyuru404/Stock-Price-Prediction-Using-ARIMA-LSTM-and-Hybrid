#!/usr/bin/env python3
"""
Direction model — STEP 1 baseline (price + technical features only).
Predicts next-day Buy / Sell / Hold for HNB. XGBoost classifier, honest walk-forward-style
chronological split, confusion matrix, must beat majority-class + persistence baselines.
Nothing tuned to inflate. Outputs to results/direction/.
"""
import warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "direction"
OUT.mkdir(parents=True, exist_ok=True)
TICKER = "HNB"
DEADZONE = 0.005   # +/-0.5%
CLASSES = ["Sell", "Hold", "Buy"]     # -1, 0, +1

# ---------------- data + features (all from PAST data; label = next day) ----------------
d = pd.read_csv(DATA / f"{TICKER}_daily_clean.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
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
feat["ret_1"] = ret1
feat["ret_5"] = c.pct_change(5)
feat["ret_10"] = c.pct_change(10)
feat["rsi_14"] = rsi(c, 14)
feat["ma5_ratio"] = c / c.rolling(5).mean() - 1
feat["ma10_ratio"] = c / c.rolling(10).mean() - 1
feat["ma20_ratio"] = c / c.rolling(20).mean() - 1
feat["momentum_10"] = c / c.shift(10) - 1
feat["vol_10"] = ret1.rolling(10).std()
feat["vol_20"] = ret1.rolling(20).std()
feat["volchg"] = v / v.rolling(5).mean() - 1
feat["dow"] = d["date"].dt.dayofweek

# label: next-day return -> Buy/Hold/Sell with dead-zone
fwd = c.shift(-1) / c - 1
label = np.where(fwd > DEADZONE, 2, np.where(fwd < -DEADZONE, 0, 1))  # 0=Sell,1=Hold,2=Buy
data = feat.copy(); data["y"] = label
data = data.dropna().iloc[:-1]          # drop last row (no next day) + NaN warmup
X = data.drop(columns="y"); y = data["y"].astype(int)

# ---------------- chronological split (train past / test future) ----------------
split = int(len(data) * 0.8)
Xtr, Xte = X.iloc[:split], X.iloc[split:]
ytr, yte = y.iloc[:split], y.iloc[split:]

# class-balanced sample weights (Hold dominates)
w = ytr.map({k: len(ytr) / (3 * (ytr == k).sum()) for k in [0, 1, 2]})

clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=3, objective="multi:softprob",
                    num_class=3, random_state=42, n_jobs=4, eval_metric="mlogloss")
clf.fit(Xtr, ytr, sample_weight=w.values)
pred = clf.predict(Xte)

# ---------------- baselines ----------------
majority = int(ytr.value_counts().idxmax())
maj_pred = np.full(len(yte), majority)
# persistence: tomorrow's direction = today's direction (sign of ret_1 with dead-zone)
pers = np.where(Xte["ret_1"] > DEADZONE, 2, np.where(Xte["ret_1"] < -DEADZONE, 0, 1))

acc = accuracy_score(yte, pred)
acc_maj = accuracy_score(yte, maj_pred)
acc_pers = accuracy_score(yte, pers)

cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
rep = classification_report(yte, pred, labels=[0, 1, 2], target_names=CLASSES, output_dict=True, zero_division=0)

# ---------------- save ----------------
pd.DataFrame(cm, index=[f"true_{x}" for x in CLASSES], columns=[f"pred_{x}" for x in CLASSES]).to_csv(OUT / "step1_confusion_matrix.csv")
rows = [{"class": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"], "f1": rep[c]["f1-score"], "support": rep[c]["support"]} for c in CLASSES]
rows.append({"class": "ACCURACY", "precision": "", "recall": "", "f1": acc, "support": len(yte)})
pd.DataFrame(rows).to_csv(OUT / "step1_metrics.csv", index=False)
imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
imp.round(4).to_csv(OUT / "step1_feature_importance.csv")

# ---------------- console ----------------
print(f"HNB next-day direction | train {len(Xtr)} test {len(Xte)} | dead-zone +/-{DEADZONE*100}%")
print(f"class balance (test): Sell {int((yte==0).sum())}  Hold {int((yte==1).sum())}  Buy {int((yte==2).sum())}")
print(f"\nAccuracy:")
print(f"  XGBoost (price+technical): {acc*100:.1f}%")
print(f"  Majority baseline        : {acc_maj*100:.1f}%   (beat it? {'YES' if acc>acc_maj else 'NO'})")
print(f"  Persistence baseline     : {acc_pers*100:.1f}%   (beat it? {'YES' if acc>acc_pers else 'NO'})")
print(f"\nConfusion matrix (rows=true, cols=pred) [Sell,Hold,Buy]:")
print(cm)
print("\nPer-class F1:", {c: round(rep[c]['f1-score'], 2) for c in CLASSES})
print("Top features:", list(imp.head(5).index))

# ---------------- plot confusion matrix ----------------
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)
ax.set_title(f"HNB next-day direction — Step 1\nAcc {acc*100:.1f}% vs majority {acc_maj*100:.1f}% / persistence {acc_pers*100:.1f}%")
fig.tight_layout(); fig.savefig(OUT / "step1_confusion_matrix.png", dpi=140)

# ---------------- summary md ----------------
beat = acc > max(acc_maj, acc_pers)
md = f"""# Direction Model — Step 1 (price + technical only)

**Target:** HNB next-day Buy / Sell / Hold (dead-zone +/-0.5%). XGBoost, chronological split.

## Result (honest)
- **XGBoost accuracy: {acc*100:.1f}%**
- Majority baseline: {acc_maj*100:.1f}%   | Persistence baseline: {acc_pers*100:.1f}%
- **Beats both baselines? {'YES' if beat else 'NO'}**

Per-class F1: Sell {rep['Sell']['f1-score']:.2f}, Hold {rep['Hold']['f1-score']:.2f}, Buy {rep['Buy']['f1-score']:.2f}.

## Read it simply
Price-only, next-day direction. Target was ~52-58%. Got {acc*100:.1f}%.
{'It clears the baselines a little — a real (small) edge from price/technical features.' if beat else 'It does NOT clearly beat the naive baselines — price+technical alone is not enough (as expected). This is the honest baseline to improve on with macro/events/sentiment.'}

## Next
Step 2 = add macro features (spread, inflation, T-bill rate, exchange rate). Measure the accuracy gain.

*Caveat: one stock, next-day horizon (hardest), single chronological split, correlation not causation.*
"""
(OUT / "step1_summary.md").write_text(md)
print(f"\nSaved to {OUT}")
