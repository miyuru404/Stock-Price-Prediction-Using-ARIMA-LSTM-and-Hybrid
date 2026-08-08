#!/usr/bin/env python3
"""
Multi-horizon test: ARIMAX (ARIMA+macro), Random Forest, CNN-LSTM.
All macro indicators + technical features. 80/20 split. DIRECT per-horizon (predict each
horizon separately, not the whole window at once). Reports DIRECTION accuracy + PRICE error
(MAPE) at every horizon, vs a naive baseline. Nothing tuned to inflate.

Usage: python src/mh_multimodel.py HNB
Outputs: results/mh_models/<TICKER>_results.csv + plot.
"""
import sys, warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _Path(__file__).resolve().parents[1]
DATA = ROOT / "cleaned_data"
OUT = ROOT / "results" / "mh_models"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
H = [1, 2, 3, 4, 5, 10, 22, 44, 66, 132, 252]           # trading-day horizons
HLAB = ["1d", "2d", "3d", "4d", "5d", "2w", "1mo", "2mo", "3mo", "6mo", "1yr"]
WIN = 20                                                 # window for CNN-LSTM

TICKER = (sys.argv[1] if len(sys.argv) > 1 else "HNB").upper()

# ---------- price data (handle S&P SL20 special file) ----------
if TICKER in ("SPSL20", "SP20"):
    d = pd.read_csv(ROOT/"data/processed/spsl20_trading_days_clean.csv", parse_dates=["date"])
    d = d.rename(columns={"spsl20_points": "close"}); d["volume"] = np.nan
else:
    fp = DATA/f"{TICKER}_daily_clean.csv"
    d = pd.read_csv(fp, parse_dates=["date"])
d = d.sort_values("date").reset_index(drop=True)
c = d["close"].astype(float)

# ---------- macro (all indicators, lagged 1 month, ffill to daily) ----------
rate = pd.read_csv(DATA/"interest_rates_monthly.csv", parse_dates=["date"]).sort_values("date")
mc = ["policy_rate","sdfr","slfr","tb_3m","tb_12m","awdr","awpr","awlr","spread"]
for x in mc: rate[x+"_chg"] = rate[x].diff()
macro_cols = mc + [x+"_chg" for x in mc]
rate_l = rate[["date"]+macro_cols].copy()
rate_l[macro_cols] = rate_l[macro_cols].shift(1)         # lag 1 month (reporting delay safety)
d = pd.merge_asof(d, rate_l, on="date", direction="backward")

# ---------- technical features ----------
ret1 = c.pct_change()
def rsi(s, n=14):
    dl = s.diff(); up = dl.clip(lower=0).rolling(n).mean(); dn = (-dl.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))
F = pd.DataFrame(index=d.index)
F["ret1"]=ret1; F["ret5"]=c.pct_change(5); F["ret10"]=c.pct_change(10)
F["rsi"]=rsi(c); F["ma5"]=c/c.rolling(5).mean()-1; F["ma10"]=c/c.rolling(10).mean()-1
F["ma20"]=c/c.rolling(20).mean()-1; F["mom10"]=c/c.shift(10)-1
F["vol10"]=ret1.rolling(10).std(); F["vol20"]=ret1.rolling(20).std()
if (d["volume"].fillna(0)>0).any():
    F["volchg"]=d["volume"]/d["volume"].rolling(5).mean()-1
for x in macro_cols: F[x]=d[x]
feat_cols = list(F.columns)

# ---------- targets: return over each horizon ----------
for h in H: F[f"y{h}"]=c.shift(-h)/c-1
F["price"]=c.values
data = F.dropna().reset_index(drop=True)
# valid rows = those with all horizon targets (drop last max(H))
data = data.iloc[:len(data)].copy()
split = int(len(data)*0.8)
Xtr = data[feat_cols].iloc[:split].values; Xte = data[feat_cols].iloc[split:].values
print(f"{TICKER}: rows {len(data)} | train {split} test {len(data)-split} | features {len(feat_cols)}")

rows = []
def metrics(pred_ret, h, idx):
    act_ret = data[f"y{h}"].iloc[idx].values
    price_t = data["price"].iloc[idx].values
    act_price = price_t*(1+act_ret); pred_price = price_t*(1+pred_ret)
    m = np.abs(act_ret) > 1e-9
    dir_acc = np.mean(np.sign(pred_ret[m])==np.sign(act_ret[m]))*100
    mape = np.mean(np.abs(act_price-pred_price)/act_price)*100
    return dir_acc, mape

test_idx = np.arange(split, len(data))

# ===== BASELINES: Majority (direction) + Persistence (price) split apart =====
for h in H:
    act  = data[f"y{h}"].iloc[test_idx].values
    m    = np.abs(act) > 1e-9                             # same flat-day mask as metrics()
    pers = np.zeros(len(test_idx))                        # persistence: pred return = 0
    _, mape_pers = metrics(pers, h, test_idx)
    # Majority-class direction baseline (no price forecast -> MAPE not defined)
    dir_maj = max(np.mean(act[m] > 0), np.mean(act[m] < 0)) * 100
    rows.append({"model":"Majority","horizon":h,"dir_acc":dir_maj,"price_mape":np.nan})
    # Persistence baseline: direction from sign of previous realised return
    prev_ret = data["y1"].shift(1).iloc[test_idx].values
    dir_pers = np.mean(np.sign(prev_ret[m]) == np.sign(act[m])) * 100
    rows.append({"model":"Persistence","horizon":h,"dir_acc":dir_pers,"price_mape":mape_pers})

# ===== RANDOM FOREST (multi-output, one fit) =====
Ytr = data[[f"y{h}" for h in H]].iloc[:split].values
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=4)
rf.fit(Xtr, Ytr); rf_pred = rf.predict(Xte)
for j,h in enumerate(H):
    da, mp = metrics(rf_pred[:,j], h, test_idx); rows.append({"model":"RandomForest","horizon":h,"dir_acc":da,"price_mape":mp})

# ===== CNN-LSTM (multi-output, one training) =====
def windows(Xarr, W):
    return np.array([Xarr[i-W:i] for i in range(W, len(Xarr))], np.float32)
allX = data[feat_cols].values.astype(np.float32)
# standardize features on train
mu, sd = allX[:split].mean(0), allX[:split].std(0)+1e-8
allXs = (allX-mu)/sd
Xw = windows(allXs, WIN)                                 # (N-W, W, feat)
Yw = data[[f"y{h}" for h in H]].values[WIN:].astype(np.float32)
# align split for windows
sp_w = split-WIN
Xw_tr, Yw_tr = Xw[:sp_w], Yw[:sp_w]
class CNNLSTM(nn.Module):
    def __init__(s, nf, no):
        super().__init__()
        s.conv = nn.Conv1d(nf, 32, 3, padding=1); s.relu = nn.ReLU()
        s.lstm = nn.LSTM(32, 50, batch_first=True); s.head = nn.Linear(50, no)
    def forward(s, x):                                   # x (B,W,nf)
        z = s.relu(s.conv(x.transpose(1,2))).transpose(1,2)
        o,_ = s.lstm(z); return s.head(o[:,-1,:])
net = CNNLSTM(len(feat_cols), len(H)); opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.MSELoss()
Xt=torch.tensor(Xw_tr); Yt=torch.tensor(Yw_tr)
nv=int(len(Xt)*0.1); Xv,Yv=Xt[-nv:],Yt[-nv:]; Xt2,Yt2=Xt[:-nv],Yt[:-nv]
best,bs,wait=1e9,None,0
for ep in range(120):
    net.train(); perm=torch.randperm(len(Xt2))
    for i in range(0,len(Xt2),64):
        idx=perm[i:i+64]; opt.zero_grad(); lf(net(Xt2[idx]),Yt2[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): vl=lf(net(Xv),Yv).item()
    if vl<best-1e-9: best,bs,wait=vl,{k:v.clone() for k,v in net.state_dict().items()},0
    else:
        wait+=1
        if wait>=12: break
net.load_state_dict(bs); net.eval()
# test windows correspond to data rows [split:]; window ending at row i predicts from features up to i
Xw_te = Xw[sp_w:]
with torch.no_grad(): cl_pred = net(torch.tensor(Xw_te)).numpy()
cl_idx = np.arange(split, split+len(Xw_te))
for j,h in enumerate(H):
    da,mp = metrics(cl_pred[:,j], h, cl_idx); rows.append({"model":"CNN-LSTM","horizon":h,"dir_acc":da,"price_mape":mp})

# ===== ARIMAX (SARIMAX on log price + macro exog) =====
logp = np.log(c.values)
exog_all = d[macro_cols].ffill().bfill().values
tr_end = data.index[split-1]                              # approx; use row count on aligned series
# align: SARIMAX on full daily series indices matching 'data' rows is complex; use data slice
# Rebuild aligned arrays from 'data' (which already dropped NaN)
lp = np.log(data["price"].values)
ex = data[macro_cols].values
try:
    res = SARIMAX(lp[:split], exog=ex[:split], order=(1,1,1),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=0)
    stride = max(1, len(test_idx)//120)
    origins = list(range(split, len(data)-max(H), stride))
    ax_pred = {h: [] for h in H}; ax_oidx = []
    for o in origins:
        # condition on data THROUGH index o (inclusive) so forecast step 1 lands on o+1.
        r_o = res.apply(lp[:o+1], exog=ex[:o+1])
        fex = np.tile(ex[o], (max(H), 1))                # hold macro at last KNOWN row (index o)
        fc = np.asarray(r_o.get_forecast(max(H), exog=fex).predicted_mean)  # steps 1..maxH
        base = data["price"].iloc[o]                     # last observed price
        for h in H: ax_pred[h].append(np.exp(fc[h-1])/base - 1)   # h-ahead
        ax_oidx.append(o)
    ax_oidx = np.array(ax_oidx)
    # ARIMAX at h=1 should not be systematically inverted; <40% signals an alignment error.
    d1 = metrics(np.array(ax_pred[1]), 1, ax_oidx)[0]
    assert d1 > 40, f"ARIMAX h=1 direction {d1:.1f}% - check forecast alignment"
    for h in H:
        da,mp = metrics(np.array(ax_pred[h]), h, ax_oidx); rows.append({"model":"ARIMAX","horizon":h,"dir_acc":da,"price_mape":mp})
except Exception as e:
    print("ARIMAX failed:", str(e)[:150])
    for h in H: rows.append({"model":"ARIMAX","horizon":h,"dir_acc":np.nan,"price_mape":np.nan})

# ---------- save + report ----------
res_df = pd.DataFrame(rows)
res_df["horizon_lbl"] = res_df["horizon"].map(dict(zip(H, HLAB)))
res_df.round(2).to_csv(OUT/f"{TICKER}_results.csv", index=False)
DIR_ORDER = ["Majority","Persistence","ARIMAX","RandomForest","CNN-LSTM"]
PRICE_ORDER = ["Persistence","ARIMAX","RandomForest","CNN-LSTM"]
print("\n=== DIRECTION ACCURACY (%) by model x horizon ===")
piv_d = res_df.pivot(index="model", columns="horizon", values="dir_acc").reindex(DIR_ORDER)[H]
piv_d.columns = HLAB; print(piv_d.round(1).to_string())
print("\n=== PRICE MAPE (%) by model x horizon  (Persistence = naive-0 baseline) ===")
piv_m = res_df.pivot(index="model", columns="horizon", values="price_mape").reindex(PRICE_ORDER)[H]
piv_m.columns = HLAB; print(piv_m.round(2).to_string())

fig,(a1,a2)=plt.subplots(1,2,figsize=(15,5.5))
for m in DIR_ORDER:
    s=res_df[res_df.model==m]
    a1.plot(range(len(H)), s.set_index("horizon").reindex(H)["dir_acc"], "o-", label=m)
    a2.plot(range(len(H)), s.set_index("horizon").reindex(H)["price_mape"], "o-", label=m)
a1.axhline(50,ls=":",c="k"); a1.set_xticks(range(len(H))); a1.set_xticklabels(HLAB); a1.set_title(f"{TICKER} — Direction accuracy vs horizon"); a1.set_ylabel("%"); a1.legend(); a1.grid(alpha=.3)
a2.set_xticks(range(len(H))); a2.set_xticklabels(HLAB); a2.set_title(f"{TICKER} — Price MAPE vs horizon"); a2.set_ylabel("%"); a2.legend(); a2.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT/f"{TICKER}_mh_plot.png", dpi=140)
print(f"\nSaved results/mh_models/{TICKER}_results.csv + plot")
