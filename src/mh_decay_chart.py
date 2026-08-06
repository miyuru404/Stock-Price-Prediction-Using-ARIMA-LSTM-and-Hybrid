#!/usr/bin/env python3
"""
Decay chart (user's method): ONE launch spot = start of test window.
From that single fixed origin the model forecasts 1d, 2d, ... 1yr ahead BLIND
(never fed the real prices inside the window). Shows:
  - actual price path (black)
  - FLAT naive line = hold last-known price (gray dashed)
  - each model's blind forecast at every horizon (points)
Left panel = price fan-out from one origin. Right panel = price-MAPE decay
(averaged over many origins, from <TICKER>_results.csv) so the accuracy-drop is robust.

Usage: python src/mh_decay_chart.py HNB
"""
import sys, warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=_Path(__file__).resolve().parents[1]; DATA=ROOT/"cleaned_data"; OUT=ROOT/"results"/"mh_models"
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
H=[1,2,3,4,5,10,22,44,66,132,252]; HLAB=["1d","2d","3d","4d","5d","2w","1mo","2mo","3mo","6mo","1yr"]; WIN=20
T=(sys.argv[1] if len(sys.argv)>1 else "HNB").upper()

# ---- price ----
if T in ("SPSL20","SP20"):
    d=pd.read_csv(ROOT/"data/processed/spsl20_trading_days_clean.csv",parse_dates=["date"]).rename(columns={"spsl20_points":"close"}); d["volume"]=np.nan
else:
    d=pd.read_csv(DATA/f"{T}_daily_clean.csv",parse_dates=["date"])
d=d.sort_values("date").reset_index(drop=True); c=d["close"].astype(float)
# ---- macro ----
rate=pd.read_csv(DATA/"interest_rates_monthly.csv",parse_dates=["date"]).sort_values("date")
mc=["policy_rate","sdfr","slfr","tb_3m","tb_12m","awdr","awpr","awlr","spread"]
for x in mc: rate[x+"_chg"]=rate[x].diff()
macro=mc+[x+"_chg" for x in mc]; rl=rate[["date"]+macro].copy(); rl[macro]=rl[macro].shift(1)
d=pd.merge_asof(d,rl,on="date",direction="backward")
# ---- features ----
r1=c.pct_change()
def rsi(s,n=14):
    dl=s.diff(); up=dl.clip(lower=0).rolling(n).mean(); dn=(-dl.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))
F=pd.DataFrame(index=d.index)
F["ret1"]=r1;F["ret5"]=c.pct_change(5);F["ret10"]=c.pct_change(10);F["rsi"]=rsi(c)
F["ma5"]=c/c.rolling(5).mean()-1;F["ma10"]=c/c.rolling(10).mean()-1;F["ma20"]=c/c.rolling(20).mean()-1
F["mom10"]=c/c.shift(10)-1;F["vol10"]=r1.rolling(10).std();F["vol20"]=r1.rolling(20).std()
if (d["volume"].fillna(0)>0).any(): F["volchg"]=d["volume"]/d["volume"].rolling(5).mean()-1
for x in macro: F[x]=d[x]
fc=list(F.columns)
for h in H: F[f"y{h}"]=c.shift(-h)/c-1
F["price"]=c.values; F["date"]=d["date"].values
data=F.dropna().reset_index(drop=True); split=int(len(data)*0.8)
Xtr=data[fc].iloc[:split].values
Ytr=data[[f"y{h}" for h in H]].iloc[:split].values

# ---- train RF ----
rf=RandomForestRegressor(300,max_depth=8,min_samples_leaf=5,random_state=SEED,n_jobs=4).fit(Xtr,Ytr)
# ---- train CNN-LSTM ----
allX=data[fc].values.astype(np.float32); mu,sd=allX[:split].mean(0),allX[:split].std(0)+1e-8; Xs=(allX-mu)/sd
Xw=np.array([Xs[i-WIN:i] for i in range(WIN,len(Xs))],np.float32); Yw=data[[f"y{h}" for h in H]].values[WIN:].astype(np.float32); spw=split-WIN
class Net(nn.Module):
    def __init__(s,nf,no):
        super().__init__(); s.cv=nn.Conv1d(nf,32,3,padding=1); s.rl=nn.ReLU(); s.ls=nn.LSTM(32,50,batch_first=True); s.h=nn.Linear(50,no)
    def forward(s,x): z=s.rl(s.cv(x.transpose(1,2))).transpose(1,2); o,_=s.ls(z); return s.h(o[:,-1,:])
net=Net(len(fc),len(H)); opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.MSELoss()
Xt=torch.tensor(Xw[:spw]); Yt=torch.tensor(Yw[:spw]); nv=int(len(Xt)*.1)
best,bs,w=1e9,None,0
for e in range(120):
    net.train(); p=torch.randperm(len(Xt)-nv)
    for i in range(0,len(p),64):
        idx=p[i:i+64]; opt.zero_grad(); lf(net(Xt[idx]),Yt[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): vl=lf(net(Xt[-nv:]),Yt[-nv:]).item()
    if vl<best-1e-9: best,bs,w=vl,{k:v.clone() for k,v in net.state_dict().items()},0
    else:
        w+=1
        if w>=12: break
net.load_state_dict(bs); net.eval()
# ---- train ARIMAX ----
lp=np.log(data["price"].values); ex=data[macro].values
ar=SARIMAX(lp[:split],exog=ex[:split],order=(1,1,1),enforce_stationarity=False,enforce_invertibility=False).fit(disp=0)

# ---- ONE launch spot = start of test ----
o=split; base=data["price"].iloc[o]
maxh=min(max(H), len(data)-1-o)                     # cap to available future
Hc=[h for h in H if h<=maxh]
# RF / CNN-LSTM forecasts at the horizons (blind, from origin o)
rf_o=rf.predict(data[fc].iloc[[o]].values)[0]                     # returns per horizon
cl_o=net(torch.tensor(Xs[o-WIN:o][None].astype(np.float32))).detach().numpy()[0]
# ARIMAX full path (blind, macro held)
r_o=ar.apply(lp[:o],exog=ex[:o]); fex=np.tile(ex[o-1],(maxh,1))
ax_path=np.exp(np.asarray(r_o.get_forecast(maxh,exog=fex).predicted_mean))   # price days 1..maxh

# actual future path from origin
fut=np.arange(o, o+maxh+1)
act_price=data["price"].iloc[fut].values
act_date=data["date"].iloc[fut].values

fig,(a1,a2)=plt.subplots(1,2,figsize=(16,6))
# ---- LEFT: single-origin fan-out ----
a1.plot(act_date, act_price, color="black", lw=2, label="Actual price")
a1.plot(act_date, np.full(len(act_date), base), color="gray", ls="--", lw=1.4, label="Naive (hold last price) — FLAT")
a1.plot(data["date"].iloc[o+1:o+1+maxh].values, ax_path, color="tab:orange", lw=1.6, label="ARIMAX")
hd=[data["date"].iloc[o+h] for h in Hc]
a1.plot(hd,[base*(1+rf_o[H.index(h)]) for h in Hc],"o-",color="tab:green",ms=5,lw=1,label="RandomForest")
a1.plot(hd,[base*(1+cl_o[H.index(h)]) for h in Hc],"s-",color="tab:red",ms=5,lw=1,label="CNN-LSTM")
a1.axvline(data["date"].iloc[o],color="blue",ls=":",lw=1);
a1.set_title(f"{T} — blind forecast from ONE launch spot (start of test)\nno real prices fed inside the window"); a1.set_ylabel("Price"); a1.grid(alpha=.3); a1.legend()
# ---- RIGHT: robust MAPE decay (many origins) ----
rp=OUT/f"{T}_results.csv"
if rp.exists():
    rr=pd.read_csv(rp)
    for m,cl in [("Naive","gray"),("ARIMAX","tab:orange"),("RandomForest","tab:green"),("CNN-LSTM","tab:red")]:
        s=rr[rr.model==m].set_index("horizon").reindex(H)
        a2.plot(range(len(H)), s["price_mape"], "o-", color=cl, label=m)
    a2.set_xticks(range(len(H))); a2.set_xticklabels(HLAB)
    a2.set_title(f"{T} — price error (MAPE %) vs horizon\n(averaged over many launch spots)"); a2.set_ylabel("MAPE %"); a2.grid(alpha=.3); a2.legend()
fig.tight_layout(); fig.savefig(OUT/f"{T}_decay_chart.png",dpi=140)
print(f"saved {T}_decay_chart.png  (launch origin row {o}, maxh {maxh})")
