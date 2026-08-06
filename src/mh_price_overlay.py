#!/usr/bin/env python3
"""Overlay: real HNB price vs model predicted price (1-day and 1-month ahead)."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path as _Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=_Path(__file__).resolve().parents[1]; DATA=ROOT/"cleaned_data"; OUT=ROOT/"results/mh_models"
np.random.seed(42); torch.manual_seed(42)
T="HNB"; HSET=[1,22]; WIN=20

d=pd.read_csv(DATA/f"{T}_daily_clean.csv",parse_dates=["date"]).sort_values("date").reset_index(drop=True)
c=d["close"].astype(float)
rate=pd.read_csv(DATA/"interest_rates_monthly.csv",parse_dates=["date"]).sort_values("date")
mc=["policy_rate","sdfr","slfr","tb_3m","tb_12m","awdr","awpr","awlr","spread"]
for x in mc: rate[x+"_chg"]=rate[x].diff()
macro=mc+[x+"_chg" for x in mc]; rl=rate[["date"]+macro].copy(); rl[macro]=rl[macro].shift(1)
d=pd.merge_asof(d,rl,on="date",direction="backward")
r1=c.pct_change()
def rsi(s,n=14):
    dl=s.diff(); up=dl.clip(lower=0).rolling(n).mean(); dn=(-dl.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))
F=pd.DataFrame(index=d.index)
F["ret1"]=r1;F["ret5"]=c.pct_change(5);F["ret10"]=c.pct_change(10);F["rsi"]=rsi(c)
F["ma5"]=c/c.rolling(5).mean()-1;F["ma10"]=c/c.rolling(10).mean()-1;F["ma20"]=c/c.rolling(20).mean()-1
F["mom10"]=c/c.shift(10)-1;F["vol10"]=r1.rolling(10).std();F["vol20"]=r1.rolling(20).std()
F["volchg"]=d["volume"]/d["volume"].rolling(5).mean()-1
for x in macro: F[x]=d[x]
fc_cols=list(F.columns)
for h in HSET: F[f"y{h}"]=c.shift(-h)/c-1
F["price"]=c.values; F["date"]=d["date"].values
data=F.dropna().reset_index(drop=True); split=int(len(data)*0.8)
Xtr=data[fc_cols].iloc[:split].values; Xte=data[fc_cols].iloc[split:].values
test_idx=np.arange(split,len(data))

# RF
Ytr=data[[f"y{h}" for h in HSET]].iloc[:split].values
rf=RandomForestRegressor(300,max_depth=8,min_samples_leaf=5,random_state=42,n_jobs=4).fit(Xtr,Ytr)
rfp=rf.predict(Xte)
# CNN-LSTM
allX=data[fc_cols].values.astype(np.float32); mu,sd=allX[:split].mean(0),allX[:split].std(0)+1e-8
Xs=(allX-mu)/sd; Xw=np.array([Xs[i-WIN:i] for i in range(WIN,len(Xs))],np.float32)
Yw=data[[f"y{h}" for h in HSET]].values[WIN:].astype(np.float32); spw=split-WIN
class Net(nn.Module):
    def __init__(s,nf,no):
        super().__init__(); s.cv=nn.Conv1d(nf,32,3,padding=1); s.rl=nn.ReLU(); s.ls=nn.LSTM(32,50,batch_first=True); s.h=nn.Linear(50,no)
    def forward(s,x): z=s.rl(s.cv(x.transpose(1,2))).transpose(1,2); o,_=s.ls(z); return s.h(o[:,-1,:])
net=Net(len(fc_cols),len(HSET)); opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.MSELoss()
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
with torch.no_grad(): clp=net(torch.tensor(Xw[spw:])).numpy()
cl_idx=np.arange(split,split+len(Xw[spw:]))
# ARIMAX
lp=np.log(data["price"].values); ex=data[macro].values
res=SARIMAX(lp[:split],exog=ex[:split],order=(1,1,1),enforce_stationarity=False,enforce_invertibility=False).fit(disp=0)
axp={h:[] for h in HSET}; axo=[]
for o in range(split,len(data)-max(HSET),max(1,len(test_idx)//150)):
    r_o=res.apply(lp[:o],exog=ex[:o]); fx=np.tile(ex[o-1],(max(HSET),1))
    f=np.asarray(r_o.get_forecast(max(HSET),exog=fx).predicted_mean); base=data["price"].iloc[o]
    for h in HSET: axp[h].append(np.exp(f[h-1]))
    axo.append(o)
axo=np.array(axo)

# plot: actual vs predicted PRICE at target date, per horizon
fig,axs=plt.subplots(len(HSET),1,figsize=(14,9),sharex=True)
N=len(data)
for k,h in enumerate(HSET):
    ax=axs[k]
    m=(test_idx+h)<N; ti=test_idx[m]
    ax.plot(data["date"].iloc[ti+h].values, data["price"].iloc[ti+h].values,color="black",lw=1.8,label="Actual HNB price")
    ax.plot(data["date"].iloc[ti+h].values, data["price"].iloc[ti].values,color="gray",lw=1.1,ls="--",alpha=.9,label="Naive (last price)")
    ax.plot(data["date"].iloc[ti+h].values, data["price"].iloc[ti].values*(1+rfp[m,k]),color="tab:green",lw=1,alpha=.8,label="RandomForest")
    mc2=(cl_idx+h)<N; ci=cl_idx[mc2]
    ax.plot(data["date"].iloc[ci+h].values, data["price"].iloc[ci].values*(1+clp[mc2,k]),color="tab:red",lw=1,alpha=.8,label="CNN-LSTM")
    ma=(axo+h)<N
    ax.plot(data["date"].iloc[axo[ma]+h].values, np.array(axp[h])[ma],color="tab:orange",lw=1.2,label="ARIMAX")
    ax.set_title(f"HNB — actual vs predicted price, {h}-day ahead"); ax.set_ylabel("Price"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"HNB_price_overlay.png",dpi=140)
print("saved HNB_price_overlay.png")
