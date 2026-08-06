#!/usr/bin/env python3
"""
Same multi-horizon test as the CSE work, applied to APPLE (AAPL), 2000 onward.
NO macro (Sri Lankan rates don't apply) -> ARIMAX becomes plain ARIMA. Technical
features only. Models: ARIMA, RandomForest, CNN-LSTM + AVG2=mean(ARIMA,RF) combo.
Blind rolling-origin, no-peek, 80/20. Reports direction + price MAPE vs naive,
saves scorecard, decay chart, ensemble MAPE chart, real-price overlay.
Usage: python src/mh_apple.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path as _Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=_Path(__file__).resolve().parents[1]; OUT=ROOT/"results"/"mh_models"; OUT.mkdir(parents=True,exist_ok=True)
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
H=[1,2,3,4,5,10,22,44,66,132,252]; HLAB=["1d","2d","3d","4d","5d","2w","1mo","2mo","3mo","6mo","1yr"]; WIN=20
CSV="/Users/miyuru/Downloads/BATS_AAPL, 1D.csv"; T="AAPL"

d=pd.read_csv(CSV); d["date"]=pd.to_datetime(d["time"],unit="s").dt.normalize()
d=d[d["date"]>="2000-01-01"].sort_values("date").reset_index(drop=True)
c=d["close"].astype(float); vol=d["Volume"].astype(float)
r1=c.pct_change()
def rsi(s,n=14):
    dl=s.diff(); up=dl.clip(lower=0).rolling(n).mean(); dn=(-dl.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))
F=pd.DataFrame(index=d.index)
F["ret1"]=r1;F["ret5"]=c.pct_change(5);F["ret10"]=c.pct_change(10);F["rsi"]=rsi(c)
F["ma5"]=c/c.rolling(5).mean()-1;F["ma10"]=c/c.rolling(10).mean()-1;F["ma20"]=c/c.rolling(20).mean()-1
F["mom10"]=c/c.shift(10)-1;F["vol10"]=r1.rolling(10).std();F["vol20"]=r1.rolling(20).std()
F["volchg"]=vol/vol.rolling(5).mean()-1
fcs=list(F.columns)
for h in H: F[f"y{h}"]=c.shift(-h)/c-1
F["price"]=c.values; F["date"]=d["date"].values
data=F.dropna().reset_index(drop=True); split=int(len(data)*0.8)
print(f"{T}: rows {len(data)} | train {split} test {len(data)-split} | features {len(fcs)} | {data['date'].iloc[0].date()} -> {data['date'].iloc[-1].date()}")
Xtr=data[fcs].iloc[:split].values; Ytr=data[[f"y{h}" for h in H]].iloc[:split].values
test_idx=np.arange(split,len(data))

def metrics(pred,h,idx):
    price_t=data["price"].iloc[idx].values; act=data["price"].iloc[idx].values*(1+data[f"y{h}"].iloc[idx].values)
    pp=price_t*(1+pred); ar=data[f"y{h}"].iloc[idx].values; m=np.abs(ar)>1e-9
    return np.mean(np.sign(pred[m])==np.sign(ar[m]))*100, np.mean(np.abs(act-pp)/act)*100

# RF
rf=RandomForestRegressor(300,max_depth=8,min_samples_leaf=5,random_state=SEED,n_jobs=4).fit(Xtr,Ytr)
# CNN-LSTM
allX=data[fcs].values.astype(np.float32); mu,sd=allX[:split].mean(0),allX[:split].std(0)+1e-8; Xs=(allX-mu)/sd
Xw=np.array([Xs[i-WIN:i] for i in range(WIN,len(Xs))],np.float32); Yw=data[[f"y{h}" for h in H]].values[WIN:].astype(np.float32); spw=split-WIN
class Net(nn.Module):
    def __init__(s,nf,no):
        super().__init__(); s.cv=nn.Conv1d(nf,32,3,padding=1); s.rl=nn.ReLU(); s.ls=nn.LSTM(32,50,batch_first=True); s.h=nn.Linear(50,no)
    def forward(s,x): z=s.rl(s.cv(x.transpose(1,2))).transpose(1,2); o,_=s.ls(z); return s.h(o[:,-1,:])
net=Net(len(fcs),len(H)); opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.MSELoss()
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
# ARIMA (no exog)
lp=np.log(data["price"].values)
ar=SARIMAX(lp[:split],order=(1,1,1),enforce_stationarity=False,enforce_invertibility=False).fit(disp=0)

# rolling origins (shared)
stride=max(1,(len(data)-max(H)-split)//150); origins=np.array(list(range(split,len(data)-max(H),stride)))
rf_o=rf.predict(data[fcs].iloc[origins].values)                      # (no x nH)
cl_o=net(torch.tensor(np.array([Xs[o-WIN:o] for o in origins],np.float32))).detach().numpy()
ar_o=np.full((len(origins),len(H)),np.nan)
for k,o in enumerate(origins):
    r_o=ar.apply(lp[:o]); f=np.exp(np.asarray(r_o.get_forecast(max(H)).predicted_mean)); base=data["price"].iloc[o]
    ar_o[k]=[f[h-1]/base-1 for h in H]

def mape_h(pred):  # pred (no x nH) returns
    out=[]
    for j,h in enumerate(H):
        base=data["price"].iloc[origins].values; act=data["price"].iloc[origins+h].values
        out.append(np.mean(np.abs(act-base*(1+pred[:,j]))/act)*100)
    return np.array(out)
def dir_h(pred):
    out=[]
    for j,h in enumerate(H):
        ar_=data[f"y{h}"].iloc[origins].values; m=np.abs(ar_)>1e-9
        out.append(np.mean(np.sign(pred[m,j])==np.sign(ar_[m]))*100)
    return np.array(out)

naive=np.zeros((len(origins),len(H)))
avg2=(ar_o+rf_o)/2; avg3=(ar_o+rf_o+cl_o)/3
MAPE=pd.DataFrame({"Naive":mape_h(naive),"ARIMA":mape_h(ar_o),"RandomForest":mape_h(rf_o),
                   "CNN-LSTM":mape_h(cl_o),"AVG2(ARIMA+RF)":mape_h(avg2),"AVG3(all)":mape_h(avg3)},index=HLAB).T
DIRt=pd.DataFrame({"Naive":[max(np.mean(data[f'y{h}'].iloc[origins]>0),np.mean(data[f'y{h}'].iloc[origins]<0))*100 for h in H],
                   "ARIMA":dir_h(ar_o),"RandomForest":dir_h(rf_o),"CNN-LSTM":dir_h(cl_o)},index=HLAB).T
print("\n=== PRICE MAPE (%) vs horizon ==="); print(MAPE.round(2).to_string())
nv=MAPE.loc["Naive"].values
print("\nHorizons where each BEATS naive (of 11):")
for m in ["ARIMA","RandomForest","CNN-LSTM","AVG2(ARIMA+RF)","AVG3(all)"]:
    beat=MAPE.loc[m].values<nv-1e-9; nlong=beat[6:].sum()
    print(f"  {m:15} {beat.sum()}/11   (long 1mo-1yr: {nlong}/5)")
print("\n=== DIRECTION ACCURACY (%) ==="); print(DIRt.round(1).to_string())
MAPE.round(3).to_csv(OUT/f"{T}_ensemble.csv")

# ---- chart 1: MAPE decay ----
fig,ax=plt.subplots(figsize=(11,6))
for m,cl,lw in [("Naive","gray",1.6),("ARIMA","tab:orange",1.6),("RandomForest","tab:green",1.6),("AVG2(ARIMA+RF)","tab:blue",2.6)]:
    ax.plot(range(len(HLAB)),MAPE.loc[m].values,"o-",color=cl,lw=lw,label=m)
for i in range(len(HLAB)):
    if MAPE.loc["AVG2(ARIMA+RF)"].values[i]<nv[i]-1e-9: ax.axvspan(i-.4,i+.4,color="tab:blue",alpha=.06)
ax.set_xticks(range(len(HLAB))); ax.set_xticklabels(HLAB); ax.set_title(f"{T} (Apple, 2000+) — price MAPE vs horizon\nblue band = combo beats naive"); ax.set_ylabel("MAPE %"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/f"{T}_ensemble_chart.png",dpi=140)

# ---- chart 2: real price overlay (1mo, 6mo) ----
fig,axs=plt.subplots(1,2,figsize=(16,6))
for ci,h in enumerate([22,132]):
    ax=axs[ci]; base=data["price"].iloc[origins].values; tgt=data["price"].iloc[origins+h].values; td=data["date"].iloc[origins+h].values
    ax.plot(td,tgt,color="black",lw=2,label="Actual price")
    ax.plot(td,base,color="gray",ls="--",lw=1.3,label="Naive (last price)")
    ax.plot(td,base*(1+avg2[:,H.index(h)]),color="tab:blue",lw=1.8,label="AVG2 combo (ARIMA+RF)")
    ax.set_title(f"AAPL — actual vs combo, {'1 month' if h==22 else '6 months'} ahead"); ax.set_ylabel("Price ($)"); ax.grid(alpha=.3); ax.legend()
fig.suptitle("Apple: real price vs combo vs naive",y=1.02); fig.tight_layout(); fig.savefig(OUT/f"{T}_price_overlay.png",dpi=140,bbox_inches="tight")
print(f"\nsaved {T}_ensemble.csv + {T}_ensemble_chart.png + {T}_price_overlay.png")
