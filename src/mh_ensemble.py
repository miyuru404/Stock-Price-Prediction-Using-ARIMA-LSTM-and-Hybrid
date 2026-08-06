#!/usr/bin/env python3
"""
Ensemble test: does COMBINING models beat the single models and naive?
Same blind, rolling-origin, no-peek protocol. At each origin every model predicts
the return over each horizon; we then form combos and score price-MAPE per horizon.

Combos (all leak-free — weights decided WITHOUT the test set):
  AVG2  = mean(ARIMAX, RF)
  AVG3  = mean(ARIMAX, RF, CNN-LSTM)
  SWITCH= per-horizon pick the model that was best on a TRAIN-tail validation slice
Usage: python src/mh_ensemble.py HNB
"""
import sys, warnings
from pathlib import Path as _Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch, torch.nn as nn

ROOT=_Path(__file__).resolve().parents[1]; DATA=ROOT/"cleaned_data"; OUT=ROOT/"results"/"mh_models"
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
H=[1,2,3,4,5,10,22,44,66,132,252]; HLAB=["1d","2d","3d","4d","5d","2w","1mo","2mo","3mo","6mo","1yr"]; WIN=20
T=(sys.argv[1] if len(sys.argv)>1 else "HNB").upper()

if T in ("SPSL20","SP20"):
    d=pd.read_csv(ROOT/"data/processed/spsl20_trading_days_clean.csv",parse_dates=["date"]).rename(columns={"spsl20_points":"close"}); d["volume"]=np.nan
else:
    d=pd.read_csv(DATA/f"{T}_daily_clean.csv",parse_dates=["date"])
d=d.sort_values("date").reset_index(drop=True); c=d["close"].astype(float)
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
if (d["volume"].fillna(0)>0).any(): F["volchg"]=d["volume"]/d["volume"].rolling(5).mean()-1
for x in macro: F[x]=d[x]
fcs=list(F.columns)
for h in H: F[f"y{h}"]=c.shift(-h)/c-1
F["price"]=c.values
data=F.dropna().reset_index(drop=True); split=int(len(data)*0.8)
Xtr=data[fcs].iloc[:split].values; Ytr=data[[f"y{h}" for h in H]].iloc[:split].values

rf=RandomForestRegressor(300,max_depth=8,min_samples_leaf=5,random_state=SEED,n_jobs=4).fit(Xtr,Ytr)
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
lp=np.log(data["price"].values); ex=data[macro].values
ar=SARIMAX(lp[:split],exog=ex[:split],order=(1,1,1),enforce_stationarity=False,enforce_invertibility=False).fit(disp=0)

def preds_at(origins):
    """return dict model-> (n_origins x n_H) predicted returns, aligned on origins."""
    P={m:np.full((len(origins),len(H)),np.nan) for m in ["ARIMAX","RandomForest","CNN-LSTM"]}
    for k,o in enumerate(origins):
        P["RandomForest"][k]=rf.predict(data[fcs].iloc[[o]].values)[0]
        P["CNN-LSTM"][k]=net(torch.tensor(Xs[o-WIN:o][None].astype(np.float32))).detach().numpy()[0]
        r_o=ar.apply(lp[:o],exog=ex[:o]); fex=np.tile(ex[o-1],(max(H),1))
        f=np.exp(np.asarray(r_o.get_forecast(max(H),exog=fex).predicted_mean))
        base=data["price"].iloc[o]; P["ARIMAX"][k]=np.array([f[h-1]/base-1 for h in H])
    return P

def mape_per_h(pred, origins):
    out=[]
    for j,h in enumerate(H):
        oi=np.array(origins); base=data["price"].iloc[oi].values
        act=data["price"].iloc[oi+h].values
        pp=base*(1+pred[:,j]); out.append(np.mean(np.abs(act-pp)/act)*100)
    return np.array(out)

# ---- validation origins (train tail) to DECIDE switch weights, no test peeking ----
val_lo=int(split*0.8); val_origins=list(range(val_lo, split-max(H), max(1,(split-max(H)-val_lo)//60)))
Pval=preds_at(val_origins)
val_mape={m:mape_per_h(Pval[m],val_origins) for m in Pval}
switch_pick=[min(["ARIMAX","RandomForest","CNN-LSTM"], key=lambda m: val_mape[m][j]) for j in range(len(H))]

# ---- test origins ----
stride=max(1,(len(data)-max(H)-split)//120); test_origins=list(range(split, len(data)-max(H), stride))
Pt=preds_at(test_origins)
naive=np.zeros((len(test_origins),len(H)))
avg2=(Pt["ARIMAX"]+Pt["RandomForest"])/2
avg3=(Pt["ARIMAX"]+Pt["RandomForest"]+Pt["CNN-LSTM"])/3
switch=np.column_stack([Pt[switch_pick[j]][:,j] for j in range(len(H))])

M={"Naive":naive,"ARIMAX":Pt["ARIMAX"],"RandomForest":Pt["RandomForest"],"CNN-LSTM":Pt["CNN-LSTM"],
   "AVG2(AR+RF)":avg2,"AVG3(all)":avg3,"SWITCH":switch}
tab=pd.DataFrame({m:mape_per_h(M[m],test_origins) for m in M}, index=HLAB).T
print(f"\n{T} — PRICE MAPE (%) by model/combo x horizon  (lower=better)")
print(tab.round(2).to_string())
nb=(tab<tab.loc["Naive"]-1e-9).sum(1)
print("\nHorizons where each BEATS naive (out of 11):")
print(nb.to_string())
print("\nSWITCH picked per horizon (chosen on validation, not test):")
print(dict(zip(HLAB,switch_pick)))
tab.round(3).to_csv(OUT/f"{T}_ensemble.csv")
print(f"\nsaved {T}_ensemble.csv")
