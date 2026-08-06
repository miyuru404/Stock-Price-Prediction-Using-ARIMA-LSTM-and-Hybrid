#!/usr/bin/env python3
"""
Real-price chart for the COMBO: actual price vs AVG2=mean(ARIMAX,RF) vs naive,
at 1-month and 6-month horizons, rolling origins across the test window.
Only ARIMAX + RandomForest trained (the two combo members). Blind, no peek.
Usage: python src/mh_ensemble_price_chart.py   (does HNB and JKH)
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path as _Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=_Path(__file__).resolve().parents[1]; DATA=ROOT/"cleaned_data"; OUT=ROOT/"results"/"mh_models"
SEED=42; np.random.seed(SEED)
H=[1,2,3,4,5,10,22,44,66,132,252]; HSHOW=[22,132]; HLAB={22:"1 month (22d)",132:"6 months (132d)"}
TICK=["HNB","JKH"]

def load(T):
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
    fcs=list(F.columns)
    for h in H: F[f"y{h}"]=c.shift(-h)/c-1
    F["price"]=c.values; F["date"]=d["date"].values
    data=F.dropna().reset_index(drop=True)
    return data,fcs,macro

fig,axs=plt.subplots(len(TICK),len(HSHOW),figsize=(17,10))
for ri,T in enumerate(TICK):
    data,fcs,macro=load(T); split=int(len(data)*0.8)
    Xtr=data[fcs].iloc[:split].values; Ytr=data[[f"y{h}" for h in H]].iloc[:split].values
    rf=RandomForestRegressor(300,max_depth=8,min_samples_leaf=5,random_state=SEED,n_jobs=4).fit(Xtr,Ytr)
    lp=np.log(data["price"].values); ex=data[macro].values
    ar=SARIMAX(lp[:split],exog=ex[:split],order=(1,1,1),enforce_stationarity=False,enforce_invertibility=False).fit(disp=0)
    N=len(data); stride=max(1,(N-max(H)-split)//200); origins=list(range(split,N-1,stride))
    rf_all=rf.predict(data[fcs].iloc[origins].values)                 # (no x nH)
    ax_ret={h:{} for h in H}
    combo_price={h:[] for h in HSHOW}; naive_price={h:[] for h in HSHOW}; act_price={h:[] for h in HSHOW}; tdate={h:[] for h in HSHOW}
    for k,o in enumerate(origins):
        r_o=ar.apply(lp[:o],exog=ex[:o]); fex=np.tile(ex[o-1],(max(H),1))
        f=np.exp(np.asarray(r_o.get_forecast(max(H),exog=fex).predicted_mean)); base=data["price"].iloc[o]
        for h in HSHOW:
            if o+h>=N: continue
            ar_ret=f[h-1]/base-1; rf_ret=rf_all[k][H.index(h)]; avg=(ar_ret+rf_ret)/2
            combo_price[h].append(base*(1+avg)); naive_price[h].append(base)
            act_price[h].append(data["price"].iloc[o+h]); tdate[h].append(data["date"].iloc[o+h])
    for ci,h in enumerate(HSHOW):
        ax=axs[ri,ci]
        ax.plot(tdate[h],act_price[h],color="black",lw=2,label="Actual price")
        ax.plot(tdate[h],naive_price[h],color="gray",ls="--",lw=1.3,label="Naive (last price)")
        ax.plot(tdate[h],combo_price[h],color="tab:blue",lw=1.8,label="AVG2 combo (ARIMAX+RF)")
        ax.set_title(f"{T} — actual vs combo, {HLAB[h]} ahead"); ax.set_ylabel("Price"); ax.grid(alpha=.3); ax.legend()
    print(f"{T} done")
fig.suptitle("Real price vs COMBO forecast (average of ARIMAX + RandomForest) vs naive",fontsize=13,y=1.01)
fig.tight_layout(); fig.savefig(OUT/"ensemble_price_chart.png",dpi=140,bbox_inches="tight")
print("saved ensemble_price_chart.png")
