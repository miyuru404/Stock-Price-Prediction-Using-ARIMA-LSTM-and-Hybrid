#!/usr/bin/env python3
"""
Bull / Bear regime detector (honest first version, rule-based).
TRUTH (hindsight answer key): standard 20% rule — a bear starts after a 20% drop from a
peak and ends after a 20% rise from the trough. Uses full series (it's the answer key).
DETECTOR (real-time, PAST DATA ONLY at each day): combines 3 signals —
  1) price below 200-day moving average (trend)
  2) price >15% below its 1-year high (drawdown)
  3) 6-month momentum negative
  -> BEAR if >=2 of 3 fire, else BULL.
Baselines: "always BULL" (base rate) and "200-day MA only".
Scored on accuracy + how many true BEAR days it catches (recall). No hindsight in detector.
Usage: python src/regime_detect.py ASPI   |   HNB | AAPL
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path as _Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=_Path(__file__).resolve().parents[1]; DATA=ROOT/"cleaned_data"; OUT=ROOT/"results"/"regime"; OUT.mkdir(parents=True,exist_ok=True)
T=(sys.argv[1] if len(sys.argv)>1 else "ASPI").upper()
if T=="AAPL":
    d=pd.read_csv("/Users/miyuru/Downloads/BATS_AAPL, 1D.csv"); d["date"]=pd.to_datetime(d["time"],unit="s").dt.normalize()
    d=d[d["date"]>="2000-01-01"].sort_values("date").reset_index(drop=True)
elif T in ("SPSL20","SP20"):
    d=pd.read_csv(ROOT/"data/processed/spsl20_trading_days_clean.csv",parse_dates=["date"]).rename(columns={"spsl20_points":"close"})
else:
    d=pd.read_csv(DATA/f"{T}_daily_clean.csv",parse_dates=["date"]).sort_values("date").reset_index(drop=True)
p=d["close"].astype(float).values; date=pd.to_datetime(d["date"]).values; N=len(p)

# ---- TRUTH: 20% peak/trough bull-bear dating (hindsight answer key) ----
truth=np.empty(N,dtype=object); state="bull"; peak=p[0]; trough=p[0]
for i,x in enumerate(p):
    if state=="bull":
        peak=max(peak,x)
        if x<=peak*0.80: state="bear"; trough=x
    else:
        trough=min(trough,x)
        if x>=trough*1.20: state="bull"; peak=x
    truth[i]=state
truth_bear=(truth=="bear")

# ---- DETECTOR: real-time, past data only ----
s=pd.Series(p)
ma200=s.rolling(200).mean().values
hi1y=s.rolling(252).max().values
mom6=np.concatenate([np.full(120,np.nan), p[120:]-p[:-120]])
sig_trend=p<ma200
sig_draw=p<0.85*hi1y
sig_mom=mom6<0
sig_count=sig_trend.astype(float)+sig_draw.astype(float)+sig_mom.astype(float)
det_bear=sig_count>=2

# valid region: where all signals exist
valid=~np.isnan(ma200)&~np.isnan(hi1y)&~np.isnan(mom6)
def scores(pred_bear):
    tb=truth_bear[valid]; pb=pred_bear[valid]
    acc=np.mean(tb==pb)*100
    rec=np.mean(pb[tb]) *100 if tb.sum()>0 else np.nan       # of true bear days, % flagged
    prec=np.mean(tb[pb])*100 if pb.sum()>0 else np.nan       # of flagged days, % truly bear
    return acc,rec,prec

always_bull=np.zeros(N,dtype=bool)
ma_only=p<ma200
print(f"{T}: days {N} | true BEAR days {truth_bear[valid].sum()} ({truth_bear[valid].mean()*100:.1f}%) | BULL base rate {(1-truth_bear[valid].mean())*100:.1f}%")
print(f"\n{'model':16}{'accuracy':>10}{'bear recall':>13}{'bear precision':>16}")
for nm,pb in [("Always BULL",always_bull),("MA200 only",ma_only),("Combined (3-sig)",det_bear)]:
    a,r,pr=scores(pb); print(f"{nm:16}{a:9.1f}%{('' if np.isnan(r) else f'{r:11.1f}%')}{('' if np.isnan(pr) else f'{pr:14.1f}%')}")

# ---- plot ----
fig,(a1,a2)=plt.subplots(2,1,figsize=(15,8),sharex=True,height_ratios=[3,1])
a1.plot(date,p,color="black",lw=1.3,label="Price")
def shade(ax,mask,color,label):
    inb=False
    for i in range(N):
        if mask[i] and not inb: st=i; inb=True
        if (not mask[i] or i==N-1) and inb:
            ax.axvspan(date[st],date[i],color=color,alpha=0.25,lw=0); inb=False
    ax.plot([],[],color=color,alpha=.5,lw=8,label=label)
shade(a1,truth_bear,"red","TRUE bear (20% rule, hindsight)")
a1.plot(date,ma200,color="tab:blue",lw=0.9,alpha=.7,label="200-day MA")
a1.set_title(f"{T} — Bull/Bear regime: price + true bear periods (red) vs detector"); a1.set_ylabel("Price"); a1.legend(loc="upper left"); a1.grid(alpha=.3)
shade(a2,det_bear,"darkred","DETECTOR bear (real-time, 2+ signals)")
a2.set_yticks([]); a2.set_title("Detector's real-time BEAR calls (no hindsight)"); a2.legend(loc="upper left"); a2.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT/f"{T}_regime.png",dpi=140)
print(f"\nsaved results/regime/{T}_regime.png")
