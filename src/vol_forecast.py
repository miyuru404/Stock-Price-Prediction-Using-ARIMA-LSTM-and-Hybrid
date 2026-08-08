#!/usr/bin/env python3
"""
VOLATILITY forecasting (new phase). Question: can we predict how WILD the price is
(volatility), even though we can't predict the price itself?

Target truth = realized daily volatility (rolling 10-day std of daily returns).
Models (predict daily volatility):
  Constant   : flat = std of TRAIN returns (the naive baseline for vol)
  EWMA(0.94) : RiskMetrics — updates with each realized return (daily-updated)
  GARCH-blind: GARCH(1,1) fit on train, forecast WHOLE test vol in one shot (pure blind = user's rule)
  GARCH-daily: GARCH(1,1) conditional vol filtered through test (daily-updated, standard risk practice)
Skill = does the model track realized vol better than the flat Constant?
Usage: python src/vol_forecast.py HNB   |   AAPL
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path as _Path
import numpy as np, pandas as pd
from arch import arch_model
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=_Path(__file__).resolve().parents[1]; DATA=ROOT/"cleaned_data"; OUT=ROOT/"results"/"volatility"; OUT.mkdir(parents=True,exist_ok=True)
T=(sys.argv[1] if len(sys.argv)>1 else "HNB").upper()

if T=="AAPL":
    d=pd.read_csv("/Users/miyuru/Downloads/BATS_AAPL, 1D.csv"); d["date"]=pd.to_datetime(d["time"],unit="s").dt.normalize()
    d=d[d["date"]>="2000-01-01"].sort_values("date").reset_index(drop=True)
elif T in ("SPSL20","SP20"):
    d=pd.read_csv(ROOT/"data/processed/spsl20_trading_days_clean.csv",parse_dates=["date"]).rename(columns={"spsl20_points":"close"})
else:
    d=pd.read_csv(DATA/f"{T}_daily_clean.csv",parse_dates=["date"]).sort_values("date").reset_index(drop=True)
price=d["close"].astype(float).reset_index(drop=True); date=d["date"].reset_index(drop=True)
ret=100*np.log(price/price.shift(1))                       # daily % log-return (arch likes %-scale)
ret=ret.dropna().reset_index(drop=True); date=date.iloc[1:].reset_index(drop=True)
N=len(ret); split=int(N*0.8)
print(f"{T}: returns {N} | train {split} | test {N-split}")

# realized-vol truth = rolling 10-day std of daily returns (annualise-free, in %/day)
realized=ret.rolling(10).std()
tr_ret=ret.iloc[:split]

# --- Constant baseline (flat train vol) ---
const=np.full(N-split, tr_ret.std())

# --- EWMA RiskMetrics lambda=0.94 (daily-updated) ---
lam=0.94; var=tr_ret.var(); ewma=[]
for i in range(N):
    if i>0: var=lam*var+(1-lam)*ret.iloc[i-1]**2
    if i>=split: ewma.append(np.sqrt(var))
ewma=np.array(ewma)

# --- GARCH(1,1) ---
am=arch_model(tr_ret, vol="GARCH", p=1, q=1, mean="Constant", dist="normal")
res=am.fit(disp="off")
# blind: analytic multi-step forecast of variance for whole test horizon
fc=res.forecast(horizon=N-split, reindex=False)
garch_blind=np.sqrt(fc.variance.values[0])                 # length = N-split
# daily-updated: conditional vol using full series params from train, filtered through test
am_full=arch_model(ret, vol="GARCH", p=1, q=1, mean="Constant", dist="normal")
cv=am_full.fix(res.params).conditional_volatility          # in-sample-style filter w/ TRAIN params (uses realized test returns to update recursion)
garch_daily=cv[split:]

truth=realized.iloc[split:].values; td=date.iloc[split:].values
def score(p):
    m=~np.isnan(truth)
    mae=np.mean(np.abs(p[m]-truth[m])); corr=np.corrcoef(p[m],truth[m])[0,1]
    return mae,corr
print("\n=== VOLATILITY forecast skill (vs realized 10-day vol) ===")
print(f"  {'model':14}{'MAE':>8}{'corr w/ realized':>20}")
rows={"Constant(flat)":const,"EWMA(0.94)":ewma,"GARCH-blind":garch_blind,"GARCH-daily":garch_daily}
for nm,p in rows.items():
    mae,corr=score(p); print(f"  {nm:14}{mae:8.3f}{corr:20.3f}")

fig,ax=plt.subplots(figsize=(14,6.5))
ax.plot(td,truth,color="black",lw=1.6,label="Realized volatility (truth)")
ax.plot(td,const,color="gray",ls="--",lw=1.3,label="Constant (flat train vol)")
ax.plot(td,garch_blind,color="tab:orange",lw=1.4,label="GARCH blind (one-shot)")
ax.plot(td,garch_daily,color="tab:blue",lw=1.4,label="GARCH daily-updated")
ax.set_title(f"{T} — VOLATILITY forecast: can we predict how wild the price is?\ntruth = 10-day realized vol; skill = tracking it + high correlation")
ax.set_ylabel("daily volatility (%)"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/f"{T}_vol_forecast.png",dpi=140)
print(f"\nsaved results/volatility/{T}_vol_forecast.png")
