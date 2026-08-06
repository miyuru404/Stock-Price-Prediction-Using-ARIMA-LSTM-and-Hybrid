#!/usr/bin/env python3
"""
PURE blind forecast (the ONLY allowed protocol):
train on first 80% of prices, then predict the ENTIRE 20% test window in one shot.
NO test-period data is ever fed to the model — not future prices, not "today's" price.
  - ARIMA(1,1,1) on log price: native multi-step forecast of full test length.
  - RandomForest: RECURSIVE — predicts 1-day return, rolls its OWN prediction forward,
    recomputes technical features from the predicted series, repeats to end of test.
  - Combo = mean(ARIMA, RF).  Naive = flat line at last training price.
Usage: python src/pure_forecast.py HNB   |   python src/pure_forecast.py AAPL
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path as _Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=_Path(__file__).resolve().parents[1]; DATA=ROOT/"cleaned_data"; OUT=ROOT/"results"/"mh_models"
SEED=42; np.random.seed(SEED)
T=(sys.argv[1] if len(sys.argv)>1 else "HNB").upper()

# ---- load price ----
if T=="AAPL":
    d=pd.read_csv("/Users/miyuru/Downloads/BATS_AAPL, 1D.csv"); d["date"]=pd.to_datetime(d["time"],unit="s").dt.normalize()
    d=d[d["date"]>="2000-01-01"].sort_values("date").reset_index(drop=True)
elif T in ("SPSL20","SP20"):
    d=pd.read_csv(ROOT/"data/processed/spsl20_trading_days_clean.csv",parse_dates=["date"]).rename(columns={"spsl20_points":"close"})
else:
    d=pd.read_csv(DATA/f"{T}_daily_clean.csv",parse_dates=["date"]).sort_values("date").reset_index(drop=True)
price=d["close"].astype(float).reset_index(drop=True); date=d["date"].reset_index(drop=True)
N=len(price); split=int(N*0.8); ntest=N-split
print(f"{T}: total {N} | train {split} | test(blind) {ntest} | {date.iloc[split].date()} -> {date.iloc[-1].date()}")

# ---- features from a price series (technical only, blind-safe) ----
def feat_row(p):
    p=pd.Series(p); r1=p.pct_change()
    def rsi(s,n=14):
        dl=s.diff(); up=dl.clip(lower=0).rolling(n).mean(); dn=(-dl.clip(upper=0)).rolling(n).mean()
        return (100-100/(1+up/dn.replace(0,np.nan))).iloc[-1]
    return np.array([r1.iloc[-1], p.pct_change(5).iloc[-1], p.pct_change(10).iloc[-1], rsi(p),
        p.iloc[-1]/p.rolling(5).mean().iloc[-1]-1, p.iloc[-1]/p.rolling(10).mean().iloc[-1]-1,
        p.iloc[-1]/p.rolling(20).mean().iloc[-1]-1, p.iloc[-1]/p.iloc[-11]-1,
        r1.rolling(10).std().iloc[-1], r1.rolling(20).std().iloc[-1]])

# ---- build training set for a 1-day-ahead RF ----
tr=price.iloc[:split].reset_index(drop=True)
X,Y=[],[]
for i in range(25,len(tr)-1):
    X.append(feat_row(tr.iloc[:i+1].values)); Y.append(tr.iloc[i+1]/tr.iloc[i]-1)
X=np.array(X); Y=np.array(Y)
rf=RandomForestRegressor(300,max_depth=8,min_samples_leaf=5,random_state=SEED,n_jobs=4).fit(X,Y)

# ---- RF recursive blind forecast over whole test ----
work=list(tr.values); rf_path=[]
for _ in range(ntest):
    ret=rf.predict(feat_row(np.array(work)).reshape(1,-1))[0]
    nxt=work[-1]*(1+ret); rf_path.append(nxt); work.append(nxt)
rf_path=np.array(rf_path)

# ---- ARIMA native multi-step ----
lp=np.log(tr.values)
ar=SARIMAX(lp,order=(1,1,1),enforce_stationarity=False,enforce_invertibility=False).fit(disp=0)
ar_path=np.exp(np.asarray(ar.get_forecast(ntest).predicted_mean))

# ---- naive flat + combo ----
naive=np.full(ntest, tr.values[-1])
combo=(ar_path+rf_path)/2
actual=price.iloc[split:].values; td=date.iloc[split:].values

def mape(p): return np.mean(np.abs(actual-p)/actual)*100
print("\n=== PURE blind forecast — MAPE over WHOLE test window (%) ===")
for nm,p in [("Naive(flat)",naive),("ARIMA",ar_path),("RandomForest",rf_path),("Combo(AR+RF)",combo)]:
    print(f"  {nm:16} {mape(p):6.2f}")
# milestone errors (single point each, since one launch)
print("\n=== error at milestones into the test (|pred-actual|/actual %) ===")
mil=[("1mo",22),("3mo",66),("6mo",132),("1yr",252)]
hdr="  {:16}"+"{:>8}"*len(mil); print(hdr.format("model",*[m for m,_ in mil]))
for nm,p in [("Naive(flat)",naive),("ARIMA",ar_path),("RandomForest",rf_path),("Combo(AR+RF)",combo)]:
    vals=[f"{abs(p[k-1]-actual[k-1])/actual[k-1]*100:6.1f}" if k-1<ntest else "  -" for _,k in mil]
    print(("  {:16}"+"{:>8}"*len(mil)).format(nm,*vals))

# ---- plot ----
fig,ax=plt.subplots(figsize=(14,7))
ax.axvline(td[0],color="blue",ls=":",lw=1,label="train ends (launch)")
ax.plot(td,actual,color="black",lw=2,label="Actual price")
ax.plot(td,naive,color="gray",ls="--",lw=1.4,label="Naive (flat = last train price)")
ax.plot(td,ar_path,color="tab:orange",lw=1.6,label="ARIMA (blind)")
ax.plot(td,rf_path,color="tab:green",lw=1.6,label="RandomForest (recursive, blind)")
ax.plot(td,combo,color="tab:blue",lw=2.2,label="Combo AR+RF (blind)")
ax.set_title(f"{T} — PURE blind forecast: train 80%, predict whole 20% test in one shot\n(no test data ever fed to the model)")
ax.set_ylabel("Price"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/f"{T}_pure_forecast.png",dpi=140)
print(f"\nsaved {T}_pure_forecast.png")
