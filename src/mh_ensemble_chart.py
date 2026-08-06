#!/usr/bin/env python3
"""Comparison chart: Naive vs single models vs AVG2 combo, per horizon, for HNB and JKH."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path as _Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=_Path(__file__).resolve().parents[1]; OUT=ROOT/"results"/"mh_models"
HLAB=["1d","2d","3d","4d","5d","2w","1mo","2mo","3mo","6mo","1yr"]
TICK=["HNB","JKH"]
show={"Naive":("gray","--"),"ARIMAX":("tab:orange","-"),"RandomForest":("tab:green","-"),"AVG2(AR+RF)":("tab:blue","-")}

fig,axs=plt.subplots(1,2,figsize=(16,6),sharey=False)
for ax,T in zip(axs,TICK):
    tab=pd.read_csv(OUT/f"{T}_ensemble.csv",index_col=0)
    x=range(len(HLAB))
    for m,(cl,ls) in show.items():
        lw=2.6 if m=="AVG2(AR+RF)" else 1.6
        ax.plot(x, tab.loc[m,HLAB].values, ls, color=cl, lw=lw, marker="o", ms=5,
                label=("AVG2 = mean(ARIMAX,RF)  <-- combo" if m=="AVG2(AR+RF)" else m))
    # shade where AVG2 beats naive
    nv=tab.loc["Naive",HLAB].values; av=tab.loc["AVG2(AR+RF)",HLAB].values
    for i in range(len(HLAB)):
        if av[i]<nv[i]-1e-9: ax.axvspan(i-0.4,i+0.4,color="tab:blue",alpha=0.06)
    ax.set_xticks(list(x)); ax.set_xticklabels(HLAB)
    ax.set_title(f"{T} — price error (MAPE %) vs horizon\nblue band = combo beats naive"); ax.set_ylabel("MAPE %  (lower = better)")
    ax.grid(alpha=.3); ax.legend()
fig.suptitle("Combining models: mean(ARIMAX, RandomForest) vs the singles vs naive", fontsize=13, y=1.02)
fig.tight_layout(); fig.savefig(OUT/"ensemble_compare.png",dpi=140,bbox_inches="tight")
print("saved ensemble_compare.png")
