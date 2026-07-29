#!/usr/bin/env python3
"""
Group correlation test: do banks and non-bank finance companies move as distinct
clusters? Compares within-group vs between-group daily-return correlations on the CSE.

If BANKS-vs-FINANCE correlation is clearly LOWER than within-BANKS and within-FINANCE,
the groups behave separately and the "they respond differently to the same events"
hypothesis is worth pursuing.
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANKS = ["HNB", "COMB", "SAMP"]
FINANCE = ["LOFC", "LOLC"]
CONTROL = ["JKH", "DIAL"]
ALL = BANKS + FINANCE + CONTROL
START = "2012-01-01"

# 1-2. load each company, filter dates, keep close
series = {}
for sym in ALL:
    df = pd.read_csv(f"cleaned_data/{sym}_daily_clean.csv", parse_dates=["date"])
    df = df[df.date >= START][["date", "close"]].rename(columns={"close": sym})
    series[sym] = df.set_index("date")[sym]

# 3. join on date, keep only days present for EVERY company
prices = pd.concat(series.values(), axis=1, join="inner").dropna()
prices = prices[ALL]  # fix column order for a clean grouped heatmap

# 4. daily percentage returns
rets = prices.pct_change().dropna()

# 5. full correlation matrix
corr = rets.corr()
print("=" * 60)
print("Daily-return correlation matrix")
print("=" * 60)
print(corr.round(3).to_string())

# 6. average pairwise correlations (exclude self-pairs)
def avg_within(group):
    return float(np.mean([corr.loc[a, b] for a, b in itertools.combinations(group, 2)]))

def avg_between(g1, g2):
    return float(np.mean([corr.loc[a, b] for a in g1 for b in g2]))

wb = avg_within(BANKS)
wf = avg_within(FINANCE)
bf = avg_between(BANKS, FINANCE)
bc = avg_between(BANKS, CONTROL)

print("\n" + "=" * 60)
print("Average pairwise correlations")
print("=" * 60)
print(f"within BANKS       {wb:.3f}")
print(f"within FINANCE     {wf:.3f}")
print(f"BANKS vs FINANCE   {bf:.3f}   <- the key number")
print(f"BANKS vs CONTROL   {bc:.3f}")

# 7. coverage
print("\n" + "=" * 60)
print(f"Common trading days: {len(prices)}")
print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")
print(f"(returns used: {len(rets)} rows)")

# verdict
sep = bf < min(wb, wf)
print("\nVERDICT:", "SEPARATE CLUSTERS — hypothesis worth pursuing"
      if sep else "SIMILAR — groups not clearly distinct")
print(f"  (BANKS-vs-FINANCE {bf:.3f} is {'LOWER' if sep else 'NOT lower'} than "
      f"within-BANKS {wb:.3f} and within-FINANCE {wf:.3f})")

# save matrix for thesis
corr.round(4).to_csv("group_correlation_matrix.csv")
print("\nSaved group_correlation_matrix.csv")

# heatmap
fig, ax = plt.subplots(figsize=(8, 7))
off = corr.values[~np.eye(len(corr), dtype=bool)]
im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=off.min(), vmax=1.0)
ax.set_xticks(range(len(ALL))); ax.set_xticklabels(ALL, rotation=0)
ax.set_yticks(range(len(ALL))); ax.set_yticklabels(ALL)
for i in range(len(ALL)):
    for j in range(len(ALL)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                fontsize=10, color="black")
# group separator lines (BANKS | FINANCE | CONTROL)
for b in (2.5, 4.5):
    ax.axhline(b, color="k", lw=1.5); ax.axvline(b, color="k", lw=1.5)
ax.set_title("CSE daily-return correlation\n(BANKS: HNB/COMB/SAMP | FINANCE: LOFC/LOLC | CONTROL: JKH/DIAL)",
             fontsize=11)
fig.colorbar(im, ax=ax, label="correlation")
fig.tight_layout()
fig.savefig("group_correlation_heatmap.png", dpi=140, bbox_inches="tight")
print("Saved group_correlation_heatmap.png")
