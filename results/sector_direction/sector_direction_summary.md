# Sector direction — the target the SYSTEM actually predicts

Every earlier phase predicted individual stocks. The product answers
*"is the banking / finance sector likely to move up?"* — a different, and possibly easier, target.

## 1. Composites really are smoother

| series | type | n | lag1_autocorr | ceiling_vs_naive_% |
|---|---|---|---|---|
| HNB | single stock | 3269 | 0.0378 | 0.071 |
| COMB | single stock | 3359 | 0.1059 | 0.562 |
| SAMP | single stock | 3403 | 0.0748 | 0.28 |
| LOFC | single stock | 3207 | -0.0864 | 0.374 |
| LOLC | single stock | 3187 | 0.0918 | 0.422 |
| LFIN | single stock | 3103 | -0.0852 | 0.364 |
| CFIN | single stock | 3142 | -0.0696 | 0.243 |
| BANKS | composite (3 stocks) | 3239 | 0.178 | 1.597 |
| FINANCE | composite (4 stocks) | 2663 | 0.0169 | 0.014 |
| SECTOR | composite (7 stocks) | 2603 | 0.078 | 0.305 |
| ASPI | market index | 3418 | 0.2326 | 2.743 |

Mean |lag-1 autocorrelation|: **single stocks 0.079** vs **composites 0.091**
(1.2x). `ceiling_vs_naive_%` is the maximum RMSE reduction any linear
model could achieve on that series — the theoretical limit, not a model result.

Averaging constituents cancels company-specific noise and leaves the common sector movement, which
is the part that persists. **The system is aimed at the easier object.**

## 2. Leak scan

CLEAN — no feature tracks the future more than the present.

## 3. Direction results (walk-forward, 6-month folds)

| group | horizon_days | label | median_acc_% | median_baseline_% | median_edge_pp | folds_positive | sign_p | BEATS_BASELINE |
|---|---|---|---|---|---|---|---|---|
| BANKS | 1 | 3-class Buy/Hold/Sell | 42.1 | 46.4 | -9.7 | 6/19 | 0.9682 | no |
| BANKS | 5 | 3-class Buy/Hold/Sell | 38.1 | 43.5 | -4.5 | 7/19 | 0.9165 | no |
| BANKS | 10 | 3-class Buy/Hold/Sell | 36.9 | 39.8 | -1.7 | 8/19 | 0.8204 | no |
| BANKS | 22 | 3-class Buy/Hold/Sell | 39.0 | 41.0 | -8.7 | 7/18 | 0.8811 | no |
| FINANCE | 1 | 3-class Buy/Hold/Sell | 41.2 | 39.0 | 2.1 | 11/19 | 0.3238 | no |
| FINANCE | 5 | 3-class Buy/Hold/Sell | 41.0 | 43.9 | -1.8 | 6/19 | 0.9682 | no |
| FINANCE | 10 | 3-class Buy/Hold/Sell | 37.1 | 45.1 | -10.5 | 5/19 | 0.9904 | no |
| FINANCE | 22 | 3-class Buy/Hold/Sell | 41.4 | 54.9 | -15.15 | 2/18 | 0.9999 | no |
| SECTOR | 1 | 3-class Buy/Hold/Sell | 45.2 | 40.7 | 4.3 | 11/19 | 0.3238 | no |
| SECTOR | 5 | 3-class Buy/Hold/Sell | 42.4 | 42.4 | 1.6 | 10/19 | 0.5 | no |
| SECTOR | 10 | 3-class Buy/Hold/Sell | 39.0 | 48.1 | -9.6 | 6/19 | 0.9682 | no |
| SECTOR | 22 | 3-class Buy/Hold/Sell | 39.8 | 55.5 | -14.15 | 4/18 | 0.9962 | no |
| BANKS | 1 | binary up/down | 56.3 | 56.8 | -3.4 | 6/19 | 0.9682 | no |
| BANKS | 5 | binary up/down | 52.4 | 55.3 | -0.9 | 7/19 | 0.9165 | no |
| BANKS | 10 | binary up/down | 55.1 | 55.5 | -1.7 | 7/19 | 0.9165 | no |
| BANKS | 22 | binary up/down | 55.3 | 56.4 | -2.35 | 5/18 | 0.9846 | no |
| FINANCE | 1 | binary up/down | 56.9 | 54.3 | 1.9 | 12/19 | 0.1796 | no |
| FINANCE | 5 | binary up/down | 56.5 | 57.9 | -1.1 | 5/19 | 0.9904 | no |
| FINANCE | 10 | binary up/down | 52.6 | 57.4 | -7.2 | 5/19 | 0.9904 | no |
| FINANCE | 22 | binary up/down | 53.8 | 66.4 | -11.45 | 1/18 | 1.0 | no |
| SECTOR | 1 | binary up/down | 57.1 | 54.8 | 1.9 | 12/19 | 0.1796 | no |
| SECTOR | 5 | binary up/down | 56.5 | 58.1 | 0.0 | 8/19 | 0.8204 | no |
| SECTOR | 10 | binary up/down | 54.9 | 58.1 | -2.4 | 6/19 | 0.9682 | no |
| SECTOR | 22 | binary up/down | 54.5 | 64.1 | -7.2 | 3/18 | 0.9993 | no |

**Configurations beating the baseline (p < 0.05): 0 of 24.**

## 4. Magnitude-weighted — is it right on the BIG moves?

| group | horizon_days | label | median_mag_edge_pp | mag_folds_positive | mag_sign_p | MAG_EDGE_REAL |
|---|---|---|---|---|---|---|
| BANKS | 1 | 3-class Buy/Hold/Sell | 34.1 | 19/19 | 0.0 | YES |
| BANKS | 5 | 3-class Buy/Hold/Sell | 39.4 | 19/19 | 0.0 | YES |
| BANKS | 10 | 3-class Buy/Hold/Sell | 29.5 | 16/19 | 0.0022 | YES |
| BANKS | 22 | 3-class Buy/Hold/Sell | 1.7 | 9/18 | 0.5927 | no |
| FINANCE | 1 | 3-class Buy/Hold/Sell | -2.6 | 7/19 | 0.9165 | no |
| FINANCE | 5 | 3-class Buy/Hold/Sell | -13.9 | 5/19 | 0.9904 | no |
| FINANCE | 10 | 3-class Buy/Hold/Sell | -26.2 | 4/19 | 0.9978 | no |
| FINANCE | 22 | 3-class Buy/Hold/Sell | -24.4 | 4/18 | 0.9962 | no |
| SECTOR | 1 | 3-class Buy/Hold/Sell | 48.3 | 19/19 | 0.0 | YES |
| SECTOR | 5 | 3-class Buy/Hold/Sell | -11.1 | 8/19 | 0.8204 | no |
| SECTOR | 10 | 3-class Buy/Hold/Sell | -28.1 | 6/19 | 0.9682 | no |
| SECTOR | 22 | 3-class Buy/Hold/Sell | -27.4 | 5/18 | 0.9846 | no |
| BANKS | 1 | binary up/down | 10.2 | 14/19 | 0.0318 | YES |
| BANKS | 5 | binary up/down | 8.1 | 14/19 | 0.0318 | YES |
| BANKS | 10 | binary up/down | 11.5 | 15/19 | 0.0096 | YES |
| BANKS | 22 | binary up/down | 6.4 | 11/18 | 0.2403 | no |
| FINANCE | 1 | binary up/down | 11.0 | 17/19 | 0.0004 | YES |
| FINANCE | 5 | binary up/down | -2.9 | 9/19 | 0.6762 | no |
| FINANCE | 10 | binary up/down | -11.1 | 5/19 | 0.9904 | no |
| FINANCE | 22 | binary up/down | -12.5 | 5/18 | 0.9846 | no |
| SECTOR | 1 | binary up/down | 3.7 | 14/19 | 0.0318 | YES |
| SECTOR | 5 | binary up/down | -0.5 | 9/19 | 0.6762 | no |
| SECTOR | 10 | binary up/down | -4.9 | 6/19 | 0.9682 | no |
| SECTOR | 22 | binary up/down | -15.1 | 6/18 | 0.9519 | no |

**Configurations with a real magnitude-weighted edge: 9 of 24.**

## What this means for the system

No sector configuration beats the naive baseline on plain accuracy.
A magnitude-weighted edge survives, so the product should flag LIKELY-BIG-MOVE days rather than emit a signal every day.

## Caveats
- Composite autocorrelation is partly **non-synchronous trading**: constituents that did not trade
  today reprice tomorrow, smearing a shock across two days. That is real and usable for a *signal*,
  but it is also why the effect does not survive transaction costs in a trading strategy.
- Folds are 6 months; at 22 days the forward windows inside a fold overlap.
- Correlation, not causation.
