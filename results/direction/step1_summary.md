# Direction Model — Step 1 (price + technical only)

**Target:** HNB next-day Buy / Sell / Hold (dead-zone +/-0.5%). XGBoost, chronological split.

## Result (honest)
- **XGBoost accuracy: 36.2%**
- Majority baseline: 39.5%   | Persistence baseline: 43.1%
- **Beats both baselines? NO**

Per-class F1: Sell 0.33, Hold 0.41, Buy 0.35.

## Read it simply
Price-only, next-day direction. Target was ~52-58%. Got 36.2%.
It does NOT clearly beat the naive baselines — price+technical alone is not enough (as expected). This is the honest baseline to improve on with macro/events/sentiment.

## Next
Step 2 = add macro features (spread, inflation, T-bill rate, exchange rate). Measure the accuracy gain.

*Caveat: one stock, next-day horizon (hardest), single chronological split, correlation not causation.*
