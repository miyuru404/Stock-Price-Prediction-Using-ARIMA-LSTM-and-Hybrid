# Price + news only (no macro), and a hybrid on the same

## Why this is not a repeat of Phase H

In Phase H, news was added **on top of** Tier-2 indicators, interest rates and sector features — it
had to prove itself while competing with ~30 other columns, several already shown to be useless.
Phase C showed macro can actively *hurt*. So news never got a clean test.

Here everything else is stripped away.

**Window: 2016-01 → 2022-06** (set by the news feed). Not comparable to other
phases — compare only within this table.

## Leak scan

CLEAN — no feature tracks the future more than the present.

## 1. DIRECTION — does news help once macro is gone?

| target | horizon_days | features | folds | median_acc_% | median_baseline_% | median_edge_pp | folds_pos | sign_p | median_mag_edge_pp | mag_folds_pos | mag_sign_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BANKS | 1 | P (price only) | 6 | 52.6 | 56.8 | -4.35 | 1/6 | 0.9844 | 4.4 | 3/6 | 0.6562 |
| BANKS | 1 | P+N (+news) | 6 | 51.0 | 56.8 | -4.8 | 0/6 | 1.0 | 2.55 | 4/6 | 0.3438 |
| BANKS | 5 | P (price only) | 6 | 46.7 | 56.5 | -9.85 | 2/6 | 0.8906 | 6.95 | 3/6 | 0.6562 |
| BANKS | 5 | P+N (+news) | 6 | 49.4 | 56.5 | -8.75 | 0/6 | 1.0 | 1.3 | 3/6 | 0.6562 |
| BANKS | 10 | P (price only) | 6 | 51.8 | 56.0 | -4.95 | 2/6 | 0.8906 | 10.75 | 5/6 | 0.1094 |
| BANKS | 10 | P+N (+news) | 6 | 49.1 | 56.0 | -1.1 | 3/6 | 0.6562 | 11.8 | 5/6 | 0.1094 |
| SPSL20 | 1 | P (price only) | 6 | 56.8 | 60.9 | -5.45 | 1/6 | 0.9844 | 18.25 | 5/6 | 0.1094 |
| SPSL20 | 1 | P+N (+news) | 6 | 57.2 | 60.9 | -5.6 | 1/6 | 0.9844 | 18.25 | 5/6 | 0.1094 |
| SPSL20 | 5 | P (price only) | 6 | 58.8 | 56.9 | 1.85 | 3/6 | 0.6562 | 21.75 | 5/6 | 0.1094 |
| SPSL20 | 5 | P+N (+news) | 6 | 54.4 | 56.9 | 0.0 | 3/6 | 0.6562 | 15.3 | 5/6 | 0.1094 |
| SPSL20 | 10 | P (price only) | 6 | 59.3 | 57.1 | 0.4 | 3/6 | 0.6562 | 32.75 | 5/6 | 0.1094 |
| SPSL20 | 10 | P+N (+news) | 6 | 57.5 | 57.1 | 2.0 | 3/6 | 0.6562 | 26.3 | 5/6 | 0.1094 |

### The isolation result: news gain (P+N minus P, matched fold by fold)

| target | horizon_days | folds | median_news_gain_pp | folds_pos | sign_p | median_news_gain_mag_pp |
|---|---|---|---|---|---|---|
| BANKS | 1 | 6 | -1.25 | 3/6 | 0.6562 | -3.0 |
| BANKS | 5 | 6 | 0.45 | 3/6 | 0.6562 | -1.75 |
| BANKS | 10 | 6 | 2.6 | 4/6 | 0.3438 | 4.2 |
| SPSL20 | 1 | 6 | -0.85 | 1/6 | 0.9844 | -1.9 |
| SPSL20 | 5 | 6 | -0.4 | 1/6 | 0.9844 | -3.75 |
| SPSL20 | 10 | 6 | -2.1 | 3/6 | 0.6562 | -2.35 |

**Horizons where news significantly helps: 0 of 6.**

## 2. PRICE — hybrid with news, no macro

| target | model | folds | RMSE | MAE | MAPE_pct | Theil_U2 | folds_beat_naive | DM_p |
|---|---|---|---|---|---|---|---|---|
| BANKS | ARIMA(2,1,0) | 6 | 1.3338 | 0.8644 | 0.9062 | 0.9848 | 4/6 | 0.298 |
| BANKS | Hybrid ARIMA+LSTM | 6 | 1.3304 | 0.8667 | 0.913 | 0.9888 | 4/6 | 0.3027 |
| BANKS | Hybrid+News | 6 | 1.3465 | 0.9049 | 0.9194 | 1.0003 | 3/6 | 0.6722 |
| BANKS | Naive | 6 | 1.363 | 0.8779 | 0.9121 | 1.0 | 0/6 | nan |
| SPSL20 | ARIMA(2,1,0) | 6 | 34.1742 | 24.738 | 0.85 | 0.9789 | 6/6 | 0.459 |
| SPSL20 | Hybrid ARIMA+LSTM | 6 | 34.4042 | 25.179 | 0.8686 | 0.9872 | 4/6 | 0.7165 |
| SPSL20 | Hybrid+News | 6 | 35.8655 | 26.3022 | 0.8894 | 1.0166 | 2/6 | 0.3987 |
| SPSL20 | Naive | 6 | 34.5704 | 24.6389 | 0.8472 | 1.0 | 0/6 | nan |

**Models beating naive on Theil's U2: 4 of 6.**

* `Hybrid ARIMA+LSTM` — LSTM on ARIMA residuals, univariate (Zhang 2003).
* `Hybrid+News` — the same, but the residual LSTM also sees the news block. If news carries
  anything, the part ARIMA cannot explain is where it should show up.

## Reading it

Removing macro does NOT rescue news. Its contribution is indistinguishable from zero even with nothing else competing for it.

## Caveats
- Window is 2016 to 2022-06; roughly 6 folds per horizon, so power is limited.
- News sentiment is **market-wide**, not company- or sector-specific. It cannot explain why banks
  move differently from the rest of the market — a known gap, still untested.
- 3 seeds averaged for every LSTM; recurrent nets remain the noisiest component here.
