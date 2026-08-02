# Hybrid ARIMA + LSTM with macro & news — and a paper-comparable evaluation

## Why this run exists

1. **The hybrid architecture.** The project is named after it, but the hybrid had only been run
   univariate on price in Stage 1. Here it is implemented properly and extended with macro + news.
2. **Comparability.** Published papers report a fixed train/test split with RMSE / MAE / MAPE / R2
   and claim superiority using the **Diebold-Mariano** test. This project had moved to
   walk-forward evaluation with baseline-relative "edge" — stricter, but not directly comparable to
   a published table. Both are now produced.

## The architecture (Zhang, 2003)

```
Y_t = L_t + N_t
  ARIMA(2, 1, 0)  -> L_hat        linear structure
  e_t = Y_t - L_hat                   what ARIMA could not explain
  LSTM(e_t)           -> N_hat        nonlinear structure in the residual
  FINAL = L_hat + N_hat
```

* **Hybrid ARIMA+LSTM** — residual LSTM sees only past residuals (classic).
* **Hybrid+MacroNews** — residual LSTM also sees ['d_policy_1m', 'd_spread_1m', 'd_tb3m_3m', 'ccpi_yoy_pct', 'usd_lkr_ret_5', 's_vader', 's_lm'].
  This is the project's extension: if macro and news carry information, the natural place for it is
  precisely the part ARIMA provably cannot explain.

LSTM: 20-step residual window, 32 hidden units, 40 epochs, averaged over 3 seeds.
ARIMA forecasts are genuine rolling **one-step-ahead** (state updated each step, parameters frozen).

## TABLE 1 — Fixed 80/20 split (this is the table to place beside a paper)

| target | model | RMSE | MAE | MAPE_% | R2 | Theil_U2 | dir_acc_% | DM_vs_naive | DM_p_vs_naive | DM_p_vs_ARIMA |
|---|---|---|---|---|---|---|---|---|---|---|
| SPSL20 | Naive (random walk) | 56.1624 | 37.058 | 0.788 | 0.9979 | 1.0 | 64.0 | nan | nan | 0.8103 |
| SPSL20 | ARIMA(2, 1, 0) | 56.435 | 36.4231 | 0.7693 | 0.9979 | 1.0049 | 61.2 | 0.24 | 0.8103 | nan |
| SPSL20 | LSTM (univariate) | 56.2062 | 36.1589 | 0.7618 | 0.9979 | 1.0008 | 63.0 | 0.031 | 0.9753 | 0.7275 |
| SPSL20 | Hybrid ARIMA+LSTM | 56.8149 | 36.4246 | 0.7676 | 0.9978 | 1.0116 | 61.5 | 0.446 | 0.6554 | 0.6243 |
| SPSL20 | Hybrid+MacroNews | 57.9297 | 38.2485 | 0.8085 | 0.9977 | 1.0315 | 61.7 | 1.319 | 0.1873 | 0.1069 |
| HNB | Naive (random walk) | 4.4506 | 2.8901 | 1.0174 | 0.9979 | 1.0 | 49.4 | nan | nan | 0.1183 |
| HNB | ARIMA(2, 1, 0) | 4.4186 | 2.8798 | 1.0106 | 0.998 | 0.9928 | 51.2 | -1.562 | 0.1183 | nan |
| HNB | LSTM (univariate) | 4.475 | 2.9254 | 1.0266 | 0.9979 | 1.0055 | 50.3 | 0.465 | 0.6419 | 0.2806 |
| HNB | Hybrid ARIMA+LSTM | 4.471 | 2.9372 | 1.0314 | 0.9979 | 1.0046 | 50.6 | 0.395 | 0.693 | 0.3041 |
| HNB | Hybrid+MacroNews | 4.4994 | 2.9731 | 1.0411 | 0.9979 | 1.011 | 50.0 | 1.839 | 0.0659 | 0.0134 |

**How to read it**
* **Theil's U2 < 1** means better than the naive random walk. **U2 >= 1 means worse.**
* **Diebold-Mariano**: H0 is equal accuracy. A *negative* DM statistic means the model has lower
  loss than the comparison; the p-value says whether that difference is significant.
  Newey-West (HAC) variance, squared-error loss.

Models with Theil's U2 < 1: **1**.
Models significantly better than naive at p < 0.05 (DM): **0**.

## TABLE 2 — Walk-forward (this project's stricter standard)

| target | model | RMSE | MAE | MAPE_pct | R2 | Theil_U2 | dir_acc_pct | folds_better_than_naive |
|---|---|---|---|---|---|---|---|---|
| HNB | ARIMA(2, 1, 0) | 2.4946 | 1.4935 | 1.1878 | 0.9819 | 1.0047 | 45.0 | 2/7 |
| HNB | Hybrid ARIMA+LSTM | 2.5682 | 1.5982 | 1.1991 | 0.982 | 1.0079 | 47.2 | 1/7 |
| HNB | Hybrid+MacroNews | 2.7 | 1.8543 | 1.259 | 0.9815 | 1.0449 | 43.8 | 0/7 |
| HNB | LSTM (univariate) | 2.554 | 1.607 | 1.1923 | 0.9819 | 1.0087 | 45.0 | 2/7 |
| HNB | Naive (random walk) | 2.5111 | 1.4854 | 1.1962 | 0.9824 | 1.0 | 43.8 | 0/7 |
| SPSL20 | ARIMA(2, 1, 0) | 37.6214 | 26.0786 | 0.8983 | 0.9856 | 0.9853 | 58.8 | 6/7 |
| SPSL20 | Hybrid ARIMA+LSTM | 38.4045 | 26.0876 | 0.9003 | 0.9856 | 0.9871 | 60.1 | 5/7 |
| SPSL20 | Hybrid+MacroNews | 47.7946 | 34.2819 | 1.0727 | 0.9855 | 0.993 | 59.8 | 4/7 |
| SPSL20 | LSTM (univariate) | 39.4822 | 26.0005 | 0.8971 | 0.9856 | 0.995 | 59.3 | 4/7 |
| SPSL20 | Naive (random walk) | 40.3966 | 26.2299 | 0.9039 | 0.9851 | 1.0 | 60.9 | 0/7 |

`folds_better_than_naive` counts folds with Theil's U2 < 1. A model that wins on the fixed split
but only ~half the folds was benefiting from that particular test window.

## Why report both

A single fixed split is what the literature uses, and it is what makes this work comparable. It is
also the weakest protocol in this project: earlier runs showed the win rate roughly DOUBLES purely
by moving the test window into the 2022 crisis. Publishing the fixed-split table alone would
overstate the result; publishing only walk-forward would be incomparable to prior work. Both is the
defensible answer, and the gap between them is itself a finding.

## Caveats
- ARIMA order is fixed at (2, 1, 0) (the source paper's choice) rather than re-selected per fold;
  Stage 1 found auto-ARIMA collapsing to (0,1,0), i.e. the naive model, on this data.
- The DM test assumes the loss differential is covariance-stationary. At one-step horizons with
  daily data that is reasonable; it would need more care at long horizons.
- News sentiment only exists from 2016 to 2022-06, so `Hybrid+MacroNews` is estimated on a shorter
  sample than the other models where the split extends beyond that.
