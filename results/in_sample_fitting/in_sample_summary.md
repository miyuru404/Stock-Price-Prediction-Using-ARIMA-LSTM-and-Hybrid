# In-sample fitting — and a control made of random numbers

## The shortcut

Cointegration / VECM / ARDL studies fit on the **full sample** and report explanatory power plus
coefficient significance. That legitimately describes a **relationship**. It is routinely read as
evidence of **forecasting** ability — which is a different claim.

## The control

Alongside the real macro block, the same models get a block of **pure random numbers** — same
column count, fixed seed, zero information.

**If noise improves in-sample fit as much as macro does, in-sample improvement is a property of
adding columns, not of the columns meaning anything.**

## Results

| target | features | model | evaluation | n | k | R2_return | adjR2_return | R2_price | MAPE_% | vs_naive_ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| SPSL20 | price only | Ridge | IN-SAMPLE | 2640 | 9 | 0.05525 | 0.05202 | 0.997987 | 0.7756 | nan |
| SPSL20 | price only | XGBoost | IN-SAMPLE | 2640 | 9 | 0.50525 | 0.50356 | 0.998812 | 0.6433 | nan |
| SPSL20 | price only | Ridge | OUT-OF-SAMPLE | 528 | 9 | 0.01479 | -0.00232 | 0.997864 | 0.7691 | 0.9713 |
| SPSL20 | price only | XGBoost | OUT-OF-SAMPLE | 528 | 9 | -0.00155 | -0.01895 | 0.99779 | 0.7739 | 0.9773 |
| SPSL20 | price + MACRO | Ridge | IN-SAMPLE | 2640 | 21 | 0.05901 | 0.05146 | 0.997994 | 0.7754 | nan |
| SPSL20 | price + MACRO | XGBoost | IN-SAMPLE | 2640 | 21 | 0.56554 | 0.56205 | 0.998919 | 0.6176 | nan |
| SPSL20 | price + MACRO | Ridge | OUT-OF-SAMPLE | 528 | 21 | -0.02127 | -0.06365 | 0.997736 | 0.8112 | 1.0245 |
| SPSL20 | price + MACRO | XGBoost | OUT-OF-SAMPLE | 528 | 21 | 0.00235 | -0.03905 | 0.997833 | 0.7635 | 0.9643 |
| SPSL20 | price + RANDOM NOISE (control) | Ridge | IN-SAMPLE | 2640 | 21 | 0.05844 | 0.05089 | 0.997988 | 0.7769 | nan |
| SPSL20 | price + RANDOM NOISE (control) | XGBoost | IN-SAMPLE | 2640 | 21 | 0.63087 | 0.62791 | 0.999066 | 0.5747 | nan |
| SPSL20 | price + RANDOM NOISE (control) | Ridge | OUT-OF-SAMPLE | 528 | 21 | 0.00202 | -0.0394 | 0.997828 | 0.7755 | 0.9795 |
| SPSL20 | price + RANDOM NOISE (control) | XGBoost | OUT-OF-SAMPLE | 528 | 21 | -0.02615 | -0.06873 | 0.99773 | 0.7855 | 0.9921 |
| BANKS | price only | Ridge | IN-SAMPLE | 3177 | 9 | 0.04013 | 0.0374 | 0.999148 | 0.8235 | nan |
| BANKS | price only | XGBoost | IN-SAMPLE | 3177 | 9 | 0.45216 | 0.45061 | 0.999454 | 0.6952 | nan |
| BANKS | price only | Ridge | OUT-OF-SAMPLE | 636 | 9 | 0.03546 | 0.0216 | 0.998376 | 0.8344 | 0.9759 |
| BANKS | price only | XGBoost | OUT-OF-SAMPLE | 636 | 9 | -0.13466 | -0.15098 | 0.99818 | 0.8909 | 1.0419 |
| BANKS | price + MACRO | Ridge | IN-SAMPLE | 3177 | 21 | 0.0493 | 0.04297 | 0.999152 | 0.8236 | nan |
| BANKS | price + MACRO | XGBoost | IN-SAMPLE | 3177 | 21 | 0.50634 | 0.50305 | 0.999494 | 0.669 | nan |
| BANKS | price + MACRO | Ridge | OUT-OF-SAMPLE | 636 | 21 | 0.04124 | 0.00845 | 0.998371 | 0.8439 | 0.987 |
| BANKS | price + MACRO | XGBoost | OUT-OF-SAMPLE | 636 | 21 | -0.05686 | -0.093 | 0.99827 | 0.8832 | 1.033 |
| BANKS | price + RANDOM NOISE (control) | Ridge | IN-SAMPLE | 3177 | 21 | 0.0429 | 0.03653 | 0.99915 | 0.8242 | nan |
| BANKS | price + RANDOM NOISE (control) | XGBoost | IN-SAMPLE | 3177 | 21 | 0.52982 | 0.52669 | 0.999523 | 0.6513 | nan |
| BANKS | price + RANDOM NOISE (control) | Ridge | OUT-OF-SAMPLE | 636 | 21 | 0.02995 | -0.00322 | 0.998371 | 0.8386 | 0.9807 |
| BANKS | price + RANDOM NOISE (control) | XGBoost | OUT-OF-SAMPLE | 636 | 21 | -0.07135 | -0.10799 | 0.998213 | 0.8812 | 1.0305 |

## Gain from adding a block

| target | model | evaluation | block | R2_gain | adjR2_gain | MAPE_improvement_% |
|---|---|---|---|---|---|---|
| BANKS | Ridge | IN-SAMPLE | price + MACRO | 0.00917 | 0.00557 | -0.012 |
| BANKS | Ridge | IN-SAMPLE | price + RANDOM NOISE (control) | 0.00277 | -0.00087 | -0.085 |
| BANKS | Ridge | OUT-OF-SAMPLE | price + MACRO | 0.00578 | -0.01315 | -1.139 |
| BANKS | Ridge | OUT-OF-SAMPLE | price + RANDOM NOISE (control) | -0.00551 | -0.02482 | -0.503 |
| BANKS | XGBoost | IN-SAMPLE | price + MACRO | 0.05418 | 0.05244 | 3.769 |
| BANKS | XGBoost | IN-SAMPLE | price + RANDOM NOISE (control) | 0.07766 | 0.07608 | 6.315 |
| BANKS | XGBoost | OUT-OF-SAMPLE | price + MACRO | 0.0778 | 0.05798 | 0.864 |
| BANKS | XGBoost | OUT-OF-SAMPLE | price + RANDOM NOISE (control) | 0.06331 | 0.04299 | 1.089 |
| SPSL20 | Ridge | IN-SAMPLE | price + MACRO | 0.00376 | -0.00056 | 0.026 |
| SPSL20 | Ridge | IN-SAMPLE | price + RANDOM NOISE (control) | 0.00319 | -0.00113 | -0.168 |
| SPSL20 | Ridge | OUT-OF-SAMPLE | price + MACRO | -0.03606 | -0.06133 | -5.474 |
| SPSL20 | Ridge | OUT-OF-SAMPLE | price + RANDOM NOISE (control) | -0.01277 | -0.03708 | -0.832 |
| SPSL20 | XGBoost | IN-SAMPLE | price + MACRO | 0.06029 | 0.05849 | 3.995 |
| SPSL20 | XGBoost | IN-SAMPLE | price + RANDOM NOISE (control) | 0.12562 | 0.12435 | 10.664 |
| SPSL20 | XGBoost | OUT-OF-SAMPLE | price + MACRO | 0.0039 | -0.0201 | 1.344 |
| SPSL20 | XGBoost | OUT-OF-SAMPLE | price + RANDOM NOISE (control) | -0.0246 | -0.04978 | -1.499 |

Median R² gain over price-only:

| | MACRO | RANDOM NOISE |
|---|---|---|
| **IN-SAMPLE** | +0.03168 | +0.04043 |
| **OUT-OF-SAMPLE** | +0.00484 | — |

## The "macro is significant" table

This is the output these papers publish — in-sample OLS of next-day return on price + macro:

| target | variable | coef | t_stat | p_value | significant_5% |
|---|---|---|---|---|---|
| SPSL20 | policy_rate | 0.000541 | 1.207 | 0.2274 | no |
| SPSL20 | spread | -6.2e-05 | -0.159 | 0.8738 | no |
| SPSL20 | tb_3m | -0.000156 | -0.497 | 0.6194 | no |
| SPSL20 | d_policy_1m | -0.000433 | -0.799 | 0.4245 | no |
| SPSL20 | d_spread_1m | 0.0005 | 0.851 | 0.3951 | no |
| SPSL20 | d_tb3m_3m | 0.000192 | 0.876 | 0.3812 | no |
| SPSL20 | term_slope | -0.000167 | -0.347 | 0.7289 | no |
| SPSL20 | ccpi_index_2021base | 2.3e-05 | 0.644 | 0.5198 | no |
| SPSL20 | ccpi_yoy_pct | 6e-06 | 0.096 | 0.9239 | no |
| SPSL20 | ccpi_mom_pct | -0.000324 | -1.503 | 0.1331 | no |
| SPSL20 | usd_lkr | -1.8e-05 | -0.756 | 0.4495 | no |
| SPSL20 | usd_lkr_ret_5 | -0.011504 | -0.642 | 0.5211 | no |
| BANKS | policy_rate | 2.5e-05 | 0.059 | 0.9533 | no |
| BANKS | spread | 0.000237 | 0.717 | 0.4736 | no |
| BANKS | tb_3m | 5.7e-05 | 0.209 | 0.8346 | no |
| BANKS | d_policy_1m | 0.000684 | 1.262 | 0.207 | no |
| BANKS | d_spread_1m | 0.001555 | 2.611 | 0.0091 | YES |
| BANKS | d_tb3m_3m | -0.000137 | -0.678 | 0.4977 | no |
| BANKS | term_slope | -0.000371 | -0.935 | 0.3497 | no |
| BANKS | ccpi_index_2021base | 0.000106 | 2.922 | 0.0035 | YES |
| BANKS | ccpi_yoy_pct | 2.7e-05 | 0.459 | 0.6465 | no |
| BANKS | ccpi_mom_pct | -0.000412 | -1.924 | 0.0544 | no |
| BANKS | usd_lkr | -6.9e-05 | -2.991 | 0.0028 | YES |
| BANKS | usd_lkr_ret_5 | 0.0175 | 1.054 | 0.2918 | no |

## Reading it

- **`R2_price` stays around 0.99 everywhere**, in-sample and out, for every feature set including
  pure noise. On a price level it is not a measure of skill.
- **`adjR2_return`** penalises parameter count. Compare it with raw `R2_return` to see how much of
  the in-sample "improvement" is just extra columns.
- **`vs_naive_ratio`** (out-of-sample only): below 1.0 means better than a random walk.

## Caveats
- Single 80/20 split for the out-of-sample side, matching the protocol these papers use.
- The OLS table uses next-day RETURN as the dependent variable. A VECM on levels would show far
  larger t-statistics still — non-stationary levels inflate significance, which is precisely why
  cointegration methods exist and precisely why their output is not a forecast.
