# Probability calibration — can the system honestly show a confidence number?

The dashboard does not just say up/down; it shows a **confidence**. That number must mean what it
says: of all the days the system prints "65%", about 65% should actually rise. A model that is
barely better than a coin flip can still be perfectly calibrated — and a well-calibrated 55% is an
honest, usable product. An **uncalibrated 85% is a lie to the user.**

Accuracy and calibration are different properties. Nothing earlier in this project tested the second.

## Method

* Targets: **BANKS** (autocorrelation 0.178, the strongest composite) and **SECTOR** (all 7).
* Horizons: **1 / 2 / 4 weeks** — what the product advertises.
* Calibrators (`raw`, `Platt`, `isotonic`) are fitted on a held-out slice **inside the training
  window**, never on the test fold. Fitting a calibrator on test data would be the same class of
  leak this project has already caught three times.
* Benchmark: **climatology** — always predict the training base rate. This is the probabilistic
  equivalent of the naive baseline.

## Key metrics

| Metric | Meaning | Good |
|---|---|---|
| **Brier skill** | 1 − Brier/Brier_climatology | **> 0** |
| **ECE** | mean gap between stated confidence and reality | **< 0.10** |
| **MCE** | worst bucket's gap | small |
| **prob_std** | how much the forecast actually varies | **not ~0** |

`prob_std` matters: a model that prints the base rate every single day is *perfectly calibrated and
completely useless*. Calibration alone is not enough — the forecast must also move.

## Results (median across walk-forward folds)

| group | horizon_days | model | calibration | accuracy_pct | AUC | Brier | Brier_clim | Brier_skill | ECE | prob_std | folds_skill_positive | USABLE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BANKS | 5 | Logistic | Platt | 52.4 | 0.496 | 0.251 | 0.2501 | -0.0036 | 0.0749 | 0.0228 | 8/19 | no |
| BANKS | 5 | Logistic | isotonic | 52.4 | 0.512 | 0.2556 | 0.2501 | -0.0195 | 0.1022 | 0.0613 | 7/19 | no |
| BANKS | 5 | Logistic | raw | 50.4 | 0.523 | 0.2583 | 0.2501 | -0.0239 | 0.1108 | 0.1073 | 6/19 | no |
| BANKS | 5 | XGBoost | Platt | 49.2 | 0.447 | 0.2543 | 0.2501 | -0.0168 | 0.0775 | 0.012 | 6/19 | no |
| BANKS | 5 | XGBoost | isotonic | 48.3 | 0.537 | 0.2552 | 0.2501 | -0.0088 | 0.0886 | 0.0424 | 7/19 | no |
| BANKS | 5 | XGBoost | raw | 52.5 | 0.516 | 0.2868 | 0.2501 | -0.1289 | 0.1882 | 0.1635 | 5/19 | no |
| BANKS | 10 | Logistic | Platt | 51.6 | 0.449 | 0.2555 | 0.251 | -0.0262 | 0.1397 | 0.0268 | 8/19 | no |
| BANKS | 10 | Logistic | isotonic | 50.5 | 0.5 | 0.2513 | 0.251 | -0.0051 | 0.1166 | 0.0379 | 8/19 | no |
| BANKS | 10 | Logistic | raw | 52.5 | 0.528 | 0.2606 | 0.251 | -0.0346 | 0.1587 | 0.1016 | 7/19 | no |
| BANKS | 10 | XGBoost | Platt | 54.0 | 0.505 | 0.2484 | 0.251 | -0.0056 | 0.1144 | 0.0209 | 9/19 | no |
| BANKS | 10 | XGBoost | isotonic | 50.0 | 0.504 | 0.2549 | 0.251 | -0.0144 | 0.1215 | 0.0456 | 7/19 | no |
| BANKS | 10 | XGBoost | raw | 52.6 | 0.527 | 0.2852 | 0.251 | -0.1427 | 0.2225 | 0.2004 | 2/19 | no |
| BANKS | 22 | Logistic | Platt | 49.7 | 0.408 | 0.254 | 0.2508 | -0.0055 | 0.1647 | 0.0431 | 7/18 | no |
| BANKS | 22 | Logistic | isotonic | 53.6 | 0.514 | 0.2576 | 0.2508 | -0.0182 | 0.1651 | 0.069 | 8/18 | no |
| BANKS | 22 | Logistic | raw | 53.2 | 0.537 | 0.2565 | 0.2508 | -0.0288 | 0.2114 | 0.1218 | 7/18 | no |
| BANKS | 22 | XGBoost | Platt | 51.65 | 0.464 | 0.2592 | 0.2508 | -0.0348 | 0.1564 | 0.0428 | 7/18 | no |
| BANKS | 22 | XGBoost | isotonic | 53.0 | 0.502 | 0.2548 | 0.2508 | -0.0221 | 0.1867 | 0.0674 | 8/18 | no |
| BANKS | 22 | XGBoost | raw | 49.95 | 0.482 | 0.299 | 0.2508 | -0.1848 | 0.2472 | 0.1942 | 4/18 | no |
| SECTOR | 5 | Logistic | Platt | 50.1 | 0.539 | 0.253 | 0.2469 | -0.0084 | 0.1126 | 0.0297 | 6/16 | no |
| SECTOR | 5 | Logistic | isotonic | 52.6 | 0.505 | 0.2565 | 0.2469 | -0.0236 | 0.1156 | 0.0783 | 6/16 | no |
| SECTOR | 5 | Logistic | raw | 53.9 | 0.535 | 0.2584 | 0.2469 | -0.024 | 0.1473 | 0.1014 | 5/16 | no |
| SECTOR | 5 | XGBoost | Platt | 53.95 | 0.5225 | 0.249 | 0.2469 | -0.0125 | 0.11 | 0.026 | 6/16 | no |
| SECTOR | 5 | XGBoost | isotonic | 50.65 | 0.5205 | 0.2527 | 0.2469 | -0.0309 | 0.1108 | 0.0754 | 4/16 | no |
| SECTOR | 5 | XGBoost | raw | 56.55 | 0.5505 | 0.2766 | 0.2469 | -0.082 | 0.18 | 0.171 | 4/16 | no |
| SECTOR | 10 | Logistic | Platt | 51.55 | 0.5375 | 0.2508 | 0.2444 | -0.0216 | 0.1792 | 0.0448 | 4/16 | no |
| SECTOR | 10 | Logistic | isotonic | 53.3 | 0.54 | 0.2504 | 0.2444 | -0.0143 | 0.1482 | 0.076 | 4/16 | no |
| SECTOR | 10 | Logistic | raw | 55.1 | 0.5375 | 0.2518 | 0.2444 | -0.0376 | 0.1598 | 0.116 | 6/16 | no |
| SECTOR | 10 | XGBoost | Platt | 47.8 | 0.528 | 0.2548 | 0.2444 | -0.03 | 0.1235 | 0.054 | 5/16 | no |
| SECTOR | 10 | XGBoost | isotonic | 50.65 | 0.4885 | 0.2561 | 0.2444 | -0.0484 | 0.13 | 0.0866 | 4/16 | no |
| SECTOR | 10 | XGBoost | raw | 50.6 | 0.508 | 0.2912 | 0.2444 | -0.1582 | 0.216 | 0.2006 | 4/16 | no |
| SECTOR | 22 | Logistic | Platt | 45.6 | 0.492 | 0.2506 | 0.2371 | -0.008 | 0.2122 | 0.0423 | 7/15 | no |
| SECTOR | 22 | Logistic | isotonic | 55.7 | 0.508 | 0.2508 | 0.2371 | -0.0557 | 0.2117 | 0.0949 | 6/15 | no |
| SECTOR | 22 | Logistic | raw | 52.8 | 0.508 | 0.2472 | 0.2371 | -0.0827 | 0.1973 | 0.1303 | 4/15 | no |
| SECTOR | 22 | XGBoost | Platt | 48.4 | 0.445 | 0.2527 | 0.2371 | -0.0293 | 0.1898 | 0.0319 | 4/15 | no |
| SECTOR | 22 | XGBoost | isotonic | 50.4 | 0.491 | 0.2626 | 0.2371 | -0.0915 | 0.2176 | 0.0812 | 2/15 | no |
| SECTOR | 22 | XGBoost | raw | 54.5 | 0.555 | 0.2916 | 0.2371 | -0.1824 | 0.2287 | 0.2323 | 4/15 | no |

**Configurations that are usable (Brier skill > 0 AND ECE < 0.10): 0 of 36.**

Best by Brier skill: **BANKS / 1 week / Logistic / Platt**
— skill -0.0036, ECE 0.0749, accuracy 52.4%.

## Reliability curves

`reliability_curves.png` — the diagonal is perfect calibration. A curve **below** the diagonal means
the system is **overconfident** (says 70%, delivers less); **above** means underconfident.

## What this means for the product

No configuration is both skilful and well calibrated. The dashboard should NOT display a raw model probability as a confidence figure.

Whatever the outcome, the honest design is the same: show the **realised track record**
("when we said 60%, it rose 58% of the time, over N predictions") rather than the model's raw
output. That is measurable, cannot be gamed, and is exactly the accuracy-history feature already
planned for the system.

## Caveats
- Walk-forward folds are 6 months; at 22 days the forward windows inside a fold overlap, so the
  effective sample there is smaller than the row count suggests.
- Isotonic regression needs a reasonable calibration slice; with ~500 training rows it can overfit
  and is the more fragile of the two methods.
- Calibration is measured on the composite index, not on individual stocks.
