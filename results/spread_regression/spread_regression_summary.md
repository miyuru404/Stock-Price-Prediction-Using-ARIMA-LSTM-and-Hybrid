# Interest-Rate Spread Regression — Result

**Test:** does bank net interest margin (spread = AWPR - AWDR) move bank and finance stock
returns **differently**? 173 monthly observations, abnormal returns (vs ASPI). The
**interaction term (d_spread x is_bank)** is the whole test.

## Verdict

**HYPOTHESIS NOT SUPPORTED (effect runs the OTHER way) — there IS a strong, significant interaction (lag-0 p=0.0002), but the signs are OPPOSITE the hypothesis: banks react NEGATIVELY to a wider spread (-2.30) and finance POSITIVELY (+2.34). And the finance-positive side is driven by extreme 2021 outliers (LOLC/LOFC bubble): excluding months with |return|>50%, the interaction p=0.000 (still significant). The margin-channel story is not supported.**

Contemporaneous (lag 0): finance slope **+2.342**, banks slope
**-2.305**, interaction **-4.647**
(SE 1.235, p = **0.000**), n=1014.

## Group interaction — all lags (banks vs finance)

| Model | Finance slope | Banks slope | Interaction | Interaction p | n |
|---|---|---|---|---|---|
| d_spread lag 0 | 2.3421 | -2.3047 | -4.6467 | 0.0002 | 1014 |
| d_spread lag 1 | 0.9191 | -1.1891 | -2.1082 | 0.3739 | 1014 |
| d_spread lag 2 | 0.1194 | -0.7138 | -0.8332 | 0.5561 | 1008 |
| d_spread lag 3 | 0.4204 | -1.4784 | -1.8988 | 0.0539 | 1002 |

Hypothesis wants: banks slope **positive**, finance slope **negative/zero**, interaction **p<0.05**.

## Per-company (contemporaneous d_spread, lag 0)

| Company | Group | Coef | p | R2 | n |
|---|---|---|---|---|---|
| HNB | Banks | -3.1845 | 0.0022 | 0.1159 | 169 |
| COMB | Banks | -1.3998 | 0.0556 | 0.0337 | 169 |
| SAMP | Banks | -2.3297 | 0.0093 | 0.0686 | 169 |
| LOFC | Finance | 4.0719 | 0.0191 | 0.0243 | 169 |
| LOLC | Finance | 3.8206 | 0.0010 | 0.0141 | 169 |
| LFIN | Finance | -0.8663 | 0.2786 | 0.0048 | 169 |
| JKH | Control | -0.0949 | 0.8642 | 0.0001 | 169 |
| DIAL | Control | -0.1362 | 0.8296 | 0.0002 | 169 |

Control (JKH, DIAL) should be near zero and non-significant — the placebo check.

## Robustness

| Variant | Finance slope | Banks slope | Interaction | Interaction p | n |
|---|---|---|---|---|---|
| d_spread ex-2022 | 3.9216 | -3.7792 | -7.7008 | 0.0060 | 942 |
| d_awpr (lag0) | 1.2986 | -1.6921 | -2.9907 | 0.0004 | 1014 |
| d_policy (lag0) | -0.0858 | -0.5355 | -0.4498 | 0.5088 | 1014 |
| d_spread winsorized 1/99pct | 1.5792 | -2.2976 | -3.8767 | 0.0000 | 1014 |
| d_spread excl |ret|>50% | 1.3525 | -2.3047 | -3.6571 | 0.0000 | 1009 |

- **ex-2022:** if the interaction loses significance here, the 2022 crisis was driving it.
- **d_awpr:** lending rate alone, for comparison.
- **d_policy:** the policy-rate proxy that already returned null in the pilot — expected weakest.

## 5 largest residuals (main lag-0 model)

      date company     grp  abn_ret  d_spread_l0   resid
2021-01-31    LOLC Finance  256.439         0.30 254.559
2021-11-30    LOFC Finance  183.218         0.68 180.448
2021-01-31    LOFC Finance   85.143         0.30  83.262
2021-01-31    LFIN Finance   80.158         0.30  78.278
2019-12-31    LOLC Finance   54.781        -0.14  53.931

## Caveats

- **173 monthly observations** — real but modest power (not thousands).
- **2022 is an enormous spread outlier** (3.41 -> 15.50) — reported with and without.
- **LFIN** prices pre-2015-07-14 back-adjusted x0.5 for an unapplied 2-for-1 split.
- **Policy framework changed 27 Nov 2024** (SDFR/SLFR -> single OPR).
- **Correlation, not causation** — this is an association between margin changes and returns,
  not proof that margin changes cause them.
- Per-company SEs are HAC (Newey-West, 3 lags); the panel interaction uses SEs **clustered by
  month** to handle contemporaneous cross-company correlation.

*Outputs: per_company_regression.csv, group_interaction_results.csv, robustness_checks.csv,
scatter_dspread_vs_abnret.png, spread_timeseries.png.*
