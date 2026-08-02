# Decomposition hybrids — the same model, leaky and honest

## The claim under test

Wavelet-LSTM, CEEMDAN-LSTM and VMD-LSTM papers report accuracy far beyond anything in this project.
This is the "latest technology" branch of the literature and the last untested explanation for why
published results exceed ours.

## The suspected mechanism

Wavelet, EMD and VMD are **global** operations: every component value at time *t* is computed from
the **whole series**, including points after *t*. Decompose first and split afterwards — the common
published pipeline — and each training component already contains test-period information, while
each test component was built using its own future.

## The experiment

Identical data, identical models, identical split. **Only the order of decompose-and-split changes.**

| Pipeline | What it does |
|---|---|
| **LEAKY** | decompose the FULL series once → build features → split |
| **HONEST** | at each *t*, decompose only `series[:t]` and keep the last component values |

## Headline

| target | decomposition | MAPE_leaky_% | MAPE_honest_% | MAPE_naive_% | improvement_leaky_% | improvement_honest_% | gain_from_leaking_pp |
|---|---|---|---|---|---|---|---|
| BANKS | CEEMDAN | 0.6864 | 1.0103 | 0.8496 | 19.2 | -6.86 | 26.06 |
| BANKS | wavelet(db4,3) | 0.7246 | 0.8883 | 0.8496 | 14.7 | -1.8 | 16.5 |
| SPSL20 | CEEMDAN | 0.6712 | 0.7725 | 0.7894 | 14.98 | 1.88 | 13.1 |
| SPSL20 | wavelet(db4,3) | 0.6735 | 0.8149 | 0.7894 | 14.69 | -0.52 | 15.21 |

`gain_from_leaking_pp` is the advantage created **purely by decomposing before splitting**.

## Full results

| target | decomposition | pipeline | model | MAPE_% | MAPE_naive_% | R2 | Theil_U2 | beats_naive | DM_p_vs_naive |
|---|---|---|---|---|---|---|---|---|---|
| SPSL20 | wavelet(db4,3) | LEAKY (decompose then split) | XGB | 0.6735 | 0.7894 | 0.998407 | 0.8635 | YES | 0.0026 |
| SPSL20 | wavelet(db4,3) | HONEST (decompose only the past) | XGB | 0.8149 | 0.8107 | 0.997331 | 1.0144 | no | 0.6751 |
| SPSL20 | CEEMDAN | LEAKY (decompose then split) | XGB | 0.6712 | 0.7894 | 0.998379 | 0.871 | YES | 0.0199 |
| SPSL20 | CEEMDAN | HONEST (decompose only the past) | Ridge | 0.7725 | 0.7873 | 0.997567 | 1.0082 | YES | 0.842 |
| BANKS | wavelet(db4,3) | LEAKY (decompose then split) | XGB | 0.7246 | 0.8496 | 0.998894 | 0.8252 | YES | 0.0 |
| BANKS | wavelet(db4,3) | HONEST (decompose only the past) | XGB | 0.8883 | 0.8726 | 0.997986 | 1.0275 | no | 0.3524 |
| BANKS | CEEMDAN | LEAKY (decompose then split) | XGB | 0.6864 | 0.8496 | 0.998975 | 0.7943 | YES | 0.0 |
| BANKS | CEEMDAN | HONEST (decompose only the past) | XGB | 1.0103 | 0.9454 | 0.997661 | 1.0241 | no | 0.3611 |

## Reading it

- **Theil's U2 below 1.0** means better than a random walk.
- If the LEAKY row shows a large advantage and the HONEST row collapses to ~1.0, the published
  advantage of decomposition hybrids on this data is an artefact of **pipeline order**, not of the
  method itself.
- Note the R2 column again: on a price level it stays near 1.0 regardless, which is exactly why
  papers reporting R2 ≈ 0.99 cannot be used to tell these two pipelines apart.

## Caveats
- One fixed 80/20 split per configuration (the protocol these papers use), not walk-forward.
- The honest CEEMDAN variant re-decomposes every 5 points rather than every point, for runtime;
  wavelet is re-decomposed at every point.
- Wavelet boundary effects mean even the honest version's most recent component values are the
  least stable — that is a genuine property of the method, not a bug in this test.
