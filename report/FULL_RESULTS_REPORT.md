# Predicting the Colombo Stock Exchange: Price, Direction and Macroeconomic Information

### Full results report — 64 logged experiments

**Market:** Colombo Stock Exchange (CSE), Sri Lanka · banking and finance sector
**Data:** 11 stocks + ASPI index, daily, 2012–2026 · CBSL/DCS macroeconomic series
**Report date:** 2026-08-02 · **Source of record:** `results_master_log.xlsx` (64 rows)

---

## 1. Headline finding

> **The direction of CSE banking and finance stocks is not predictable from price history,
> technical indicators, or any classic macroeconomic variable — at any horizon from one day to one
> year, on any of 11 stocks, using linear models, gradient boosting or recurrent neural networks.**

Every configuration tested was measured against naive baselines (majority class and persistence).
**None beat them.** The single most promising result was pre-registered, re-tested across 21
independent walk-forward windows, and refuted.

Two secondary findings are positive and defensible:

1. **Apparent skill is a property of the test period, not the model** — confirmed three times
   independently.
2. **Sequence order carries a little information that summary features discard** — LSTM beats the
   same data flattened at all three horizons tested, though not enough to reach the baseline.

---

## 2. How the question evolved

| Stage | Question | Outcome |
|---|---|---|
| 1 | Can we forecast the **price level**? | No — nothing beats "tomorrow = today" |
| 2 | Can we forecast the **direction** (Buy/Hold/Sell)? | No — this report |
| 3 | Does **macroeconomic data** improve direction forecasting? | No — the core contribution |

The scope changed once, from price to direction, for a substantive reason: price proved to be a
random walk, so the level was the wrong target. It was **not** changed again. Direction remained
the target for all 34 direction experiments, including those that failed.

---

## 3. Stage 1 — Price forecasting (15 experiments)

ARIMA, LSTM, Hybrid ARIMA+LSTM, GRU, XGBoost, Prophet, and four transformer architectures
(Informer, Autoformer, PatchTST, TFT), across 80/20, 50/50 and 40/60 splits, one-step and
multi-step.

| Result | Evidence |
|---|---|
| No model beats naive one-step | Best MAPE 0.77% vs naive 0.79% — a tie |
| Naive wins multi-step outright | ARIMA 33.3%, LSTM 35.9%, **naive 33.2%** |
| ARIMA collapses to a random walk | Auto-selected order was **(0,1,0)** on HNB — literally "no change" |
| Transformers add nothing | 6.9–14% MAPE vs ARIMA's ~6.3% |
| An apparent "LSTM win" was seed noise | Disappeared on re-run with different seeds |

**Conclusion:** the price level is a random walk. Continuing to forecast it would have measured
nothing but the persistence baseline.

---

## 4. Stage 2 — Direction, and the stepwise macro ablation

### 4.1 Design

Every experiment used the same protocol:

- **Target:** Buy / Hold / Sell over the next *h* trading days, dead-zone **±0.5% × √h**
  (a fixed dead-zone would leave zero "Hold" days at one year)
- **Method:** DIRECT — a separate model per horizon, no recursion
- **Horizons:** 1, 5, 10, 15, 22, 44, 66, 132, 252 trading days
- **Split:** 80/20 chronological, never shuffled, with an **h-bar purge gap** so no training label
  reaches into the test window
- **Models:** Logistic Regression **and** XGBoost on every run; Ridge and XGBoost regressor for the
  return-% target; LSTM/GRU for the sequence test
- **Baselines:** majority class, persistence at the **matched horizon**, train-mean drift (returns),
  and "always guess the winning side" (sign accuracy)
- **Macro lagged by publication date:** rates 35 days, CPI 21 days, IIP 50 days, money supply 65
  days (the CBSL file states "two months after the end of month")
- **All macro fed as changes, never levels** (see §6.1)

Each phase adds one block of features to the previous one, holding rows, split and seeds identical,
so each gain is cleanly attributable.

### 4.2 The ablation result

| Phase | Added | Features | Mean gain | Beats baseline |
|---|---|---|---|---|
| **A** | Tier-1 technical (returns, MAs, momentum, volatility) | 9 | — (floor) | 0 / 9 |
| **B** | + Tier-2 technical (RSI, MACD, volume) | 15 | **−0.9 pp** | 0 / 9 |
| **C** | + interest rates, levels **and** changes | 28 | **−2.8 pp** | 0 / 9 |
| **C2** | + interest rates, **changes only** | 22 | **+3.4 pp** vs C | 0 / 9 |
| **D** | + sector (ASPI, peers, relative strength, beta) | 38 | **+2.6 pp** | 0 / 9 |
| **E** | + daily USD/LKR | 42 | **+0.0 pp** | 0 / 9 |
| **E2** | + oil, US 10Y, DXY, CPI | 54 | **−1.9 pp** | 0 / 9 |
| **M** | + money supply (M1, M2, M2b, M4) | 47 | **−0.7 pp** | 0 / 9 |
| **I** | + industrial production | 42 | **−6.4 pp** | 0 / 9 |

Only **one** step ever helped consistently: **Phase D (+2.6 pp)**, replicated across 11 stocks at
**+3.8 pp**, positive in 66 of 99 stock-horizon cells and 9 of 11 stocks.

**It still never crossed the baseline.** Sector context lifts the model from *clearly worse than a
naive guess* to *roughly a coin flip* at 1 day–1 week, and does nothing at longer horizons.

### 4.3 All five classic macroeconomic variables

The reference paper (Naik & Padhi, 2012) used five variables. All are now tested here under an
out-of-sample forecasting protocol:

| Variable | Phase | Gain | Their finding |
|---|---|---|---|
| Industrial production | G | **−6.4 pp** | positive, significant (their strongest) |
| Money supply | F | **−0.7 pp** | positive, significant |
| T-bill / policy rate | C | **−2.8 pp** (levels harmful) | insignificant |
| Exchange rate | E | **+0.0 pp** | insignificant |
| Inflation | E | part of **−1.9 pp** | negative, significant |

**None improves direction forecasting.** Adding *all* macro (25 of 99 cells beating baseline) is
worse than sector features alone (27 of 99).

---

## 5. Why this does not contradict the literature

This is the intellectual core of the project, and it resolves an apparent conflict.

| | Naik & Padhi (2012) and similar | This project |
|---|---|---|
| Method | Johansen cointegration / VECM | supervised direct forecasting |
| Question | is there a long-run **relationship**? | can you **predict** the next move? |
| Sample use | full sample, in-sample fit | train on past, grade on unseen future |
| Target | index **level** | **direction** of the move |
| Result | macro is significant | macro adds nothing |

A cointegrating relationship says two series drift together over years. It does **not** imply
out-of-sample directional forecasting power.

> **Macroeconomic variables explain. They do not predict.**

Both statements can be true simultaneously, and this project demonstrates the gap empirically
rather than asserting it.

---

## 6. Methodological findings

These are results in their own right, and several were discovered by nearly being fooled.

### 6.1 Macro must be fed as changes, never levels

Phase C used rate **levels** and **changes** together. It was not merely unhelpful — it was
**unstable**: the gain swung from −22.3 to +10.3 pp across horizons, one-year accuracy collapsed to
**4.5%**, and return RMSE blew out to **1.37×** the null.

Diagnosis: a rate *level* such as "15%" occurs in exactly one era. A tree memorises
"2022 → crash", and that mapping never recurs. Rate *changes* are stationary and reusable.

Dropping levels and keeping only changes recovered **+3.4 pp**, with one-year accuracy going
**4.5% → 49.7%**.

### 6.2 Apparent skill is a property of the test period (confirmed 3×)

| Run | Evidence |
|---|---|
| Money supply | calm window 17–27 of 99 cells beat baseline; **crisis window 43–51 of 99** — for price-only features too |
| 1-day walk-forward re-test | 2016–2020: **1 of 10** folds positive · 2021–2024: **6 of 8** |
| LSTM sequence test | 2019–2020: **0 of 2** folds positive · 2021–2025: **5 of 5** |

Same code, same features — only the period changes, and the measured "skill" roughly doubles.

**Implication:** a single 80/20 split can make a model look twice as good depending on which years
land in the test set. Walk-forward or multi-window evaluation should be the default, not an extra.

### 6.3 Sequence order carries a little real information

LSTM/GRU read the raw sequence of the last 30 days; every other model read a flat row of summary
features, which discards ordering. Against the **same data flattened**:

| Horizon | LSTM | GRU | Logistic (flattened) |
|---|---|---|---|
| 1 day | **+2.9 pp** | +1.3 | −1.7 |
| 1 week | **−0.4** | +0.0 | −2.1 |
| 1 month | **−5.4** | −8.3 | −9.9 |

LSTM beats the flattened control at **all three** horizons. Order matters slightly — the only time
in the project that added model complexity genuinely helped. It is still **not enough to reach the
baseline** (best: 5/7 folds, p = 0.227).

### 6.4 The recurring "attention without payoff" fingerprint

| Feature block | Share of XGBoost importance | Accuracy gain |
|---|---|---|
| Tier-2 indicators | 41% | −0.9 pp |
| Monthly macro | 64% | −2.8 pp |
| Sector | 34% | +2.6 pp |
| Daily macro + global + CPI | 27% | −1.9 pp |
| Money supply | 29% | −0.7 pp |
| Industrial production | 15% | −6.4 pp |

The model consistently *uses* new features heavily and gains nothing. **Feature importance is not
evidence of predictive value** — a point worth making explicitly, since importance plots are
routinely presented as if they were.

### 6.5 The simpler model usually wins

Logistic Regression beat XGBoost in **72%** of head-to-head comparisons (39 vs 14).

If a nonlinear pattern existed — "rates matter *only when* volatility is high" — XGBoost would find
it and Logistic could not. That XGBoost loses indicates it is fitting noise, which is itself
evidence of absent signal rather than of a poor algorithm.

---

## 7. Candidate findings that were chased down and killed

Every promising result was named in advance, then re-tested. All died. This is the project's main
methodological contribution.

| Candidate | How it looked | How it died |
|---|---|---|
| HNB 1–3 week return **sign edge** (+1.7/+1.7/+1.1 pp) | first positive sign edges in the study | replication on 11 stocks: **6 of 33 cells (18%)** vs ~50% for a coin flip |
| Pooled **1-week** edge (+2.7 pp) | pooled model beat the pooled baseline | pooled baseline is unfair; re-scored per stock → **−0.5 pp** |
| Pooled **1-day** edge (+2.5 pp, 6/7 stocks, p = 0.062) | strongest result in the project | walk-forward on 21 windows: **6–7 of 21 folds positive, median −1.9 pp, p = 0.99** |
| 2-week / 3-week edges (+0.8, +1.5 pp) | two more positive horizons | caused by a **bug** — persistence baseline used a fixed 5-day lookback instead of the matched horizon |
| 55–57% return **sign accuracy** at 1–3 months | inside the realistic 52–58% target band | "always guess up" scored the same or better — the test window simply rose |
| Ridge beating naive on return RMSE (8 of 9 horizons) | looked like genuine skill | sat within ~1% of the train-mean drift — it learned drift, not signal |
| Large macro co-movement lifts (M2 +17.0, T-bill −19.3 pp) | large and "significant" | backed by only **7–24 independent windows**; overlapping windows inflated the test |

### Two bugs, both biasing toward a false positive

1. **Persistence baseline used a fixed 5-day lookback** instead of the matched horizon → faked
   +0.8 pp and +1.5 pp edges.
2. **A pooled baseline is not a fair benchmark** — pooling stocks with different class balances
   weakens the majority guess → faked +2.7 pp.

Both errors made results look *better*. This is the expected asymmetry: mistakes that produce
exciting results survive longer, because favourable outcomes attract less scrutiny.

---

## 8. What the models actually learned

Depth-3 decision tree on the pooled panel, every rule scored on **unseen** test data
(1-day horizon, baseline 38.0%):

| Rule | Says | Fired | Correct |
|---|---|---|---|
| calm market **and** flat ASPI **and** no big drop | Hold | 692 | **49.3%** |
| calm market **and** ASPI up **and** not extended | Hold | 457 | 46.4% |
| jumpy market **and** below MA5 **and** calm 10-day | Buy | 504 | 41.1% |

The rules are sensible — mean reversion after spikes, momentum continuation in calm markets — and
they are simply not reliable enough to be useful.

Direct co-movement counting (no model) shows where the only real association lies:

| Indicator | Lift at 1 day | Lift at 1 month |
|---|---|---|
| **ASPI (market)** | **+8.3 pp** | +7.1 |
| **Peer banks** | **+6.1 pp** | +6.4 |
| Policy rate change | −0.7 | −8.8 |
| USD/LKR | −1.6 | −11.8 |
| Money supply | −0.4 | −1.7 |
| Industrial production | −0.0 | +0.9 |

Only **market and peer co-movement** is real and well-sampled — which is market beta, not
forecasting power. Macro lifts are ≈ 0 wherever the sample is large.

---

## 9. Data collected and cleaned

| Dataset | Frequency | Coverage | Notes |
|---|---|---|---|
| 11 CSE stocks + ASPI | daily | 2012–2026 | OHLCV |
| Interest rates (policy, SDFR, SLFR, T-bills, AWDR, AWPR, AWLR, spread) | monthly | 2003–2026 | |
| **Inflation (CCPI + NCPI)** | monthly | **2003–2026** | chained across **four** base years |
| **USD/LKR** | **daily** | 2010–2026 | dense from 2014 |
| **Money supply (M1, M2, M2b, M4)** | monthly | 2010–**2024-08** | CBSL export is ~2 years stale |
| **Industrial production (IIP)** | monthly | 2016–2026 | **not** seasonally adjusted |
| Global factors (oil, US 10Y, DXY) | daily | 2015–2026 | |
| Policy-rate events | event | 2012–2026 | 35 events |

Two cleaning problems required real handling:

**CPI arrives in four base years** (2002, 2006/07, 2013, 2021), each restarting at 100.
Concatenating them would create fake 50% crashes at every rebase. They were **chained** onto the
2021 base using overlap ratios. Validation: native-base months match the source PDFs to **0.05 pp**,
and the September 2022 peak reproduces at **68.7% YoY** against an actual of ~70%.

**IIP is not seasonally adjusted.** April falls 10–34 points *every single year* (Sinhala/Tamil New
Year) and March always peaks. Month-on-month change would therefore encode the holiday calendar,
not the economy, so only **year-on-year** features were used; the column is named
`iip_mom_pct_DO_NOT_USE` to prevent later misuse.

---

## 10. Limitations

- **One market, one sector.** Results may not generalise beyond CSE banking and finance.
- **Single-country macro.** Sri Lanka's 2020–2023 crisis is unusual; the regime effect in §6.2 may
  be specific to it.
- **Stocks are not independent.** All 11 share the ASPI market factor, so every p-value quoted is
  *optimistic*, not conservative.
- **Long horizons have very few independent windows** — at 252 days, ~2–12 per stock. No claim at
  6 months or 1 year should be treated as statistically meaningful, in either direction.
- **Money supply ends 2024-08**, so it could only be tested on an earlier window.
- **Not tested:** foreign investor flows, dividend/ex-dividend dates, earnings announcements, news
  sentiment. These are event-driven and textual sources, deliberately reserved for a later stage.
- **Correlation, not causation** throughout.

---

## 11. Conclusion

Across **64 logged experiments**, three modelling paradigms (statistical, tree-based, deep
learning), 11 stocks, 9 horizons, per-stock and pooled designs, and every classic macroeconomic
variable used in the reference literature:

1. **CSE banking/finance stock direction cannot be predicted** by the information tested.
2. **Macroeconomic data does not improve on plain price history** — at monthly or daily frequency,
   in levels or changes, alone or combined.
3. **The only reliable co-movement is market beta** — stocks follow the index, which is descriptive,
   not predictive.
4. **Apparent skill is period-dependent**, demonstrated three times independently.
5. **Every candidate positive result failed replication**, including one that reached p = 0.062
   before being refuted on 21 independent windows.

The contribution is not a forecasting model. It is a **rigorously bounded negative result**, with
the evaluation methodology — matched baselines, purged splits, publication lags, pre-registered
claims, replication across stocks, and walk-forward re-testing — as the transferable output.

---

## Appendix A — Reproducibility

| Script | Purpose |
|---|---|
| `src/clean_macro_sources.py` | CBSL/DCS PDFs, HTML and CSV → clean CSVs, with quality report |
| `src/direction_multi_horizon.py` | Phase A — 9 horizons, direction + return targets |
| `src/direction_phase_ablation.py` | Phases A → B → C → C2 → D (add a phase = add one list) |
| `src/direction_sector_sweep.py` | Replication of Phase A vs D across 11 stocks |
| `src/direction_daily_macro.py` | Phase E — daily FX, oil, US 10Y, DXY, CPI |
| `src/direction_money_supply.py` | Phase F — money supply, window moved to 2024-08 |
| `src/direction_industrial_production.py` | Phase G — industrial production |
| `src/direction_pooled_and_rules.py` | Co-movement table, pooled panel model, readable rules |
| `src/direction_1day_retest.py` | Walk-forward re-test of the 1-day claim (21 folds) |
| `src/direction_lstm_gru.py` | LSTM/GRU for direction, with flattened control |
| `src/build_results_log.py` | Regenerates `results_master_log.xlsx` |

All outputs live under `results/direction/`. Seeds fixed throughout; recurrent models averaged over
three seeds by majority vote.

## Appendix B — Reading any results table in this project

- **`edge_pp` is the only number that matters** — model accuracy minus the best naive baseline.
  Raw accuracy is meaningless on its own: 78% can be poor (if the baseline also scores 78%) and 40%
  can be less poor.
- **Never compare accuracy across horizons or windows.** Baselines differ. Compare edge.
- **Never compare RMSE across horizons.** A year moves more than a day. Compare the ratio to the
  matched null.
- **Check `indep_windows`** before believing any long-horizon cell.
