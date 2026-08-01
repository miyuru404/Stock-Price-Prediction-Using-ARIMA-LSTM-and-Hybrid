# Multi-Horizon Direction + Return Testing — Plan & Handoff

> Read this to continue the project in a NEW chat without losing progress. Self-contained.
> Style rule: explain SIMPLE (caveman) — short, plain words, key points only.

---

## 0. The big picture (what we learned so far)

- **Price level = unpredictable.** ARIMA, LSTM, Hybrid, Prophet, transformers — NONE beat a naive
  baseline, at any timescale or split. Price is a random walk. (Proven, many tests.)
- **Direction (Buy/Sell/Hold) is the honest reframe** — predict *which way*, not the number.
- **Step 1 baseline (done):** HNB next-day direction, price+technical only, XGBoost = **36.2%**,
  loses to persistence (43.1%). So **price alone has no edge** — the floor to beat.
- **Sector finding (real):** banks & finance are separate clusters; banks' returns react
  **negatively** to a widening lending-deposit **spread** (short-run). Matches Sri Lankan bank
  literature (higher rates hurt bank margins/ROE). Long-run link is weak (ARDL/VECM).
- **The whole project question now:** *can adding real info (macro / events / sentiment) push
  direction accuracy above the naive baseline?*
- **Phase A done (2026-08-01):** HNB, 9 horizons (1d→1yr), Tier-1 only, direct models.
  **0 of 9 horizons beat the baseline.** Longer horizon makes it *worse*, not better
  (−3.5 pp at 1d → −47.1 pp at 1yr), because long-horizon drift inflates the dumb baseline.
  Return % target: Ridge beats naive-0 but only ties train-mean drift → **drift, not signal**.
  Logistic > XGBoost at every horizon → no nonlinear structure in Tier-1 features.
  → `results/direction/multi_horizon/mh_summary.md`, `src/direction_multi_horizon.py`.
- **Two nulls added (keep using them):** *train-mean drift* for return RMSE, and
  *always-guess-the-winning-side* for sign accuracy. Without them, 55–57% sign accuracy at
  1–3 months looks like a hit but is not.
- **Phase B done (2026-08-01):** + Tier-2 (RSI, MACD, volume). Same rows/split/seeds as A, only
  features change. **Mean gain −1.1 pp; positive at 1/9 horizons; 0/9 beat baseline.**
  XGBoost gives Tier-2 **41%** of its importance (macd_signal is the top feature at 6/9 horizons)
  and still gains nothing → the indicators are **redundant with price**, not additive.
  Return %: Tier-2 made RMSE slightly worse at 9/9. → `results/direction/phase_ablation/`,
  `src/direction_phase_ablation.py` (reusable — add a new phase by adding a feature list).
- **Technical features are now exhausted.** Phases A and B together say: nothing in the price
  chart predicts HNB at any horizon from 1 day to 1 year. The remaining hope is Phase C+
  (macro / events / sentiment) — information that is *not* in the chart.
- **Phase C done (2026-08-01):** + macro interest rates (13 features), 35-day publication-lag
  guard so no rate is seen before it was released. **Mean gain −2.8 pp; 0/9 beat baseline.**
  Not just negative — **unstable**: gain swings −22.3 to +10.3 pp, and return RMSE blows out to
  1.37× the train-mean null (A/B sat at ~0.99). That is overfitting, not weak signal.
- **BIG METHODOLOGY LESSON (Phase C2 diagnostic):** rate **levels** are non-stationary and get
  memorised ("rates were 15% in 2022") — worthless out of sample. Dropping levels and keeping only
  **changes** recovered **+3.4 pp** (1-year went 4.5% → 49.7%). **Always feed macro as changes,
  never levels.** Applies to every future phase.
- Even in change form, macro still beats the baseline at **0/9** horizons. Rates explain bank
  returns *contemporaneously* (the earlier spread work) but do **not forecast** them.
- **Phase D done (2026-08-01):** + sector (ASPI returns/vol, relative strength vs ASPI and vs peer
  banks, peer-bank & finance composites, 60d corr and beta). Built **on C2**, not C.
  **Mean gain +2.6 pp — the first consistently positive step (6/9 horizons).** Still **0/9** beat
  the baseline, but short horizons are now near-parity: **1 week −0.2 pp**, 1 day −1.7, 2 weeks −1.6.
- **Two soft signals worth re-testing (NOT findings yet):**
  1. Phase D **return sign edge is positive at 1, 2 and 3 weeks** (+1.7 / +1.7 / +1.1 pp) — first
     positive sign edges in the study, and at three adjacent horizons.
  2. Importance splits by horizon: **sector dominates short** (42% at 1 day), **macro dominates
     long** (49% at 1 year). Mirror images.
  Both are from one stock and one split. **Confirm on other tickers before claiming anything.**
- **Ranking of what helps (HNB, direction):** sector +2.6 > macro Δ +3.4 vs C but 0 vs A/B >
  Tier-2 −0.9 > macro levels −2.8. Nothing crosses the baseline.
- **Sector sweep done (2026-08-01):** Phase A vs D on **11 stocks** (3 banks, 4 finance, 4 control),
  9 horizons = 99 cells. Peer composites exclude the target stock. → `src/direction_sector_sweep.py`,
  `results/direction/sector_sweep/`.
  - **Sector gain REPLICATES: +3.8 pp mean, positive in 66/99 cells, 9 of 11 stocks.**
    HNB was not a fluke. Best LFIN +13.2, SAMP +9.4, LOFC +8.2; worst CTC −11.0.
  - **The edge does NOT.** Phase D beats the baseline in 27/99 cells (Phase A: 14/99) — but a sign
    test across stocks gives **p = 0.500 at 1 day and 1 week** (6/11 stocks positive = exact coin
    flip) and worse at every longer horizon. **Nothing is significant anywhere.**
  - **HNB's 1–3 week sign edge is REFUTED:** 6/33 cells positive (18%) vs ~50% for a coin flip.
    Naming the claim before re-testing it is what caught this — keep doing that.
- **What sector features actually do:** they lift the model from *clearly worse than the baseline*
  up to *coin flip* at 1 day–1 week, and do nothing at longer horizons. Real effect, no edge.
- **Standing rule now proven twice:** a gain that replicates is still not a win. Only
  `edge_pp > 0`, holding across stocks, counts.

---

## PHASES AT A GLANCE (6 steps × 9 horizons × 2 targets)

| Phase | Add to features | Data status |
|---|---|---|
| **A** | Tier-1 technical (momentum, volatility, MA, returns) | HAVE |
| **B** | + Tier-2 technical (RSI, MACD, volume) | HAVE |
| **C** | + macro (rates, inflation, FX, money supply) | rates HAVE; rest COLLECT |
| **D** | + sector (spread, is_bank, ASPI) | HAVE |
| **E** | + events (dividends, rate decisions) | rate events HAVE; dividends COLLECT |
| **F** | + news sentiment | COLLECT |

Each phase is run across **9 horizons** (1d → 1yr) and **2 targets** (direction + return %).
The accuracy **gain from each phase** is the result.

---

## 1. This testing series — the plan

**Two targets, side by side (see if both improve together):**
1. **Direction** — Buy / Sell / Hold (±0.5% dead-zone → later maybe magnitude bands).
2. **Return %** — the % move over the horizon (NOT the price level → no random-walk trap).

**Multiple horizons (test all):**
`1 day, 1 week (5d), 2 weeks (10d), 3 weeks (15d), 4 weeks / 1 month (22d), 2 months (44d),
3 months (66d), 6 months (132d), 1 year (252d).`

**Method:** DIRECT — a **separate model per horizon** (no recursion / no compounding). Features use
only data up to "today"; label is the move to "today+horizon".

**Add indicators in STEPS (ablation) — measure accuracy gain at each step:**
- **Step A** = Tier-1 technical only (baseline).
- **Step B** = + Tier-2 technical.
- **Step C** = + macro (interest/inflation/FX/money supply).
- **Step D** = + sector (spread, is_bank, ASPI).
- **Step E** = + events (dividends, rate decisions).
- **Step F** = + news sentiment.
The **gain from each step = the scientific finding.**

**Sector-aware:** run on banks + finance + control (not just HNB).

---

## 2. Indicator ranking (add in this order)

| Tier | Indicators | Best for | Status |
|---|---|---|---|
| **1** | momentum(10d), volatility(20d std), MA ratios(5/10/20), recent returns(1/5/10) | short horizon | HAVE (compute) |
| **2** | RSI(14), MACD, volume change | short/mid | HAVE (compute) |
| **3 macro** | T-bill/policy rate, inflation, money supply, exchange rate | long horizon (monthly) | rates HAVE; inflation/FX/M2 COLLECT |
| **3b sector** | spread, is_bank, ASPI market return | banks, mid/long | HAVE |
| **4 events/text** | dividend flag, rate-decision flag, **news sentiment** | event days | COLLECT |

Rule: **technical → short horizon; macro/sentiment → long horizon.**

---

## 3. Models to use (simple → strong)

**Direction (classification):**
1. Logistic Regression (dumb baseline)
2. **XGBoost** (main)
3. LightGBM, CatBoost, Random Forest

**Return (regression):**
1. Linear / ElasticNet (baseline)
2. XGBoost / LightGBM regressor

**Later (need more data / long horizon):** LSTM, GRU, Bi-LSTM.

**Always compare to baselines at EVERY horizon:** persistence + majority-class (direction);
naive "no change" (return).

---

## 4. Data — HAVE vs COLLECT

**HAVE (in `cleaned_data/`):**
- Daily OHLCV: HNB, COMB, SAMP, LOFC, LOLC, LFIN, CFIN, JKH, DIAL, CTC, DIST, MELS (2012–2026)
- ASPI daily; HNB hourly
- Monthly rates: policy, sdfr, slfr, tb_3m, tb_12m, awdr, awpr, awlr, **spread**
- Rate-decision events (35); sector groupings

**COLLECTED 2026-08-01** (cleaned by `src/clean_macro_sources.py`, originals in `raw_exports/cbsl/`):
| File | Freq | Coverage | Verdict |
|---|---|---|---|
| `cleaned_data/usd_lkr_daily.csv` | **daily** | 2010-05 → 2026-07 (dense from 2014) | **BEST — use this** |
| `cleaned_data/inflation_monthly.csv` | monthly | 2003-01 → 2026-07 | OK, but monthly = flat within month |
| `cleaned_data/money_supply_monthly.csv` | monthly | 2010-01 → **2024-08** | ⚠ **ends inside the test window** |

- CCPI arrives in 4 base years (2002/2006-07/2013/2021) — chained into one continuous index on
  2021=100 via overlap ratios. Validated: native-base months match the PDFs to 0.05 pp; Sep-2022
  peak reproduces at 68.7% YoY. Splice factors in `inflation_ccpi_splice_factors.csv`.
- **Money supply warning:** the CBSL export has 24 blank trailing months. Test window is 2024–2026,
  so adding M1/M2 would delete almost the whole test set. Use it for explanation, NOT the direction test.
- USD/LKR before 2014 is patchy (30–162 obs/yr vs ~240 after). Start at 2014.

**STILL TO COLLECT:**
- Foreign investor net buy/sell, daily, per stock — CSE (highest value remaining)
- Dividend dates + amounts per stock — CSE
- CSE announcements / earnings dates + EPS
- News sentiment (economynext + FinBERT) — scrape + score

---

## 5. Evaluation rules (honesty)

- Chronological / walk-forward split. **Never shuffle.**
- Metrics: **accuracy, per-class precision/recall/F1, confusion matrix** (direction);
  RMSE + directional accuracy (return).
- **Must beat majority + persistence** or it adds nothing.
- Realistic target ~**52–58%** direction. **85%+ = data leak** → investigate.
- Long horizons = few real windows (overlap ≠ independent) → don't over-claim significance.
- Correlation, not causation.

---

## 6. Done / To-do

| Task | Status |
|---|---|
| Step 1 baseline (HNB, next-day, price-only, XGBoost) | ✅ done — 36.2%, loses to persistence |
| Multi-horizon runner (all horizons, Tier-1, XGBoost+Logistic+baselines) | ✅ done — **0/9 horizons beat baseline** |
| Return % target alongside direction | ✅ done — RMSE ≈ train-mean drift; sign edge 0/9 (real) |
| Phase B ablation (+Tier-2: RSI, MACD, volume) | ✅ done — **gain −1.1 pp, 0/9 beat baseline** |
| Phase C ablation (+macro rates) | ✅ done — **gain −2.8 pp, 0/9 beat baseline** |
| Phase D ablation (+sector: ASPI, peers) | ✅ done — **gain +2.6 pp (first positive), still 0/9** |
| Sector-aware run: repeat Phase D on banks/finance/control | ✅ done — **gain replicates, edge does not** |
| Phase E (events: dividends, rate decisions) | ⏳ NEXT |
| Phase F (news sentiment) | ⏳ to do |
| Collect macro (inflation/FX/M2), dividends, sentiment | ⏳ to do |
| Sector-aware run (banks/finance/control) | ⏳ to do |
| Calibrated confidence + economic (trading) backtest | ⏳ later |

---

## 7. Working notes for the AI (rules to keep)

- **Explain caveman-simple.** Short, plain words, key points only.
- **Never run git commit/push** — give the user the commands.
- **After every test, add a row to `results_master_log.xlsx`** (edit `src/build_results_log.py`, re-run).
- Environments: main `.venv` (has xgboost, statsmodels, sklearn); `.venv-tf` (torch, prophet, mlflow).
  Python 3.14 — some libs need care (TF absent; prophet needs libomp; lightgbm/catboost may need install).
- Key scripts: `src/direction_step1_baseline.py`, `src/spread_*`, `src/build_results_log.py`.
- Outputs live in `results/` (per-topic subfolders); master log = `results_master_log.xlsx`.

---

## 8. First move in the new chat
Build the **multi-horizon direction runner**: HNB, horizons [1,5,10,15,22,44,66,132,252] days,
Tier-1 technical features, **direct** per-horizon models (Logistic + XGBoost), vs persistence +
majority baselines. Also output the **return %** version. Report an accuracy-vs-horizon table +
plot. Then repeat for banks/finance/control. This shows: *does accuracy rise as horizon grows, and
does any horizon beat the baseline?*
