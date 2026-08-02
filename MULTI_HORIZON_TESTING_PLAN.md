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
- **Phase E done (2026-08-01):** daily macro, all 11 stocks. → `src/direction_daily_macro.py`,
  `results/direction/daily_macro/`. Tested the "monthly frequency was the problem" hypothesis.
  - **It wasn't.** Daily USD/LKR gain = **+0.0 pp** (positive in 57/99 cells = coin flip).
  - Adding daily oil / US 10Y / DXY / CPI made it **worse: −1.9 pp** (−6.0 pp at 1 year).
  - **0 of 36 phase×horizon combinations reach p < 0.05.** Best is 1 day, 6/11 stocks, p = 0.500 —
    the same exact coin flip the sector sweep found.
  - **The macro question is now closed at BOTH frequencies.** Not a data-frequency problem;
    the information is not there.
- **The recurring fingerprint (3rd sighting):** the model always *spends* importance on new
  features — Tier-2 41%, monthly macro 64%, daily macro+global+CPI 27% — and never gains accuracy.
  Using a feature ≠ profiting from it. Expect this again in any future phase.
- **Phase F done (2026-08-01):** money supply + the full classic macro set, **window moved** to
  2024-08 (where the CBSL money data ends) so M1/M2 got a fair test instead of being excluded.
  Test window becomes **2022-03 → 2024-08 — it contains the 2022 crisis.**
  → `src/direction_money_supply.py`, `results/direction/money_supply/`.
  - **Money supply gain: −0.7 pp**, positive in 50/99 cells = exact coin flip. Adds nothing.
  - Full classic set (money + rates + FX + inflation) = 4 of the 5 Naik & Padhi variables.
    **Only industrial production (IIP) is still uncollected** — the last macro gap.
- **⚠ WINDOW SENSITIVITY — important for the write-up.** Moving the test window into the crisis
  roughly **doubles** the win rate for *every* phase, including price-only (17–27/99 → 43–51/99).
  Results are **window-dependent, not model-dependent**. This argues for reporting walk-forward or
  multi-window results in the dissertation rather than a single 80/20 split.
- **The 1-day near-miss, and why it is not a finding:** 2 of 36 cells reached p < 0.05 (chance
  gives ~1.8). Both at 1 day — and the **strongest is Phase A, the PRICE-ONLY model**
  (9/11 stocks, +3.9 pp, p = 0.033). Adding money supply *shrinks* it to +3.2 pp. So it is the
  crisis window, not macro. Checking the null model is what stopped this becoming a false claim.
- **Phase G done (2026-08-02):** industrial production (DCS IIP, entered by hand, 2016-01 → 2026-05).
  → `src/direction_industrial_production.py`, `results/direction/industrial_production/`.
  - **IIP gain: −6.4 pp**, positive in only 25/99 cells, **0/36 significant.** It makes things worse.
  - This was the **strongest** variable in the Naik & Padhi reference paper. It still fails here.
  - **Seasonality guard:** DCS IIP is NOT seasonally adjusted — April drops 10–34 points every year
    (Sinhala/Tamil New Year), March always peaks. Only YoY-based features used; the mom column is
    literally named `iip_mom_pct_DO_NOT_USE`. Feeding mom would let the model "predict" the holiday.
  - Note: sample starts 2016, so this table is not comparable to the 2012-start phases.
- **★ THE MACRO CHAPTER IS COMPLETE.** All five classic macro variables from the reference paper
  are now tested against the same protocol:

  | Variable | Phase | Gain |
  |---|---|---|
  | Industrial production | G | −6.4 pp |
  | Money supply (M1/M2/M2b/M4) | F | −0.7 pp |
  | T-bill / policy rate | C | −2.8 pp (levels actively harmful) |
  | Exchange rate (daily) | E | +0.0 pp |
  | Inflation (CCPI) | E | part of −1.9 pp |

  **None beats plain price history.** Adding ALL macro (25/99 cells) is *worse* than sector-only
  (27/99). The headline result is now fully evidenced.
- **Why this does not contradict the literature:** papers like Naik & Padhi use cointegration/VECM
  to ask *"is there a long-run relationship?"* (in-sample, on levels). This project asks
  *"can you predict the next move?"* (out-of-sample, on direction). Macro **explains**; it does not
  **predict**. That distinction is the intellectual core of the write-up.
- **Pooled panel + co-movement + rules done (2026-08-02):** → `src/direction_pooled_and_rules.py`,
  `results/direction/pooled_rules/`. Three tests answering "can the model just learn *if X up then
  stock up*?"
  - **Co-movement (no model, pure counting):** market/peer indicators DO carry a real
    **+6 to +10 pp** lift at 1 day–1 month, on huge samples. Macro lifts are ~0 at short horizons.
    All the eye-catching macro lifts (T-bill −19.3, M2 +17.0) sit at 3 mo–1 yr where only
    **7–24 independent windows** exist → artifacts, discard.
  - **Pooled panel model:** 7 stocks stacked = **17,770 training rows** (vs ~2,700 per stock),
    split by DATE not row. Beat the pooled baseline at 1 day (+3.3) and 1 week (+2.7).
  - **⚠ FAIRNESS CHECK CAUGHT AN ARTIFACT:** pooling stocks with different class balances weakens
    the majority baseline. Re-scored per stock against each stock's OWN baseline:
    **1 week collapsed to −0.5 pp (3/7 stocks)** — the pooled win was fake.
    **1 day survived: 6/7 stocks, median +2.5 pp, p = 0.062.**
  - **That 1-day cell was the strongest result in the whole project — and it is now REFUTED.**
    See the walk-forward re-test below.
  - **Readable rules:** 17 plain-English rules with out-of-sample hit rates. Best 1-day rule
    (calm market + flat ASPI + no big drop → Hold) scores **49.3% vs 38.0% baseline**.
    The 1-month rule showing 68.5% fires only 317 times → noise.
- **★ 1-DAY CLAIM RE-TESTED AND REFUTED (2026-08-02):** → `src/direction_1day_retest.py`,
  `results/direction/retest_1day/`. Walk-forward, **21 non-overlapping 6-month windows**,
  retrained from scratch each fold, verdict rule fixed **in advance**, 1-week carried as a control,
  both models reported separately (no cherry-picking the winner per fold).

  | Horizon | Model | Folds positive | Median of fold medians | p | Survives? |
  |---|---|---|---|---|---|
  | 1 day | Logistic | 7/21 | **−1.8 pp** | 0.96 | **NO** |
  | 1 day | XGBoost | 6/21 | **−1.9 pp** | 0.99 | **NO** |
  | 1 week (control) | XGBoost | 7/21 | −6.0 pp | 0.96 | NO |

  Chance alone would give ~10.5 positive folds. We got 6–7. The +2.5 pp was **one lucky window**.
- **Window regime effect confirmed a SECOND time:** folds in 2021–2024 are mostly positive
  (6 of 8), folds in 2016–2020 mostly negative (1 of 10). Apparent skill is a property of the
  **period**, not the model. This is now backed by two independent runs and belongs in the
  methodology chapter as an argument for walk-forward reporting.
- **No per-stock consistency either:** LOFC and CFIN positive in 62% of folds, HNB in 19%.
  A real sector edge would show across the sector, not in two names.
- **LSTM / GRU for direction done (2026-08-02):** → `src/direction_lstm_gru.py`,
  `results/direction/lstm_gru/`. The last untested **model class** — and a genuinely different
  hypothesis: recurrent nets read the raw **sequence** of the last 30 days, while every earlier
  model read a flat row of summary features that discards the order.
  Pooled, 7 walk-forward 12-month folds, 3 seeds majority-voted, per-stock scoring.
  - **No recurrent model beats the naive baseline at any horizon.** Best is LSTM at 1 day:
    5/7 folds, +2.9 pp, **p = 0.227**.
  - **BUT the flat control worked and revealed something real:** LSTM beats the *same data
    flattened* at **all three** horizons — 1 day **+2.9 vs −1.7**, 1 week −0.4 vs −2.1,
    1 month −5.4 vs −9.9. **The order of recent days does carry a little signal that summary
    features throw away.** First time in the project that added complexity actually helped.
    Not enough to reach the baseline, but honest and worth reporting.
  - Closes the "did you try deep learning on direction?" question, and finally applies LSTM to the
    direction target (the project is named after it).
- **★ REGIME EFFECT — THIRD independent confirmation.** LSTM at 1 day: **0 of 2** folds positive in
  2019–2020, **5 of 5** positive in 2021–2025. Same shape as the money-supply run and the 1-day
  re-test. Same code, same features — only the period changes. This is the project's strongest
  methodological finding and it also explains away the 5/7 folds above.
- **Phase H done (2026-08-02) — NEWS SENTIMENT, the last untested source:**
  → `src/direction_phase_h_sentiment.py`, `src/finance_lexicon.py`,
  `results/direction/sentiment/`, `cleaned_data/news_sentiment_daily.csv`.
  149,240 Dailymirror/Newsfirst articles (Kaggle), **9,785 market-relevant**, 2016 → **2022-06**.
  - **Sentiment gain: −0.8 / −0.7 / +2.4 / +1.6 pp** at 1d / 1wk / 2wk / 1mo. All p ≥ 0.50.
    **No phase beats the naive baseline at any horizon.**
  - **Two lexicons (VADER + finance) correlate only 0.416** — two reasonable measures of the same
    articles substantially disagree. The signal is **fragile**, not just weak. Running one lexicon
    alone would have hidden this. Always run two.
  - **Look-ahead guard mattered:** CSE closes 14:30; **36%** of articles were published after the
    close and were reassigned to the NEXT trading day. Without it those articles report the very
    move being predicted.
  - **The regime confound did NOT appear here** — and it inverted: calm 2019–2020 gain **+0.85**
    (9/16 folds), crisis 2021–2022 **−0.45** (3/12). First phase free of the period effect.
  - **Honest limitation: only 7 folds per horizon = low power.** State this as "no detectable
    effect", not a firm null.
  - Upgrade path: drop the official Loughran-McDonald master dictionary into
    `cleaned_data/loughran_mcdonald_master.csv` and the script uses it automatically, no code change.
- **The "attention without payoff" fingerprint has now appeared SIX times:** Tier-2 41%,
  monthly macro 64%, sector 34%, daily macro 27%, money supply 29%, sentiment 30% — every one with
  zero or negative gain. Feature importance is not evidence of predictive value.
- **Twitter dataset REJECTED, do not use:** `SriLankaTweets.csv` covers **2022-07-10 16:55–20:51**
  — a single four-hour scrape, one calendar day, only 50% English. No time series exists in it.
- **★★ DIAGNOSTICS (2026-08-02) — the null is REAL, and one thing SURVIVED.**
  → `src/direction_diagnostics.py`, `results/direction/diagnostics/`.
  1. **POSITIVE CONTROL — the pipeline is not blind.** Same code, same folds. Using **today's**
     market/macro/news to predict **today's** direction: **68.6% vs 57.4% baseline, +8.2 pp,
     17/19 folds, p = 0.0004.** Using the *same information one day earlier*: **−4.1 pp, 3/19.**
     **Gap 13.7 pp.** This is the explain-vs-predict distinction demonstrated in one table, and it
     proves every earlier null is a real null rather than a broken evaluation.
     (I had to fix a leak in my own control first — it read 100% because the "lagged" features
     still contained today's return.)
  2. **Binary up/down did NOT rescue it** (−4.8 / −1.1 / −2.1 pp). The 3-class dead-zone was not
     hiding a signal.
  3. **★ MAGNITUDE-WEIGHTED EDGE — THE FIRST RESULT TO SURVIVE.** The model is *wrong more often*
     than the baseline (−4.3 pp plain accuracy) but *right on the big moves*:
     **+7.9 pp at 1 day (15/19 folds, p = 0.0096)** and **+13.2 pp at 1 week (15/19, p = 0.0096)**.
     **Accuracy was the wrong metric all along.**
  4. **But it is not exploitable.** Long/short backtest at h=1 (the only non-overlapping horizon):
     gross **+22.19%** vs buy-and-hold +6.49%, median **40 position flips** per 6-month fold →
     net **+17.35% @10bps · +11.24% @25bps · +0.05% @50bps**. At realistic CSE costs the edge is
     **exactly zero** (8/19 folds, p = 0.82).
  → Write-up line: *a statistically significant magnitude-weighted edge exists at 1-day and 1-week
  horizons and is entirely consumed by transaction costs* — a real finding AND an economic limit.
- **Price forecasting revisited with macro + news (2026-08-02):**
  → `src/price_with_macro_news.py`, `results/price_macro_news/`.
  **Naive unbeaten: 0 of 133 cases** at either horizon. Best model ARIMA(1,1,1) at 1.004× naive.
  **All 8 macro/news ablation combinations are NEGATIVE** (−0.008 to −0.078 pp) — adding macro and
  news makes price forecasting *worse*. Stage 1's univariate conclusion survives the multivariate test.
- **CSE illiquidity fact:** ~10% of bank/finance days close **exactly unchanged**
  (LOFC **36.0%**, HNB 12.9%, LOLC 7.2%). Part of why the naive baseline is so strong and why the
  Hold class dominates. Belongs in the limitations section.
- **⚠ KNOWN GAP — sentiment was MARKET-WIDE, not company-specific.** One national daily score was
  applied identically to all 7 stocks, so it could never explain why HNB differs from COMB — and
  ASPI/peer returns already capture market-wide moves far better. Sentiment was largely redundant
  by construction. Firm-specific news matching (by company name/ticker mention) is untested.
- **No multimodal architecture was used.** Text was reduced to a lexicon score before the model saw
  it, then concatenated into the flat feature table. A true multimodal design (text encoder +
  numeric encoder, fused at a hidden layer) is untested — reasonable "future work", though the
  0.42 lexicon agreement suggests limited upside.
- **★ INDEX TARGETS + A THIRD LEAK CAUGHT (2026-08-02):** S&P SL 20 and ASPI added to
  `src/price_with_macro_news.py`.
  - **The leak:** SPSL20 and ASPI do not share a trading calendar (207 SPSL20 dates absent from
    ASPI, 962 the other way). Forward-filling ASPI across those gaps let it carry a move that
    SPSL20 only recorded on its **next** row. `aspi_ret_1` correlated **0.742** with SPSL20's
    FUTURE return but only **0.374** with its own same-day return, and **halved MAPE** —
    ratio 0.496 vs a true 0.970.
  - **Fix:** (1) no forward-fill across mismatched calendars — missing dates become NaN and drop;
    (2) index targets get **no ASPI features at all** (same market measured twice, ~0.95 correlated).
  - **★ PERMANENT GUARD ADDED:** `leak_scan()` checks every feature's correlation with the
    same-day vs next-day return on every run and flags anything that tracks the future more
    closely. Writes `leak_scan.csv`. Now reports **clean**.
  - **Corrected index result:** best is Ridge price-only, **0.970 (1d) / 0.959 (5d)** vs naive,
    beating it in only 10/17 and 9/17 folds. Macro and news make it **worse** (1.00–1.10).
  - **Why indices differ from stocks — a real finding:** lag-1 return autocorrelation is
    **+0.237 (SPSL20)** and **+0.233 (ASPI)** versus **+0.038 (HNB)**. That is
    **non-synchronous trading**: illiquid constituents (LOFC unchanged on **37.8%** of days)
    reprice a day late and smear market shocks across two days. A **one-parameter AR(1)** captures
    the entire index "edge" — statistically real, economically useless, since exploiting it means
    trading the very illiquid names that cause it.
- **★ THREE leak-class bugs have now been caught in this project, and ALL THREE flattered the
  model:** (1) persistence baseline using a fixed 5-day lookback instead of the matched horizon;
  (2) a pooled baseline treated as a fair benchmark; (3) ASPI forward-filled across calendars.
  Errors that produce exciting results survive longer because favourable outcomes attract less
  scrutiny. This asymmetry belongs in the methodology chapter.
- **One-step vs multi-step:** all price and direction tests use the **DIRECT** method — the h-step
  move is predicted in one shot, never by iterating 1-day forecasts. Stage 1's multi-step was
  **recursive**, which is why naive dominated so heavily there (33% MAPE vs 2.9% here).
- **Two bugs caught in this run — keep both guards forever:**
  1. Persistence baseline must use the **matched horizon** (`past_h`), not a fixed 5-day lookback.
     The wrong version faked +0.8 and +1.5 pp edges at 2 and 3 weeks.
  2. A **pooled baseline is not a fair benchmark**. Always re-score per stock against its own
     baseline and sign-test across stocks.
- **SCOPE IS FIXED — do not change it again.** The project is: start from price only, then add
  macroeconomic variables **step by step**, and measure whether macro beats plain historical
  time-series forecasting. That question is the dissertation. Direction stays the target.
  Sentiment / dividends / earnings are a **later, separate stage** (different data type, harder to
  collect) and are not part of the macro chapter.
- **Volatility is deferred and OPTIONAL — it is NOT a replacement target.** If used at all, it is
  one short supporting section explaining *why* direction fails (the movement is real and its size
  is predictable; only the sign is not). Do not let a failed test trigger a scope change.

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
| Phase E (daily macro: FX, oil, US10Y, DXY, CPI) | ✅ done — **+0.0 pp, 0/36 significant** |
| Phase F (money supply + full classic macro set) | ✅ done — **−0.7 pp, macro chapter closed** |
| Phase G (industrial production) | ✅ done — **−6.4 pp. MACRO CHAPTER COMPLETE.** |
| Collect foreign reserves, trade balance (optional extras) | ⏳ optional |
| Later stage: events (dividends/earnings) then news sentiment | ⏳ separate chapter |
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
