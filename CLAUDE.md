# CLAUDE.md — S&P SL 20 Forecasting Project

This file gives Claude (and any collaborator) the full context, methodology, and conventions for this project. Read it before doing any modeling work here.

---

## 1. Project overview

We forecast the **S&P SL 20 index** of the Colombo Stock Exchange and compare three modeling setups on the **same data and the same train/test split**:

1. **ARIMA** — traditional linear time-series model.
2. **LSTM** — recurrent neural network for nonlinear patterns.
3. **Hybrid ARIMA + LSTM** — ARIMA models the linear part, LSTM models ARIMA's residual (nonlinear) part, and the two are summed.

The aim is to measure and compare the **accuracy of all three** and determine which forecasts the index best.

### Relationship to the source paper
Based on *"A Comparative Analysis of Time Series Models for Predicting the S&P SL 20 Index of the CSE"* (IRC-OUSL 2025, Vithushan & Kethmi). The paper compares **ARIMA vs. LSTM separately** on 2010–2018 data.

**How this project differs / extends it:**
- We use **our own dataset spanning 2015–2026, which INCLUDES the Sri Lankan crisis years (2020–2023)** that the paper deliberately excluded. Testing the models through that volatility is a core goal.
- We add a **third setup: the combined ARIMA + LSTM hybrid**, so the final comparison is three-way, not two-way.

Because the data window and volatility differ, our metrics will NOT match the paper's Table 4 — that is expected and intended.

---

## 2. Data

**Original source:** `cse_indices_macro_clean.csv` — multiple series, daily, 2015-01-01 → 2026-04-16.

**We use ONLY the S&P SL 20 series (`spsl20_points`). Univariate. No macro variables.**

**Working files (already created):**
| File | Rows | Use |
|------|------|-----|
| `spsl20_dataset.csv` | 2,946 | Raw separated series (`date, spsl20_points`), every calendar day |
| `spsl20_trading_days_clean.csv` | 2,664 | **PRIMARY FILE — use this.** Weekends and carry-forward holiday duplicates removed |

**Data-quality note:** the original CSV fills weekends/holidays by repeating the previous close (e.g. 2015-01-01 and 01-02 both = 4089.1399). Those fake flat days mislead both models, so all modeling uses `spsl20_trading_days_clean.csv`.

---

## 3. Train / test methodology (critical)

The model **learns on past data and is graded on data it has never seen.** This is how we measure real forecasting accuracy rather than memorization.

**Rules:**
1. **Split by time, never randomly.** This is sequential stock data — shuffling would let future information leak into the past and produce fake-good scores.
2. **80 / 20 chronological split** (DEFAULT):
   - **Training set** = earliest ~80% of trading days (≈ 2015 → 2023).
   - **Test set** = latest ~20% of trading days (≈ 2024 → 2026), held out and never seen during training.
3. **Identical split for all three setups.** ARIMA, LSTM, and Hybrid must train and be evaluated on exactly the same dates, or the comparison is invalid.
4. **No leakage.** Any scaling (MinMax for LSTM) is fit on the **training portion only**, then applied to the test portion.
5. A small **validation slice** may be carved from the end of the training set for LSTM early stopping.

**Optional extension (documented, not default):** walk-forward / rolling-origin evaluation, where the model retrains as it steps through the test period. More realistic for crisis-era volatility but slower and noisier for a clean 3-way comparison. Only use if explicitly requested.

---

## 4. The three setups in detail

### Setup 1 — ARIMA
1. ADF stationarity test → choose differencing order `d` (paper used log-transform + `d=1`).
2. Inspect ACF / PACF → choose `p`, `q`. Start from **ARIMA(2,1,0)** (paper's choice); also run `auto_arima` to search.
3. Fit on training set.
4. Residual diagnostics: Ljung-Box, Jarque-Bera, heteroskedasticity.
5. Forecast the test window; record MAE, MAPE, RMSE.

### Setup 2 — LSTM
1. MinMax-scale to [0,1] — **scaler fit on training data only.**
2. Build sliding windows: last `n` days (default 60, try 30) → predict next day.
3. Network: 1–2 stacked LSTM layers (~50 units each) + dropout (~0.2) → Dense(1).
4. Loss = MSE, optimizer = Adam, early stopping on validation slice.
5. Predict on test set, inverse-transform to price units; record MAE, MAPE, RMSE.

### Setup 3 — Hybrid ARIMA + LSTM (residual/error-correction)
Series = linear part + nonlinear part → `Yₜ = Lₜ + Nₜ`.

```
Yₜ ──► ARIMA ──► L̂ₜ (linear forecast)
             │
   residual  eₜ = Yₜ − L̂ₜ   (nonlinear leftover)
             │
             └──► LSTM (trained on residuals) ──► N̂ₜ
                                                   │
   FINAL:  Ŷₜ = L̂ₜ + N̂ₜ
```

Steps:
1. Fit ARIMA (Setup 1); get fitted values + test forecast `L̂ₜ`.
2. Compute residuals `eₜ = Yₜ − L̂ₜ` on the training portion.
3. Train LSTM (Setup 2 architecture) **on the residual series**: MinMax-scale residuals (fit on train only), window of ~30–60 past residuals → predict next residual.
4. Predict residuals `N̂ₜ` across the test set, inverse-transform.
5. **Final forecast = ARIMA forecast + LSTM residual forecast.** Record MAE, MAPE, RMSE.

---

## 5. Evaluation & comparison

All three setups scored on the **identical test set** using the paper's three metrics:
- **MAE** — mean absolute error (price units).
- **MAPE** — mean absolute percentage error.
- **RMSE** — root mean squared error (penalizes large misses).

Deliverables:
- One comparison table (mirror the paper's Table 4) with all three setups.
- One overlay plot: actual vs. ARIMA vs. LSTM vs. Hybrid over the test period.
- Short conclusion naming the best model per metric and noting trade-offs (especially behavior through the crisis years).

---

## 6. Project structure

```
LSTM + ARIM/
├── cse_indices_macro_clean.csv          original (all series)
├── spsl20_dataset.csv                    separated raw SPSL20
├── spsl20_trading_days_clean.csv         separated, cleaned — USE THIS
├── CLAUDE.md                             this file
├── WORKFLOW_and_CHECKLIST.md             stage-by-stage workflow + checklist
├── 01_arima.py        (or .ipynb)        Setup 1
├── 02_lstm.py                            Setup 2
├── 03_hybrid.py                          Setup 3
└── 04_compare_results.py                 metrics table + overlay plot
```

Build order: **01 → 02 → 03 → 04**, each with its own evaluation, then the combined comparison.

---

## 7. Environment

Python 3 with:
```
pandas  numpy  matplotlib
statsmodels  pmdarima         # ARIMA + auto_arima + ADF
scikit-learn                  # MinMaxScaler, metrics
tensorflow / keras            # LSTM
```

---

## 8. Conventions & rules of thumb
- Always use `spsl20_trading_days_clean.csv` unless a task explicitly says otherwise.
- Keep the 80/20 chronological split constant across all setups; do not shuffle.
- Fit scalers on training data only — never touch the test set during fitting.
- Set random seeds where possible so LSTM runs are reproducible.
- Report all three metrics (MAE, MAPE, RMSE) for every setup, on the same test window.
- Expect results to differ from the paper because of the crisis-year data — interpret, don't "fix" to match.

---

## 9. Status / next step
Datasets separated ✔ · Workflow documented ✔ · Methodology agreed ✔.
**Next:** build `01_arima.py` (Setup 1), then LSTM, then Hybrid, then the comparison.
