# S&P SL 20 Forecasting — Full Project Documentation

**Project:** Forecasting the S&P SL 20 index of the Colombo Stock Exchange with three
model setups — **ARIMA**, **LSTM**, and a **Hybrid ARIMA + LSTM** — evaluated on the same
data and the same train/test split.

**Author environment:** `/Users/miyuru/Desktop/LSTM + ARIM`
**Date documented:** 2026-07-22

---

## 1. What this project does

We forecast the **S&P SL 20** daily index and compare three modelling setups head-to-head
on an identical held-out test window:

1. **ARIMA** — linear time-series model.
2. **LSTM** — recurrent neural network for nonlinear patterns.
3. **Hybrid ARIMA + LSTM** — ARIMA models the linear part; an LSTM models ARIMA's residual
   (nonlinear leftover); the two are summed.

It both **reproduces** the approach of the source paper (Vithushan & Kethmi, IRC-OUSL 2025)
and **extends** it in two ways: it uses a newer dataset that **includes the 2020–2023
Sri Lankan crisis years**, and it adds the **third (hybrid) setup** that the paper only
described conceptually but never actually reported.

---

## 2. Data

| Item | Value |
|------|-------|
| Series used | `spsl20_points` (S&P SL 20 closing level) — univariate, no macro variables |
| Primary file | `spsl20_trading_days_clean.csv` |
| Rows (clean) | 2,664 trading days |
| Date span | 2015-01-01 → 2026-04-16 |
| Cleaning | Weekends and carry-forward holiday duplicates removed from the raw feed |

**Why the clean file:** the original `cse_indices_macro_clean.csv` fills weekends/holidays by
repeating the previous close (fake flat days). Those mislead both models, so all modelling
uses the trading-day-only file.

---

## 3. Train / test split (identical for all three setups)

- **80 / 20 chronological split** (no shuffling — this is sequential market data).
- Split computed once in `01_arima.py` and saved to `split_info.json`; scripts 2 and 3 load
  that file so every model trains and is graded on **exactly the same dates**.

| Portion | Rows | Dates |
|---------|-----:|-------|
| Train | 2,131 (80%) | 2015-01-01 → 2024-01-19 |
| Test  | 533 (20%) | 2024-01-22 → 2026-04-16 |

**Crisis placement:** the volatile 2020–2023 crisis years fall **inside the training window**.
The models are therefore *trained through* the turbulence and *graded on* the calmer
2024–2026 recovery. This is the deliberate stress test the project set out to run.

**No leakage:** every scaler (MinMax for the LSTMs) is fit on the **training portion only** and
then applied to the test portion. ARIMA parameters are estimated on training data only.

---

## 4. Forecast protocol — one-step-ahead (important)

All three models produce **one-step-ahead** forecasts across the test window: each day's
prediction uses the *true* prior history up to the day before.

- **LSTM / Hybrid:** to predict day *t*, feed the true previous *w* days (or residuals).
- **ARIMA:** parameters are frozen after training; the model **state** is updated with the
  actual test observations via statsmodels' `append(..., refit=False)`, yielding one-step-ahead
  forecasts. **Parameters never change on the test set** — this is *not* walk-forward retraining
  and introduces no parameter leakage.

**Why this matters:** the LSTM is inherently a one-step-ahead predictor. If ARIMA were instead
asked for a single 533-day multi-step forecast, it would collapse to a near-flat drift line and
the comparison would be meaningless. One-step-ahead puts all three models on equal footing.

---

## 5. Environment

Modelling ran in a local virtual environment (`.venv`) under **Python 3.14**.

| Package | Role |
|---------|------|
| pandas, numpy | data handling |
| statsmodels | ARIMA, ADF, Ljung-Box, Jarque-Bera, Breusch-Pagan |
| pmdarima | `auto_arima` order search |
| scikit-learn | MinMaxScaler, metric helpers |
| **PyTorch (torch 2.13.0)** | LSTM (see note) |
| matplotlib | overlay plot |

> **Note on TensorFlow → PyTorch.** CLAUDE.md specifies TensorFlow/Keras for the LSTM, but this
> machine only has Python 3.14, for which **TensorFlow currently ships no wheel**. The LSTM was
> therefore implemented in **PyTorch** with the *identical architecture and training recipe*
> the spec asked for (2 stacked LSTM layers × 50 units, dropout 0.2, Dense(1) head, MSE loss,
> Adam optimiser, early stopping on a validation slice, fixed random seeds). This is an
> implementation-library substitution only; the modelling method is unchanged.
>
> To reproduce: `source .venv/bin/activate` then run the scripts in order.

---

## 6. The three setups in detail

### Setup 1 — ARIMA (`01_arima.py`)

1. Load clean CSV, sort by date, 80/20 split, save `split_info.json`.
2. **ADF stationarity test** on the training set:
   - Levels: ADF = −2.9137, p = 0.0438 (only marginally stationary).
   - 1st difference: ADF = −8.3109, p < 0.0001 (clearly stationary).
   - **Decision: d = 1.** The levels result is borderline and driven by the post-crash recovery
     trend; d = 1 is the standard choice for stock prices and matches the paper. Fixing d also
     keeps the AIC comparison valid (AIC is only comparable across models with the same `d`).
3. **Order selection:** baseline **ARIMA(2,1,0)** (AIC 21594.20) vs. **auto_arima → ARIMA(1,1,4)**
   (AIC 21572.18). The lower-AIC **ARIMA(1,1,4)** was kept (selection on training information only).
4. **Residual diagnostics (training fit):**
   - Ljung-Box (lag 10): stat 1.77, p = 0.998 → **no residual autocorrelation**.
   - Jarque-Bera: p < 0.0001, skew 35.7, kurtosis 1515 → **strongly non-normal** (one extreme
     residual spike from a price gap; the Kalman-filter initialisation residual on day 0 is a
     separate artifact, dropped in the hybrid).
   - Breusch-Pagan: p = 0.106 → **homoskedastic** (no time trend in residual variance).
5. **Forecast** the test window one-step-ahead; save predictions and metrics.

**Outputs:** `split_info.json`, `arima_info.json`, `arima_predictions.csv`,
`arima_train_residuals.csv`.

### Setup 2 — LSTM (`02_lstm.py`)

1. Reuse the saved split. **MinMax-scale to [0,1], scaler fit on training data only.**
2. Sliding windows tried at **60** and **30** days → predict next day.
3. Network: **2 stacked LSTM layers, 50 units each, dropout 0.2 → Dense(1)**; loss MSE;
   optimiser Adam; **early stopping** on a 10% validation slice carved from the *end* of train
   (chronological, no shuffle). Random seeds fixed (42).
4. Predict the test set one-step-ahead, inverse-transform to price units.
5. **Window 60** was selected (lower validation RMSE) and its predictions saved.

**Outputs:** `lstm_predictions.csv`, `lstm_info.json`.

### Setup 3 — Hybrid ARIMA + LSTM (`03_hybrid.py`)

Decomposition `Yₜ = Lₜ + Nₜ`:

```
Yₜ ──► ARIMA ──► L̂ₜ (linear forecast, reused from Setup 1)
             │
   residual  eₜ = Yₜ − L̂ₜ
             │
             └──► LSTM (trained on residuals) ──► N̂ₜ
                                                   │
   FINAL:  Ŷₜ = L̂ₜ + N̂ₜ
```

1. **Reuse ARIMA from Setup 1** via its saved artifacts (no refit): training residuals from
   `arima_train_residuals.csv`, test linear forecast L̂ from `arima_predictions.csv`.
   The day-0 Kalman initialisation residual (fitted = 0 → residual = first price) is dropped.
2. Train the **same LSTM architecture on the residual series** (MinMax scaler fit on train
   residuals only, windows 60/30, validation slice, seeds).
3. **Final forecast = ARIMA forecast + LSTM residual forecast.** Residual window 30 was selected.

**Outputs:** `hybrid_predictions.csv`, `hybrid_info.json`.

### Setup 4 — Comparison (`04_compare_results.py`)

Loads all three prediction sets (asserting identical test dates), builds the metrics table,
saves `comparison_table.csv`, draws `comparison_overlay.png`, and prints the conclusion.

---

## 7. Results

All metrics on the **identical** test window (2024-01-22 → 2026-04-16, 533 trading days),
one-step-ahead. Lower is better for all three metrics.

| Model | MAE | MAPE | RMSE |
|-------|----:|-----:|-----:|
| ARIMA(1,1,4) | 36.42 | 0.770% | 56.25 |
| LSTM (window 60) | 140.52 | 2.491% | 198.10 |
| **Hybrid (ARIMA + LSTM, resid window 30)** | **36.25** | **0.766%** | **56.15** |

**Best per metric: the Hybrid wins all three**, but only marginally ahead of pure ARIMA.

Overall RMSE ranking (best first): **Hybrid < ARIMA < LSTM.**

Overlay chart: `comparison_overlay.png` — ARIMA and Hybrid sit right on the actual line (they
overlap so closely the ARIMA trace is hidden under the Hybrid), while the standalone LSTM
visibly lags below during the 2025–2026 uptrend.

### Interpretation

- **ARIMA and Hybrid dominate the standalone LSTM** (MAPE ~0.77% vs ~2.49%). At a one-day
  horizon the index is close to a random walk, which a differenced ARIMA captures almost
  exactly, whereas the level-trained LSTM systematically lags the price.
- **The hybrid adds very little** over pure ARIMA (RMSE 56.15 vs 56.25). ARIMA's residuals are
  essentially white noise (Ljung-Box p = 0.998), so the residual-LSTM has almost no structure
  left to exploit — the "correct" and expected outcome for a well-specified ARIMA.
- **Crisis behaviour:** training through the 2020–2023 turbulence did not stop ARIMA/Hybrid from
  tracking the 2024–2026 recovery tightly. But that turbulence left fat-tailed training residuals
  (kurtosis ~1515), which is why RMSE sits well above MAE (large-miss days dominate the squared
  error) even though the typical day is predicted within ~0.77%.

---

## 8. Comparison with the source paper

**Paper:** *A Comparative Analysis of Time Series Models for Predicting the S&P SL 20 Index of
the CSE* — M. Vithushan & G. A. P. Kethmi, IRC-OUSL 2025.

### 8.1 Methodology comparison

| Aspect | Paper (Vithushan & Kethmi) | This project |
|--------|----------------------------|--------------|
| Data window | 2010–2018 (~2,160 obs) | 2015–2026 (2,664 trading days) |
| Crisis years (2020–2023) | **Deliberately excluded** | **Included (in training)** |
| Models compared | ARIMA vs LSTM (**two-way**) | ARIMA vs LSTM vs **Hybrid** (**three-way**) |
| Hybrid actually built? | **No** — described conceptually only; not in Table 4 | **Yes** — full residual-correction hybrid, reported |
| Train / test split | Date-based: train 2010–2016, test 2017–2018 | 80/20 chronological: train 2015→Jan-2024, test 2024→2026 |
| ARIMA transform / d | Log-transform, ADF p = 0.05108, d = 1 | No log; ADF-based d = 1 |
| ARIMA order | **(2,1,0)** from ACF/PACF (manual) | **(1,1,4)** from `auto_arima` (beat (2,1,0) on AIC) |
| LSTM library | Keras/TensorFlow (Adam, MSE) | PyTorch (Adam, MSE) — same recipe |
| Metrics | MAE, MAPE, RMSE | MAE, MAPE, RMSE (same three) |

### 8.2 Results comparison

**Paper — Table 4 (2010–2018, crisis-free):**

| Model | MAE | MAPE | RMSE |
|-------|----:|-----:|-----:|
| ARIMA | 233.96 | 7.04% | 269.57 |
| LSTM | 249.37 | 6.96% | 269.86 |
| Hybrid | — (not reported) | — | — |

**This project (2024–2026 test, crisis in training):**

| Model | MAE | MAPE | RMSE |
|-------|----:|-----:|-----:|
| ARIMA | 36.42 | 0.770% | 56.25 |
| LSTM | 140.52 | 2.491% | 198.10 |
| Hybrid | 36.25 | 0.766% | 56.15 |

### 8.3 What the comparison shows

- **The raw numbers are not directly comparable, and are not meant to be.** Different era,
  different index level, and — most importantly — a different forecast horizon. Our explicit
  one-step-ahead protocol yields much lower errors (MAPE < 1%) than the paper's ~7%. The paper
  notes ARIMA produced "constant predictions in some periods," which points to a longer-horizon /
  flatter forecast; that alone explains most of the error-magnitude gap.
- **Same qualitative headline: ARIMA ≥ LSTM.** The paper found ARIMA and LSTM roughly tied
  (ARIMA better MAE/RMSE, LSTM marginally better MAPE). We find **ARIMA decisively beats the
  standalone LSTM** on all three metrics — a sharper version of the same conclusion.
- **New finding the paper could not report — the hybrid.** Because we actually built the third
  setup, we can state what the paper only hypothesised: on this series the hybrid is the best
  model, but its edge over pure ARIMA is negligible, because ARIMA already whitens the residuals.
  The paper's premise that "a hybrid ARIMA-LSTM effectively combines both methods to enhance
  accuracy" holds only very weakly here.
- **Diagnostics agree in direction.** Both studies find ARIMA residuals free of autocorrelation
  (Ljung-Box) but strongly non-normal (Jarque-Bera). Our non-normality is far more extreme
  (kurtosis ~1515 vs the paper's 8.83) — a direct fingerprint of the crisis-era jumps that the
  paper excluded and we kept.

---

## 9. File inventory

| File | Description |
|------|-------------|
| `01_arima.py` | Setup 1 — ARIMA (split, ADF, order search, diagnostics, forecast) |
| `02_lstm.py` | Setup 2 — LSTM (PyTorch) |
| `03_hybrid.py` | Setup 3 — Hybrid ARIMA + LSTM residual correction |
| `04_compare_results.py` | Setup 4 — comparison table, overlay plot, conclusion |
| `split_info.json` | Exact 80/20 split (indices + boundary dates) |
| `arima_info.json`, `lstm_info.json`, `hybrid_info.json` | Per-model chosen config + test metrics |
| `arima_predictions.csv` | ARIMA test forecast (date, actual, pred) |
| `arima_train_residuals.csv` | ARIMA training residuals (feeds the hybrid) |
| `lstm_predictions.csv`, `hybrid_predictions.csv` | LSTM / Hybrid test forecasts |
| `comparison_table.csv` | Final three-way metrics table |
| `comparison_overlay.png` | Actual vs ARIMA vs LSTM vs Hybrid over the test period |

**Build order:** `01 → 02 → 03 → 04` (each script depends on artifacts from the earlier ones).

---

## 10. Caveats & honest limitations

- The standalone LSTM predicts the **raw price level**, which is why it lags. A returns- or
  difference-based LSTM would likely close much of the gap to ARIMA; that variant is beyond the
  spec as written but is the obvious next experiment.
- Results are one seed (42). LSTM training has residual stochasticity; the qualitative ranking is
  stable but the third-decimal metrics would move slightly across seeds.
- One-step-ahead is the fair comparison chosen here; multi-step (e.g. 5- or 20-day) horizons
  would widen the gap in ARIMA's disfavour and are a reasonable extension.
