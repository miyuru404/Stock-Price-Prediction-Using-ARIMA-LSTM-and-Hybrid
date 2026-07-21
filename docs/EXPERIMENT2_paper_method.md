# Experiment 2 — Paper-Style (Multi-Step) Forecast

**Purpose:** Re-run the ARIMA / LSTM / Hybrid comparison on the **same S&P SL 20 data**, but using the **forecasting method the source paper (26.pdf) used** — a true out-of-sample **multi-step** forecast — so our numbers are directly comparable to the paper's Table 4.

This does NOT replace Experiment 1. It runs alongside it and writes to **new filenames**, so we end up with both stories side by side:
- **Experiment 1 (already done):** one-step-ahead ("easy mode"). Model sees the real previous day each step.
- **Experiment 2 (this file):** multi-step ("hard mode", paper's way). Model forecasts the whole test window forward using **only its own predictions** — it never sees a real test value.

---

## 0. Important: does the model "remember" the test data from last time?

No. Each script run trains a **fresh model from scratch on the training slice only**; the test slice is held out and never used for fitting. Models have no memory between runs. Re-running is clean. The only real leakage risks are *within a run* (fitting model/scaler on test data, or tuning settings against the test score) — this spec avoids all of them: scaler fit on train only, ARIMA parameters estimated on train only, model selection on a validation slice, test touched only for final scoring.

---

## 1. Data & split (identical to Experiment 1)

- File: `spsl20_trading_days_clean.csv` (univariate S&P SL 20, crisis years included).
- Reuse `split_info.json` — the **same 80/20 chronological split** used by scripts 01–03, so all experiments share identical train/test dates.
- Never shuffle. Fit any scaler on the training portion only.

---

## 2. What changes vs. Experiment 1 (the core idea)

The ONLY thing that changes is **how the test window is forecast**. In Experiment 1, each test-day prediction used the *true* recent history. Here, the model is cut off from real test values and must roll forward on its own guesses — exactly the harder, more honest test the paper reported.

### Setup 1 — ARIMA (multi-step / dynamic)
- Follow the paper's modeling choices: **log-transform** the series, **d = 1**, baseline **ARIMA(2,1,0)** (also run `auto_arima` and keep the lower-AIC order, as before).
- Fit on the training set.
- Produce a **single dynamic forecast for the entire test horizon in one shot**: `get_forecast(steps=len(test))` (equivalently `predict(..., dynamic=True)` from the first test point). Do **NOT** call `append(test)` or feed any real test observation back.
- Inverse the log-transform (`exp`) before scoring.

### Setup 2 — LSTM (recursive multi-step)
- Same architecture as Experiment 1 (2 stacked LSTM layers, 50 units, dropout 0.2, Dense(1), MSE, Adam, early stopping on a validation slice). MinMax scaler fit on **train only**.
- Seed the model with the **last `w` days of the training set**. Predict day 1 of the test window.
- **Feed that prediction back in** as the newest input, drop the oldest day, and predict day 2. Repeat recursively across the whole test window — the input window fills up with the model's own predictions, never real test values.
- Inverse-transform, then score.

### Setup 3 — Hybrid ARIMA + LSTM (multi-step)
- ARIMA part = the multi-step ARIMA forecast above (the linear estimate `L̂ₜ` for the whole test horizon).
- LSTM part = an LSTM trained on the **training residuals**, then rolled forward **recursively** (same self-feeding scheme as Setup 2) to predict residuals `N̂ₜ` across the test horizon.
- Final forecast = `L̂ₜ + N̂ₜ`.

---

## 3. Add a naive baseline (do this in both experiments)

Include a **persistence / naive baseline** so we can tell whether the models add real skill:
- **One-step naive:** prediction for day *t* = actual value of day *t−1*.
- **Multi-step naive (flat):** prediction for every test day = the **last training value** (a flat line). This is the honest "no-skill" reference for multi-step.
- Report MAE / MAPE / RMSE for the baseline alongside the models. **A model is only genuinely useful if it clearly beats the naive baseline.**

---

## 4. Outputs (new filenames — do NOT overwrite Experiment 1)

Write:
- `ms_arima_predictions.csv`, `ms_lstm_predictions.csv`, `ms_hybrid_predictions.csv`
- `ms_arima_info.json`, `ms_lstm_info.json`, `ms_hybrid_info.json`
- `naive_baseline_info.json` (both one-step and multi-step naive metrics)
- `comparison_multistep.csv` — MAE/MAPE/RMSE for ARIMA, LSTM, Hybrid, Naive (multi-step)
- `comparison_overlay_multistep.png` — actual vs. all forecasts over the test period

Keep Experiment 1's files (`arima_predictions.csv`, etc.) untouched.

---

## 5. Final comparison to produce

One combined table with three blocks so nothing is misread:

| Model | One-step MAE | One-step MAPE | One-step RMSE | Multi-step MAE | Multi-step MAPE | Multi-step RMSE |
|-------|-------------|---------------|---------------|----------------|-----------------|-----------------|
| Naive |  |  |  |  |  |  |
| ARIMA |  |  |  |  |  |  |
| LSTM  |  |  |  |  |  |  |
| Hybrid|  |  |  |  |  |  |

Plus a short written conclusion covering:
1. How our **multi-step** numbers compare to the paper's Table 4 (ARIMA MAE 233.96 / MAPE 7.04% / RMSE 269.57; LSTM MAE 249.37 / MAPE 6.96% / RMSE 269.86).
2. Whether each model actually **beats the naive baseline** (the real test of skill).
3. Why the one-step numbers are so much lower (easy vs. hard test), stated plainly so a reader isn't misled.
4. How the models behaved through the 2020–2023 crisis stretch.

---

## 6. Prompt to run this in Claude Code

> Read `EXPERIMENT2_paper_method.md` and follow it exactly. Also read `CLAUDE.md` for the shared data/split rules. Reuse `split_info.json` so the split is identical to Experiment 1. Build three scripts — `05_arima_multistep.py`, `06_lstm_multistep.py`, `07_hybrid_multistep.py` — implementing the multi-step (paper-style) forecasts described here, plus a naive baseline, and `08_compare_multistep.py` for the combined table and overlay plot. Do not overwrite any Experiment 1 output files; use the `ms_` / `comparison_multistep` filenames specified. Print MAE/MAPE/RMSE for every model and the naive baseline, and run each script after writing it. Install any missing dependencies.
