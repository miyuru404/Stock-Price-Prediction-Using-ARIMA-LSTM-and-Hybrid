# Experiment 3 — XGBoost & GRU

**Purpose:** Add two more models to the S&P SL 20 comparison — **XGBoost** (gradient-boosted trees, a different ML family) and **GRU** (a lighter RNN, LSTM's cousin) — evaluated the **same way** as ARIMA/LSTM/Hybrid so everything sits in one table.

Runs alongside Experiments 1 and 2. Same data, same split, same naive baseline, **both** forecasting protocols (one-step and multi-step). New filenames — nothing from earlier experiments is overwritten.

---

## 0. Shared rules (identical to Experiments 1 & 2)

- Data: `spsl20_trading_days_clean.csv` (univariate S&P SL 20). Reuse `split_info.json` for the exact 80/20 chronological split.
- Never shuffle. Any scaler / feature statistics fit on the **training portion only**.
- Report **MAE, MAPE, RMSE** on the held-out test window (533 days) for every model.
- Include the same **naive baselines** already used: one-step naive (day t = actual t-1) and multi-step naive (flat = last training value).
- Models are stochastic (GRU especially) -> set seeds; for GRU, run 3 seeds and report mean +/- std.

---

## 1. Model A — XGBoost (gradient-boosted trees)

Trees don't understand time directly, so we turn the series into a supervised table using **lag features**.

**Feature engineering (fit on training rows only):**
- Lags: value at t-1, t-2, t-3, t-5, t-10, t-20.
- Rolling stats: rolling mean and rolling std over windows 5, 10, 20 (use only past values — no future leak).
- Optional calendar: day-of-week, month.
- Target: next-day *difference* (see note below).

**Important — predict differences, not raw levels.** Trees cannot extrapolate beyond the range of values seen in training. Our test period rises above anything in training, so a level-target XGBoost will saturate. Train it to predict the **daily change** (value_t - value_{t-1}), then reconstruct the level by adding the change to the previous value. This gives the model a fair chance on a trending series.

**One-step forecast:** for each test day, build features from the **true** prior values, predict the change, add to the true previous value.

**Multi-step forecast:** seed from the last training values, predict the change, add it to the model's **own** previous prediction, and roll forward — features rebuilt from predicted values, never real test values.

**Settings:** reasonable defaults (n_estimators~500, max_depth~4-6, learning_rate~0.05, early stopping on a validation slice carved from the end of train). LightGBM is an acceptable drop-in if XGBoost is unavailable.

---

## 2. Model B — GRU (recurrent neural network)

Reuse the LSTM pipeline exactly; only swap nn.LSTM -> nn.GRU.

- Architecture: 2 stacked GRU layers x 50 units, dropout 0.2, Dense(1) head, MSE loss, Adam, early stopping on a validation slice.
- MinMax scaler fit on **train only**. Windows: try 60 and 30, select by validation RMSE.
- **One-step forecast:** each test day predicted from the true prior w days.
- **Multi-step forecast:** seed with last w training days, feed each prediction back recursively across the whole horizon.
- Run 3 seeds; report mean +/- std of the test metrics.

---

## 3. Outputs (new filenames — do NOT overwrite earlier experiments)

- `xgb_predictions_onestep.csv`, `xgb_predictions_multistep.csv`, `xgb_info.json`
- `gru_predictions_onestep.csv`, `gru_predictions_multistep.csv`, `gru_info.json` (per-seed and mean+/-std)
- `comparison_all_models.csv` — the master table below
- `comparison_overlay_all.png` — actual vs. all models (one-step panel + multi-step panel)

---

## 4. Master comparison table to produce

| Model | 1-step MAE | 1-step MAPE | 1-step RMSE | Multi MAE | Multi MAPE | Multi RMSE |
|-------|-----------|-------------|-------------|-----------|------------|------------|
| Naive |  |  |  |  |  |  |
| ARIMA |  |  |  |  |  |  |
| LSTM  |  |  |  |  |  |  |
| Hybrid|  |  |  |  |  |  |
| XGBoost |  |  |  |  |  |  |
| GRU   |  |  |  |  |  |  |

Pull the ARIMA/LSTM/Hybrid/Naive rows from the existing Experiment 1 & 2 result files; add the two new rows.

**Written conclusion should cover:** does XGBoost (a different ML family) beat ARIMA? Does GRU match or beat LSTM? Does either beat the naive baseline — one-step and multi-step? Keep the honest framing: multi-step is the test that reflects real skill.

---

## 5. Prompt to run in Claude Code

> Read `EXPERIMENT3_xgboost_gru.md` and `CLAUDE.md`. Reuse `split_info.json` for the identical split. Build `09_xgboost.py` and `10_gru.py` implementing both the one-step and multi-step forecasts exactly as specified (XGBoost predicts daily differences and reconstructs levels; GRU reuses the LSTM pipeline with nn.GRU, 3 seeds, mean+/-std). Then build `11_compare_all.py` to assemble the master table (`comparison_all_models.csv`) by combining these with the existing Experiment 1 & 2 result files, plus a combined overlay plot. Do not overwrite any existing output files. Print MAE/MAPE/RMSE for every model under both protocols, and run each script after writing it. Install any missing dependencies (xgboost).
