# S&P SL 20 Forecasting — Executive Summary

**One-line result:** On the 2024–2026 test window, the **Hybrid ARIMA + LSTM** is the most
accurate model on all three metrics, but only *marginally* ahead of pure **ARIMA**; the
**standalone LSTM** is a distant third.

---

## What we did

We forecast the **S&P SL 20** index (Colombo Stock Exchange) and compared three setups on the
**same data and the same split**: ARIMA, LSTM, and a Hybrid where an LSTM corrects ARIMA's
residuals. Data: 2,664 cleaned trading days, 2015–2026. Split: **80/20 chronological** — train
2015→Jan 2024 (with the **2020–2023 crisis years inside training**), test Jan 2024→Apr 2026 (533
days). All models forecast **one-step-ahead**; no shuffling; scalers fit on training data only.

Built as four scripts: `01_arima.py → 02_lstm.py → 03_hybrid.py → 04_compare_results.py`.
(LSTM implemented in **PyTorch** — Python 3.14 has no TensorFlow wheel — with the same
architecture: 2×50-unit LSTM, dropout 0.2, Dense(1), Adam, MSE, early stopping.)

## Results

| Model | MAE | MAPE | RMSE |
|-------|----:|-----:|-----:|
| ARIMA(1,1,4) | 36.42 | 0.770% | 56.25 |
| LSTM (window 60) | 140.52 | 2.491% | 198.10 |
| **Hybrid** | **36.25** | **0.766%** | **56.15** |

**Winner: Hybrid** (best on MAE, MAPE, RMSE). RMSE ranking: **Hybrid < ARIMA < LSTM.**

## Why

- At a one-day horizon the index is near a random walk, which the differenced ARIMA nails; the
  level-trained LSTM lags the price.
- The hybrid barely improves on ARIMA because ARIMA's residuals are already **white noise**
  (Ljung-Box p = 0.998) — little nonlinear signal left to model.
- Trained through the crisis, ARIMA/Hybrid still track the recovery tightly, but crisis-era
  jumps leave fat-tailed residuals, so RMSE (56) sits well above MAE (36).

## Versus the source paper (Vithushan & Kethmi, IRC-OUSL 2025)

| | Paper | This project |
|--|-------|--------------|
| Data | 2010–2018, **crisis excluded** | 2015–2026, **crisis in training** |
| Comparison | ARIMA vs LSTM (2-way) | ARIMA vs LSTM vs **Hybrid** (3-way) |
| Hybrid reported? | No (conceptual only) | **Yes** |
| ARIMA order | (2,1,0) manual | (1,1,4) auto_arima |
| Metrics | ARIMA 233.96 / 7.04% / 269.57 · LSTM 249.37 / 6.96% / 269.86 | see table above |

**Takeaways:** (1) the error magnitudes differ mainly because of a different era and our explicit
one-step-ahead horizon, so the raw numbers aren't directly comparable — as expected. (2) Both
studies agree **ARIMA ≥ LSTM**; ours is a sharper version (ARIMA clearly beats the standalone
LSTM). (3) We add the evidence the paper lacked: the hybrid *is* best, but its gain over ARIMA is
negligible on this series.

## Deliverables

- Full write-up: `PROJECT_DOCUMENTATION.md`
- Metrics table: `comparison_table.csv` · Overlay chart: `comparison_overlay.png`
- Scripts + per-model configs/predictions (see documentation §9).
