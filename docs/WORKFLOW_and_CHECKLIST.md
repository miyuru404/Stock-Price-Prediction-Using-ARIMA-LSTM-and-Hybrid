# S&P SL 20 Forecasting — Workflow & Checklist

**Goal:** Reproduce and extend the IRC-OUSL 2025 paper *"A Comparative Analysis of Time Series Models for Predicting the S&P SL 20 Index of the CSE."* We build three models — **ARIMA** (linear), **LSTM** (nonlinear), and a **Hybrid ARIMA+LSTM** — and compare them on the same S&P SL 20 series so we can see which performs best.

The paper's own result (Table 4): ARIMA MAE 233.96 / MAPE 7.04% / RMSE 269.57; LSTM MAE 249.37 / MAPE 6.96% / RMSE 269.86. They were close, and the paper argues a **hybrid** combines ARIMA's short-term linear strength with LSTM's nonlinear strength for better accuracy. That hybrid is exactly what we're testing.

---

## 1. Data

**Source:** `cse_indices_macro_clean.csv` (2015-01-01 → 2026-04-16). It holds several series; we use only **`spsl20_points`** as requested.

**Files created for this task:**

- `spsl20_dataset.csv` — the raw separated series: `date, spsl20_points` (2,946 rows, no missing values).
- `spsl20_trading_days_clean.csv` — same series with weekend rows and carry-forward duplicate days removed (2,664 rows). Use this for modeling; use the raw file if you want to keep a strict daily calendar.

**Data-quality note:** the raw CSV fills every calendar day, including weekends and holidays, by repeating the previous close (e.g. 2015-01-01 and 01-02 both = 4089.1399). That inflates row count and injects fake "flat" days that mislead both ARIMA and LSTM. The clean file fixes this. Decide up front which one you'll standardize on — I recommend the clean trading-days file.

**Scope decision:** The paper is **univariate** (S&P SL 20 alone), and you said to use only SPSL20 data — so we stay univariate. The macro columns (crude oil, US 10y yield, DXY, USD/LKR, ASPI) are available later if you want a multivariate LSTM extension.

---

## 2. Workflow

### Stage A — Preparation (shared by all models)
1. Load `spsl20_trading_days_clean.csv`, set `date` as index, sort ascending.
2. Plot the series; visually check trend, volatility, and any structural breaks.
3. **Split chronologically** (never shuffle time series): train = earliest ~80%, test = latest ~20%. Keep the exact same split for all three models so the comparison is fair.

### Stage B — ARIMA (linear model)
1. Test stationarity with the **ADF test**. If non-stationary, difference the series (this sets `d`). The paper used log-transform + `d=1`.
2. Read **ACF / PACF** plots to pick `p` and `q`. The paper landed on **ARIMA(2,1,0)** — a sensible starting point; also try `auto_arima` to search parameters.
3. Fit ARIMA on the training set.
4. Run diagnostics: Ljung-Box (residual autocorrelation), Jarque-Bera (normality), heteroskedasticity.
5. Forecast the test window and save predictions.

### Stage C — LSTM (nonlinear model)
1. **Scale** the series to [0,1] with MinMaxScaler (fit on training data only — never on test).
2. Build **sliding windows**: use the last *n* days (e.g. 30 or 60) to predict the next day.
3. Define the network: one or more LSTM layers → Dense(1) output. Loss = **MSE**, optimizer = **Adam** (as in the paper).
4. Train with early stopping / a validation slice; watch for overfitting.
5. Predict on the test set, then **inverse-transform** back to price units.

### Stage D — Hybrid ARIMA + LSTM (the point of the exercise)
The standard residual hybrid (Kulshreshtha & Vijayalakshmi, 2020, cited in the paper):
1. Fit ARIMA and get its fitted/forecast values — this captures the **linear** part.
2. Compute **residuals** = actual − ARIMA prediction. These residuals hold the **nonlinear** structure ARIMA missed.
3. Train the **LSTM on the residuals** (same windowing approach).
4. **Final forecast = ARIMA forecast + LSTM-predicted residual.**

### Stage E — Comparison & evaluation
1. Score every model on the identical test set with **MAE, MAPE, RMSE** (the paper's three metrics).
2. Put results in one comparison table (mirror Table 4) and overlay all forecasts vs. actual on one plot.
3. Conclude which model wins on each metric and note trade-offs.

---

## 3. Checklist

**Setup**
- [ ] Confirm working on `spsl20_trading_days_clean.csv` (or justify using the raw file)
- [ ] Environment ready: `pandas, numpy, matplotlib, statsmodels, pmdarima, scikit-learn, tensorflow/keras`
- [ ] Series loaded, date-indexed, sorted, plotted

**Split**
- [ ] Chronological train/test split defined (~80/20) and reused identically across all models

**ARIMA**
- [ ] ADF stationarity test run; differencing order `d` chosen
- [ ] ACF/PACF inspected; `p`,`q` chosen (start ARIMA(2,1,0), also try auto_arima)
- [ ] Model fitted; residual diagnostics checked
- [ ] Test-set forecast saved

**LSTM**
- [ ] Data scaled (scaler fit on train only)
- [ ] Sliding-window sequences built (window size chosen)
- [ ] Network defined (LSTM → Dense), MSE loss, Adam optimizer
- [ ] Trained with early stopping; predictions inverse-transformed

**Hybrid**
- [ ] ARIMA residuals computed
- [ ] LSTM trained on residuals
- [ ] Final = ARIMA + LSTM residual forecast

**Evaluation**
- [ ] MAE, MAPE, RMSE computed for all three on the same test set
- [ ] Comparison table + overlay forecast plot produced
- [ ] Best model identified with trade-off notes

---

## 4. Suggested file structure for the project

```
LSTM + ARIM/
├── cse_indices_macro_clean.csv          (original, all series)
├── spsl20_dataset.csv                    (separated raw SPSL20)
├── spsl20_trading_days_clean.csv         (separated, cleaned — use this)
├── WORKFLOW_and_CHECKLIST.md             (this file)
├── 01_arima.py / .ipynb
├── 02_lstm.py / .ipynb
├── 03_hybrid.py / .ipynb
└── 04_compare_results.py / .ipynb
```

---

## 5. Next step
Once you've picked which dataset file to standardize on, the next deliverable is the ARIMA notebook (Stage B). Say the word and I'll build the three model scripts one at a time, each with its own evaluation, then the final comparison.
