# Comparing My Results with the 47 Research Papers

## 0. The honest ground rule first

**You cannot line up raw RMSE / MAE / MAPE numbers across papers.** Each paper uses a
different stock, a different price scale, a different period, and a different metric.
"RMSE 10" on Google says nothing about "RMSE 269" on the S&P SL20 — they are different
rulers. Comparing them directly would be dishonest.

**What CAN be compared, and what actually matters:**
1. Did the paper use a **naive / random-walk baseline**? (almost none did)
2. Was the **evaluation sound** (proper time split, out-of-sample, no leakage)?
3. Where do the **honestly-evaluated** papers land — and do they match me?

On those three, a real comparison is possible. It is done below.

---

## 1. The one paper that IS directly comparable — Paper 26 (my source paper)

Paper 26 (Vithushan & Kethmi, 2025) forecasts the **same index (S&P SL20)** with the
**same two base models (ARIMA vs LSTM)**. This is my true benchmark.

| | Paper 26 | My project |
|---|---|---|
| Index | S&P SL20 | S&P SL20 (+ 8 stocks + Apple) |
| Models | ARIMA, LSTM | ARIMA, LSTM, hybrid, RF, XGB, GRU, CNN-LSTM, GARCH |
| Result | ARIMA ≈ LSTM (MAE 233.96 vs 249.37; LSTM MAPE ~6%) | ARIMA ≈ LSTM ≈ **naive** — nobody beats random walk |
| Data window | 2010–2018 (**crisis years excluded on purpose**) | 2015–2026 (**crisis years included**) |
| Naive baseline? | ❌ No | ✅ Yes (every test) |
| Blind vs rolling protocol tested? | ❌ No | ✅ Yes (caught the difference) |
| Horizons | 1 | 11 (1 day → 1 year) |
| Volatility studied? | ❌ No | ✅ Yes (GARCH, works) |

**Key point:** their headline "ARIMA ≈ LSTM, MAPE ~6%" is *consistent with my finding*.
They just never added the naive baseline that would have shown neither model beats a
random walk. **I reproduced their conclusion and then went past it.**

---

## 2. The literature's reported accuracy — and why most of it is inflated

Many papers report spectacular numbers: **88.7%, 89.7%, 90%, 99%, 100%, R² 0.98+**.
On the surface it looks like everyone beat me. But the survey's **own "Limitations"
column** explains why most of those numbers are not real forecasting skill.

### The 5 recurring flaws (with the papers that have them)

**FLAW 1 — Tested on the training data (in-sample) → fake ~100%.**
- **Paper 27 (CSE announcements):** reports up to **100%** accuracy, but the survey notes
  it "come[s] from testing the models on the same data used for training… When proper
  evaluation (cross-validation/holdout) is used, accuracy drops to **35–47%**." That is a
  60-point collapse.
- **Paper 33 (CSE):** MLP "99% accuracy" with only a **2-day input window** — a classic
  one-step-ahead illusion (same trap I found: 1-day looks perfect because tomorrow ≈ today).

**FLAW 2 — Reporting SENTIMENT-classification accuracy as if it were PRICE prediction.**
- **Paper 23 (CSE, honest):** the real stock-prediction accuracy was only **58.37%** —
  "only slightly better than random guessing." The impressive **88–95%** only applies to
  the sentiment-labelling task, NOT the price forecast. Many papers blur these two.
- **Paper 29 (AAPL):** the **83–92%** "only shows how well the models match the sentiment
  labels… not whether sentiment can predict stock price."

**FLAW 3 — Data leakage / unexplained near-perfect one-step results.**
- **Paper 21:** the survey flags "unexplained **near-perfect 1-day-ahead results**" — the
  exact leakage red flag I caught and fixed in my own ARIMAX.
- **Paper 30 (multimodal, >90%):** "does not clearly explain how training and testing data
  were separated, creating a risk of **data leakage**"; only 3 months of data.
- **Paper 20 (CSE, Dinushan):** best results came from **SHUFFLED** data + the previous
  month's value — shuffling time-series is leakage (future leaks into past).

**FLAW 4 — Removing the hard cases to inflate the score.**
- **Paper 31 (China MMAN):** **61.20%** accuracy, and even that is helped because it
  "removes small price-change cases, which are harder to predict."

**FLAW 5 — Tiny test window / tiny dataset.**
- **Paper 25 (CSE hybrid ARIMA+ANN):** out-of-sample test is **just one week** — cannot
  represent real performance.
- **Paper 1 (CSE hybrid, 65%):** only **43–90 instances** per company.
- **Paper 30:** 3 months, one sector. **Paper 2:** flagged as "**mock data**."

### What happens when evaluation IS sound

Strip the flaws away and the honest papers all collapse into the **55–65% zone** — right
where I am:

| Paper | Honest? | Real accuracy |
|---|---|---|
| Paper 27 (after proper CV) | ✅ | 35–47% |
| Paper 26 (ARIMA vs LSTM, S&P SL20) | ✅ | ARIMA ≈ LSTM, ~6% MAPE, no edge over naive |
| Paper 23 (CSE, walk-forward) | ✅ | **58.37%** |
| Paper 31 (China) | ⚠️ | 61.2% (and inflated by dropping flat cases) |
| Paper 19 (S&P 500) | ✅ | 60% directional |
| Paper 14 (AAPL, EMD-LSTM) | ✅ | 70.56% (uses aggressive decomposition) |
| **My blind direction test** | ✅ | **~50–55% (coin flip), naive unbeaten** |
| **Paper 12 (review)** | ✅ | 90% at **3 days**, "**drops significantly as window extends to 10 days**" — my horizon-decay finding, confirmed by a review of 20+ papers |

---

## 3. The macro / econometric papers (34–45) — different question, and they back me up

Papers 34–45 are not price-prediction models; they study macro relationships. Two matter
a lot for my project:

- **Paper 43 (Fernando 2018, EGARCH):** finds the CSE is **"not semi-strong efficient in
  the short run"** and shows **volatility clustering** with **asymmetric ("bad news bigger")
  effects.** → This is direct literature support for **my positive volatility result**
  (GARCH tracks realized vol at 0.88–0.91). Volatility *should* be predictable, and it is.
- **Papers 39, 42, 44 (ARDL/VECM):** interest rates have a **negative** long/short-run link
  with CSE prices/bank performance. → Matches my earlier spread-regression finding
  (banks react negatively to rate widening).

---

## 4. Where I stand (the honest verdict)

1. **My accuracy is not "worse" than the literature — it is the HONEST version of it.**
   The papers that evaluated properly (23, 26, 31, and 27-after-fixing) land in the same
   50–65% zone I do. The ones that "beat" me mostly did so through leakage, in-sample
   testing, mislabeled sentiment accuracy, dropped hard cases, or one-week tests — every
   one of which I explicitly avoided.

2. **I did the naive-baseline comparison that almost none of them did.** Across 47 papers,
   a proper random-walk baseline is essentially absent. That is my methodological edge.

3. **I confirmed the literature's own quiet admissions:** short-window accuracy looks great
   and collapses with horizon (Paper 12), one-step results are misleadingly perfect
   (Paper 21), and CSE price prediction is barely above chance (Paper 23).

4. **My positive contribution is where the honest literature points:** not price (random
   walk), but **volatility** (Paper 43's EGARCH world) — which I show works across HNB and
   Apple.

**One-line summary for the supervisor:**
> "The literature's high accuracies mostly come from flawed evaluation (leakage, in-sample
> testing, sentiment-accuracy mislabeled as price accuracy, or 1-day/1-week windows). When
> I evaluate honestly — with a naive baseline the papers omit — my results match the
> honestly-evaluated papers (55–65%, no edge over random walk on price), and I extend the
> field with a multi-horizon blind benchmark and a working volatility model."
