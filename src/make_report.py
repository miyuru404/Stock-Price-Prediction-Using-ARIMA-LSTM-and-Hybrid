"""
Build a single comprehensive PDF report covering:
  - the source paper's findings (Vithushan & Kethmi, IRC-OUSL 2025)
  - Experiment 1: one-step-ahead ARIMA / LSTM / Hybrid
  - Experiment 2: multi-step (paper-style) ARIMA / LSTM / Hybrid + naive baseline
  - both overlay charts and all comparison tables

Output: SPSL20_Full_Report.pdf
"""

from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
FIG = _ROOT / "results" / "figures"
REPORT = _ROOT / "report"
REPORT.mkdir(parents=True, exist_ok=True)

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, HRFlowable)

OUT = str(REPORT / "SPSL20_Full_Report.pdf")

# ---------------------------------------------------------------- palette
NAVY = colors.HexColor("#1a2a4f")
BLUE = colors.HexColor("#2b6cb0")
LGREY = colors.HexColor("#eef1f6")
MGREY = colors.HexColor("#d5dae4")
GREEN = colors.HexColor("#2f855a")
RED = colors.HexColor("#c53030")

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=17,
                    spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=BLUE, fontSize=13,
                    spaceBefore=10, spaceAfter=5)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=10, leading=15,
                      alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=14, bulletIndent=2,
                        spaceAfter=2)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.5, textColor=colors.grey,
                       alignment=TA_CENTER)
CAP = ParagraphStyle("CAP", parent=BODY, fontSize=8.5, textColor=colors.grey,
                     alignment=TA_CENTER, spaceBefore=2)
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], textColor=NAVY, fontSize=24,
                       leading=28, alignment=TA_CENTER)
SUBTITLE = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=13, leading=18,
                          alignment=TA_CENTER, textColor=BLUE)

story = []


def P(t, style=BODY):
    story.append(Paragraph(t, style))


def bullets(items, style=BULLET):
    for it in items:
        story.append(Paragraph(f"&bull;&nbsp;&nbsp;{it}", style))


def spacer(h=8):
    story.append(Spacer(1, h))


def rule():
    story.append(HRFlowable(width="100%", thickness=0.6, color=MGREY,
                            spaceBefore=6, spaceAfter=8))


def make_table(data, col_widths, header_bg=NAVY, highlight_rows=None,
               highlight_col=None, font_size=9):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.4, MGREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LGREY]),
    ]
    if highlight_rows:
        for r in highlight_rows:
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e6f4ea")))
            style.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


# ============================================================ TITLE PAGE
spacer(90)
P("Forecasting the S&amp;P SL 20 Index", TITLE)
spacer(6)
P("ARIMA vs. LSTM vs. Hybrid — One-Step and Multi-Step Analysis", SUBTITLE)
spacer(30)
P("A complete record of two forecasting experiments on the Colombo Stock Exchange "
  "S&amp;P SL 20 index (2015&ndash;2026), benchmarked against the source paper "
  "of Vithushan &amp; Kethmi (IRC-OUSL 2025).", ParagraphStyle(
      "lead", parent=BODY, alignment=TA_CENTER, fontSize=11, leading=16))
spacer(40)
meta_tbl = Table([
    ["Index", "S&P SL 20 (Colombo Stock Exchange)"],
    ["Data window", "2015-01-01 to 2026-04-16 (2,664 trading days)"],
    ["Test window", "2024-01-22 to 2026-04-16 (533 trading days)"],
    ["Models", "ARIMA, LSTM, Hybrid ARIMA+LSTM, Naive baseline"],
    ["Experiments", "1 = one-step-ahead   |   2 = multi-step (paper-style)"],
    ["Report date", "2026-07-22"],
], colWidths=[4 * cm, 11 * cm])
meta_tbl.setStyle(TableStyle([
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LGREY]),
    ("GRID", (0, 0), (-1, -1), 0.4, MGREY),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(meta_tbl)
story.append(PageBreak())

# ============================================================ 1. OVERVIEW
P("1. Project Overview", H1)
rule()
P("This project forecasts the daily <b>S&amp;P SL 20</b> index of the Colombo Stock "
  "Exchange (CSE) and compares three modelling setups on <b>identical data and an "
  "identical train/test split</b>:")
bullets([
    "<b>ARIMA</b> &mdash; a traditional linear time-series model.",
    "<b>LSTM</b> &mdash; a recurrent neural network for nonlinear patterns.",
    "<b>Hybrid ARIMA + LSTM</b> &mdash; ARIMA models the linear part; an LSTM models "
    "ARIMA's residual (nonlinear leftover); the two are summed.",
])
spacer()
P("The work both <b>reproduces</b> and <b>extends</b> the source paper. We reproduce its "
  "ARIMA-vs-LSTM comparison, but on a newer dataset that <b>includes the 2020&ndash;2023 "
  "Sri Lankan crisis years</b> (which the paper deliberately excluded), and we add the "
  "<b>Hybrid</b> setup the paper described but never actually reported.")
spacer()
P("Crucially, we run the comparison <b>two ways</b>, because how you forecast the test "
  "window changes the numbers by an order of magnitude:")
bullets([
    "<b>Experiment 1 &mdash; one-step-ahead (&lsquo;easy mode&rsquo;).</b> To predict each "
    "day, the model is given the <i>true</i> recent history up to the day before.",
    "<b>Experiment 2 &mdash; multi-step (&lsquo;hard mode&rsquo;, the paper&rsquo;s way).</b> "
    "The model forecasts the whole test window forward using <i>only its own predictions</i> "
    "&mdash; it never sees a real test value. This is what makes our numbers comparable to the "
    "paper&rsquo;s Table 4.",
])
spacer()
P("A <b>naive persistence baseline</b> is included throughout so we can tell whether the "
  "models add genuine skill or merely echo yesterday&rsquo;s price.")

# ============================================================ 2. DATA & SPLIT
P("2. Data and Train/Test Split", H1)
rule()
P("All modelling uses <b>spsl20_trading_days_clean.csv</b> &mdash; the univariate S&amp;P "
  "SL 20 closing level with weekends and carry-forward holiday duplicates removed (those "
  "fake flat days mislead both models). The split is <b>80/20 chronological</b> (never "
  "shuffled &mdash; this is sequential market data), computed once and reused by every "
  "script so all models train and are graded on exactly the same dates.")
spacer()
story.append(make_table([
    ["Portion", "Rows", "Share", "Dates"],
    ["Training", "2,131", "80%", "2015-01-01  to  2024-01-19"],
    ["Test (held out)", "533", "20%", "2024-01-22  to  2026-04-16"],
], [3.6 * cm, 2.2 * cm, 2 * cm, 7 * cm]))
spacer(6)
P("<b>Where the crisis sits:</b> the volatile 2020&ndash;2023 crisis years fall inside the "
  "<i>training</i> window. The models are therefore trained through the turbulence and graded "
  "on the calmer, strongly rising 2024&ndash;2026 recovery &mdash; the deliberate stress test "
  "of this project. <b>No leakage:</b> every MinMax scaler is fit on the training portion only, "
  "and ARIMA parameters are estimated on training data only.", SMALL)

story.append(PageBreak())

# ============================================================ 3. THE PAPER
P("3. The Source Paper &mdash; Findings", H1)
rule()
P("<b>Vithushan, M. &amp; Kethmi, G. A. P.</b> &mdash; <i>A Comparative Analysis of Time "
  "Series Models for Predicting the S&amp;P SL 20 Index of the CSE</i>, IRC-OUSL 2025.")
spacer()
P("<b>What the paper did:</b>")
bullets([
    "Used daily S&amp;P SL 20 data for <b>2010&ndash;2018</b> (~2,160 observations), "
    "<b>deliberately excluding</b> later years because of &lsquo;unusual circumstances&rsquo; "
    "(the crisis).",
    "Compared <b>ARIMA vs. LSTM</b> only &mdash; a two-way study. A hybrid is mentioned in "
    "the introduction but <b>never reported</b> in the results.",
    "ARIMA: log-transformed series, ADF p = 0.05108 (non-stationary), order <b>(2,1,0)</b> "
    "chosen from ACF/PACF. LSTM: Adam optimiser, MSE loss.",
    "Split by date: train 2010&ndash;2016, test 2017&ndash;2018.",
])
spacer()
P("<b>The paper&rsquo;s results (Table 4):</b>")
spacer(4)
story.append(make_table([
    ["Model", "MAE", "MAPE", "RMSE"],
    ["ARIMA", "233.96", "7.04%", "269.57"],
    ["LSTM", "249.37", "6.96%", "269.86"],
    ["Hybrid", "not reported", "—", "—"],
], [5 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm]))
spacer(6)
P("<b>The paper&rsquo;s conclusion:</b> ARIMA and LSTM perform similarly. ARIMA is marginally "
  "better on MAE and RMSE (suited to linear trends); LSTM is marginally better on MAPE "
  "(captures nonlinear patterns). The paper notes ARIMA produced &lsquo;constant predictions "
  "in some periods&rsquo; &mdash; a fingerprint of the flat multi-step forecast that becomes "
  "important in our Experiment 2.")

# ============================================================ 4. EXPERIMENT 1
P("4. Experiment 1 &mdash; One-Step-Ahead Forecast", H1)
rule()
P("<b>Method.</b> Each test-day prediction uses the true prior history. For the LSTM/Hybrid, "
  "the previous <i>w</i> real days (or residuals) feed the next-day prediction. For ARIMA, "
  "parameters are frozen after training and the model <i>state</i> is updated with actual test "
  "observations (statsmodels <font face='Courier'>append(refit=False)</font>) &mdash; this is "
  "<b>not</b> retraining and adds no parameter leakage. Model details:")
bullets([
    "<b>ARIMA:</b> ADF-based d = 1; auto_arima selected <b>ARIMA(1,1,4)</b> (lower AIC than "
    "the (2,1,0) baseline). Residual diagnostics: Ljung-Box p = 0.998 (no autocorrelation), "
    "Breusch-Pagan p = 0.106 (homoskedastic), Jarque-Bera p &lt; 0.0001 (non-normal &mdash; "
    "one extreme crisis-era residual spike).",
    "<b>LSTM (PyTorch):</b> 2 stacked LSTM layers &times; 50 units, dropout 0.2, Dense(1), "
    "MSE loss, Adam, early stopping on a validation slice; MinMax scaler fit on train only; "
    "seeds fixed. Window 60 selected.",
    "<b>Hybrid:</b> ARIMA(1,1,4) forecast + an LSTM trained on ARIMA&rsquo;s training "
    "residuals (window 30).",
])
spacer(4)
P("<b>Experiment 1 results</b> (test window, 533 days; lower is better):")
spacer(4)
story.append(make_table([
    ["Model", "MAE", "MAPE", "RMSE"],
    ["Naive (one-step)", "37.06", "0.788%", "56.16"],
    ["ARIMA (1,1,4)", "36.42", "0.770%", "56.25"],
    ["LSTM (window 60)", "140.52", "2.491%", "198.10"],
    ["Hybrid (best)", "36.25", "0.766%", "56.15"],
], [5.5 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm], highlight_rows=[4]))
spacer(6)
P("<b>Reading it:</b> the Hybrid wins on all three metrics, but only marginally ahead of pure "
  "ARIMA &mdash; and both barely beat the naive baseline. At a one-day horizon the index is "
  "almost a random walk (tomorrow &asymp; today), so persistence alone already scores &lt;1% "
  "MAPE. The standalone LSTM lags the price and is the clear laggard. The Hybrid adds almost "
  "nothing over ARIMA because ARIMA&rsquo;s residuals are essentially white noise (Ljung-Box "
  "p = 0.998), leaving no structure for the residual-LSTM to exploit.")
spacer(4)
story.append(Image(str(FIG / "comparison_overlay.png"), width=16 * cm, height=8 * cm))
P("Figure 1. Experiment 1 (one-step). ARIMA and Hybrid sit on the actual line (ARIMA hidden "
  "under Hybrid); the standalone LSTM lags below during the uptrend.", CAP)

story.append(PageBreak())

# ============================================================ 5. EXPERIMENT 2
P("5. Experiment 2 &mdash; Multi-Step (Paper-Style) Forecast", H1)
rule()
P("<b>Method.</b> The model is cut off from real test values and must roll the whole 533-day "
  "horizon forward on its own guesses &mdash; the harder, honest test the paper used.")
bullets([
    "<b>ARIMA:</b> paper recipe &mdash; log-transform, d = 1, baseline (2,1,0); auto_arima "
    "selected <b>ARIMA(0,1,1)</b>. A single dynamic forecast for the entire horizon "
    "(<font face='Courier'>get_forecast(steps=533)</font>); no test value fed back; log "
    "inverted before scoring.",
    "<b>LSTM (recursive):</b> seeded with the last 60 training days, predicts day 1, then "
    "feeds each prediction back as the newest input and rolls forward &mdash; the window fills "
    "with the model&rsquo;s own outputs.",
    "<b>Hybrid:</b> multi-step ARIMA forecast + a residual-LSTM rolled forward recursively.",
    "<b>Naive (flat):</b> every test day = the last training value &mdash; the honest "
    "&lsquo;no-skill&rsquo; reference for multi-step.",
])
spacer(4)
P("<b>Experiment 2 results</b> (test window, 533 days; lower is better):")
spacer(4)
story.append(make_table([
    ["Model", "MAE", "MAPE", "RMSE"],
    ["Naive (flat)", "1788.9", "33.23%", "2162.5"],
    ["ARIMA (0,1,1) log", "1792.8", "33.32%", "2165.9"],
    ["LSTM (recursive)", "1913.9", "35.93%", "2277.0"],
    ["Hybrid (multi-step)", "1792.1", "33.30%", "2165.3"],
], [5.5 * cm, 3.1 * cm, 3.1 * cm, 3.1 * cm], highlight_rows=[1]))
spacer(6)
P("<b>Reading it:</b> every multi-step forecast collapses to a near-flat line around 3000 "
  "while the index climbs to ~6700. The <b>naive flat line is actually the best</b> (lowest "
  "RMSE); ARIMA and the Hybrid essentially tie it, and the recursive LSTM does slightly worse. "
  "<b>None of the models adds real multi-step skill</b> over 533 days &mdash; in a trending "
  "market, forecasting the whole horizon from a single origin is close to hopeless, and "
  "&lsquo;flat at the last value&rsquo; is a hard reference to beat.")
spacer(4)
story.append(Image(str(FIG / "comparison_overlay_multistep.png"), width=16 * cm, height=8 * cm))
P("Figure 2. Experiment 2 (multi-step). All forecasts flatten near 3000 (ARIMA, Hybrid and "
  "Naive overlap; the LSTM drifts slightly lower) while the actual index rises to ~6700.", CAP)

story.append(PageBreak())

# ============================================================ 6. COMBINED
P("6. Combined Comparison &mdash; One-Step vs. Multi-Step", H1)
rule()
P("The single most important table in this report. The same models, same data, same test "
  "window &mdash; only the <i>forecasting protocol</i> differs between the two blocks.")
spacer(6)
story.append(make_table([
    ["Model", "1-step\nMAE", "1-step\nMAPE", "1-step\nRMSE",
     "Multi\nMAE", "Multi\nMAPE", "Multi\nRMSE"],
    ["Naive", "37.06", "0.79%", "56.16", "1788.9", "33.23%", "2162.5"],
    ["ARIMA", "36.42", "0.77%", "56.25", "1792.8", "33.32%", "2165.9"],
    ["LSTM", "140.52", "2.49%", "198.10", "1913.9", "35.93%", "2277.0"],
    ["Hybrid", "36.25", "0.77%", "56.15", "1792.1", "33.30%", "2165.3"],
], [2.6 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm],
    font_size=8.5))
spacer(6)
P("The ~40&times; jump in every error metric from left block to right block is the difference "
  "between the easy and the hard test &mdash; not a change in the models. This is exactly why "
  "one-step numbers must never be quoted as if they measured genuine forecasting skill.", SMALL)

# ============================================================ 7. VS PAPER
P("7. How Our Numbers Compare to the Paper", H1)
rule()
P("Our <b>multi-step</b> block is the fair comparison to the paper&rsquo;s Table 4, because it "
  "uses the same forecasting method. Yet our errors are far larger (ARIMA MAPE 33% vs. the "
  "paper&rsquo;s 7%). <b>The cause is the test window, not the models.</b>")
bullets([
    "The paper&rsquo;s 2017&ndash;2018 test period was roughly flat, so its multi-step "
    "forecasts &mdash; which collapse toward a flat line &mdash; stayed close to the truth.",
    "Our 2024&ndash;2026 window is a strong post-crisis bull run (~3050 &rarr; ~6700), so any "
    "flat or converging multi-step forecast is badly wrong. Same method, much harder horizon.",
    "Both studies agree on the qualitative headline: <b>ARIMA &ge; LSTM</b>. Our one-step "
    "result sharpens this (ARIMA clearly beats the standalone LSTM); the paper found them "
    "near-tied.",
    "We supply the evidence the paper lacked: the Hybrid <i>is</i> marginally best one-step, "
    "but its gain over ARIMA is negligible &mdash; ARIMA already whitens the residuals.",
])

# ============================================================ 8. CONCLUSIONS
P("8. Conclusions and Key Takeaways", H1)
rule()
bullets([
    "<b>One-step:</b> Hybrid &lt; ARIMA &lt; Naive &lt;&lt; LSTM. The Hybrid is best but the "
    "margin over ARIMA and even the naive baseline is tiny &mdash; daily persistence dominates.",
    "<b>Multi-step:</b> Naive &le; Hybrid &asymp; ARIMA &lt; LSTM. No model beats the flat "
    "baseline; long-horizon forecasting of this trending index is effectively no-skill.",
    "<b>Beat the baseline?</b> Only marginally, and only one-step. The honest reading is that "
    "neither ARIMA, LSTM, nor the Hybrid delivers reliable multi-step skill on this series.",
    "<b>Crisis effect:</b> the 2020&ndash;2023 crisis (in the training window) taught a "
    "volatile, largely trendless regime, so the fitted ARIMA carries almost no drift &mdash; "
    "which is exactly why its multi-step forecast flattens and undershoots the rising recovery. "
    "It also left fat-tailed residuals (why RMSE &raquo; MAE throughout).",
    "<b>Honesty note:</b> sub-1% one-step MAPE looks impressive but mostly reflects the "
    "easy test; the multi-step block is the one to trust for real forecasting ability.",
])

# ============================================================ 9. REPRODUCIBILITY
P("9. Environment and Reproducibility", H1)
rule()
P("Modelling ran in a local virtual environment under <b>Python 3.14</b> with pandas, numpy, "
  "statsmodels, pmdarima, scikit-learn, matplotlib, and <b>PyTorch</b>. <b>Note:</b> the "
  "project spec named TensorFlow/Keras for the LSTM, but Python 3.14 currently has no "
  "TensorFlow wheel, so the LSTM was implemented in PyTorch with the <i>identical</i> "
  "architecture and training recipe (2&times;50-unit LSTM, dropout 0.2, Dense(1), MSE, Adam, "
  "early stopping, fixed seeds). This is a library substitution only; the method is unchanged.")
spacer(4)
story.append(make_table([
    ["Script", "Produces"],
    ["01_arima.py / 02_lstm.py / 03_hybrid.py", "Experiment 1 (one-step) predictions + metrics"],
    ["04_compare_results.py", "Experiment 1 table + overlay (Figure 1)"],
    ["05_arima_multistep.py / 06 / 07", "Experiment 2 (multi-step) predictions + metrics"],
    ["08_compare_multistep.py", "Naive baseline, combined table, overlay (Figure 2)"],
    ["make_report.py", "This PDF report"],
], [7.5 * cm, 8 * cm], font_size=8.5))
spacer(6)
P("Build order 01 &rarr; 08. Every script reuses <font face='Courier'>split_info.json</font> "
  "so the train/test split is identical across all experiments. Experiment 1 and Experiment 2 "
  "write to separate filenames and do not overwrite each other.", SMALL)

# ---------------------------------------------------------------- build
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.1 * cm, "S&P SL 20 Forecasting — Full Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(MGREY)
    canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        title="S&P SL 20 Forecasting — Full Report",
                        author="LSTM + ARIMA project")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {OUT}")
