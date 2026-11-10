# Macroeconomic Variable Forecasting — US CPI Inflation

Quantum project #2. Forecasts US consumer-price inflation with a classic
seasonal-ARIMA model and backtests it honestly against naive baselines.

The point isn't nailing the number — it's the process: pull real data, difference
out trend and seasonality, hold out a window, and show the model beats the naive
"tomorrow looks like today" forecast.

## What it does

- **Data:** FRED series `CPIAUCSL` (US CPI), pulled from the public keyless CSV — no API key.
- **Model selection:** SARIMA order chosen by **AIC** over a curated candidate set (the classic monthly "airline" family + richer variants) on `log(CPI)` — not a hard-coded guess. Current winner: `SARIMA(1,1,1)(0,1,1)₁₂`.
- **Missing months stay missing:** BLS never published October 2025 (government shutdown). The month is kept as a gap, not dropped, so every year-over-year comparison spans exactly 12 months (an `assert` enforces it) and the Kalman filter skips it. October 2025 and October 2026 therefore have no YoY figure and are not scored.
- **Backtest:** **walk-forward** over the last 36 months — at each month we forecast 1/3/6/12 months ahead using only past data, then score by horizon. This is how a forecaster is actually judged, not one static multi-year projection.
- **Honest baselines:** random walk on the *inflation rate* (freeze the last YoY) and seasonal naive (YoY from a year ago). These are genuinely hard to beat, so the comparison is meaningful.
- **Diagnostics:** ADF stationarity, Ljung-Box residual autocorrelation, AIC.
- **Output:** tables to stdout + `cpi_forecast.png` (level forecast with 80% band, and 1-step walk-forward vs. actual inflation).

## Result (as of Oct 2026 data) — RMSE on YoY inflation, by horizon

| Horizon | SARIMA | RW-on-inflation | Seasonal naive |
|--------:|-------:|----------------:|---------------:|
| 1 month  | **0.25** | 0.29 | 1.38 |
| 3 months | 0.57 | **0.57** | 1.11 |
| 6 months | 0.66 | **0.56** | 0.81 |
| 12 months| 1.03 | **0.75** | 0.75 |

The honest story: **SARIMA wins one month ahead; at three months it is a tie
(0.572 vs 0.568) and at 6–12 months a random walk on the inflation rate is
unbeatable** — a well-known result in the
inflation-forecasting literature (Atkeson–Ohanian). 1-step bias is ~0, but the
residuals do **not** pass Ljung-Box (p < 0.001 once the filter's 27 warm-up months
are excluded): errors stay positively autocorrelated for almost a year, so there is
pattern left that this model does not capture. The 12-month projection sees
inflation easing from 3.72% (Sep 2026) to 3.11% (Oct 2027), 80% band 1.04% to 5.23%.

## Run

**CLI** (metrics + `cpi_forecast.png`):

```bash
pip install -r requirements.txt
python3 cpi_forecast.py
```

**Web UI** — an animated, step-by-step "working paper" (editorial macro-research
style: paper white, Fraunces serif, tabular mono numbers, a single signal-red)
that walks through the model, AIC selection, diagnostics, the honest backtest,
and the 12-month forecast with a chart. Pick the inflation measure (CPI / Core
CPI / PCE / Core PCE):

```bash
python3 server.py   # -> http://localhost:8000
```

Backend is stdlib-only (`http.server`); the model runs in `cpi_forecast.analyze()`
and results are cached per series. `render.yaml`-style deploy works the same as
project #1 (live Python backend, so not static hosting).

## Notes

- Order selection is a curated shortlist, not an exhaustive grid — swap in
  `pmdarima.auto_arima` if you want full tuning.
- Walk-forward re-filters with fixed params each origin (`.apply`, no refit) — fast and honest.
- `python.org` Python ships no CA bundle, so the fetch uses `certifi`.
