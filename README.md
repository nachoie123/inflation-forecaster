# Macroeconomic Variable Forecasting — US CPI Inflation

Quantum project #2. Forecasts US consumer-price inflation with a classic
seasonal-ARIMA model and backtests it honestly against naive baselines.

The point isn't nailing the number — it's the process: pull real data, difference
out trend and seasonality, hold out a window, and show the model beats the naive
"tomorrow looks like today" forecast.

## What it does

- **Data:** FRED series `CPIAUCSL` (US CPI), pulled from the public keyless CSV — no API key.
- **Model selection:** SARIMA order chosen by **AIC** over a curated candidate set (the classic monthly "airline" family + richer variants) on `log(CPI)` — not a hard-coded guess. Current winner: `SARIMA(1,1,1)(0,1,1)₁₂`.
- **Backtest:** **walk-forward** over the last 36 months — at each month we forecast 1/3/6/12 months ahead using only past data, then score by horizon. This is how a forecaster is actually judged, not one static multi-year projection.
- **Honest baselines:** random walk on the *inflation rate* (freeze the last YoY) and seasonal naive (YoY from a year ago). These are genuinely hard to beat, so the comparison is meaningful.
- **Diagnostics:** ADF stationarity, Ljung-Box residual autocorrelation, AIC.
- **Output:** tables to stdout + `cpi_forecast.png` (level forecast with 80% band, and 1-step walk-forward vs. actual inflation).

## Result (as of Jun 2026 data) — RMSE on YoY inflation, by horizon

| Horizon | SARIMA | RW-on-inflation | Seasonal naive |
|--------:|-------:|----------------:|---------------:|
| 1 month  | **0.24** | 0.32 | 2.29 |
| 3 months | **0.49** | 0.51 | 1.93 |
| 6 months | 0.64 | **0.53** | 1.45 |
| 12 months| 1.15 | **0.83** | 0.83 |

The honest story: **SARIMA wins at short horizons (1–3 months); at 6–12 months a
random walk on the inflation rate is unbeatable** — a well-known result in the
inflation-forecasting literature (Atkeson–Ohanian). Residuals pass Ljung-Box
(p ≈ 0.95, white noise) and 1-step bias is ~0. The 12-month projection sees
inflation easing from ~3.7% toward the Fed's 2% target.

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
