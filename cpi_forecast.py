"""
Quantum #2 — Macroeconomic variable forecasting: US CPI inflation.

Classic econometrics done honestly:
  - SARIMA order chosen by AIC over a small candidate set (the "airline model"
    family), not a hard-coded guess.
  - Walk-forward backtest: at each month in the holdout we forecast 1..12 months
    ahead using only past data, then score by horizon. This is how a forecaster
    is actually judged — not a single static multi-year projection.
  - Honest baselines: random walk on the *inflation rate* (freeze last YoY) and
    seasonal naive (YoY from a year ago). These are genuinely hard to beat, so
    the comparison means something.
  - Diagnostics: ADF stationarity, Ljung-Box residual autocorrelation, AIC.

Data: FRED series CPIAUCSL (keyless public CSV, no API key).
Run:  python3 cpi_forecast.py   ->  metrics to stdout + cpi_forecast.png
"""
import io
import ssl
import urllib.request
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import certifi  # python.org builds ship no CA bundle; use certifi's
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.simplefilter("ignore")  # statsmodels convergence chatter, not our concern

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
# Whitelist of monthly price indices — all make YoY inflation meaningful and use
# the exact same pipeline. Whitelisting also stops user input reaching the URL (SSRF).
SERIES = {
    "CPIAUCSL": "US CPI — all items",
    "CPILFESL": "US Core CPI — ex food & energy",
    "PCEPI":    "US PCE price index",
    "PCEPILFE": "US Core PCE — the Fed's preferred gauge",
}
DEFAULT_SERIES = "CPIAUCSL"
HOLDOUT = 36          # months of walk-forward backtest
FORECAST_H = 12       # months to forecast out-of-sample
HORIZONS = (1, 3, 6, 12)

# Candidate orders: the classic monthly-seasonal ("airline") family plus a few
# richer variants. AIC picks; we don't hard-code one and hope.
# ponytail: curated shortlist, not a full grid search. Swap in pmdarima.auto_arima
# if you want exhaustive tuning — this covers the orders that actually matter for CPI.
CANDIDATES = [
    ((0, 1, 1), (0, 1, 1, 12)),   # airline model — the default to beat
    ((1, 1, 1), (0, 1, 1, 12)),
    ((2, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 2), (0, 1, 1, 12)),
    ((2, 1, 2), (0, 1, 1, 12)),
    ((1, 1, 1), (1, 1, 1, 12)),
    ((2, 1, 2), (1, 1, 1, 12)),
]


def load_series(series_id=DEFAULT_SERIES):
    if series_id not in SERIES:
        raise ValueError(f"unknown series {series_id!r}")
    req = urllib.request.Request(FRED_CSV.format(series_id), headers={"User-Agent": "quantum-cpi/1.0"})
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df.columns = ["date", "cpi"]
    s = pd.Series(df["cpi"].values, index=pd.to_datetime(df["date"]), name="cpi")
    return s.asfreq("MS").dropna()   # SARIMAX needs an explicit monthly freq


load_cpi = load_series  # back-compat alias for main()


def yoy(cpi):
    """Year-over-year inflation, %."""
    return 100.0 * (cpi / cpi.shift(12) - 1.0)


def _fit(logcpi, order, seasonal):
    return SARIMAX(logcpi, order=order, seasonal_order=seasonal,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def select_order(logcpi):
    """Pick the (order, seasonal) with lowest AIC on the full sample."""
    scored = []
    for order, seasonal in CANDIDATES:
        try:
            scored.append((_fit(logcpi, order, seasonal).aic, order, seasonal))
        except Exception:
            continue
    scored.sort()
    return scored


def walk_forward(cpi, order, seasonal):
    """
    Rolling-origin backtest over the last HOLDOUT months. Fit once, then roll the
    filter forward one real observation at a time (fast, no refit), forecasting
    1..max(HORIZONS) ahead from each origin. Returns tidy rows: (h, actual, sarima,
    rw_yoy, snaive) — all YoY inflation %, comparable across models.
    """
    log = np.log(cpi)
    lvl = cpi.values.astype(float)
    yoy_act = yoy(cpi).values
    n, max_h = len(cpi), max(HORIZONS)
    start = n - HOLDOUT

    base_res = _fit(log.iloc[:start], order, seasonal)   # fit params ONCE, on pre-holdout data
    rows = []
    for o in range(start, n):                       # o = # observed months; origin is position o-1
        # re-filter the first o months with the fixed params (fast, no refit)
        res = base_res.apply(log.iloc[:o]) if o > start else base_res
        fc = np.exp(np.asarray(res.forecast(max_h)))  # forecast levels for positions o..o+max_h-1
        last_known_yoy = yoy_act[o - 1]
        for h in HORIZONS:
            tp = o + h - 1                          # target position (0-based)
            if tp >= n:
                continue
            bp = tp - 12                            # YoY base position
            base = lvl[bp] if bp < o else fc[bp - o]   # actual if known, else model's own forecast
            sar = 100.0 * (fc[h - 1] / base - 1.0)
            snaive = yoy_act[tp - 12] if tp - 12 >= 0 else np.nan   # inflation a year ago
            rows.append((h, yoy_act[tp], sar, last_known_yoy, snaive))
    return pd.DataFrame(rows, columns=["h", "actual", "sarima", "rw_yoy", "snaive"])


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def score_by_horizon(bt):
    out = []
    for h in HORIZONS:
        s = bt[bt.h == h]
        out.append((h, len(s),
                    rmse(s.actual, s.sarima),
                    rmse(s.actual, s.rw_yoy),
                    rmse(s.actual, s.snaive)))
    return pd.DataFrame(out, columns=["h_months", "n", "SARIMA", "RW_YoY", "SeasNaive"])


def one_step_series(cpi, bt):
    """h=1 walk-forward predictions, indexed by target date, for plotting."""
    idx = cpi.index[len(cpi) - HOLDOUT:]
    return pd.Series(bt[bt.h == 1]["sarima"].values, index=idx)


def final_forecast(cpi, order, seasonal):
    fit = _fit(np.log(cpi), order, seasonal)
    fc = fit.get_forecast(FORECAST_H)
    future = pd.date_range(cpi.index[-1], periods=FORECAST_H + 1, freq="MS")[1:]
    lvl = pd.Series(np.exp(np.asarray(fc.predicted_mean)), index=future)
    ci = np.exp(np.asarray(fc.conf_int(alpha=0.20)))     # 80% band
    return lvl, pd.DataFrame(ci, index=future, columns=["lo", "hi"]), fit


def diagnostics(fit, cpi):
    lb = acorr_ljungbox(fit.resid[13:], lags=[12, 24], return_df=True)
    adf_p = adfuller(np.log(cpi).diff().dropna())[1]
    return fit.aic, lb["lb_pvalue"].to_dict(), adf_p


def plot(cpi, one_step, fc_lvl, ci):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    recent = cpi.iloc[-72:]
    ax1.plot(recent.index, recent.values, label="CPI (actual)", color="#222")
    ax1.plot(fc_lvl.index, fc_lvl.values, "--", label=f"SARIMA forecast (+{FORECAST_H}m)", color="#c48a2c")
    ax1.fill_between(ci.index, ci["lo"], ci["hi"], color="#c48a2c", alpha=0.18, label="80% interval")
    ax1.set_title("US CPI (CPIAUCSL) — level + 12-month SARIMA forecast")
    ax1.legend(); ax1.grid(alpha=0.3)

    infl = yoy(cpi).iloc[-72:]
    ax2.plot(infl.index, infl.values, label="YoY inflation (actual)", color="#222")
    ax2.plot(one_step.index, one_step.values, label="SARIMA 1-step (walk-forward)", color="#c48a2c")
    ax2.axhline(2.0, ls=":", color="gray", label="Fed 2% target")
    ax2.set_title("YoY inflation — actual vs. 1-step-ahead walk-forward"); ax2.set_ylabel("%")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("cpi_forecast.png", dpi=130)


def _fmt_order(order, seasonal):
    return f"({order[0]},{order[1]},{order[2]})({seasonal[0]},{seasonal[1]},{seasonal[2]})[12]"


@lru_cache(maxsize=8)   # ponytail: full run is ~10s; FRED updates monthly, so cache per series
def analyze(series_id=DEFAULT_SERIES):
    """Run the whole pipeline for one series and return a JSON-able dict for the web UI."""
    cpi = load_series(series_id)
    log = np.log(cpi)
    scored = select_order(log)
    _, order, seasonal = scored[0]

    bt = walk_forward(cpi, order, seasonal)
    table = score_by_horizon(bt)
    fc_lvl, ci, fit = final_forecast(cpi, order, seasonal)
    aic, lb, adf_p = diagnostics(fit, cpi)

    hist = yoy(cpi).dropna().iloc[-48:]
    combined = pd.concat([cpi, fc_lvl])
    base12 = combined.shift(12).loc[fc_lvl.index]     # YoY base level for each forecast month
    fc_yoy = 100.0 * (fc_lvl / base12 - 1.0)
    lo_yoy = 100.0 * (ci["lo"] / base12 - 1.0)
    hi_yoy = 100.0 * (ci["hi"] / base12 - 1.0)

    return {
        "series": series_id,
        "series_name": SERIES[series_id],
        "start": cpi.index[0].strftime("%Y-%m"),
        "end": cpi.index[-1].strftime("%Y-%m"),
        "n": int(len(cpi)),
        "latest_yoy": round(float(yoy(cpi).iloc[-1]), 2),
        "candidates": [{"order": _fmt_order(o, s), "aic": round(a, 1)} for a, o, s in scored],
        "chosen": _fmt_order(order, seasonal),
        "backtest": [
            {"h": int(r.h_months), "n": int(r.n),
             "sarima": round(r.SARIMA, 3), "rw": round(r.RW_YoY, 3), "snaive": round(r.SeasNaive, 3),
             "winner": min(("SARIMA", r.SARIMA), ("RW", r.RW_YoY), ("SeasNaive", r.SeasNaive),
                           key=lambda x: x[1])[0]}
            for _, r in table.iterrows()
        ],
        "bias": round(float((bt[bt.h == 1].actual - bt[bt.h == 1].sarima).mean()), 3),
        "diag": {"aic": round(aic, 1), "lb12": round(float(lb[12]), 3),
                 "lb24": round(float(lb[24]), 3), "adf_p": round(float(adf_p), 4)},
        "history": [{"date": d.strftime("%Y-%m"), "yoy": round(float(v), 2)} for d, v in hist.items()],
        "forecast": [
            {"date": d.strftime("%Y-%m"), "yoy": round(float(fc_yoy[d]), 2),
             "lo": round(float(lo_yoy[d]), 2), "hi": round(float(hi_yoy[d]), 2)}
            for d in fc_lvl.index
        ],
    }


def _selfcheck():
    idx = pd.date_range("2000-01-01", periods=25, freq="MS")
    doubling = pd.Series([100 * 2 ** (i / 12) for i in range(25)], index=idx)
    assert abs(yoy(doubling).iloc[-1] - 100.0) < 1e-6, "yoy() broken"
    assert rmse([1, 2, 3], [1, 2, 3]) == 0, "rmse() broken"
    assert rmse([1, np.nan, 3], [1, 9, 3]) == 0, "rmse() should skip NaN pairs"


def main():
    _selfcheck()
    cpi = load_cpi()
    print(f"CPI loaded: {cpi.index[0].date()} … {cpi.index[-1].date()} ({len(cpi)} months)")
    print(f"Latest YoY inflation: {yoy(cpi).iloc[-1]:.2f}%\n")

    scored = select_order(np.log(cpi))
    print("Model selection by AIC (lower = better):")
    for aic, order, seasonal in scored:
        print(f"  SARIMA{order}{seasonal[:3]}s12   AIC={aic:9.1f}")
    _, order, seasonal = scored[0]
    print(f"  -> chosen: SARIMA{order}{seasonal[:3]}s12\n")

    bt = walk_forward(cpi, order, seasonal)
    table = score_by_horizon(bt)
    print(f"Walk-forward backtest (last {HOLDOUT} months) — YoY inflation RMSE by horizon:")
    print("  h(mo)   n   SARIMA   RW_YoY  SeasNaive   winner")
    for _, r in table.iterrows():
        best = min(("SARIMA", r.SARIMA), ("RW_YoY", r.RW_YoY), ("SeasNaive", r.SeasNaive), key=lambda x: x[1])[0]
        print(f"   {int(r.h_months):>3} {int(r.n):>4}  {r.SARIMA:>6.3f}  {r.RW_YoY:>6.3f}   {r.SeasNaive:>7.3f}   {best}")
    bias = float((bt[bt.h == 1].actual - bt[bt.h == 1].sarima).mean())
    print(f"  1-step mean bias (actual - pred): {bias:+.3f} pp\n")

    fc_lvl, ci, fit = final_forecast(cpi, order, seasonal)
    aic, lb, adf_p = diagnostics(fit, cpi)
    print("Diagnostics (full-sample fit):")
    print(f"  AIC = {aic:.1f}")
    print(f"  Ljung-Box p-value  lag12={lb[12]:.3g}  lag24={lb[24]:.3g}  (p>0.05 = residuals ~ white noise)")
    print(f"  ADF on Δlog(CPI): p={adf_p:.4f}  (p<0.05 = differenced series is stationary)\n")

    next_infl = yoy(pd.concat([cpi, fc_lvl])).loc[fc_lvl.index]
    print(f"Out-of-sample forecast (next {FORECAST_H} months), implied YoY inflation:")
    for d, v in next_infl.items():
        print(f"  {d.date()}  {v:5.2f}%")

    plot(cpi, one_step_series(cpi, bt), fc_lvl, ci)
    print("\nSaved cpi_forecast.png")


if __name__ == "__main__":
    main()
