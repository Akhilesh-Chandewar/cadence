"""Pure per-series diagnostics (spec §7.2) — no LLM, no I/O, fully testable.

Each function takes one series (numpy array or a single unique_id's frame slice)
and returns plain values. DiagnosticAgent composes these into the diagnostics.json
contract; Phase 2's known-answer tests run them against the synthetic fixtures.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

# strength at/above which seasonality is called "present" (Hyndman's rule of thumb)
SEASONALITY_STRENGTH_THRESHOLD = 0.6


# |ACF| above ~2/sqrt(n) is outside the 95% white-noise band
def _acf_significance_band(n: int) -> float:
    return 2.0 / np.sqrt(n)


# a series is "intermittent" when at least this share of observations is zero
ZERO_FRACTION_THRESHOLD = 0.5


def compute_trend(y: np.ndarray) -> dict:
    """Linear-fit slope significance (spec §7.2 option 1)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return {"present": False, "direction": "none", "slope": 0.0, "pvalue": 1.0}
    t = np.arange(n, dtype=float)
    fit = sps.linregress(t, y)
    present = bool(fit.pvalue < 0.05 and fit.slope != 0.0)
    return {
        "present": present,
        "direction": ("up" if fit.slope > 0 else "down") if present else "none",
        "slope": float(fit.slope),
        "pvalue": float(fit.pvalue),
    }


def stl_strength(y: np.ndarray, period: int) -> float:
    """Seasonal strength Fs = max(0, 1 - Var(resid)/Var(seasonal+resid)); 0 when degenerate."""
    y = np.asarray(y, dtype=float)
    if period < 2 or len(y) < 2 * period:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = STL(y, period=period, robust=True).fit()
    denom = np.var(np.asarray(res.seasonal) + np.asarray(res.resid))
    if denom <= 0:
        return 0.0
    return float(max(0.0, 1.0 - np.var(np.asarray(res.resid)) / denom))


def compute_seasonality(y: np.ndarray, period: int) -> dict:
    """STL strength + ACF cross-check (spec §7.2: 'STL decomposition; ACF peak detection')."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if period < 2 or n < 2 * period:
        return {"present": False, "period": period, "strength": 0.0, "acf_at_period": 0.0}

    strength = stl_strength(y, period)
    centered = y - y.mean()
    acf_at_period = float(pd.Series(centered).autocorr(lag=period))
    if np.isnan(acf_at_period):
        acf_at_period = 0.0

    band = _acf_significance_band(n)
    present = bool(strength >= SEASONALITY_STRENGTH_THRESHOLD and acf_at_period > band)
    return {
        "present": present,
        "period": period,
        "strength": round(strength, 4),
        "acf_at_period": round(acf_at_period, 4),
    }


def compute_stationarity(y: np.ndarray) -> dict:
    """ADF (H0: unit root) AND KPSS (H0: stationary) — run both, per spec §7.2.

    Degenerate series (short perfect repeats, zero-variance tails) can break the
    tests' internal lag math (KPSS autolag overflows to inf); those fall back to
    a fixed lag count, then to an `ambiguous` verdict rather than a crash.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 8 or np.var(y) == 0:
        return {"adf_pvalue": 1.0, "kpss_pvalue": 1.0, "verdict": "ambiguous"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS p-value table-range warnings
        try:
            adf_pvalue: float | None = float(adfuller(y, autolag="AIC")[1])
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            adf_pvalue = None
        try:
            kpss_pvalue: float | None = float(kpss(y, regression="c", nlags="auto")[1])
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            # gamma_hat can be inf on perfectly-repeating series — retry fixed lag
            try:
                kpss_pvalue = float(kpss(y, regression="c", nlags=max(1, len(y) // 4))[1])
            except (ValueError, np.linalg.LinAlgError, OverflowError):
                kpss_pvalue = None

    if adf_pvalue is None or kpss_pvalue is None:
        return {
            "adf_pvalue": 1.0 if adf_pvalue is None else round(adf_pvalue, 6),
            "kpss_pvalue": 1.0 if kpss_pvalue is None else round(kpss_pvalue, 6),
            "verdict": "ambiguous",
        }

    adf_stationary = adf_pvalue < 0.05  # reject unit root
    kpss_stationary = kpss_pvalue >= 0.05  # fail to reject stationarity

    if adf_stationary and kpss_stationary:
        verdict = "stationary"
    elif not adf_stationary and not kpss_stationary:
        verdict = "non_stationary"
    else:
        verdict = "ambiguous"
    return {
        "adf_pvalue": round(adf_pvalue, 6),
        "kpss_pvalue": round(kpss_pvalue, 6),
        "verdict": verdict,
    }


def compute_outliers(y: np.ndarray, period: int = 1) -> tuple[int, np.ndarray]:
    """Rolling-window IQR flags (spec §7.2: flagged, never silently removed).

    Window is 2*period+1 (odd, centered) so seasonal peaks don't flag themselves.
    Returns (count, boolean mask).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return 0, np.zeros(n, dtype=bool)
    window = max(9, 2 * period + 1) | 1  # force odd
    s = pd.Series(y)
    q25 = s.rolling(window, center=True, min_periods=max(3, window // 2)).quantile(0.25)
    q75 = s.rolling(window, center=True, min_periods=max(3, window // 2)).quantile(0.75)
    iqr = q75 - q25
    # 3x (not Tukey's 1.5x): quartiles from small rolling windows are noisy, and
    # 1.5x flags ~5% of clean gaussian points; 3x keeps the flag rate near zero
    lower, upper = q25 - 3.0 * iqr, q75 + 3.0 * iqr
    mask = ((y < lower) | (y > upper)).to_numpy()
    mask &= np.isfinite(y)
    return int(mask.sum()), mask


def compute_intermittency(y: np.ndarray) -> dict:
    """Demand-shaped check (spec §7.2): mostly zeros → Croston/TSB territory."""
    y = np.asarray(y, dtype=float)
    zero_fraction = float((y == 0).mean()) if len(y) else 0.0
    return {
        "intermittent": bool(zero_fraction >= ZERO_FRACTION_THRESHOLD),
        "zero_fraction": round(zero_fraction, 4),
    }


def compute_missingness(ds: pd.Series, freq: str) -> dict:
    """Gap analysis against the expected regular grid (canonical validator rejects
    NaN y upstream, so 'missing' here means missing timestamps).

    Returns pct_missing plus the expected row count the grid implies.
    """
    ds = pd.DatetimeIndex(pd.Series(ds).sort_values())
    if len(ds) < 2:
        return {"pct_missing": 0.0, "n_expected": len(ds), "n_gaps": 0}
    full = pd.date_range(ds.min(), ds.max(), freq=freq)
    n_gaps = max(0, len(full) - len(ds))
    return {
        "pct_missing": round(n_gaps / len(full), 6) if len(full) else 0.0,
        "n_expected": len(full),
        "n_gaps": n_gaps,
    }
