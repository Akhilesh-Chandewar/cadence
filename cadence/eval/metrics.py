"""Metrics: MASE and sMAPE — the spec §9 primary scoring pair (Phase 1).

WQL (quantile models, Phase 5+) lands here too when probabilistic tiers arrive.
"""

from __future__ import annotations

import numpy as np


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_hist: np.ndarray,
    seasonality: int,
) -> float:
    """Mean Absolute Scaled Error.

    Scale-free (comparable across series, spec §9): MAE of the forecast divided by
    the MAE of the one-step naive forecast on the training history, i.e.
    scale = mean(|y_hist[m:] - y_hist[:-m]|) with m = seasonality.

    MASE < 1 beats the seasonal-naive baseline on the same history.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    if seasonality < 1:
        raise ValueError(f"seasonality must be >= 1, got {seasonality}")
    if len(y_hist) <= seasonality:
        raise ValueError(
            f"history too short for MASE: {len(y_hist)} points, seasonality={seasonality}"
        )
    scale = np.mean(np.abs(y_hist[seasonality:] - y_hist[:-seasonality]))
    if scale == 0:
        raise ValueError("MASE undefined: training history is constant (naive scale is zero)")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error, bounded in [0, 2].

    2.0 when the forecast is exactly wrong on a zero target; 0.0 when perfect.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    # Convention: when both are 0 the term is 0 (perfect), not undefined.
    terms = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom))
    return float(2.0 * np.mean(terms))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss at quantile level q in (0, 1).

    L_q(y, ŷ) = q·(y - ŷ) when y >= ŷ, else (1 - q)·(ŷ - y). Zero only when the
    forecast sits exactly on the empirical q-quantile boundary everywhere.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    losses = np.where(diff >= 0, q * diff, (1 - q) * -diff)
    return float(np.mean(losses))


def wql(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Weighted Quantile Loss at level q (spec §9 probabilistic metric).

    WQL_q = Σ pinball losses / Σ |y| — scale-free, comparable across series
    (the M5 formulation; M5 multiplies the numerator by 2, which rescales but
    preserves ranking).
    """
    y_true = np.asarray(y_true, dtype=float)
    denom = float(np.abs(y_true).sum())
    if denom == 0:
        raise ValueError("WQL undefined: all-true-values are zero (scale is zero)")
    return pinball_loss(y_true, y_pred, q) * len(y_true) / denom


def wql_from_interval(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: int = 95) -> float:
    """WQL averaged over the two quantiles an (1-level)% interval exposes.

    A model reporting only lo/hi at level L gives quantiles at alpha=(1-L/100)/2
    and 1-alpha; this averages their WQL — a fair §9 score for interval-only
    probabilistic outputs.
    """
    alpha = (100 - level) / 200
    return float((wql(y_true, lo, alpha) + wql(y_true, hi, 1 - alpha)) / 2)
