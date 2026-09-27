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
