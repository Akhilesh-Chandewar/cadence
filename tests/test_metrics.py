"""Known-answer tests for the §9 metric primitives."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.eval.metrics import mase, smape


class TestMASE:
    def test_perfect_forecast_is_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert mase(y, y, y, seasonality=1) == 0.0

    def test_naive_forecast_is_one(self) -> None:
        y_hist = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y_true = y_hist[1:]
        y_pred = y_hist[:-1]  # one-step naive
        assert mase(y_true, y_pred, y_hist, seasonality=1) == pytest.approx(1.0)

    def test_scale_free_across_series_magnitudes(self) -> None:
        """Same relative error on a x1000 series should give the same MASE."""
        rng = np.random.default_rng(0)
        hist = np.cumsum(rng.normal(0, 1, 50)) + 100
        y_true = hist[-5:] + rng.normal(0, 1, 5)
        y_pred = hist[-5:] + rng.normal(0, 1, 5)
        small = mase(y_true, y_pred, hist, seasonality=1)
        big = mase(y_true * 1000, y_pred * 1000, hist * 1000, seasonality=1)
        assert small == pytest.approx(big, rel=1e-9)

    def test_invalid_seasonality_raises(self) -> None:
        with pytest.raises(ValueError, match="seasonality"):
            mase(np.ones(3), np.ones(3), np.ones(5), seasonality=0)

    def test_short_history_raises(self) -> None:
        with pytest.raises(ValueError, match="history too short"):
            mase(np.ones(2), np.ones(2), np.ones(2), seasonality=3)

    def test_constant_history_raises(self) -> None:
        with pytest.raises(ValueError, match="constant"):
            mase(np.ones(2), np.ones(2), np.full(10, 5.0), seasonality=1)

    def test_seasonal_scale_is_lag_m_not_mth_difference(self) -> None:
        """Regression: scale must be the lag-m |difference|, not np.diff(n=m).

        On a linear ramp the m-th difference is ~0 while the lag-m difference
        equals the slope — the buggy version collapsed the scale to zero.
        """
        ramp = np.arange(50, dtype=float)  # slope 1/step
        y_true = ramp[-4:] + 7.0  # off by 7 per step
        y_pred = ramp[-4:]
        # lag-7 diff = 7 everywhere; forecast error = 7 → MASE = 7/7 = 1.0
        # (exactly ties the seasonal-naive baseline — the correct semantics)
        assert mase(y_true, y_pred, ramp, seasonality=7) == pytest.approx(1.0)


class TestSMAPE:
    def test_perfect_is_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert smape(y, y) == 0.0

    def test_wrong_on_zero_target_is_two(self) -> None:
        y_true = np.zeros(3)
        y_pred = np.ones(3)
        assert smape(y_true, y_pred) == pytest.approx(2.0)

    def test_both_zero_is_zero_not_nan(self) -> None:
        y = np.zeros(4)
        assert smape(y, y) == 0.0

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(1)
        a, b = rng.random(10) + 0.1, rng.random(10) + 0.1
        assert smape(a, b) == pytest.approx(smape(b, a))

    def test_bounded(self) -> None:
        rng = np.random.default_rng(2)
        a, b = rng.random(50), rng.random(50) * 100
        assert 0.0 <= smape(a, b) <= 2.0
