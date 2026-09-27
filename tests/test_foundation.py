"""Tier 4 (Chronos-Bolt) tests.

Network-free by default: the HF weights download is gated behind
CADENCE_TEST_CHRONOS=1 (run manually after `uv sync --extra chronos`).
Everything else stubs `_pipeline_quantiles`, verifying frame shapes, the
interval-column contract, and §9 scorer compatibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadence.models.base import Forecaster, MissingDependencyError, require_group
from tests.conftest import make_canonical_df


class TestDependencyGating:
    def test_missing_chronos_gives_extra_hint(self):
        with pytest.raises(MissingDependencyError, match="uv sync --extra chronos"):
            require_group("definitely_not_chronos_xyz", "extra:chronos")

    def test_protocol_compliance(self):
        pytest.importorskip("chronos")
        from cadence.models.foundation import ChronosForecaster

        assert isinstance(ChronosForecaster(), Forecaster)


class TestStubbedPipeline:
    """Verify the wrapper's frame logic with a deterministic fake pipeline."""

    @pytest.fixture
    def fc(self):
        pytest.importorskip("chronos")
        from cadence.config.default_config import ChronosConfig
        from cadence.models.foundation import ChronosForecaster

        forecaster = ChronosForecaster(config=ChronosConfig())
        # stub the heavy pipeline: deterministic linear quantiles
        forecaster._pipeline = object()  # anything non-None
        forecaster._freq = "D"
        forecaster._df = make_canonical_df(uids=["a", "b"], n=60, freq="D")
        forecaster._pipeline_quantiles = self._fake_quantiles  # type: ignore[method-assign]
        return forecaster

    @staticmethod
    def _fake_quantiles(context: np.ndarray, h: int, quantile_levels: list[float]):
        """Deterministic stand-in: (h, n_levels) quantiles increasing across levels."""
        last = float(context[-1])
        mean = np.full(h, last)
        q = last + np.arange(h, dtype=float)[:, None] + np.array(quantile_levels)[None, :]
        return q, mean

    def test_predict_wide_frame_with_intervals(self, fc):
        out = fc.predict(h=5)

        assert len(out) == 10  # 2 series x 5 steps
        assert {
            "unique_id",
            "ds",
            "Chronos-Bolt",
            "Chronos-Bolt-lo-95",
            "Chronos-Bolt-hi-95",
        } <= set(out.columns)
        assert (out["Chronos-Bolt-lo-95"] <= out["Chronos-Bolt-hi-95"]).all()
        # future timestamps continue after each series' last observed ds
        for uid, grp in fc._df.groupby("unique_id"):
            last = grp["ds"].max()
            assert (out.loc[out["unique_id"] == uid, "ds"] > last).all()

    def test_point_only_when_level_empty(self, fc):
        out = fc.predict(h=4, level=())
        assert "Chronos-Bolt-lo-95" not in out.columns
        assert "Chronos-Bolt" in out.columns

    def test_series_too_short_raises(self, fc):
        fc._df = make_canonical_df(uids=["tiny"], n=1, freq="D")
        with pytest.raises(ValueError, match="too short"):
            fc.predict(h=3)

    def test_cross_validation_structure(self, fc):
        cv = fc.cross_validation(h=6, n_windows=3)

        assert {"unique_id", "ds", "cutoff", "y", "Chronos-Bolt"} <= set(cv.columns)
        assert len(cv) == 36  # 2 series x 3 windows x 6 steps
        # expanding context: later windows have later cutoffs
        for _uid, grp in cv.groupby("unique_id"):
            assert grp["cutoff"].is_monotonic_increasing
            assert grp.groupby("cutoff").size().eq(6).all()

    def test_cv_too_short_raises(self, fc):
        fc._df = make_canonical_df(uids=["tiny"], n=20, freq="D")
        with pytest.raises(ValueError, match="too short"):
            fc.cross_validation(h=6, n_windows=3)

    def test_cv_compatible_with_shared_scorer(self, fc):
        """Zoo-wide guarantee: the §9 scorer consumes tier-4 CV frames unchanged."""
        from cadence.eval.backtest import _score_frame

        cv = fc.cross_validation(h=6, n_windows=3)
        scores = _score_frame(cv, ["Chronos-Bolt"], seasonality=7, train_history=fc._df)
        assert len(scores) == 6  # 2 series x 3 windows
        assert all(np.isfinite(s.mase) and np.isfinite(s.smape) for s in scores)


class TestRealWeights:
    """Manual gate: downloads HF weights (~200MB). CADENCE_TEST_CHRONOS=1 to run."""

    @pytest.mark.skipif(
        not __import__("os").environ.get("CADENCE_TEST_CHRONOS"),
        reason="set CADENCE_TEST_CHRONOS=1 to download weights and run",
    )
    def test_real_chronos_bolt_end_to_end(self):
        from cadence.config.default_config import ChronosConfig
        from cadence.connectors.csv_connector import CSVConnector
        from cadence.models.foundation import ChronosForecaster

        df = CSVConnector().load("data/sample/air_passengers.csv").df
        fc = ChronosForecaster(config=ChronosConfig(model_id="amazon/chronos-bolt-small")).fit(df)

        out = fc.predict(h=12)
        assert len(out) == 12
        assert (out["Chronos-Bolt-lo-95"] <= out["Chronos-Bolt-hi-95"]).all()
        # zero-shot sanity: forecast stays in a plausible band around the data
        assert 50 < out["Chronos-Bolt"].mean() < 1500

        cv = fc.cross_validation(h=12, n_windows=3)
        from cadence.eval.backtest import _score_frame

        scores = _score_frame(cv, ["Chronos-Bolt"], seasonality=12, train_history=df)
        assert all(np.isfinite(s.mase) for s in scores)
