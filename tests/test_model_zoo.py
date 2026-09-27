"""Phase 4 model-zoo tests: one Forecaster contract across tiers.

ML/DL tier tests skip (not fail) when the optional dependency groups aren't
synced — the core suite stays green on `uv sync --group dev` alone.
"""

from __future__ import annotations

import pytest

from cadence.models.base import Forecaster, MissingDependencyError, require_group
from cadence.models.classical import ClassicalForecaster, ClassicalModelConfig
from tests.conftest import make_canonical_df


class TestDependencyGating:
    def test_missing_module_gives_actionable_error(self):
        with pytest.raises(MissingDependencyError, match="uv sync --group ml"):
            require_group("definitely_not_a_real_module_xyz", "ml")

    def test_present_module_passes(self):
        require_group("pandas", "core")  # any installed module passes

    def test_protocol_compliance(self):
        # runtime-checkable protocol: verifies the method surface exists
        assert isinstance(ClassicalForecaster(ClassicalModelConfig()), Forecaster)
        from cadence.models.deep_learning import DLForecaster
        from cadence.models.ml import MLForecaster

        assert isinstance(MLForecaster(), Forecaster)
        assert isinstance(DLForecaster(), Forecaster)


@pytest.fixture(scope="module")
def ml_df():
    return make_canonical_df(uids=["a", "b"], n=80, freq="D")


class TestMLTier:
    def test_requires_fit(self):
        pytest.importorskip("mlforecast")
        from cadence.models.ml import MLForecaster

        with pytest.raises(RuntimeError, match="fit"):
            MLForecaster().predict(h=3)

    def test_predict_wide_frame_with_intervals(self, ml_df):
        pytest.importorskip("mlforecast")
        from cadence.models.ml import MLForecaster

        fc = MLForecaster().fit(ml_df)
        out = fc.predict(h=7)

        assert len(out) == 14  # 2 series x 7 steps
        assert "MLForecast-LightGBM" in out.columns
        assert "MLForecast-LightGBM-lo-95" in out.columns
        assert "MLForecast-LightGBM-hi-95" in out.columns
        lo = out["MLForecast-LightGBM-lo-95"]
        hi = out["MLForecast-LightGBM-hi-95"]
        assert (lo <= hi).all()  # interval sanity

    def test_predict_beyond_calibration_horizon_raises(self, ml_df):
        pytest.importorskip("mlforecast")
        from cadence.config.default_config import MLModelConfig
        from cadence.models.ml import MLForecaster

        fc = MLForecaster(config=MLModelConfig(interval_horizon=6)).fit(ml_df)
        with pytest.raises(ValueError, match="calibration horizon"):
            fc.predict(h=7)

    def test_cross_validation_structure_and_scoring(self, ml_df):
        pytest.importorskip("mlforecast")
        from cadence.eval.backtest import _score_frame
        from cadence.models.ml import MLForecaster

        fc = MLForecaster().fit(ml_df)
        cv = fc.cross_validation(h=7, n_windows=3)

        assert {"unique_id", "ds", "cutoff", "y", "MLForecast-LightGBM"} <= set(cv.columns)
        # cutoffs are per (series x window); the two series have offset date ranges
        assert cv.groupby("unique_id")["cutoff"].nunique().eq(3).all()

        # zoo-wide guarantee: the Phase 1 scorer consumes any tier's CV frame
        scores = _score_frame(cv, ["MLForecast-LightGBM"], seasonality=7, train_history=ml_df)
        assert len(scores) == 6  # 2 series x 3 windows
        assert all(s.n_forecast == 7 for s in scores)
        assert all(s.smape >= 0.0 for s in scores)


class TestDLTier:
    def test_requires_fit(self):
        pytest.importorskip("neuralforecast")
        from cadence.models.deep_learning import DLForecaster

        with pytest.raises(RuntimeError, match="fit"):
            DLForecaster().predict()

    def test_unsupported_model_rejected(self):
        pytest.importorskip("neuralforecast")
        from cadence.config.default_config import DLModelConfig
        from cadence.models.deep_learning import DLForecaster

        fc = DLForecaster(config=DLModelConfig(models=["GPT5"]))
        with pytest.raises(ValueError, match="unsupported DL model"):
            fc._build_models(h=6, freq="D")

    def test_fit_predict_and_horizon_fixity(self):
        pytest.importorskip("neuralforecast")
        from cadence.config.default_config import DLModelConfig
        from cadence.connectors.csv_connector import CSVConnector
        from cadence.models.deep_learning import DLForecaster

        df = CSVConnector().load("data/sample/air_passengers.csv").df
        cfg = DLModelConfig(models=["NHITS"], input_size=24, max_steps=3)
        fc = DLForecaster(config=cfg, horizon=6).fit(df)

        out = fc.predict()
        assert len(out) == 6  # horizon-fixed
        assert "NHITS" in out.columns

        # zoo-wide alias works only for the fitted horizon
        assert len(fc.forecast()) == 6
        with pytest.raises(ValueError, match="horizon-fixed"):
            fc.forecast(h=13)

    def test_cross_validation_smoke(self):
        pytest.importorskip("neuralforecast")
        from cadence.config.default_config import DLModelConfig
        from cadence.connectors.csv_connector import CSVConnector
        from cadence.models.deep_learning import DLForecaster

        df = CSVConnector().load("data/sample/air_passengers.csv").df
        cfg = DLModelConfig(models=["NHITS"], input_size=24, max_steps=3)
        fc = DLForecaster(config=cfg).fit(df)
        cv = fc.cross_validation(h=6, n_windows=3)

        assert {"unique_id", "ds", "cutoff", "y", "NHITS"} <= set(cv.columns)
        assert cv["cutoff"].nunique() == 3
