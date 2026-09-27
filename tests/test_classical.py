"""ClassicalForecaster + inference helper tests (statsforecast Tier 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cadence.models.classical import (
    MODEL_REGISTRY,
    ClassicalForecaster,
    ClassicalModelConfig,
    infer_frequency,
    infer_season_length,
    normalize_frequency,
)
from tests.conftest import make_canonical_df


class TestInference:
    def test_infer_frequency_daily(self) -> None:
        assert infer_frequency(make_canonical_df(n=30, freq="D")) == "D"

    def test_infer_frequency_monthly(self) -> None:
        assert infer_frequency(make_canonical_df(n=30, freq="MS")) == "MS"

    def test_season_length_daily_is_weekly(self) -> None:
        assert infer_season_length(make_canonical_df(freq="D")) == 7

    def test_season_length_monthly_is_yearly(self) -> None:
        assert infer_season_length(make_canonical_df(freq="MS")) == 12

    def test_season_length_hourly(self) -> None:
        assert infer_season_length(make_canonical_df(freq="D"), freq="h") == 24

    def test_unknown_frequency_defaults_to_one(self) -> None:
        assert infer_season_length(make_canonical_df(freq="D"), freq="S") == 1

    def test_normalize_frequency_aliases(self) -> None:
        assert normalize_frequency("ME") == "M"
        assert normalize_frequency("W-SUN") == "W"


class TestConfig:
    def test_unknown_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown model"):
            ClassicalModelConfig(models=("NotAModel",))

    def test_registry_has_phase1_models(self) -> None:
        assert {"AutoARIMA", "AutoETS", "AutoTheta", "Croston"} <= set(MODEL_REGISTRY)


class TestClassicalForecaster:
    def test_forecast_requires_fit(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            ClassicalForecaster().forecast(h=3)

    def test_forecast_produces_intervals(self, air_passengers_csv: Path) -> None:
        from cadence.connectors.csv_connector import CSVConnector

        df = CSVConnector().load(air_passengers_csv).df
        fc = ClassicalForecaster(ClassicalModelConfig(models=("AutoARIMA", "AutoETS"))).fit(df)
        out = fc.forecast(h=12)

        assert len(out) == 12
        for name in ("AutoARIMA", "AutoETS"):
            assert name in out.columns
            assert f"{name}-lo-95" in out.columns
            assert f"{name}-hi-95" in out.columns
            # intervals bracket the point forecast row-wise
            assert (out[f"{name}-lo-95"] <= out[name]).all()
            assert (out[name] <= out[f"{name}-hi-95"]).all()
        # future timestamps continue after the last observed one (naive UTC out)
        last_naive = df["ds"].dt.tz_convert("UTC").dt.tz_localize(None).max()
        assert out["ds"].min() > last_naive

    def test_cross_validation_structure(self, air_passengers_csv: Path) -> None:
        from cadence.connectors.csv_connector import CSVConnector

        df = CSVConnector().load(air_passengers_csv).df
        fc = ClassicalForecaster(ClassicalModelConfig(models=("AutoETS",))).fit(df)
        cv = fc.cross_validation(h=6, n_windows=3)

        assert {"unique_id", "ds", "cutoff", "y", "AutoETS"} <= set(cv.columns)
        assert len(cv) == 18  # 3 windows x 6 steps, single series
        assert cv["cutoff"].nunique() == 3
        for _, grp in cv.groupby("cutoff"):
            assert len(grp) == 6

    def test_accepts_tz_aware_input(self) -> None:
        # canonical fixtures are tz-aware; statsforecast gets naive-UTC internally
        df = make_canonical_df(n=40, freq="D")
        fc = ClassicalForecaster(ClassicalModelConfig(models=("AutoETS",))).fit(df)
        out = fc.forecast(h=4)
        assert len(out) == 4
