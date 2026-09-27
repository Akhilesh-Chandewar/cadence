"""Fixture integrity tests — the known-answer inputs Phases 1–2 will rely on."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cadence.connectors.csv_connector import CSVConnector

EXPECTED_FIXTURES = {
    "air_passengers.csv": ("air_passengers", 144),
    "synthetic_stationary.csv": ("synthetic_stationary", 200),
    "synthetic_trending.csv": ("synthetic_trending", 200),
    "synthetic_seasonal.csv": ("synthetic_seasonal", 196),
}


@pytest.mark.parametrize("filename", list(EXPECTED_FIXTURES))
def test_fixture_exists_and_validates(filename: str, sample_dir: Path) -> None:
    frame = CSVConnector().load(sample_dir / filename)
    uid, n_rows = EXPECTED_FIXTURES[filename]
    assert frame.source_meta.row_count == n_rows
    assert frame.source_meta.series_count == 1
    assert frame.df["unique_id"].iloc[0] == uid
    assert str(frame.df["ds"].dt.tz) == "UTC"


def test_air_passengers_known_shape(air_passengers_csv: Path) -> None:
    frame = CSVConnector().load(air_passengers_csv)
    df = frame.df
    assert df["ds"].min() == pd.Timestamp("1949-01-01", tz="UTC")
    assert df["ds"].max() == pd.Timestamp("1960-12-01", tz="UTC")
    # Classic series: strictly increasing yearly peaks (multiplicative seasonality + trend)
    yearly_max = df.groupby(df["ds"].dt.year)["y"].max()
    assert yearly_max.is_monotonic_increasing


def test_synthetic_series_known_answers(sample_dir: Path) -> None:
    """Smoke checks on the deterministic generators (deep stats land in Phase 2)."""
    stationary = CSVConnector().load(sample_dir / "synthetic_stationary.csv").df
    trending = CSVConnector().load(sample_dir / "synthetic_trending.csv").df
    seasonal = CSVConnector().load(sample_dir / "synthetic_seasonal.csv").df

    # stationary: mean ~10, no drift — first-half vs second-half means nearly equal
    half = len(stationary) // 2
    assert abs(stationary["y"][:half].mean() - stationary["y"][half:].mean()) < 1.0

    # trending: strong positive slope
    slope = trending["y"].diff().mean()
    assert slope == pytest.approx(0.15, abs=0.05)

    # seasonal: strong period-7 signal — lag-7 autocorrelation far exceeds lag-4
    s = seasonal["y"] - seasonal["y"].mean()
    acf7 = s.autocorr(lag=7)
    acf4 = s.autocorr(lag=4)
    assert acf7 > 0.7
    assert acf7 > acf4 + 0.4
