"""Known-answer tests for the §7.2 pure stats functions, run on the fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cadence.agents.diagnostic_stats import (
    compute_intermittency,
    compute_missingness,
    compute_outliers,
    compute_seasonality,
    compute_stationarity,
    compute_trend,
    stl_strength,
)
from cadence.connectors.csv_connector import CSVConnector


@pytest.fixture(scope="module")
def stationary():
    return CSVConnector().load("data/sample/synthetic_stationary.csv").df


@pytest.fixture(scope="module")
def trending():
    return CSVConnector().load("data/sample/synthetic_trending.csv").df


@pytest.fixture(scope="module")
def seasonal():
    return CSVConnector().load("data/sample/synthetic_seasonal.csv").df


class TestTrend:
    def test_trending_fixture_detected_up(self, trending):
        t = compute_trend(trending["y"].to_numpy())
        assert t["present"] is True
        assert t["direction"] == "up"
        assert t["pvalue"] < 0.001

    def test_stationary_fixture_no_trend(self, stationary):
        t = compute_trend(stationary["y"].to_numpy())
        assert t["present"] is False

    def test_too_short_series_reports_absent(self):
        t = compute_trend(np.array([1.0, 2.0]))
        assert t["present"] is False and t["direction"] == "none"


class TestSeasonality:
    def test_seasonal_fixture_finds_period_7(self, seasonal):
        s = compute_seasonality(seasonal["y"].to_numpy(), period=7)
        assert s["present"] is True
        assert s["strength"] > 0.6
        assert s["acf_at_period"] > 0.5

    def test_stationary_fixture_no_seasonality(self, stationary):
        s = compute_seasonality(stationary["y"].to_numpy(), period=7)
        assert s["present"] is False
        assert s["strength"] < 0.3

    def test_too_short_for_period_returns_absent(self):
        s = compute_seasonality(np.arange(10.0), period=7)
        assert s["present"] is False

    def test_stl_strength_perfect_sine_is_near_one(self):
        t = np.arange(140.0)
        y = np.sin(2 * np.pi * t / 7.0)
        assert stl_strength(y, period=7) > 0.9


class TestStationarity:
    def test_stationary_fixture_verdict(self, stationary):
        st = compute_stationarity(stationary["y"].to_numpy())
        assert st["verdict"] == "stationary"
        assert st["adf_pvalue"] < 0.05 and st["kpss_pvalue"] >= 0.05

    def test_trending_fixture_verdict(self, trending):
        st = compute_stationarity(trending["y"].to_numpy())
        assert st["verdict"] == "non_stationary"
        assert st["adf_pvalue"] >= 0.05 and st["kpss_pvalue"] < 0.05

    def test_both_tests_always_reported(self, seasonal):
        st = compute_stationarity(seasonal["y"].to_numpy())
        assert 0.0 <= st["adf_pvalue"] <= 1.0
        assert 0.0 <= st["kpss_pvalue"] <= 1.0
        assert st["verdict"] in {"stationary", "non_stationary", "ambiguous"}

    def test_constant_series_is_ambiguous_not_crash(self):
        st = compute_stationarity(np.full(20, 5.0))
        assert st["verdict"] == "ambiguous"


class TestOutliers:
    def test_injected_spike_is_flagged(self):
        rng = np.random.default_rng(7)
        y = rng.normal(0, 1, 120)
        y[60] = 25.0  # one obvious spike
        count, mask = compute_outliers(y, period=1)
        assert count >= 1 and mask[60]

    def test_clean_series_flags_little(self):
        rng = np.random.default_rng(8)
        y = rng.normal(0, 1, 200)
        count, _ = compute_outliers(y, period=1)
        assert count < 10  # gaussian tails flag a few; half the series is a bug

    def test_short_series_returns_zero(self):
        count, mask = compute_outliers(np.array([1.0, 2.0, 3.0]), period=1)
        assert count == 0 and len(mask) == 3


class TestIntermittency:
    def test_mostly_zero_series_is_intermittent(self):
        y = np.zeros(100)
        y[:10] = 5.0
        r = compute_intermittency(y)
        assert r["intermittent"] is True and r["zero_fraction"] == pytest.approx(0.9)

    def test_continuous_demand_is_not(self):
        r = compute_intermittency(np.abs(np.random.default_rng(9).normal(5, 1, 100)))
        assert r["intermittent"] is False


class TestMissingness:
    def test_gaps_detected_against_grid(self):
        ds = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        ds = ds.delete([3, 7])  # two dropped days
        r = compute_missingness(pd.Series(ds), "D")
        assert r["n_gaps"] == 2
        assert r["pct_missing"] == pytest.approx(2 / 10)

    def test_complete_series_has_no_gaps(self, seasonal):
        r = compute_missingness(seasonal["ds"], "D")
        assert r["n_gaps"] == 0 and r["pct_missing"] == 0.0
