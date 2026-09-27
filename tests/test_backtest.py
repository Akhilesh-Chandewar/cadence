"""Rolling-backtest tests (spec §9 behavior)."""

from __future__ import annotations

import pandas as pd
import pytest

from cadence.connectors.csv_connector import CSVConnector
from cadence.eval.backtest import rolling_backtest
from tests.conftest import make_canonical_df


@pytest.fixture(scope="module")
def air_df():
    return CSVConnector().load("data/sample/air_passengers.csv").df


class TestSpecMinimums:
    def test_fewer_than_three_windows_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 3 rolling windows"):
            rolling_backtest(make_canonical_df(n=50), horizon=4, n_windows=2)


class TestBacktestBehavior:
    def test_monthly_series_seasonality_and_structure(self, air_df) -> None:
        result = rolling_backtest(air_df, horizon=12, n_windows=3)

        assert result.seasonality == 12  # MS freq → yearly seasonality
        assert len(result.aggregate) == 2  # AutoARIMA + AutoETS
        assert (result.aggregate["windows"] == 3).all()
        assert len(result.per_window) == 6  # 2 models x 3 windows
        assert (result.aggregate["mase_mean"] > 0).all()
        assert result.aggregate["smape_mean"].between(0, 2).all()
        # every window scored on the full horizon
        assert all(s.n_forecast == 12 for s in result.per_window)

    def test_ranking_sorted_and_consistency_exposed(self, air_df) -> None:
        result = rolling_backtest(air_df, horizon=12, n_windows=3)
        ranking = result.ranking(metric="mase")

        assert list(ranking["mase_mean"]) == sorted(ranking["mase_mean"])
        assert {"model", "mase_mean", "mase_max", "windows"} <= set(ranking.columns)

    def test_multi_series_scored_independently(self) -> None:
        df = pd.concat(
            [
                make_canonical_df(uids=["a"], n=60, freq="D"),
                make_canonical_df(uids=["b"], n=60, freq="D"),
            ],
            ignore_index=True,
        )
        result = rolling_backtest(df, horizon=7, n_windows=3, models=("AutoETS",))

        assert set(result.aggregate["unique_id"]) == {"a", "b"}
        assert len(result.aggregate) == 2
        assert result.seasonality == 7

    def test_cv_frame_has_cutoffs(self, air_df) -> None:
        result = rolling_backtest(air_df, horizon=12, n_windows=3, models=("AutoETS",))
        assert "cutoff" in result.cv_frame.columns
        assert result.cv_frame["cutoff"].nunique() == 3
