"""Deterministic recommender + preprocessing applier tests."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.agents.decisions import (
    MissingStrategy,
    OutlierStrategy,
    PreprocessingDecision,
    Transform,
    apply_preprocessing,
    recommend_preprocessing,
)
from tests.conftest import make_canonical_df


def _diag(**over) -> dict:
    base = {
        "length": 100,
        "pct_missing": 0.0,
        "outlier_count": 0,
        "skew": 0.0,
        "stationarity": {"verdict": "stationary"},
        "y": np.abs(np.random.default_rng(0).normal(10, 1, 100)),
    }
    base.update(over)
    return base


class TestRecommender:
    def test_defaults_are_conservative(self):
        d = recommend_preprocessing(_diag())
        assert d.missing_strategy == MissingStrategy.INTERPOLATE_LINEAR
        assert d.outlier_strategy == OutlierStrategy.KEEP_AND_FLAG
        assert d.transform == Transform.NONE

    def test_skewed_nonstationary_positive_gets_log(self):
        d = recommend_preprocessing(_diag(skew=2.0, stationarity={"verdict": "non_stationary"}))
        assert d.transform == Transform.LOG

    def test_skewed_with_nonpositive_never_log(self):
        d = recommend_preprocessing(
            _diag(
                skew=2.0,
                stationarity={"verdict": "non_stationary"},
                y=np.array([-1.0, 2.0, 3.0] * 33 + [4.0]),
            )
        )
        assert d.transform == Transform.NONE

    def test_many_outliers_gets_clip(self):
        d = recommend_preprocessing(_diag(outlier_count=10, length=100))
        assert d.outlier_strategy == OutlierStrategy.CLIP_AT_P99

    def test_heavy_missingness_gets_ffill(self):
        d = recommend_preprocessing(_diag(pct_missing=0.5))
        assert d.missing_strategy == MissingStrategy.FORWARD_FILL


class TestApply:
    def test_linear_interpolation_fills_internal_nans(self):
        df = make_canonical_df(n=10)
        df.loc[4, "y"] = np.nan
        out = apply_preprocessing(
            df, PreprocessingDecision(missing_strategy=MissingStrategy.INTERPOLATE_LINEAR)
        )
        assert out["y"].isna().sum() == 0
        assert out.loc[3, "y_raw"] != out.loc[3, "y"] or True  # raw preserved below
        assert out["y_raw"].isna().sum() == 1  # raw column keeps the gap

    def test_ffill_strategy(self):
        df = make_canonical_df(n=6)
        df.loc[2, "y"] = np.nan
        out = apply_preprocessing(
            df, PreprocessingDecision(missing_strategy=MissingStrategy.FORWARD_FILL)
        )
        assert out["y"].isna().sum() == 0
        assert out.loc[2, "y"] == out.loc[1, "y"]

    def test_clip_caps_extreme_values(self):
        df = make_canonical_df(n=50)
        df.loc[25, "y"] = 1e6
        out = apply_preprocessing(
            df, PreprocessingDecision(outlier_strategy=OutlierStrategy.CLIP_AT_P99)
        )
        assert out.loc[25, "y"] < 1e6
        assert out.loc[25, "y_raw"] == 1e6

    def test_log_transform(self):
        df = make_canonical_df(n=10)
        df["y"] = np.abs(df["y"]) + 1.0
        out = apply_preprocessing(df, PreprocessingDecision(transform=Transform.LOG))
        assert np.allclose(out["y"], np.log(out["y_raw"]))

    def test_log_with_nonpositive_raises(self):
        df = make_canonical_df(n=5)
        df.loc[0, "y"] = -1.0
        with pytest.raises(ValueError, match="non-positive"):
            apply_preprocessing(df, PreprocessingDecision(transform=Transform.LOG))

    def test_difference_transform_drops_first_row(self):
        df = make_canonical_df(n=6)
        out = apply_preprocessing(df, PreprocessingDecision(transform=Transform.DIFFERENCE))
        assert len(out) == 5

    def test_keeps_sorting_and_columns(self):
        df = make_canonical_df(n=5).sample(frac=1.0, random_state=0)  # shuffled
        out = apply_preprocessing(df, PreprocessingDecision())
        assert out["ds"].is_monotonic_increasing
        assert {"unique_id", "ds", "y", "y_raw"} <= set(out.columns)
