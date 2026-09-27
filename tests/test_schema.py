"""Spec §5 canonical-schema validation contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cadence.connectors.base import (
    CANONICAL_COLUMNS,
    SourceMeta,
    validate_canonical_df,
)
from cadence.connectors.csv_connector import ColumnMapping
from tests.conftest import make_canonical_df


class TestRequiredColumns:
    def test_missing_column_raises_with_clear_error(self) -> None:
        df = make_canonical_df().drop(columns=["y"])
        with pytest.raises(Exception, match="missing required column.*y"):
            validate_canonical_df(df, source_type="test")

    def test_extra_columns_tolerated_as_covariates(self) -> None:
        df = make_canonical_df()
        df["promo_flag"] = 1
        frame = validate_canonical_df(df, source_type="test")
        assert frame.covariate_columns == ["promo_flag"]


class TestNulls:
    def test_null_unique_id_rejected(self) -> None:
        df = make_canonical_df()
        df.loc[0, "unique_id"] = None
        with pytest.raises(Exception, match="unique_id contains nulls"):
            validate_canonical_df(df, source_type="test")

    def test_nan_y_rejected(self) -> None:
        df = make_canonical_df()
        df.loc[2, "y"] = np.nan
        with pytest.raises(Exception, match="y non-numeric"):
            validate_canonical_df(df, source_type="test")

    def test_unparseable_ds_rejected(self) -> None:
        df = make_canonical_df()
        df["ds"] = df["ds"].astype(object)
        df.loc[1, "ds"] = "not-a-date"
        with pytest.raises(Exception, match="ds unparseable"):
            validate_canonical_df(df, source_type="test")


class TestDuplicates:
    def test_duplicate_pair_rejected_not_dropped(self) -> None:
        df = make_canonical_df()
        dup = df.iloc[[5]].copy()
        dup["y"] = 999.0  # same (unique_id, ds), different value — must reject
        df = pd.concat([df, dup], ignore_index=True)
        with pytest.raises(Exception, match="duplicate \\(unique_id, ds\\)"):
            validate_canonical_df(df, source_type="test")


class TestMonotonicity:
    def test_non_monotonic_ds_within_series_rejected(self) -> None:
        df = make_canonical_df()
        df.iloc[10], df.iloc[5] = df.iloc[5].copy(), df.iloc[10].copy()
        with pytest.raises(Exception, match="not monotonic"):
            validate_canonical_df(df, source_type="test")

    def test_monotonic_multi_series_passes(self, multi_series_df: pd.DataFrame) -> None:
        frame = validate_canonical_df(multi_series_df, source_type="test")
        assert frame.source_meta.series_count == 3


class TestNormalizationAndMeta:
    def test_ds_normalized_to_utc(self) -> None:
        df = make_canonical_df(tz=None)  # naive timestamps
        frame = validate_canonical_df(df, source_type="test")
        assert str(frame.df["ds"].dt.tz) == "UTC"

    def test_source_meta_populated(self) -> None:
        df = make_canonical_df(uids=["x", "y"], n=10)
        frame = validate_canonical_df(df, source_type="unit_test")
        assert isinstance(frame.source_meta, SourceMeta)
        assert frame.source_meta.source_type == "unit_test"
        assert frame.source_meta.row_count == 20
        assert frame.source_meta.series_count == 2

    def test_canonical_columns_constant(self) -> None:
        assert CANONICAL_COLUMNS == frozenset({"unique_id", "ds", "y"})


class TestColumnMapping:
    def test_mapping_validator_rejects_blank(self) -> None:
        with pytest.raises(Exception, match="non-blank"):
            ColumnMapping(y="  ")
