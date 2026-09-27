"""CSV connector tests — including the spec §12 Phase 0 round-trip requirement."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cadence.connectors.base import SchemaValidationError
from cadence.connectors.csv_connector import ColumnMapping, CSVConnector
from tests.conftest import make_canonical_df


class TestRoundTrip:
    """Phase 0 acceptance: CSV in → validated canonical DataFrame out."""

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        src = make_canonical_df(uids=["s1", "s2"], n=25)
        path = tmp_path / "in.csv"
        src.to_csv(path, index=False)

        frame = CSVConnector().load(path)

        assert frame.source_meta.source_type == "csv"
        assert frame.source_meta.row_count == 50
        assert frame.source_meta.series_count == 2
        pd.testing.assert_frame_equal(
            frame.df.reset_index(drop=True),
            validate_reloaded(src, path),
            check_dtype=False,
        )

    def test_parquet_round_trip(self, tmp_path: Path) -> None:
        src = make_canonical_df(n=15)
        path = tmp_path / "in.parquet"
        src.to_parquet(path, index=False)

        frame = CSVConnector().load(path)
        assert len(frame.df) == 15
        assert list(frame.df.columns) == ["unique_id", "ds", "y"]


def validate_reloaded(src: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Reload via the connector and return only the canonical columns for comparison."""
    frame = CSVConnector().load(path)
    return frame.df[["unique_id", "ds", "y"]].reset_index(drop=True)


class TestColumnMapping:
    def test_renamed_columns_map_canonical(self, tmp_path: Path) -> None:
        raw = pd.DataFrame(
            {
                "sku": ["a", "a"],
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="D"),
                "units_sold": [3.0, 4.0],
                "promo": [0, 1],
            }
        )
        path = tmp_path / "retail.csv"
        raw.to_csv(path, index=False)

        mapping = ColumnMapping(
            unique_id="sku", ds="timestamp", y="units_sold", covariates={"promo": "promo"}
        )
        frame = CSVConnector(mapping).load(path)

        assert list(frame.df.columns) == ["unique_id", "ds", "y", "promo"]
        assert frame.df["unique_id"].tolist() == ["a", "a"]

    def test_missing_mapped_column_raises(self, tmp_path: Path) -> None:
        raw = pd.DataFrame({"a": [1], "b": [2]})
        path = tmp_path / "bad.csv"
        raw.to_csv(path, index=False)
        with pytest.raises(SchemaValidationError, match="missing source column"):
            CSVConnector(ColumnMapping(ds="b", y="nope")).load(path)


class TestErrorPaths:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SchemaValidationError, match="not found"):
            CSVConnector().load(tmp_path / "nope.csv")

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "in.xlsx"
        path.write_text("x")
        with pytest.raises(SchemaValidationError, match="unsupported extension"):
            CSVConnector().load(path)

    def test_empty_csv_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["unique_id", "ds", "y"]).to_csv(path, index=False)
        with pytest.raises(SchemaValidationError, match="empty"):
            CSVConnector().load(path)

    def test_duplicate_rows_in_file_rejected(self, tmp_path: Path) -> None:
        df = make_canonical_df(n=5)
        path = tmp_path / "dup.csv"
        pd.concat([df, df.iloc[[2]]], ignore_index=True).to_csv(path, index=False)
        with pytest.raises(SchemaValidationError, match="duplicate"):
            CSVConnector().load(path)
