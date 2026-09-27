"""CSV/Parquet connector (spec §6): map source rows → canonical schema.

The only job of this module is column mapping + delegation to the shared validator.
No business logic, no LLM calls — plumbing stays deterministic (spec §7.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from cadence.connectors.base import SchemaValidationError, validate_canonical_df


class ColumnMapping(BaseModel):
    """Which source column plays which canonical role (spec §6: column-mapping config)."""

    unique_id: str = Field(default="unique_id")
    ds: str = Field(default="ds")
    y: str = Field(default="y")
    # source column -> canonical covariate column name (kept as-is by default)
    covariates: dict[str, str] = Field(default_factory=dict)

    @field_validator("unique_id", "ds", "y")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("column mapping entries must be non-blank")
        return v


class CSVConnector:
    """Reads a CSV or Parquet file and emits a validated canonical DataFrame."""

    def __init__(self, mapping: ColumnMapping | None = None, **read_kwargs: Any) -> None:
        self.mapping = mapping or ColumnMapping()
        self.read_kwargs = read_kwargs

    def read(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise SchemaValidationError(f"input file not found: {path}")

        if path.suffix.lower() == ".parquet":
            raw = pd.read_parquet(path, **self.read_kwargs)
        elif path.suffix.lower() in {".csv", ".txt"}:
            raw = pd.read_csv(path, **self.read_kwargs)
        else:
            raise SchemaValidationError(
                f"unsupported extension {path.suffix!r} — expected .csv or .parquet"
            )

        return self.canonicalize(raw)

    def canonicalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        m = self.mapping
        required = [m.unique_id, m.ds, m.y]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise SchemaValidationError(
                f"column mapping refers to missing source column(s) {missing}; "
                f"available columns: {list(raw.columns)}"
            )

        unknown_covs = [c for c in m.covariates if c not in raw.columns]
        if unknown_covs:
            raise SchemaValidationError(
                f"covariate mapping refers to missing source column(s) {unknown_covs}"
            )

        out = pd.DataFrame()
        out["unique_id"] = raw[m.unique_id]
        out["ds"] = raw[m.ds]
        out["y"] = raw[m.y]
        for src, dst in m.covariates.items():
            out[dst] = raw[src]

        return out

    def load(self, path: str | Path):
        """read + validate → CanonicalFrame (the connector's public contract, spec §6)."""
        return validate_canonical_df(self.read(path), source_type="csv")
