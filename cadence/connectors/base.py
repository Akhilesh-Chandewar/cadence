"""Canonical schema contract and validation (spec §5).

Every connector's only job is to emit rows in this shape; nothing downstream ever
looks at a source-specific format again.

Canonical long-format schema (Nixtla-ecosystem convention — StatsForecast /
MLForecast / NeuralForecast consume it with zero glue code):

    unique_id: str     # which series this row belongs to (SKU id, sensor id, ...)
    ds: datetime       # timestamp, tz-aware (normalized to UTC on the wire)
    y: float           # the target value

Any column NOT in {unique_id, ds, y} is treated as an exogenous/covariate column;
its known-future vs past-only distinction is tracked separately (see CovariateRole).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

CANONICAL_COLUMNS: frozenset[str] = frozenset({"unique_id", "ds", "y"})


class CadenceError(Exception):
    """Base class for all Cadence errors — clear, actionable, never silent."""


class SchemaValidationError(CadenceError):
    """Raised when a source cannot be mapped/validated into the canonical schema."""


class CovariateRole(StrEnum):
    """Spec §5: known-future vs past-only changes which models can use a covariate."""

    KNOWN_FUTURE = "known_future"  # e.g. holiday calendar, planned promotion
    PAST_ONLY = "past_only"  # e.g. another sensor reading


class SourceMeta(BaseModel):
    """Audit-trail metadata attached to every ingestion (spec §6)."""

    source_type: str
    ingested_at: datetime
    row_count: int
    series_count: int
    extra: dict[str, Any] = Field(default_factory=dict)


class CanonicalRow(BaseModel):
    """Pydantic validation model for one canonical-schema row (spec §5, Phase 0)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    unique_id: str = Field(min_length=1)
    ds: datetime
    y: float

    @field_validator("unique_id")
    @classmethod
    def _unique_id_not_null(cls, v: str) -> str:
        if not v or v.lower() in {"nan", "none", "null"}:
            raise ValueError("unique_id must be non-null")
        return v

    @field_validator("y")
    @classmethod
    def _y_numeric(cls, v: float) -> float:
        if v != v:  # NaN check
            raise ValueError("y must be numeric, got NaN")
        return v


class CanonicalFrame(BaseModel):
    """A validated canonical DataFrame plus its audit metadata.

    Wraps (rather than subclasses) a pandas DataFrame so the Pydantic contract stays
    explicit: you cannot construct one without passing validation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame
    source_meta: SourceMeta

    @property
    def covariate_columns(self) -> list[str]:
        return [c for c in self.df.columns if c not in CANONICAL_COLUMNS]


def validate_canonical_df(df: pd.DataFrame, *, source_type: str) -> CanonicalFrame:
    """Enforce the spec §5 rules on a canonical-shaped DataFrame.

    Rules (hard failures, per spec — reject with clear errors, never silently drop):
    - required columns {unique_id, ds, y} present
    - unique_id non-null
    - ds parseable as datetime (normalized to UTC)
    - y numeric
    - duplicate (unique_id, ds) pairs rejected
    - ds monotonic non-decreasing within each unique_id

    Row-level Pydantic validation (CanonicalRow) runs on a sample of rows; whole-frame
    vectorized checks below cover the rest.
    """
    missing = CANONICAL_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"canonical schema violation: missing required column(s) {sorted(missing)}; "
            f"got columns {list(df.columns)}"
        )

    if df.empty:
        raise SchemaValidationError("canonical schema violation: DataFrame is empty")

    out = df.copy()

    # unique_id: non-null, coerced to str
    if out["unique_id"].isna().any():
        raise SchemaValidationError("canonical schema violation: unique_id contains nulls")
    out["unique_id"] = out["unique_id"].astype(str)

    # ds: parseable, tz-normalized to UTC
    out["ds"] = pd.to_datetime(out["ds"], utc=True, errors="coerce")
    if out["ds"].isna().any():
        bad_idx = out.index[out["ds"].isna()][:5].tolist()
        raise SchemaValidationError(
            f"canonical schema violation: ds unparseable at row(s) {bad_idx} (first 5 shown)"
        )

    # y: strictly numeric
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    if out["y"].isna().any():
        bad_idx = out.index[out["y"].isna()][:5].tolist()
        raise SchemaValidationError(
            f"canonical schema violation: y non-numeric at row(s) {bad_idx} (first 5 shown)"
        )

    # duplicate (unique_id, ds) pairs — rejected, not dropped (spec §5)
    dup_mask = out.duplicated(subset=["unique_id", "ds"], keep=False)
    if dup_mask.any():
        example = out.loc[dup_mask, ["unique_id", "ds"]].head(5)
        raise SchemaValidationError(
            f"canonical schema violation: duplicate (unique_id, ds) pairs found "
            f"({int(dup_mask.sum())} rows); examples:\n{example}"
        )

    # monotonic ds within each unique_id
    for uid, grp in out.groupby("unique_id", sort=False):
        if not grp["ds"].is_monotonic_increasing:
            raise SchemaValidationError(
                f"canonical schema violation: ds not monotonic for unique_id={uid!r}"
            )

    # Row-level Pydantic validation on a deterministic sample (cheap belt-and-braces)
    sample_idx = out.index[:: max(1, len(out) // 100)][:100]
    for i in sample_idx:
        row = out.loc[i]
        CanonicalRow(unique_id=row["unique_id"], ds=row["ds"].to_pydatetime(), y=float(row["y"]))

    meta = SourceMeta(
        source_type=source_type,
        ingested_at=datetime.now().astimezone(),
        row_count=len(out),
        series_count=out["unique_id"].nunique(),
    )
    return CanonicalFrame(df=out, source_meta=meta)
