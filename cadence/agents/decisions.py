"""Typed §7.2 preprocessing decisions + deterministic fallback recommender.

The LLM (§7.7) returns one of these validated models; when the LLM is disabled
the deterministic recommender below supplies the same contract so the harness
runs with no keys and no network.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import pandas as pd  # only for type hints


class MissingStrategy(StrEnum):
    INTERPOLATE_LINEAR = "interpolate_linear"
    FORWARD_FILL = "forward_fill"
    MODEL_BASED = "model_based"  # Phase 3+; deterministic fallback never picks this


class OutlierStrategy(StrEnum):
    KEEP_AND_FLAG = "keep_and_flag"
    CLIP_AT_P99 = "clip_at_p99"
    REMOVE = "remove"


class Transform(StrEnum):
    NONE = "none"
    LOG = "log"
    DIFFERENCE = "difference"


class PreprocessingDecision(BaseModel):
    """The `recommended_preprocessing` block of the diagnostics.json contract (§7.2)."""

    missing_strategy: MissingStrategy = MissingStrategy.INTERPOLATE_LINEAR
    outlier_strategy: OutlierStrategy = OutlierStrategy.KEEP_AND_FLAG
    transform: Transform = Transform.NONE
    reason: str = Field(default="deterministic defaults (LLM disabled)")


class SeriesDiagnostics(BaseModel):
    """diagnostics.json shape for one series (§7.2 example)."""

    unique_id: str
    length: int
    freq: str
    pct_missing: float
    trend: dict
    seasonality: dict
    stationarity: dict
    outlier_count: int
    intermittent: bool
    recommended_preprocessing: PreprocessingDecision


def recommend_preprocessing(diag: dict, reason: str | None = None) -> PreprocessingDecision:
    """Deterministic fallback rules — the LLM-disabled path of §7.7.

    Same inputs the LLM would see, same output contract. Conservative by design:
    interpolate small gaps, flag outliers rather than touch them, transform only
    when both skew and non-stationarity suggest it would help classical models.
    """
    # log only helps right-skewed, strictly positive, non-stationary series
    skew = float(diag.get("skew", 0.0))
    verdict = diag.get("stationarity", {}).get("verdict")
    transform = (
        Transform.LOG
        if (skew > 1.0 and (diag["y"] > 0).all() and verdict == "non_stationary")
        else Transform.NONE
    )
    # clip only when outliers are common enough to distort scale-sensitive fits
    outlier_share = diag["outlier_count"] / max(1, diag["length"])
    outlier_strategy = (
        OutlierStrategy.CLIP_AT_P99 if outlier_share > 0.05 else OutlierStrategy.KEEP_AND_FLAG
    )
    # forward-fill for mostly-empty series, interpolate for small gaps
    missing_strategy = (
        MissingStrategy.FORWARD_FILL
        if diag.get("pct_missing", 0.0) > 0.2
        else MissingStrategy.INTERPOLATE_LINEAR
    )
    return PreprocessingDecision(
        missing_strategy=missing_strategy,
        outlier_strategy=outlier_strategy,
        transform=transform,
        reason=reason or "deterministic fallback (LLM disabled)",
    )


def apply_preprocessing(df: pd.DataFrame, decision: PreprocessingDecision) -> pd.DataFrame:
    """Apply the chosen strategies to one series' canonical frame (pure function).

    Interpolation fills only *internal* gaps; the raw series is preserved in a
    `y_raw` column so the ReportAgent can show what changed.
    """
    out = df.sort_values("ds").copy()
    out["y_raw"] = out["y"].astype(float)

    # missing values (NaN targets can only appear via user-supplied data paths)
    if out["y"].isna().any():
        if decision.missing_strategy == MissingStrategy.INTERPOLATE_LINEAR:
            out["y"] = out["y"].interpolate(method="linear", limit_direction="both")
        elif decision.missing_strategy == MissingStrategy.FORWARD_FILL:
            out["y"] = out["y"].ffill().bfill()

    # outliers: clip flagged values at the 1st/99th percentile of the raw series
    if decision.outlier_strategy == OutlierStrategy.CLIP_AT_P99 and out["y_raw"].notna().any():
        p01, p99 = out["y_raw"].quantile([0.01, 0.99])
        outlier_mask = (out["y_raw"] < p01) | (out["y_raw"] > p99)
        if int(outlier_mask.sum()) > 0:
            out.loc[outlier_mask, "y"] = out["y_raw"].clip(p01, p99)[outlier_mask]

    # transforms
    if decision.transform == Transform.LOG:
        if (out["y"] <= 0).any():
            raise ValueError("log transform requested but series has non-positive values")
        out["y"] = np.log(out["y"])
    elif decision.transform == Transform.DIFFERENCE:
        out["y"] = out["y"].diff()
        out = out.dropna(subset=["y"])

    return out
