"""CadenceState — the shared mutable state passed node to node (spec §7.6).

Per-series channels (diagnostics, candidate_models, backtest_scores, forecasts)
use a dict-merge reducer so partial updates from one node never clobber another
series' entries; `errors` accumulates append-only (§7.6: accumulate, don't crash).
DataFrames pass through by reference — the graph runs in-process without a
checkpointer, so serialization is not required.

`source_config` is the graph's input channel (path + column mapping); it is an
addition to the spec's field list because IngestAgent needs its raw source config
(§7.1 input) to have somewhere to live.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer: shallow-merge per-series dicts (right wins on key clashes)."""
    return {**(left or {}), **(right or {})}


class CadenceState(TypedDict, total=False):
    # graph input (§7.1): {"path": ..., "column_mapping": {...} | None}
    source_config: dict

    # spec §7.6 fields
    source_meta: dict
    raw_df: Any  # pd.DataFrame in canonical schema
    diagnostics: Annotated[dict[str, dict], merge_dicts]  # per unique_id (json-safe)
    cleaned_df: Any  # pd.DataFrame
    candidate_models: Annotated[dict[str, dict], merge_dicts]  # per unique_id shortlists
    backtest_scores: Annotated[dict[str, list], merge_dicts]  # per unique_id score rows
    forecasts: Annotated[dict[str, dict], merge_dicts]  # per unique_id forecast payloads
    report: dict
    rendered: dict  # {"markdown": str, "html": str} — ReportAgent output (Phase 8)
    errors: Annotated[list[dict], operator.add]  # append-only accumulation
