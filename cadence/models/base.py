"""Shared model-zoo contract (spec §7.4/§8): one interface across all tiers.

Heavy tiers (ml, dl, foundation) import their libraries lazily — the core install
must work without torch/JAX (spec §10 dependency-group layout). `require_group`
turns an absent library into an actionable error naming the uv command that fixes
it, instead of a bare ModuleNotFoundError deep in a stack trace.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


class MissingDependencyError(ImportError):
    """Raised when a tier's library group hasn't been synced."""


def require_group(module: str, group: str, package: str | None = None) -> None:
    """Import `module`, raising an actionable error if the dependency group is missing."""
    try:
        __import__(module)
    except ModuleNotFoundError as exc:
        name = package or module.split(".")[0]
        raise MissingDependencyError(
            f"{name} is required for this tier but is not installed. "
            f"Sync the optional dependency group with:  uv sync --group {group}"
        ) from exc


@runtime_checkable
class Forecaster(Protocol):
    """The zoo-wide contract (§12 Phase 4): every tier exposes the same three calls.

    - fit(df): canonical (unique_id, ds, y) frame in, fitted state stored
    - predict(h): h-step forecast from the fitted state (wide statsforecast-style frame)
    - cross_validation(h, n_windows, step_size): rolling-origin CV frame with
      per-window predictions for backtesting (§9)
    """

    def fit(self, df: pd.DataFrame) -> Forecaster: ...

    def predict(self, h: int) -> pd.DataFrame: ...

    def cross_validation(
        self, h: int, n_windows: int = 3, step_size: int | None = None
    ) -> pd.DataFrame: ...
