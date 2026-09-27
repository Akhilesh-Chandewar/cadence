"""Rolling-window backtesting with MASE/sMAPE scoring (spec §9, Phase 1).

Wraps statsforecast's cross_validation (order-respecting rolling origin — never a
random split) and scores every window with the §9 primary pair. Per-window scores
are kept so downstream phases can judge "consistent winner across windows" rather
than a single aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from cadence.eval.metrics import mase, smape
from cadence.models.classical import (
    ClassicalForecaster,
    ClassicalModelConfig,
    infer_season_length,
)

# columns in the CV frame that are not model outputs
_NON_MODEL_COLS = {"unique_id", "ds", "cutoff", "y"}


@dataclass
class WindowScore:
    """One rolling window's score for one model on one series."""

    unique_id: str
    model: str
    window: int  # 1..n_windows; larger = later in time
    mase: float
    smape: float
    n_forecast: int


@dataclass
class BacktestResult:
    per_window: list[WindowScore]
    aggregate: pd.DataFrame  # unique_id, model, mase_mean/max, smape_mean/max, windows
    cv_frame: pd.DataFrame  # long CV frame for inspection/debugging
    seasonality: int = field(default=1)

    def ranking(self, metric: str = "mase") -> pd.DataFrame:
        """Models best-first by mean score, with worst-window (max) alongside.

        Ranking uses the mean; the max column exposes consistency (spec §7.4
        "clear, consistent winner across windows" — the ensemble trigger in Phase 6).
        """
        if metric not in {"mase", "smape"}:
            raise ValueError("metric must be 'mase' or 'smape'")
        return self.aggregate.sort_values(f"{metric}_mean").reset_index(drop=True)


def _model_columns(cv: pd.DataFrame) -> list[str]:
    """Point-forecast columns in the wide CV frame (interval cols end in -lo/-hi)."""
    return [
        c
        for c in cv.columns
        if c not in _NON_MODEL_COLS and not (c.endswith("-lo") or "-lo-" in c or "-hi-" in c)
    ]


def _score_frame(
    cv: pd.DataFrame,
    models: list[str],
    seasonality: int,
    train_history: pd.DataFrame,
) -> list[WindowScore]:
    """Score the wide CV frame per (series, model, window).

    MASE is scaled by the naive one-step MAE on the series' training history —
    the same data statsforecast fitted on — so scores are comparable across models
    and windows of the same series.
    """
    scores: list[WindowScore] = []
    for uid, sub in cv.groupby("unique_id", sort=True):
        hist = train_history.loc[train_history["unique_id"] == uid, "y"].to_numpy(float)
        for model in models:
            for w, (_, wgrp) in enumerate(sub.groupby("cutoff", sort=True), start=1):
                y_true = wgrp["y"].to_numpy(float)
                y_pred = wgrp[model].to_numpy(float)
                scores.append(
                    WindowScore(
                        unique_id=uid,
                        model=model,
                        window=w,
                        mase=mase(y_true, y_pred, hist, seasonality),
                        smape=smape(y_true, y_pred),
                        n_forecast=len(wgrp),
                    )
                )
    return scores


def rolling_backtest(
    df: pd.DataFrame,
    horizon: int,
    n_windows: int = 3,
    step_size: int | None = None,
    models: tuple[str, ...] | list[str] = ("AutoARIMA", "AutoETS"),
    level: tuple[int, ...] = (95,),
    seasonality: int | None = None,
) -> BacktestResult:
    """Run the §9 backtest: >=3 rolling windows, MASE + sMAPE per window per model.

    Args:
        df: canonical (unique_id, ds, y) frame.
        horizon: forecast steps per window.
        n_windows: rolling windows (spec §9 minimum 3 — enforced).
        step_size: window advance; defaults to horizon (contiguous windows).
        models: registry names to compare.
        level: prediction-interval levels to request in CV output.
        seasonality: MASE scale period; inferred from data when None.

    Returns:
        BacktestResult with per-window scores, a per-series aggregate table, and the
        raw CV frame.
    """
    if n_windows < 3:
        raise ValueError(f"spec §9: at least 3 rolling windows are required (got {n_windows})")

    fc = ClassicalForecaster(ClassicalModelConfig(models=tuple(models), level=level)).fit(df)
    seasonality = seasonality or infer_season_length(df)
    cv = fc.cross_validation(h=horizon, n_windows=n_windows, step_size=step_size)

    scores = _score_frame(cv, list(models), seasonality, fc._df)  # noqa: SLF001 — same package
    per_model = pd.DataFrame([s.__dict__ for s in scores])
    aggregate = per_model.groupby(["unique_id", "model"], as_index=False).agg(
        mase_mean=("mase", "mean"),
        mase_max=("mase", "max"),
        smape_mean=("smape", "mean"),
        smape_max=("smape", "max"),
        windows=("window", "nunique"),
    )
    return BacktestResult(
        per_window=scores,
        aggregate=aggregate,
        cv_frame=cv,
        seasonality=seasonality,
    )
