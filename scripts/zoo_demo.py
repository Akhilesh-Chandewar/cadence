"""Phase 4 demo (spec §12): one fixture, every available tier, one scorer.

Runs the §9 rolling backtest for the classical tier and the equivalent CV for
tiers 2–3, scoring everything with the same MASE/sMAPE machinery, so the output
is a like-for-like tier comparison. Tiers whose dependency groups aren't synced
are skipped with the uv command that adds them.

Usage:
    uv run python scripts/zoo_demo.py [path/to.csv] [horizon]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from cadence.connectors.csv_connector import CSVConnector
from cadence.eval.backtest import _score_frame, rolling_backtest
from cadence.models.base import MissingDependencyError


def score_cv(cv: pd.DataFrame, model: str, seasonality: int, history: pd.DataFrame) -> pd.DataFrame:
    scores = _score_frame(cv, [model], seasonality, history)
    rows = [
        {
            "model": s.model,
            "mase_mean": s.mase,
            "smape_mean": s.smape,
            "windows": 1,
        }
        for s in scores
    ]
    return pd.DataFrame(rows).groupby("model", as_index=False).mean(numeric_only=True)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample/air_passengers.csv")
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    frame = CSVConnector().load(path)
    df = frame.df
    print(f"loaded {path}  rows={frame.source_meta.row_count}  h={horizon}  windows=3\n")

    tables: list[pd.DataFrame] = []

    # Tier 1 — classical (always available)
    print("tier 1: classical (statsforecast)...")
    result = rolling_backtest(df, horizon=horizon, n_windows=3)
    tables.append(result.aggregate[["model", "mase_mean", "smape_mean"]].assign(tier="classical"))

    # Tier 2 — ML
    try:
        from cadence.models.ml import MLForecaster

        print("tier 2: MLForecast-LightGBM...")
        fc = MLForecaster().fit(df)
        cv = fc.cross_validation(h=horizon, n_windows=3)
        tables.append(score_cv(cv, "MLForecast-LightGBM", result.seasonality, df).assign(tier="ml"))
    except MissingDependencyError as exc:
        print(f"tier 2 skipped: {exc}")

    # Tier 3 — deep learning (tiny budget: demo, not a leaderboard run)
    try:
        from cadence.config.default_config import DLModelConfig
        from cadence.models.deep_learning import DLForecaster

        print("tier 3: N-HiTS (max_steps=20, CPU)...")
        fc = DLForecaster(config=DLModelConfig(max_steps=20), horizon=horizon).fit(df)
        cv = fc.cross_validation(h=horizon, n_windows=3)
        tables.append(score_cv(cv, "NHITS", result.seasonality, df).assign(tier="deep_learning"))
    except MissingDependencyError as exc:
        print(f"tier 3 skipped: {exc}")

    # Tier 4 — zero-shot foundation model
    try:
        from cadence.models.foundation import ChronosForecaster

        print("tier 4: Chronos-Bolt (zero-shot)...")
        fc = ChronosForecaster().fit(df)
        cv = fc.cross_validation(h=horizon, n_windows=3)
        tables.append(
            score_cv(cv, "Chronos-Bolt", result.seasonality, df).assign(tier="foundation")
        )
    except MissingDependencyError as exc:
        print(f"tier 4 skipped: {exc}")

    combined = pd.concat(tables, ignore_index=True)[
        ["tier", "model", "mase_mean", "smape_mean"]
    ].sort_values("mase_mean")
    print("\n=== §9 rolling backtest, MASE best-first ===")
    print(combined.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
