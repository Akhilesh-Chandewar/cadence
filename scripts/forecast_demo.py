"""Phase 1 end-to-end demo (spec §12): CSV in → backtest → forecast + intervals.

Usage:
    uv run python scripts/forecast_demo.py [path/to.csv] [horizon]

Defaults to the AirPassengers fixture with a 12-step horizon.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cadence.connectors.csv_connector import CSVConnector
from cadence.eval.backtest import rolling_backtest
from cadence.models.classical import ClassicalForecaster, ClassicalModelConfig


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample/air_passengers.csv")
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    frame = CSVConnector().load(path)
    df = frame.df
    print(
        f"loaded {path}  rows={frame.source_meta.row_count}  "
        f"series={frame.source_meta.series_count}"
    )

    print(f"\nrolling backtest (h={horizon}, 3 windows, spec §9)...")
    result = rolling_backtest(df, horizon=horizon, n_windows=3)
    print(result.aggregate.to_string(index=False))

    best = result.ranking("mase").iloc[0]["model"]
    print(f"\nbacktest winner by MASE: {best}")

    print(f"\nrefitting all candidates on full history, forecasting h={horizon}...")
    fc = ClassicalForecaster(ClassicalModelConfig(level=(95,))).fit(df)
    forecast = fc.forecast(h=horizon)
    cols = ["unique_id", "ds"] + [c for c in forecast.columns if c not in {"unique_id", "ds"}]
    print(forecast[cols].tail(horizon).to_string(index=False))


if __name__ == "__main__":
    main()
