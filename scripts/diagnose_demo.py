"""Phase 2 end-to-end demo (spec §12): CSV → per-series diagnostics + cleaned frame.

Usage:
    uv run python scripts/diagnose_demo.py [path/to.csv]

Defaults to the seasonal fixture. The LLM stays disabled (deterministic fallback)
unless CADENCE_DEMO_LLM=1 is set AND an LLMClient is wired in code.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cadence.agents.diagnostic_agent import DiagnosticAgent
from cadence.config.default_config import CadenceConfig
from cadence.connectors.csv_connector import CSVConnector


async def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample/synthetic_seasonal.csv")
    frame = CSVConnector().load(path)
    print(f"loaded {path}  rows={frame.source_meta.row_count}\n")

    agent = DiagnosticAgent(CadenceConfig())  # LLM disabled → deterministic path
    result = await agent.diagnose(frame.df)

    for diag in result.diagnostics:
        print(f"=== {diag.unique_id} ===")
        print(f"  length={diag.length}  freq={diag.freq}  pct_missing={diag.pct_missing}")
        print(f"  trend:        {diag.trend}")
        print(f"  seasonality:  {diag.seasonality}")
        print(f"  stationarity: {diag.stationarity}")
        print(f"  outliers={diag.outlier_count}  intermittent={diag.intermittent}")
        pp = diag.recommended_preprocessing
        print(
            f"  preprocessing: missing={pp.missing_strategy} outliers={pp.outlier_strategy} "
            f"transform={pp.transform}"
        )
        print(f"  reason: {pp.reason}")

    if result.errors:
        print("\nerrors (accumulated, run continued):")
        for err in result.errors:
            print(f"  {err}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
