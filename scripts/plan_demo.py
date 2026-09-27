"""Phase 3 demo (spec §12): diagnostics → PlannerAgent shortlists per series.

Runs the full diagnose → plan path on a fixture with the LLM disabled
(deterministic rule table) and prints each series' shortlist + rule hits.

Usage:
    uv run python scripts/plan_demo.py [path/to.csv]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from cadence.agents.diagnostic_agent import DiagnosticAgent
from cadence.agents.planner_agent import PlannerAgent
from cadence.config.default_config import CadenceConfig
from cadence.connectors.csv_connector import CSVConnector


async def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample/synthetic_seasonal.csv")
    frame = CSVConnector().load(path)
    print(f"loaded {path}  rows={frame.source_meta.row_count}\n")

    cfg = CadenceConfig()  # LLM disabled → deterministic rule table only
    diag_result = await DiagnosticAgent(cfg).diagnose(frame.df)

    if diag_result.errors:
        print("diagnostic errors (run continued):", diag_result.errors, "\n")

    planner = PlannerAgent(cfg)
    for diag in diag_result.diagnostics:
        shortlist = await planner.plan(diag)
        print(f"=== {diag.unique_id} ({diag.length} pts, freq={diag.freq}) ===")
        print(f"  rule hits:   {shortlist.rule_hits}")
        print(f"  borderline:  {shortlist.borderline}  arbitration: {shortlist.arbitration}")
        for c in shortlist.candidates:
            avail = "ready" if c.is_available else f"phase {c.implemented_in_phase}"
            print(f"    [{c.tier:12s}] {c.name:20s} ({avail})  — {c.reason}")


if __name__ == "__main__":
    asyncio.run(main())
