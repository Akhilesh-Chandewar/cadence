"""Phase 6 demo (spec §12): the full §7.4 chain — diagnose → plan → forecast.

Per series: backtests every implemented shortlist candidate, applies the
clear+consistent best-vs-ensemble rule, and prints the decision, weights,
score table (with §9 WQL where intervals exist) and the final forecast.
Uses auto_llm_config: the LLM arbitrates preprocessing/borderline calls
whenever a provider key is present, otherwise deterministic fallbacks.

Usage:
    uv run python scripts/forecast_agent_demo.py [path/to.csv] [horizon]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from cadence.agents.diagnostic_agent import DiagnosticAgent
from cadence.agents.forecast_agent import ForecastAgent
from cadence.agents.planner_agent import PlannerAgent
from cadence.config.default_config import CadenceConfig
from cadence.connectors.csv_connector import CSVConnector
from cadence.llm.litellm_client import auto_llm_config


async def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample/air_passengers.csv")
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    frame = CSVConnector().load(path)
    print(f"loaded {path}  rows={frame.source_meta.row_count}\n")

    cfg = CadenceConfig(
        llm=auto_llm_config(),
        forecast__horizon=horizon,  # type: ignore[call-arg]
    )
    mode = f"ON — {cfg.llm.provider}/{cfg.llm.model}" if cfg.llm.enabled else "off (deterministic)"
    print(f"llm: {mode}   horizon: {horizon}\n")

    # 1) diagnose (§7.2)
    diag_result = await DiagnosticAgent(cfg).diagnose(frame.df)
    if diag_result.errors:
        print("diagnostic errors:", diag_result.errors)

    # 2) plan (§7.3)
    planner = PlannerAgent(cfg)
    shortlists = [await planner.plan(d) for d in diag_result.diagnostics]
    for sl in shortlists:
        print(
            f"shortlist[{sl.unique_id}]: {[c.name for c in sl.candidates]}"
            f"  (borderline={sl.borderline}, {sl.arbitration})"
        )

    # 3) forecast (§7.4) — transforms passed so forecasts land in original space
    transforms = {
        d.unique_id: d.recommended_preprocessing.transform for d in diag_result.diagnostics
    }
    agent = ForecastAgent(cfg)
    result = agent.run(diag_result.cleaned_df, shortlists, transforms=transforms)

    for err in result.errors:
        print(f"error[{err['unique_id']}]: {err['error']}")
    for out in result.outputs:
        print(f"\n=== {out.unique_id} — decision: {out.decision} ===")
        if out.weights:
            for name, w in sorted(out.weights.items(), key=lambda kv: -kv[1]):
                print(f"  {name:20s} weight={w:.3f}")
        print("\n  backtest scores (§9, mean across windows):")
        cols = ["model", "mase_mean", "smape_mean", "wql"]
        print(out.scores[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        label = {"best": "winner", "ensemble": "weighted ensemble"}[out.decision]
        print(f"\n  final h={len(out.point)} forecast ({label}):")
        merged = out.point.merge(out.intervals, on=["unique_id", "ds"], how="left")
        print(merged.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    asyncio.run(main())
