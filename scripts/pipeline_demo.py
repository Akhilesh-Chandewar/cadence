"""Phase 7 demo (spec §12): the full graph — ingest → diagnose → plan → forecast → report.

One `run_pipeline` call drives the whole §7.6 state machine; the LLM participates
wherever a provider key is present (auto_llm_config), otherwise deterministic
fallbacks. Prints the shared state's journey and the final per-series report.

Usage:
    uv run python scripts/pipeline_demo.py [path/to.csv] [horizon]
"""

from __future__ import annotations

import sys
from pathlib import Path

from cadence.config.default_config import CadenceConfig
from cadence.graph.cadence_graph import run_pipeline


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample/air_passengers.csv")
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    cfg = CadenceConfig()
    cfg.forecast.horizon = horizon
    cfg.llm.enabled = True  # auto_llm_config already ran inside build_cadence_graph

    print(f"pipeline: {path}  horizon={horizon}\n")
    final_state = run_pipeline({"path": str(path)}, cfg)

    meta = final_state.get("source_meta", {})
    print(
        f"ingest:   {meta.get('row_count')} rows, {meta.get('series_count')} series "
        f"(source={meta.get('source_type')})"
    )
    for uid, diag in (final_state.get("diagnostics") or {}).items():
        s = diag.get("seasonality", {})
        print(
            f"diagnose: {uid}: len={diag.get('length')} freq={diag.get('freq')} "
            f"seasonal={s.get('present')} (period {s.get('period')})"
        )
    for uid, sl in (final_state.get("candidate_models") or {}).items():
        print(f"plan:     {uid}: {[c['name'] for c in sl['candidates']]}")
    for uid, fc in (final_state.get("forecasts") or {}).items():
        print(f"forecast: {uid}: decision={fc['decision']} selected={fc['selected']}")
        weights = fc.get("weights") or {}
        if weights:
            for name, w in sorted(weights.items(), key=lambda kv: -kv[1]):
                print(f"            {name:20s} {w:.3f}")

    print("\n=== report (plain-language summary per series) ===")
    for uid, rep in (final_state.get("report") or {}).items():
        pp = rep["preprocessing"]
        first, last = rep["forecast"][0], rep["forecast"][-1]
        print(
            f"\n{uid}: {rep['length']} points ({rep['freq']}), "
            f"preprocessing: missing={pp['missing']}, outliers={pp['outliers']}, "
            f"transform={pp['transform']}"
        )
        print(f"  decision: {rep['decision']} — {rep['selected']}")
        best = min(rep["scores"], key=lambda r: r["mase_mean"])
        print(f"  best backtest MASE: {best['mase_mean']:.3f} ({best['model']})")
        print(
            f"  forecast starts {str(first['ds'])[:10]} (yhat={first['yhat']:.2f}), "
            f"ends {str(last['ds'])[:10]} (yhat={last['yhat']:.2f})"
        )

    if final_state.get("errors"):
        print("\nerrors accumulated (§7.6, run continued):")
        for e in final_state["errors"]:
            uid = e.get("unique_id") or "-"
            print(f"  [{e['stage']}] {uid}: {e['error']}")


if __name__ == "__main__":
    main()
