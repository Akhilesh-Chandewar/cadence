"""FastAPI layer (spec §12 Phase 8): POST /forecast, GET /diagnostics/{unique_id}.

Thin transport over the §7.6 graph — the API never reimplements pipeline logic,
it builds a source_config, runs the graph, and serves slices of the final
CadenceState. Formats: json (the machine report), markdown, html.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from cadence.agents.report_agent import ReportAgent
from cadence.config.default_config import CadenceConfig
from cadence.graph.cadence_graph import run_pipeline

app = FastAPI(
    title="Cadence",
    version="0.1.0",
    description="Time series analysis harness (spec §7.5/§8 API layer)",
)


class ForecastRequest(BaseModel):
    # v1 is local-first single-user (§3 non-goals); auth/tenancy comes with v2.
    source_config: dict = Field(
        description='{"path": "..."} | {"connection_string": ..., "query"/"table": ..., '
        '"column_mapping": {...}?} | {"url": ..., "records_path": ..., ...} — see §6',
    )
    horizon: int = Field(default=12, ge=1, le=720)
    format: Literal["json", "markdown", "html"] = "json"
    use_llm: bool = Field(
        default=False,
        description="enable §7.7 LLM arbitration (requires a provider key in .env)",
    )


def _run(source_config: dict, horizon: int, use_llm: bool = False) -> dict:
    cfg = CadenceConfig()
    cfg.forecast.horizon = horizon
    cfg.llm.enabled = use_llm  # §7.7: off by default; factory fails fast without a key
    try:
        return run_pipeline(source_config, cfg)
    except Exception as exc:  # ingest failures raise by design (§7.6)
        raise HTTPException(status_code=400, detail=f"pipeline failed: {exc}") from exc


@app.post("/forecast")
def forecast(req: ForecastRequest):
    state = _run(req.source_config, req.horizon, req.use_llm)
    report = state.get("report") or {}
    if req.format == "json":
        return {
            "source_meta": state.get("source_meta"),
            "report": report,
            "errors": state.get("errors", []),
        }
    md = ReportAgent().render_markdown(report, state.get("source_meta"), state.get("errors"))
    if req.format == "markdown":
        return {"markdown": md}
    return {"html": ReportAgent().render_html(md)}


@app.get("/diagnostics/{unique_id}")
def diagnostics(unique_id: str, path: str = Query(..., description="CSV/Parquet source path")):
    # diagnostics-only fast path: ingest + diagnose nodes are enough; the full
    # graph still runs (phases are cheap except forecasting, which we simply
    # don't serve here) — Phase 9+ can add an interrupt if runs get heavy.
    state = _run({"path": path}, horizon=1)
    diag = (state.get("diagnostics") or {}).get(unique_id)
    if diag is None:
        known = ", ".join(state.get("diagnostics") or {}) or "none"
        raise HTTPException(
            status_code=404, detail=f"unknown unique_id {unique_id!r}; known: {known}"
        )
    return {"unique_id": unique_id, "diagnostics": diag}
