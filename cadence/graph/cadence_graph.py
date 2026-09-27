"""LangGraph wiring (spec §7.6): the five Cadence agents as one state machine.

Ingest → Diagnose → Plan → Forecast → Report, passing a single CadenceState.
Ingest is the only node that RAISES on failure (nothing downstream is possible
without a canonical frame) — it carries a graph-level RetryPolicy for transient
I/O errors. Diagnose/Plan/Forecast nodes CATCH and ACCUMULATE per-series errors
(§7.6: one bad series never halts the run); LLM transients are retried inside the
§7.7 client before the deterministic fallback engages.

Nodes write only json-safe payloads into per-series channels (DataFrames flow by
reference through raw_df/cleaned_df; report stays a dict) so the state stays
inspectable and checkpointable.
"""

from __future__ import annotations

import asyncio

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from cadence.agents.diagnostic_agent import DiagnosticAgent
from cadence.agents.diagnostic_agent import SeriesDiagnostics as DiagnosticsModel
from cadence.agents.forecast_agent import ForecastAgent
from cadence.agents.planner_agent import ModelShortlist, PlannerAgent
from cadence.config.default_config import CadenceConfig
from cadence.connectors.api_connector import RESTConnector
from cadence.connectors.base import CadenceError, SchemaValidationError
from cadence.connectors.csv_connector import ColumnMapping, CSVConnector
from cadence.connectors.sql_connector import SQLConnector
from cadence.llm.litellm_client import auto_llm_config
from cadence.state import CadenceState

# transient-error predicate for the ingest node's RetryPolicy
_TRANSIENT = ("timeout", "temporarily", "connection reset", "connection refused", "econn")


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, CadenceError) and any(t in str(exc).lower() for t in _TRANSIENT)


def _err(stage: str, uid: str | None, exc: Exception) -> dict:
    return {
        "stage": stage,
        "unique_id": uid,
        "error": f"{type(exc).__name__}: {exc}",
    }


def build_cadence_graph(config: CadenceConfig | None = None) -> object:
    """Compile the five-agent graph. Returns the compiled LangGraph app."""
    cfg = config or CadenceConfig(llm=auto_llm_config())
    graph = StateGraph(
        CadenceState
    )  # ------------------------------------------------------------- 1. ingest

    async def ingest_node(state: CadenceState) -> dict:
        source = state.get("source_config") or {}
        mapping = source.get("column_mapping")
        col_map = ColumnMapping(**mapping) if mapping else None

        if "path" in source:  # CSV/Parquet (§6 row 1)
            frame = CSVConnector(col_map).load(source["path"])
        elif "connection_string" in source:  # SQL (§6 row 2, Phase 9)
            connector = SQLConnector(source["connection_string"], col_map)
            frame = connector.load(query=source.get("query"), table=source.get("table"))
        elif "url" in source:  # REST API (§6 row 3, Phase 9)
            connector = RESTConnector(
                url=source["url"],
                mapping=col_map,
                records_path=source.get("records_path"),
                page_param=source.get("page_param"),
                size_param=source.get("size_param"),
                page_size=source.get("page_size", 500),
                next_path=source.get("next_path"),
                headers=source.get("headers"),
            )
            frame = connector.load()
        else:
            raise SchemaValidationError(
                "source_config needs one of: 'path' (csv/parquet), "
                "'connection_string' (sql), 'url' (rest api)"
            )
        return {
            "raw_df": frame.df,
            "source_meta": frame.source_meta.model_dump(mode="json"),
        }

    # ------------------------------------------------------------ 2. diagnose
    async def diagnose_node(state: CadenceState) -> dict:
        agent = DiagnosticAgent(cfg)
        diag = await agent.diagnose(state["raw_df"])
        return {
            "diagnostics": {d.unique_id: d.model_dump(mode="json") for d in diag.diagnostics},
            "cleaned_df": diag.cleaned_df,
            "errors": diag.errors,
        }

    # ---------------------------------------------------------------- 3. plan
    async def plan_node(state: CadenceState) -> dict:
        planner = PlannerAgent(cfg)
        shortlists: dict[str, dict] = {}
        errors: list[dict] = []
        diagnostics = state.get("diagnostics") or {}
        for uid, payload in diagnostics.items():
            try:
                diag_model = DiagnosticsModel(**payload)
                sl = await planner.plan(diag_model)
                shortlists[uid] = sl.model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001 — §7.6 accumulate
                errors.append(_err("plan", uid, exc))
        return {
            "candidate_models": shortlists,
            "errors": errors,
        }

    # ------------------------------------------------------------ 4. forecast
    async def forecast_node(state: CadenceState) -> dict:
        agent = ForecastAgent(cfg)
        shortlists = [
            ModelShortlist(**payload) for payload in (state.get("candidate_models") or {}).values()
        ]
        transforms = {
            uid: (payload.get("recommended_preprocessing") or {}).get("transform", "none")
            for uid, payload in (state.get("diagnostics") or {}).items()
        }
        from cadence.agents.decisions import Transform

        transform_enum = {
            uid: Transform(t) for uid, t in transforms.items() if t in Transform._value2member_map_
        }
        cleaned = state.get("cleaned_df")
        if cleaned is None or cleaned.empty:
            return {"errors": [_err("forecast", None, RuntimeError("no cleaned data"))]}
        result = agent.run(cleaned, shortlists, transforms=transform_enum)

        def _json_safe(rows: list[dict]) -> list[dict]:
            """NaN is not JSON-compliant (pandas turns None wql into NaN) — null it."""
            return [
                {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
                for row in rows
            ]

        payloads: dict[str, dict] = {}
        for out in result.outputs:
            payloads[out.unique_id] = {
                "decision": out.decision,
                "selected": out.selected,
                "weights": out.weights,
                "point": _json_safe(out.point.to_dict("records")),
                "intervals": _json_safe(out.intervals.to_dict("records"))
                if out.intervals is not None
                else None,
                "scores": _json_safe(out.scores.to_dict("records")),
            }
        errors = [{**e, "stage": "forecast"} for e in result.errors]
        return {
            "forecasts": payloads,
            "backtest_scores": {uid: p["scores"] for uid, p in payloads.items()},
            "errors": errors,
        }

    # ------------------------------------------------------------- 5. report
    async def report_node(state: CadenceState) -> dict:
        from cadence.agents.report_agent import ReportAgent

        forecasts = state.get("forecasts") or {}
        diagnostics = state.get("diagnostics") or {}
        report: dict[str, dict] = {}
        for uid, fc in forecasts.items():
            diag = diagnostics.get(uid, {})
            pp = diag.get("recommended_preprocessing") or {}
            report[uid] = {
                "series": uid,
                "length": diag.get("length"),
                "freq": diag.get("freq"),
                "preprocessing": {
                    "missing": pp.get("missing_strategy"),
                    "outliers": pp.get("outlier_strategy"),
                    "transform": pp.get("transform"),
                    "reason": pp.get("reason"),
                },
                "decision": fc["decision"],
                "selected": fc["selected"],
                "weights": fc["weights"],
                "scores": fc["scores"],
                "forecast": fc["point"],
                "intervals": fc["intervals"],
            }
        agent = ReportAgent()
        markdown = agent.render_markdown(report, state.get("source_meta"), state.get("errors"))
        return {
            "report": report,
            "rendered": {"markdown": markdown, "html": agent.render_html(markdown)},
        }

    retry = RetryPolicy(max_attempts=3, retry_on=_is_transient)
    graph.add_node("ingest", ingest_node, retry_policy=retry)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("plan", plan_node)
    graph.add_node("forecast", forecast_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "diagnose")
    graph.add_edge("diagnose", "plan")
    graph.add_edge("plan", "forecast")
    graph.add_edge("forecast", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_pipeline(source_config: dict, config: CadenceConfig | None = None) -> dict:
    """Sync convenience entry: build + run the graph, return final CadenceState."""
    app = build_cadence_graph(config)
    return asyncio.run(app.ainvoke({"source_config": source_config, "errors": []}))
