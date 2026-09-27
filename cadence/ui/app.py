"""Cadence live pipeline UI (Phase 10) — Streamlit dashboard.

Every §7.6 graph stage (ingest → diagnose → plan → forecast → report) renders
as its own checkpoint block as soon as it completes, driven by the graph's
stream_pipeline event generator (LangGraph updates mode). Shows: ingest meta +
data preview, per-series diagnostics (trend/seasonality/stationarity/preprocessing
with the §7.7 LLM-vs-deterministic reason), the §7.3 shortlist, the §7.4 decision
with ensemble weights + backtest score table + forecast chart, and the rendered
report — plus a raw checkpoint inspector and the accumulated error list.

Run:  uv sync --group ui && uv run streamlit run cadence/ui/app.py
"""

from __future__ import annotations

import asyncio
import queue
import threading

import pandas as pd
import streamlit as st

from cadence.config.default_config import CadenceConfig
from cadence.graph.cadence_graph import stream_pipeline

st.set_page_config(page_title="Cadence — live pipeline", page_icon="📈", layout="wide")

STAGES = ["ingest", "diagnose", "plan", "forecast", "report"]


# ----------------------------------------------------------------- helpers
def _drain_events(source_config: dict, cfg: CadenceConfig, q: queue.Queue[dict]) -> None:
    """Run stream_pipeline on a worker loop, forwarding events to the queue."""

    async def _consume() -> None:
        try:
            async for event in stream_pipeline(source_config, cfg):
                q.put(event)
        except Exception as exc:  # noqa: BLE001 — surface to the UI
            q.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            q.put(None)

    asyncio.run(_consume())


def _mark_active(placeholder, node: str) -> None:
    placeholder.info(f"⏳ running **{node}**…")


def _diag_badge(label: str, payload: dict, key: str) -> None:
    present = payload.get("present") or payload.get("verdict") in {"non_stationary"}
    tone = "🟢" if present else "⚪"
    detail = payload.get("strength") or payload.get("verdict") or payload.get("direction") or ""
    st.metric(label, f"{tone} {detail}" if detail else tone, border=True)


def _render_ingest(checkpoint: dict) -> None:
    meta = checkpoint.get("source_meta") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("rows", meta.get("row_count", "—"))
    c2.metric("series", meta.get("series_count", "—"))
    c3.metric("source", meta.get("source_type", "—"))
    c4.metric("ingested", str(meta.get("ingested_at", "—"))[:19])
    # checkpoints are json-safe (no DataFrames); read the file for a preview
    path = st.session_state.get("_source_path")
    if path:
        try:
            preview = pd.read_csv(path)
            st.caption("source preview")
            st.dataframe(preview.head(8), use_container_width=True, hide_index=True)
        except Exception:  # noqa: BLE001 — preview is best-effort
            pass


def _render_diagnose(checkpoint: dict) -> None:
    diagnostics = checkpoint.get("diagnostics") or {}
    for uid, diag in diagnostics.items():
        with st.container(border=True):
            st.markdown(
                f"**`{uid}`** — {diag.get('length')} points, freq `{diag.get('freq')}`, "
                f"missing {diag.get('pct_missing', 0):.1%}"
            )
            t, s, stn = st.columns(4)
            with t:
                _diag_badge("trend", diag.get("trend", {}), "trend")
            with s:
                seas = diag.get("seasonality", {})
                present = seas.get("present")
                st.metric(
                    "seasonality",
                    f"{'🟢' if present else '⚪'} p={seas.get('period')} · "
                    f"{seas.get('strength', 0):.2f}",
                    border=True,
                )
            with stn:
                stn_payload = diag.get("stationarity", {})
                st.metric(
                    "stationarity",
                    stn_payload.get("verdict", "—"),
                    f"ADF {stn_payload.get('adf_pvalue', 0):.2f} · "
                    f"KPSS {stn_payload.get('kpss_pvalue', 0):.2f}",
                    border=True,
                )
            with st.container():
                pp = diag.get("recommended_preprocessing", {})
                tone = "🤖 LLM" if "LLM" in (pp.get("reason") or "") else "⚙️ deterministic"
                st.markdown(
                    f"**preprocessing** ({tone}): missing=`{pp.get('missing_strategy')}` · "
                    f"outliers=`{pp.get('outlier_strategy')}` · transform=`{pp.get('transform')}`"
                )
                if pp.get("reason"):
                    st.caption(pp["reason"])


def _render_plan(checkpoint: dict) -> None:
    for uid, sl in (checkpoint.get("candidate_models") or {}).items():
        with st.container(border=True):
            tone = "🤖 LLM-arbitrated" if sl.get("arbitration") == "llm" else "⚙️ rule table"
            st.markdown(
                f"**`{uid}`** — shortlist ({tone})"
                + (" · *borderline*" if sl.get("borderline") else "")
            )
            for c in sl.get("candidates", []):
                ready = (
                    "✅"
                    if c.get("is_available", True)
                    else f"⏳ phase {c.get('implemented_in_phase')}"
                )
                st.markdown(f"- `{c['name']}` *{c['tier']}* {ready} — {c['reason']}")
            if sl.get("rule_hits"):
                st.caption("rule hits: " + ", ".join(f"`{h}`" for h in sl["rule_hits"]))


def _render_forecast(checkpoint: dict) -> None:
    for uid, fc in (checkpoint.get("forecasts") or {}).items():
        with st.container(border=True):
            decision = fc.get("decision")
            st.markdown(f"**`{uid}`** — decision: **{decision}**")
            weights = fc.get("weights") or {}
            if weights:
                cols = st.columns(len(weights))
                for col, (name, w) in zip(
                    cols, sorted(weights.items(), key=lambda kv: -kv[1]), strict=True
                ):
                    col.metric(name, f"{w:.3f}", border=True)
            scores = pd.DataFrame(fc.get("scores") or [])
            if not scores.empty:
                st.markdown("**backtest scores (§9 rolling windows)**")
                st.dataframe(
                    scores[["model", "mase_mean", "mase_max", "smape_mean", "wql"]].sort_values(
                        "mase_mean"
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            point = pd.DataFrame(fc.get("point") or [])
            intervals = pd.DataFrame(fc.get("intervals") or [])
            if not point.empty:
                chart = point[["ds", "yhat"]].copy()
                if not intervals.empty:
                    chart = chart.merge(
                        intervals[["ds", "yhat-lo-95", "yhat-hi-95"]], on="ds", how="left"
                    )
                chart["ds"] = pd.to_datetime(chart["ds"])
                st.line_chart(chart.set_index("ds"))


def _render_report(checkpoint: dict) -> None:
    rendered = checkpoint.get("rendered") or {}
    if rendered.get("markdown"):
        st.markdown(rendered["markdown"])


_RENDERERS = {
    "ingest": _render_ingest,
    "diagnose": _render_diagnose,
    "plan": _render_plan,
    "forecast": _render_forecast,
    "report": _render_report,
}


# ------------------------------------------------------------------- app
st.title("📈 Cadence — live pipeline")
st.caption(
    "ingest → diagnose → plan → forecast → report — each checkpoint renders "
    "as its graph node completes (§7.6)"
)

with st.sidebar:
    st.header("Run settings")
    default_path = "data/sample/air_passengers.csv"
    path = st.text_input("CSV / Parquet path", value=default_path)
    horizon = st.slider("Horizon", 1, 48, 12)
    use_llm = st.toggle(
        "§7.7 LLM arbitration",
        value=False,
        help="Needs a provider key in .env (Groq default). Off = deterministic fallbacks.",
    )
    run_clicked = st.button("▶ Run pipeline", type="primary", use_container_width=True)
    st.divider()
    st.caption(
        "Specify a different source by editing `source_config` in the code — "
        "SQL/REST connectors use the same graph."
    )

# keep the latest checkpoints in session state so rerenders keep the view
state: dict = st.session_state.setdefault("_checkpoints", {})
state.setdefault("errors", [])

if run_clicked:
    st.session_state["_checkpoints"] = {}
    st.session_state["_source_path"] = path
    state = st.session_state["_checkpoints"]
    state["errors"] = []

    cfg = CadenceConfig()
    cfg.forecast.horizon = horizon
    cfg.llm.enabled = use_llm

    q: queue.Queue[dict | None] = queue.Queue()
    worker = threading.Thread(target=_drain_events, args=({"path": path}, cfg, q), daemon=True)
    worker.start()

    progress = st.progress(0.0)
    status = st.empty()
    events_seen = 0
    while True:
        event = q.get()
        if event is None:
            break
        etype = event.get("type")
        if etype == "start":
            continue
        if etype == "error":
            st.error(f"pipeline failed: {event['error']}")
            break
        if etype == "stage":
            node = event["node"]
            checkpoint = event["checkpoint"]
            events_seen += 1
            progress.progress(min(1.0, events_seen / len(STAGES)))
            _mark_active(status, node)
            _RENDERERS[node](checkpoint)
            state[node] = checkpoint
    progress.progress(1.0)
    status.success("✅ pipeline complete")

# persisted views (rerender from session state on later script reruns)
if not run_clicked and state.get("errors"):
    with st.expander("⚠️ errors accumulated (§7.6)", expanded=True):
        for e in state["errors"]:
            st.error(f"`[{e.get('stage')}]` {e.get('unique_id') or '-'}: {e.get('error')}")

if not run_clicked and state.get("rendered"):
    _render_report(state)

with st.expander("🔍 raw checkpoints (machine view)"):
    st.json({k: v for k, v in state.items() if k != "errors"}, expanded=False)
