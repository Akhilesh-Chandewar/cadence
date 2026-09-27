"""Phase 10 tests: the stream_pipeline event feed that drives the UI."""

from __future__ import annotations

import asyncio

import pandas as pd


def _collect(source_config: dict) -> list[dict]:
    from cadence.config.default_config import CadenceConfig
    from cadence.graph.cadence_graph import stream_pipeline

    async def run() -> list[dict]:
        return [event async for event in stream_pipeline(source_config, CadenceConfig())]

    return asyncio.run(run())


class TestStreamPipeline:
    def test_event_sequence_covers_all_stages_in_order(self):
        events = _collect({"path": "data/sample/synthetic_seasonal.csv"})

        assert events[0]["type"] == "start"
        assert events[-1]["type"] == "done"
        stages = [e["node"] for e in events if e["type"] == "stage"]
        assert stages == ["ingest", "diagnose", "plan", "forecast", "report"]

    def test_checkpoints_are_json_safe(self):
        """UI contract: no DataFrames (or other non-serializable) in checkpoints."""
        events = _collect({"path": "data/sample/synthetic_seasonal.csv"})
        for event in events:
            if event["type"] != "stage":
                continue
            for key, value in event["checkpoint"].items():
                assert not isinstance(value, pd.DataFrame), f"{key} leaked a DataFrame"

    def test_stage_checkpoints_carry_expected_payloads(self):
        events = _collect({"path": "data/sample/synthetic_seasonal.csv"})
        by_node = {e["node"]: e["checkpoint"] for e in events if e["type"] == "stage"}

        assert "source_meta" in by_node["ingest"]
        assert "synthetic_seasonal" in by_node["diagnose"]["diagnostics"]
        shortlist = by_node["plan"]["candidate_models"]["synthetic_seasonal"]
        assert len(shortlist["candidates"]) >= 2
        assert "synthetic_seasonal" in by_node["forecast"]["forecasts"]
        assert by_node["report"]["rendered"]["markdown"].startswith("# Cadence")

    def test_ingest_error_streams_not_raises(self, tmp_path):
        events = _collect({"path": str(tmp_path / "missing.csv")})
        assert events[-1]["type"] == "error"
        assert "not found" in events[-1]["error"]


class TestUIImports:
    def test_app_module_imports_cleanly(self):
        """Smoke: the Streamlit script is importable (bare mode) with no side effects."""
        import importlib

        module = importlib.import_module("cadence.ui.app")
        assert module.STAGES == ["ingest", "diagnose", "plan", "forecast", "report"]


class TestRenderers:
    """Execute every renderer against real checkpoints (Streamlit bare mode).

    Regression guard: the import smoke test alone did NOT catch a
    st.columns(4)-into-3-variables unpack crash inside _render_diagnose.
    """

    def test_renderers_execute_on_real_checkpoints(self, monkeypatch):
        import cadence.ui.app as ui

        events = _collect({"path": "data/sample/synthetic_seasonal.csv"})
        by_node = {e["node"]: e["checkpoint"] for e in events if e["type"] == "stage"}

        monkeypatch.setattr(ui.st, "session_state", {"_source_path": None}, raising=False)

        ui._render_ingest(by_node["ingest"])
        ui._render_diagnose(by_node["diagnose"])
        ui._render_plan(by_node["plan"])
        ui._render_forecast(by_node["forecast"])
        ui._render_report(by_node["report"])


class TestSSEEndpoint:
    def test_pipeline_stream_emits_events(self, tmp_path):
        import shutil

        from fastapi.testclient import TestClient

        from cadence.api.main import app

        shutil.copy("data/sample/synthetic_seasonal.csv", tmp_path / "s.csv")
        client = TestClient(app)
        with client.stream(
            "GET",
            "/pipeline/stream",
            params={"path": str(tmp_path / "s.csv"), "horizon": 6},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = "".join(resp.iter_text())

        types = [
            line.split(": ", 1)[1].split('"')[3]
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        assert types[0] == "start" and types[-1] == "done"
        assert types.count("stage") == 5
