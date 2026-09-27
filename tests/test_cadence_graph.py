"""LangGraph wiring tests (§7.6): reducers, happy path, error accumulation."""

from __future__ import annotations

import operator

import pytest

from cadence.state import merge_dicts


class TestReducers:
    def test_merge_dicts_right_wins(self):
        assert merge_dicts({"a": 1}, {"a": 2, "b": 3}) == {"a": 2, "b": 3}

    def test_merge_dicts_none_safe(self):
        assert merge_dicts(None, {"a": 1}) == {"a": 1}
        assert merge_dicts({"a": 1}, None) == {"a": 1}

    def test_operator_add_accumulates(self):
        state = {"errors": [{"e": 1}]}
        state["errors"] = operator.add(state["errors"], [{"e": 2}])
        assert state["errors"] == [{"e": 1}, {"e": 2}]


class TestHappyPath:
    def test_full_graph_runs_end_to_end(self, tmp_path):
        import asyncio
        import shutil

        from cadence.config.default_config import CadenceConfig
        from cadence.graph.cadence_graph import build_cadence_graph

        shutil.copy("data/sample/synthetic_seasonal.csv", tmp_path / "s.csv")
        app = build_cadence_graph(CadenceConfig())  # LLM disabled: deterministic, fast
        state = asyncio.run(
            app.ainvoke({"source_config": {"path": str(tmp_path / "s.csv")}, "errors": []})
        )

        assert state["source_meta"]["source_type"] == "csv"
        assert set(state["diagnostics"]) == {"synthetic_seasonal"}
        shortlist = state["candidate_models"]["synthetic_seasonal"]
        assert len(shortlist["candidates"]) >= 2
        assert "synthetic_seasonal" in state["forecasts"]
        fc = state["forecasts"]["synthetic_seasonal"]
        assert fc["decision"] in {"best", "ensemble"}
        assert len(fc["point"]) > 0
        assert "synthetic_seasonal" in state["report"]
        # §7.6: clean run → no errors anywhere
        assert state["errors"] == []

    def test_report_contains_reasoning(self, tmp_path):
        import asyncio
        import shutil

        from cadence.config.default_config import CadenceConfig
        from cadence.graph.cadence_graph import build_cadence_graph

        shutil.copy("data/sample/air_passengers.csv", tmp_path / "a.csv")
        app = build_cadence_graph(CadenceConfig())
        state = asyncio.run(
            app.ainvoke({"source_config": {"path": str(tmp_path / "a.csv")}, "errors": []})
        )
        rep = state["report"]["air_passengers"]
        assert rep["decision"] in {"best", "ensemble"}
        assert rep["selected"]
        assert rep["preprocessing"]["missing"] is not None
        assert rep["forecast"] and len(rep["forecast"]) == 12


class TestErrorAccumulation:
    def test_ingest_failure_raises_after_retries(self, tmp_path):
        import asyncio

        from cadence.config.default_config import CadenceConfig
        from cadence.graph.cadence_graph import build_cadence_graph

        app = build_cadence_graph(CadenceConfig())
        from cadence.connectors.base import SchemaValidationError

        with pytest.raises((SchemaValidationError, Exception)) as exc_info:
            asyncio.run(
                app.ainvoke(
                    {"source_config": {"path": str(tmp_path / "missing.csv")}, "errors": []}
                )
            )
        assert "not found" in str(exc_info.value).lower()

    def test_no_cross_node_crash_on_bad_state(self):
        """Nodes tolerate malformed upstream payloads (§7.6: accumulate, don't crash)."""
        import asyncio

        from cadence.config.default_config import CadenceConfig
        from cadence.graph.cadence_graph import build_cadence_graph

        app = build_cadence_graph(CadenceConfig())
        # plan node receives a diagnostics entry that violates the §7.2 contract;
        # forecast node receives no candidate_models at all — both must accumulate
        state = asyncio.run(
            app.ainvoke(
                {
                    "source_config": {"path": "data/sample/synthetic_seasonal.csv"},
                    "errors": [],
                    # simulate: skip diagnose output by pre-seeding broken diagnostics
                    "diagnostics": {"broken": {"unique_id": "broken"}},
                    "cleaned_df": None,
                }
            )
        )
        stages = {e["stage"] for e in state["errors"]}
        assert "plan" in stages or "forecast" in stages
