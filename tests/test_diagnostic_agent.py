"""DiagnosticAgent end-to-end tests: disabled-LLM path, fake-LLM path, error accumulation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from cadence.agents.decisions import PreprocessingDecision, Transform
from cadence.agents.diagnostic_agent import DiagnosticAgent
from cadence.config.default_config import CadenceConfig, LLMConfig
from cadence.connectors.csv_connector import CSVConnector
from tests.conftest import make_canonical_df


class FakeLLMClient:
    """§7.7 contract double: returns a canned validated decision, or raises."""

    def __init__(self, decision: PreprocessingDecision | Exception) -> None:
        self.decision = decision
        self.calls = 0

    async def complete(self, messages: list[dict], response_schema: type) -> Any:
        self.calls += 1
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


@pytest.fixture(scope="module")
def seasonal_df():
    return CSVConnector().load("data/sample/synthetic_seasonal.csv").df


class TestDisabledLLMPath:
    async def test_single_series_end_to_end(self, seasonal_df):
        agent = DiagnosticAgent(CadenceConfig())  # llm.enabled=False by default
        result = await agent.diagnose(seasonal_df)

        assert result.errors == []
        diag = result.by_id("synthetic_seasonal")
        assert diag.length == 196
        assert diag.freq == "D"
        assert diag.seasonality["period"] == 7
        assert diag.seasonality["present"] is True
        assert diag.trend["direction"] == "up"
        assert diag.intermittent is False
        # cleaned frame preserves all rows and adds y_raw
        assert len(result.cleaned_df) == len(seasonal_df)
        assert "y_raw" in result.cleaned_df.columns

    async def test_multi_series_all_diagnosed(self):
        df = pd.concat(
            [make_canonical_df(uids=["a"], n=40), make_canonical_df(uids=["b"], n=40)],
            ignore_index=True,
        )
        result = await DiagnosticAgent(CadenceConfig()).diagnose(df)
        assert {d.unique_id for d in result.diagnostics} == {"a", "b"}
        assert len(result.cleaned_df) == 80

    async def test_no_llm_calls_when_disabled(self, seasonal_df):
        llm = FakeLLMClient(PreprocessingDecision())
        await DiagnosticAgent(CadenceConfig(), llm=llm).diagnose(seasonal_df)
        assert llm.calls == 0


class TestLLMArbitration:
    async def test_llm_decision_is_used_when_enabled(self, seasonal_df):
        llm = FakeLLMClient(
            PreprocessingDecision(transform=Transform.NONE, reason="LLM says keep as-is")
        )
        cfg = CadenceConfig(llm=LLMConfig(enabled=True))
        result = await DiagnosticAgent(cfg, llm=llm).diagnose(seasonal_df)

        assert llm.calls == 1
        diag = result.by_id("synthetic_seasonal")
        assert diag.recommended_preprocessing.reason == "LLM says keep as-is"

    async def test_llm_failure_falls_back_deterministically(self, seasonal_df):
        llm = FakeLLMClient(RuntimeError("rate limited"))
        cfg = CadenceConfig(llm=LLMConfig(enabled=True, max_retries=2))
        result = await DiagnosticAgent(cfg, llm=llm).diagnose(seasonal_df)

        assert llm.calls == 2  # retried, then gave up
        assert result.errors == []  # fallback is not an error
        diag = result.by_id("synthetic_seasonal")
        assert "deterministic fallback" in diag.recommended_preprocessing.reason

    async def test_nonpositive_series_never_gets_log(self):
        y = [(-x) for x in range(1, 41)]  # strictly negative, trending
        df = make_canonical_df(n=40)
        df["y"] = y
        llm = FakeLLMClient(PreprocessingDecision(transform=Transform.LOG))
        cfg = CadenceConfig(llm=LLMConfig(enabled=True))
        result = await DiagnosticAgent(cfg, llm=llm).diagnose(df)

        diag = result.by_id("series_1")
        assert diag.recommended_preprocessing.transform == Transform.NONE
        assert "log dropped" in diag.recommended_preprocessing.reason


class TestErrorAccumulation:
    async def test_bad_series_logged_not_fatal(self):
        good = make_canonical_df(uids=["good"], n=40)
        bad = make_canonical_df(uids=["bad"], n=40)
        bad["y"] = "not-a-number"  # breaks numeric diagnostics for this series only
        df = pd.concat([good, bad], ignore_index=True)

        result = await DiagnosticAgent(CadenceConfig()).diagnose(df)

        # good series still processed
        assert [d.unique_id for d in result.diagnostics] == ["good"]
        assert len(result.cleaned_df) == 40
        # bad series logged with context
        assert len(result.errors) == 1
        assert result.errors[0]["unique_id"] == "bad"
        assert result.errors[0]["stage"] == "diagnostic"
