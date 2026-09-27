"""PlannerAgent tests: §7.3 rule rows, shortlist bounds, borderline + LLM paths."""

from __future__ import annotations

from typing import Any

from cadence.agents.decisions import PreprocessingDecision
from cadence.agents.diagnostic_agent import SeriesDiagnostics
from cadence.agents.planner_agent import (
    CandidateModel,
    PlannerAgent,
    ShortlistDecision,
    Tier,
    _is_borderline,
    rule_table_shortlist,
)
from cadence.config.default_config import CadenceConfig, LLMConfig, PlannerConfig


def make_diag(**over) -> SeriesDiagnostics:
    """A plain non-seasonal stationary series by default."""
    base = dict(
        unique_id="s",
        length=100,
        freq="D",
        pct_missing=0.0,
        trend={"present": False, "direction": "none", "slope": 0.0, "pvalue": 1.0},
        seasonality={"present": False, "period": 1, "strength": 0.1, "acf_at_period": 0.05},
        stationarity={"adf_pvalue": 0.5, "kpss_pvalue": 0.5, "verdict": "stationary"},
        outlier_count=0,
        intermittent=False,
        zero_fraction=0.0,
        recommended_preprocessing=PreprocessingDecision(),
    )
    base.update(over)
    return SeriesDiagnostics(**base)


class FakeLLMClient:
    def __init__(self, decision: Any = None) -> None:
        self.decision = decision
        self.calls = 0
        self.last_payload: dict | None = None

    async def complete(self, messages: list[dict], response_schema: type) -> Any:
        self.calls += 1
        self.last_payload = messages[-1]["content"]
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


class TestRuleTable:
    def test_intermittent_row_gives_croston_tsb(self):
        candidates, hits, _ = rule_table_shortlist(
            make_diag(intermittent=True, zero_fraction=0.9), CadenceConfig()
        )
        names = {c.name for c in candidates}
        assert {"Croston", "TSB"} <= names
        assert "intermittent" in hits

    def test_short_seasonal_row_gives_classical_trio(self):
        candidates, hits, _ = rule_table_shortlist(
            make_diag(
                length=150,
                seasonality={"present": True, "period": 7, "strength": 0.8, "acf_at_period": 0.7},
            ),
            CadenceConfig(),
        )
        names = {c.name for c in candidates}
        assert {"AutoARIMA", "AutoETS", "AutoTheta"} <= names
        assert "short_history_with_seasonality" in hits

    def test_many_series_with_covariates_gives_ml(self):
        cfg = CadenceConfig(planner=PlannerConfig(scale="many", has_covariates=True))
        candidates, hits, _ = rule_table_shortlist(make_diag(length=2000), cfg)
        assert "MLForecast-LightGBM" in {c.name for c in candidates}
        assert "many_series_with_covariates" in hits

    def test_long_history_gives_deep_learning(self):
        candidates, hits, _ = rule_table_shortlist(make_diag(length=1500), CadenceConfig())
        names = {c.name for c in candidates}
        assert {"N-HiTS", "TFT"} <= names
        assert "long_history" in hits

    def test_fast_baseline_row_gives_chronos(self):
        candidates, hits, _ = rule_table_shortlist(make_diag(length=15), CadenceConfig())
        assert "Chronos-Bolt" in {c.name for c in candidates}
        assert "fast_baseline_or_cold_start" in hits

    def test_prefer_fast_flag_gives_chronos(self):
        cfg = CadenceConfig(planner=PlannerConfig(prefer_fast=True))
        candidates, hits, _ = rule_table_shortlist(make_diag(), cfg)
        assert "Chronos-Bolt" in {c.name for c in candidates}
        assert "fast_baseline_or_cold_start" in hits

    def test_empty_shortlist_impossible(self):
        # no rule row matches (many-scale, no covariates, mid-length, no seasonality)
        candidates, hits, _ = rule_table_shortlist(make_diag(length=500), CadenceConfig())
        assert len(candidates) >= 2
        assert "default_classical" in hits

    def test_shortlist_within_2_to_5(self):
        # trigger as many rows as possible at once
        candidates, _, _ = rule_table_shortlist(
            make_diag(
                length=1500,
                intermittent=True,
                zero_fraction=0.8,
                seasonality={"present": True, "period": 7, "strength": 0.8, "acf_at_period": 0.7},
            ),
            CadenceConfig(planner=PlannerConfig(scale="many", has_covariates=True)),
        )
        assert 2 <= len(candidates) <= 5

    def test_two_tier_ambiguity_rule(self):
        # borderline length near 200 with only classical candidates → foundation added
        candidates, hits, borderline = rule_table_shortlist(
            make_diag(
                length=185,
                seasonality={"present": True, "period": 7, "strength": 0.8, "acf_at_period": 0.7},
            ),
            CadenceConfig(),
        )
        assert borderline is True
        assert "ambiguous_two_tiers" in hits
        assert {c.tier for c in candidates} >= {"classical", "foundation"}


class TestBorderlineDetection:
    def test_mid_length_series_not_borderline(self):
        assert _is_borderline(make_diag(length=500), CadenceConfig()) is False

    def test_length_near_200_is_borderline(self):
        assert _is_borderline(make_diag(length=185), CadenceConfig()) is True

    def test_seasonality_strength_near_threshold_is_borderline(self):
        diag = make_diag(
            seasonality={"present": False, "period": 7, "strength": 0.55, "acf_at_period": 0.1}
        )
        assert _is_borderline(diag, CadenceConfig()) is True


class TestPlannerAgent:
    async def test_deterministic_when_llm_disabled(self):
        shortlist = await PlannerAgent(CadenceConfig()).plan(make_diag(length=185))
        assert shortlist.arbitration == "deterministic"
        assert 2 <= len(shortlist.candidates) <= 5

    async def test_llm_not_called_for_clear_cases(self):
        llm = FakeLLMClient()
        shortlist = await PlannerAgent(CadenceConfig(llm=LLMConfig(enabled=True)), llm=llm).plan(
            make_diag(length=500)
        )
        assert llm.calls == 0
        assert shortlist.arbitration == "deterministic"

    async def test_llm_arbitrates_borderline_when_enabled(self):
        llm = FakeLLMClient(
            ShortlistDecision(
                candidates=[
                    CandidateModel(name="AutoARIMA", tier=Tier.CLASSICAL, reason="kept"),
                    CandidateModel(name="N-HiTS", tier=Tier.DEEP_LEARNING, reason="long enough"),
                ]
            )
        )
        shortlist = await PlannerAgent(CadenceConfig(llm=LLMConfig(enabled=True)), llm=llm).plan(
            make_diag(length=185)
        )

        assert llm.calls == 1
        assert shortlist.arbitration == "llm"
        assert [c.name for c in shortlist.candidates] == ["AutoARIMA", "N-HiTS"]

    async def test_llm_failure_falls_back(self):
        llm = FakeLLMClient(RuntimeError("quota"))
        shortlist = await PlannerAgent(
            CadenceConfig(llm=LLMConfig(enabled=True, max_retries=2)), llm=llm
        ).plan(make_diag(length=185))

        assert llm.calls == 2
        assert shortlist.arbitration == "deterministic"  # deterministic shortlist stands
        assert 2 <= len(shortlist.candidates) <= 5

    async def test_llm_cannot_invent_models(self):
        llm = FakeLLMClient(
            ShortlistDecision(
                candidates=[CandidateModel(name="MagicModel", tier=Tier.ML, reason="hallucination")]
            )
        )
        shortlist = await PlannerAgent(CadenceConfig(llm=LLMConfig(enabled=True)), llm=llm).plan(
            make_diag(length=185)
        )
        # hallucinated-only shortlist rejected → deterministic fallback
        assert shortlist.arbitration == "deterministic"
        assert all(c.name != "MagicModel" for c in shortlist.candidates)
