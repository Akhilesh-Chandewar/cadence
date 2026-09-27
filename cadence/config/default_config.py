"""Run configuration defaults (spec §10, §7.7).

Everything tunable lives here; agents receive config, never import it directly.
LLM settings follow §7.7: provider-agnostic strings plus a hard off-switch.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Spec §7.7 — one injectable client, provider swap is config, disabled mode mandatory."""

    enabled: bool = False  # off by default: harness must work with no keys, no network
    provider: str = "openai"  # any LiteLLM provider prefix
    model: str = "gpt-4o-mini"  # resolved as f"{provider}/{model}" by LiteLLM
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)


class BacktestConfig(BaseModel):
    """Spec §9 — rolling-window CV defaults."""

    n_windows: int = Field(default=3, ge=3)  # minimum 3 windows per spec §9
    horizon: int = Field(default=12, ge=1)
    step_size: int = Field(default=1, ge=1)


class PlannerConfig(BaseModel):
    """Spec §7.3 — run config + rule-table thresholds."""

    horizon: int = Field(default=12, ge=1)
    scale: Literal["single", "many"] = "single"  # tens (interactive) vs thousands of series
    has_covariates: bool = False
    prefer_fast: bool = False  # "fast reasonable baseline now"
    short_history: int = Field(default=200, ge=1)  # §7.3 "<~200 points"
    long_history: int = Field(default=1000, ge=1)
    ambiguous_window: float = Field(default=0.2, gt=0, lt=1)  # ±20% around a threshold = borderline
    llm_arbitration: bool = True  # only effective when llm.enabled; planner-level off-switch


class CadenceConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    random_seed: int = 42


def default_config() -> CadenceConfig:
    return CadenceConfig()
