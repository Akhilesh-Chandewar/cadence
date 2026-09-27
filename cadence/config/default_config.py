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
    provider: str = "groq"  # project default (user choice); any LiteLLM prefix works
    model: str = "openai/gpt-oss-120b"  # resolved as f"{provider}/{model}" by LiteLLM
    api_key_env: str | None = None  # env var holding the key; per-provider default when None
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)


class BacktestConfig(BaseModel):
    """Spec §9 — rolling-window CV defaults."""

    n_windows: int = Field(default=3, ge=3)  # minimum 3 windows per spec §9
    horizon: int = Field(default=12, ge=1)
    step_size: int = Field(default=1, ge=1)


class MLModelConfig(BaseModel):
    """Tier 2 (§8): MLForecast + LightGBM hyperparameters."""

    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 7, 14])
    # lag → rolling-mean window size (mlforecast wants transform *instances*;
    # the config carries just the sizes and the wrapper constructs them)
    rolling_mean_windows: dict[int, int] = Field(default_factory=lambda: {7: 3, 14: 3})
    date_features: list[str] | None = None  # inferred from freq when None
    # conformal-interval calibration horizon: predict(h) emits lo/hi columns for h <= this
    interval_horizon: int = Field(default=24, ge=1)
    num_leaves: int = Field(default=31, ge=2)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    n_estimators: int = Field(default=300, ge=1)


class DLModelConfig(BaseModel):
    """Tier 3 (§8): NeuralForecast deep-learning hyperparameters."""

    models: list[str] = Field(default_factory=lambda: ["NHITS"])  # NHITS | TFT
    input_size: int = Field(default=2 * 12, ge=2)  # 2x horizon is the common default
    max_steps: int = Field(default=100, ge=1)  # kept small for CPU-friendly tests
    scaler_type: str = "robust"


class ChronosConfig(BaseModel):
    """Tier 4 (§8): Chronos-Bolt zero-shot foundation model."""

    model_id: str = "amazon/chronos-bolt-small"  # bolt-small: fastest CPU default
    device: str = "cpu"
    torch_dtype: str = "float32"  # bfloat16 on GPU in production


class ForecastConfig(BaseModel):
    """Spec §7.4 — best-vs-ensemble decision + backtest shape."""

    horizon: int = Field(default=12, ge=1)
    n_windows: int = Field(default=3, ge=3)  # §9 minimum
    winner_margin: float = Field(default=0.05, gt=0)  # "clear": best beats #2 by ≥5%
    consistency_ratio: float = Field(default=0.7, gt=0, le=1)  # "consistent": window-win share
    ensemble_top_k: int = Field(default=3, ge=2)  # members in the inverse-error ensemble


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
    ml: MLModelConfig = Field(default_factory=MLModelConfig)
    dl: DLModelConfig = Field(default_factory=DLModelConfig)
    chronos: ChronosConfig = Field(default_factory=ChronosConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    random_seed: int = 42


def default_config() -> CadenceConfig:
    return CadenceConfig()
