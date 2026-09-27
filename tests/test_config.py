"""Config defaults + §7.7 LLM client contract tests."""

from __future__ import annotations

import pytest

from cadence.config.default_config import BacktestConfig, LLMConfig, default_config
from cadence.llm.client import DisabledLLMClient, LLMClient, LLMDisabledError


class TestDefaults:
    def test_llm_disabled_by_default(self) -> None:
        cfg = default_config()
        assert cfg.llm.enabled is False  # §7.7: must run with no keys, no network

    def test_backtest_meets_spec_minimums(self) -> None:
        bt = BacktestConfig()
        assert bt.n_windows >= 3  # spec §9: at least 3 rolling windows

    def test_provider_swap_is_config_only(self) -> None:
        cfg = LLMConfig(provider="anthropic", model="claude-sonnet-4-5")
        assert cfg.provider != "openai"  # swap is a string change, per §7.7


class TestDisabledLLMClient:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(DisabledLLMClient(), LLMClient)

    def test_rejects_enabled_config(self) -> None:
        with pytest.raises(ValueError, match="enabled=false"):
            DisabledLLMClient(LLMConfig(enabled=True))

    async def test_complete_raises_with_actionable_error(self) -> None:
        client = DisabledLLMClient()
        with pytest.raises(LLMDisabledError, match="deterministic fallback"):
            await client.complete([{"role": "user", "content": "hi"}], response_schema=dict)
