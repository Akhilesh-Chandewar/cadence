"""Provider-agnostic LLM access — one client protocol for all agents (spec §7.7).

Phase 0 ships the protocol + the mandatory disabled no-op. The LiteLLM-backed
implementation arrives with the first agent that needs it (Phase 2/3); the point of
the protocol is that agents never import a provider SDK directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from cadence.config.default_config import LLMConfig


@runtime_checkable
class LLMClient(Protocol):
    """Spec §7.7: structured output only — every call returns a validated Pydantic model."""

    async def complete(
        self, messages: list[dict], response_schema: type[BaseModel]
    ) -> BaseModel: ...


class LLMDisabledError(RuntimeError):
    """Raised when complete() is called while the client is in disabled mode."""


class DisabledLLMClient:
    """Spec §7.7: with llm.enabled=false every agent must still work.

    Agents should branch on ``llm.enabled`` *before* deciding to call; if a call slips
    through anyway, this explicit error beats a silent hallucination.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig(enabled=False)
        if self.config.enabled:
            raise ValueError("DisabledLLMClient requires llm.enabled=false")

    @property
    def enabled(self) -> bool:
        return False

    async def complete(self, messages: list[dict], response_schema: type[BaseModel]) -> BaseModel:
        raise LLMDisabledError(
            "LLM is disabled (llm.enabled=false); use the deterministic fallback path per spec §7.7"
        )
