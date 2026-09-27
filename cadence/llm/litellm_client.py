"""LiteLLM-backed §7.7 client — the provider-agnostic default implementation.

Structured output only (spec §7.7): the response schema is injected into the
prompt, the reply must contain JSON, and it is validated into the Pydantic model
before it ever reaches an agent. Malformed output raises → agents retry → then
fall back deterministically. Any litellm-supported provider works: OpenAI,
Anthropic, Gemini, Groq, OpenRouter, Ollama (local), ...

Key resolution order (per provider):
    1. config.api_key_env named env var (explicit override)
    2. the provider's conventional env var (PROVIDER_KEY_ENVS)
    3. process env as-is (litellm resolves keys itself, e.g. Ollama needs none)
A `.env` file at the project root is loaded first (keys are never read from
config files so nothing secret lands in git).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pydantic import BaseModel

from cadence.config.default_config import LLMConfig
from cadence.llm.client import DisabledLLMClient, LLMClient
from cadence.models.base import require_group

# provider prefix -> conventional key env var
PROVIDER_KEY_ENVS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
    "xai": "XAI_API_KEY",
}

# providers that need no key (local runtimes)
KEYLESS_PROVIDERS = {"ollama"}

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMKeyError(RuntimeError):
    """Raised when the LLM is enabled but no API key can be resolved."""


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing env vars win, no new deps."""
    env_path = Path(path) if path else Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_api_key(config: LLMConfig) -> str | None:
    """Resolve the API key for the configured provider, or None when absent."""
    load_dotenv()
    candidates = [config.api_key_env] if config.api_key_env else []
    default_env = PROVIDER_KEY_ENVS.get(config.provider)
    if default_env:
        candidates.append(default_env)
    for env_name in candidates:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _model_string(config: LLMConfig) -> str:
    """litellm model string: f"{provider}/{model}" — passthrough only when the model
    already starts with the provider prefix. Model IDs may themselves contain '/'
    (e.g. groq serving 'openai/gpt-oss-120b'), so a bare '/' check is wrong."""
    if config.model.startswith(f"{config.provider}/"):
        return config.model
    return f"{config.provider}/{config.model}"


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (handles fences and chatter)."""
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"no JSON object found in LLM reply: {text[:200]!r}")
    return json.loads(match.group(0))


def _response_field(reply: object, name: str) -> object:
    """Read a field off a litellm response, tolerating mapping- or attribute-style
    access (the real ModelResponse supports both; some wrappers only one)."""
    if isinstance(reply, dict):
        return reply[name]
    try:
        return reply[name]  # type: ignore[index]
    except (TypeError, KeyError, IndexError):
        return getattr(reply, name)


class LiteLLMClient:
    """Default LLMClient implementation (spec §7.7)."""

    def __init__(self, config: LLMConfig) -> None:
        if not config.enabled:
            raise ValueError(
                "LiteLLMClient requires llm.enabled=true; use DisabledLLMClient otherwise"
            )
        self.config = config
        self.api_key = resolve_api_key(config)

    async def complete(self, messages: list[dict], response_schema: type[BaseModel]) -> BaseModel:
        require_group("litellm", "llm")
        import litellm

        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        user_content = (
            f"{messages[-1]['content']}\n\n"
            f"Respond with a single JSON object that validates against this JSON schema "
            f"(no other text):\n{schema_json}"
        )
        reply = await litellm.acompletion(
            model=_model_string(self.config),
            messages=[
                *messages[:-1],
                {"role": "user", "content": user_content},
            ],
            api_key=self.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_seconds,
        )
        choices = _response_field(reply, "choices")
        message = _response_field(choices[0], "message")
        content = _response_field(message, "content")
        return response_schema.model_validate(_extract_json(content))


def auto_llm_config() -> LLMConfig:
    """Project-default LLM config: enabled iff a key resolves for the default provider.

    Used by demos/entry points so the LLM participates at every stage whenever a
    key is present, while tests and keyless environments stay deterministic.
    """
    enabled_cfg = LLMConfig(enabled=True)
    if resolve_api_key(enabled_cfg) is None and enabled_cfg.provider not in KEYLESS_PROVIDERS:
        return LLMConfig(enabled=False)
    return enabled_cfg


def make_llm_client(config: LLMConfig) -> LLMClient:
    """§7.7 factory: disabled config → DisabledLLMClient; enabled config → LiteLLMClient.

    Raises LLMKeyError at construction time when enabled but no key is resolvable —
    a config error should fail fast, not surface as a mid-run agent fallback.
    """
    if not config.enabled:
        return DisabledLLMClient(config)
    client = LiteLLMClient(config)
    if client.api_key is None and config.provider not in KEYLESS_PROVIDERS:
        env_hint = config.api_key_env or PROVIDER_KEY_ENVS.get(
            config.provider, "<PROVIDER>_API_KEY"
        )
        raise LLMKeyError(
            f"llm.enabled=true but no API key found for provider {config.provider!r}. "
            f"Set {env_hint} (in the environment or a project-root .env file), or set "
            "llm.enabled=false to run the deterministic path."
        )
    return client
