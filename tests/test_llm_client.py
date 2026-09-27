"""§7.7 LiteLLM client tests: JSON extraction, key resolution, factory paths.

No network: litellm.acompletion is monkeypatched with a fake async client so the
full agent→client→provider path runs end-to-end in CI.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from cadence.config.default_config import LLMConfig
from cadence.llm.client import DisabledLLMClient, LLMDisabledError
from cadence.llm.litellm_client import (
    LiteLLMClient,
    LLMKeyError,
    _extract_json,
    _model_string,
    make_llm_client,
    resolve_api_key,
)


class Box(BaseModel):
    value: int


class TestJSONExtraction:
    def test_plain_json(self):
        assert _extract_json('{"value": 3}') == {"value": 3}

    def test_json_in_code_fence(self):
        text = 'Here you go:\n```json\n{"value": 7}\n```\nDone.'
        assert _extract_json(text) == {"value": 7}

    def test_json_with_chatter(self):
        text = 'Sure! {"value": 42} — hope that helps.'
        assert _extract_json(text) == {"value": 42}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="no JSON object"):
            _extract_json("I cannot help with that.")


class TestModelString:
    def test_prefix_added(self):
        assert (
            _model_string(LLMConfig(provider="openai", model="gpt-4o-mini")) == "openai/gpt-4o-mini"
        )

    def test_provider_prefix_passthrough(self):
        cfg = LLMConfig(provider="groq", model="groq/openai/gpt-oss-120b")
        assert _model_string(cfg) == "groq/openai/gpt-oss-120b"

    def test_nested_provider_prefix_still_gets_provider(self):
        # groq model IDs contain '/' themselves — the provider prefix must survive
        cfg = LLMConfig(provider="groq", model="openai/gpt-oss-120b")
        assert _model_string(cfg) == "groq/openai/gpt-oss-120b"


class TestKeyResolution:
    def test_conventional_env_used(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert resolve_api_key(LLMConfig(provider="anthropic")) == "sk-test"

    def test_explicit_api_key_env_wins(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "custom")
        monkeypatch.setenv("OPENAI_API_KEY", "conventional")
        cfg = LLMConfig(provider="openai", api_key_env="MY_KEY")
        assert resolve_api_key(cfg) == "custom"

    def test_missing_key_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)  # no .env file here
        assert resolve_api_key(LLMConfig(provider="openai")) is None

    def test_dotenv_loaded(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        (tmp_path / ".env").write_text('GROQ_API_KEY="gsk-dotenv"\n')
        monkeypatch.chdir(tmp_path)
        assert resolve_api_key(LLMConfig(provider="groq")) == "gsk-dotenv"

    def test_real_env_wins_over_dotenv(self, monkeypatch, tmp_path):
        (tmp_path / ".env").write_text("MISTRAL_API_KEY=from-file\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
        assert resolve_api_key(LLMConfig(provider="mistral")) == "from-env"


class TestFactory:
    def test_disabled_config_gives_disabled_client(self):
        assert isinstance(make_llm_client(LLMConfig(enabled=False)), DisabledLLMClient)

    def test_enabled_without_key_fails_fast(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)  # no .env in scope
        with pytest.raises(LLMKeyError, match="OPENAI_API_KEY"):
            make_llm_client(LLMConfig(enabled=True, provider="openai"))

    def test_keyless_provider_needs_no_key(self):
        client = make_llm_client(LLMConfig(enabled=True, provider="ollama", model="llama3.1"))
        assert isinstance(client, LiteLLMClient)

    def test_enabled_with_key_gives_litellm_client(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.chdir(tmp_path)
        cfg = LLMConfig(enabled=True, provider="openai")
        assert isinstance(make_llm_client(cfg), LiteLLMClient)

    def test_auto_config_enabled_when_default_provider_key_present(self, monkeypatch, tmp_path):
        from cadence.llm.litellm_client import auto_llm_config

        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.chdir(tmp_path)
        assert auto_llm_config().enabled is True

    def test_auto_config_disabled_without_key(self, monkeypatch, tmp_path):
        from cadence.llm.litellm_client import auto_llm_config

        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = auto_llm_config()
        assert cfg.enabled is False  # deterministic fallback per §7.7


class FakeLiteLLM:
    """Monkeypatch target standing in for litellm.acompletion."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    async def acompletion(self, **kwargs):
        self.calls.append(kwargs)

        class _Msg:
            content = self.content

        class _Choice:
            message = _Msg()

        class _Reply:
            choices = [_Choice()]

        return _Reply()


class TestEndToEnd:
    async def test_complete_validates_schema(self, monkeypatch, tmp_path):
        pytest.importorskip("litellm")  # optional dep group; skip when not synced
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.chdir(tmp_path)
        client = make_llm_client(LLMConfig(enabled=True, provider="openai", model="gpt-4o-mini"))
        fake = FakeLiteLLM('{"value": 9}')
        monkeypatch.setattr("litellm.acompletion", fake.acompletion)

        result = await client.complete([{"role": "user", "content": "gimme"}], Box)

        assert isinstance(result, Box) and result.value == 9
        sent = fake.calls[0]
        assert sent["model"] == "openai/gpt-4o-mini"
        assert sent["api_key"] == "sk-test"
        # schema is injected into the user prompt
        assert "JSON schema" in sent["messages"][-1]["content"]

    async def test_malformed_reply_raises_for_agent_retry(self, monkeypatch, tmp_path):
        pytest.importorskip("litellm")  # optional dep group; skip when not synced
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.chdir(tmp_path)
        client = make_llm_client(LLMConfig(enabled=True, provider="openai", model="gpt-4o-mini"))
        monkeypatch.setattr("litellm.acompletion", FakeLiteLLM("no json here, sorry").acompletion)
        with pytest.raises(ValueError, match="no JSON object"):
            await client.complete([{"role": "user", "content": "gimme"}], Box)

    async def test_disabled_client_still_raises(self):
        with pytest.raises(LLMDisabledError):
            await DisabledLLMClient().complete([], response_schema=Box)
