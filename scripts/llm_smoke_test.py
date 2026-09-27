"""Live §7.7 smoke test: config → factory → one real structured LLM call.

Verifies the whole chain with a real provider: key resolution, the model string,
the JSON-schema-in-prompt round trip, and Pydantic validation of the reply.

Usage:
    uv run python scripts/llm_smoke_test.py [--provider openai] [--model gpt-4o-mini]

Defaults come from CADENCE_LLM_PROVIDER / CADENCE_LLM_MODEL env vars (or .env),
falling back to openai/gpt-4o-mini. The API key is read from the environment or
.env per PROVIDER_KEY_ENVS — it is never a CLI flag (keeps it out of shell history).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from pydantic import BaseModel, Field

from cadence.config.default_config import LLMConfig
from cadence.llm.litellm_client import PROVIDER_KEY_ENVS, load_dotenv, make_llm_client


class SmoothingChoice(BaseModel):
    """Toy §7.7-style structured decision for the smoke test."""

    method: str = Field(description="one of: interpolate_linear, forward_fill, drop")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=os.environ.get("CADENCE_LLM_PROVIDER", "groq"))
    parser.add_argument(
        "--model", default=os.environ.get("CADENCE_LLM_MODEL", "openai/gpt-oss-120b")
    )
    args = parser.parse_args()

    config = LLMConfig(enabled=True, provider=args.provider, model=args.model)
    try:
        client = make_llm_client(config)
    except Exception as exc:  # LLMKeyError / MissingDependencyError
        print(f"setup failed: {exc}")
        sys.exit(1)

    key_hint = "(resolved from env)" if client.api_key else "(keyless provider)"
    print(f"provider={config.provider}  model={config.model}  {key_hint}")
    print("calling...\n")

    messages = [
        {
            "role": "user",
            "content": (
                "A daily sales series has 2% missing values scattered as single-day gaps, "
                "no long stretches of missingness. Choose a missing-value strategy."
            ),
        },
    ]

    try:
        decision = asyncio.run(client.complete(messages, response_schema=SmoothingChoice))
    except Exception as exc:
        print(f"LLM call failed: {type(exc).__name__}: {exc}")
        sys.exit(1)

    print("structured response received and validated:")
    print(f"  method:     {decision.method}")
    print(f"  confidence: {decision.confidence}")
    print(f"  reason:     {decision.reason}")
    env_name = PROVIDER_KEY_ENVS.get(config.provider, "keyless")
    print(f"\n§7.7 chain OK ({config.provider}/{config.model}, key via {env_name})")


if __name__ == "__main__":
    main()
