"""OpenAI-compatible chat completion client."""

from __future__ import annotations

import os
from dataclasses import dataclass

from synthetic_data_gen.generation_schema import GENERATION_BATCH_SCHEMA
from synthetic_data_gen.prompts import SYSTEM_PROMPT
from synthetic_data_gen.types import (
    ModelName,
    OpenAIApiKey,
    OpenAIBaseUrl,
    PromptText,
    RawModelOutput,
)


@dataclass
class OpenAICompatibleClient:
    model: ModelName
    base_url: OpenAIBaseUrl
    api_key: OpenAIApiKey
    temperature: float
    max_tokens: int = 4096
    timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=str(self.base_url),
            api_key=str(self.api_key),
            timeout=self.timeout_seconds,
        )

    def generate(self, prompt: PromptText) -> RawModelOutput:
        completion = self._client.chat.completions.create(
            model=str(self.model),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "finance_router_generation_batch",
                    "schema": GENERATION_BATCH_SCHEMA,
                    "strict": True,
                },
            },
        )
        content = completion.choices[0].message.content or ""
        return RawModelOutput(str(content))


def openai_api_key_from_env() -> OpenAIApiKey:
    return OpenAIApiKey(os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "-")


def assert_openai_model_available(
    *,
    model: ModelName,
    base_url: OpenAIBaseUrl,
    api_key: OpenAIApiKey,
) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=str(base_url), api_key=str(api_key), timeout=20.0)
    try:
        models = {ModelName(item.id) for item in client.models.list().data}
    except Exception as exc:
        raise RuntimeError(f"Could not reach OpenAI-compatible server at {base_url}.") from exc
    if models and model not in models:
        sample = ", ".join(sorted(models)[:5])
        raise RuntimeError(f"Model {model!r} is not served at {base_url}. Available: {sample}")
