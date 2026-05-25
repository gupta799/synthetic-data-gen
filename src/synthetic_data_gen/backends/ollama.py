"""Ollama backend for local Apple Silicon/MPS-style runs."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from synthetic_data_gen.generation_schema import GENERATION_BATCH_SCHEMA
from synthetic_data_gen.prompts import SYSTEM_PROMPT
from synthetic_data_gen.types import ModelName, OllamaBaseUrl, PromptText, RawModelOutput


@dataclass
class OllamaClient:
    model: ModelName
    base_url: OllamaBaseUrl
    temperature: float
    max_tokens: int = 4096
    timeout_seconds: float = 180.0

    def generate(self, prompt: PromptText) -> RawModelOutput:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": str(self.model),
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "format": GENERATION_BATCH_SCHEMA,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                f"Could not generate with Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc

        message = response_payload.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not content:
            raise RuntimeError("Ollama response did not include message.content.")
        return RawModelOutput(str(content))


def list_ollama_models(base_url: OllamaBaseUrl) -> set[ModelName]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. Is `ollama serve` running?"
        ) from exc
    models = payload.get("models", [])
    return {ModelName(str(item.get("name"))) for item in models if item.get("name")}


def assert_ollama_model_available(*, model: ModelName, base_url: OllamaBaseUrl) -> None:
    models = list_ollama_models(base_url)
    if model not in models:
        raise RuntimeError(
            f"Ollama model {model!r} is not installed. Run `ollama pull {model}` and retry."
        )
