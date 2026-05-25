"""Ollama/DeepAgents client for synthetic prompt generation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from synthetic_data_gen.prompts import SYSTEM_PROMPT


class GenerationClient(Protocol):
    model: str

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class DeepAgentOllamaClient:
    model: str = "gemma4:e2b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.8

    def __post_init__(self) -> None:
        from deepagents import create_deep_agent
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            format="json",
        )

        self._agent = create_deep_agent(
            model=llm,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
        )

    def generate(self, prompt: str) -> str:
        result = self._agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            return str(result)
        last = messages[-1]
        content = getattr(last, "content", None)
        if content is None and isinstance(last, dict):
            content = last.get("content")
        return str(content)


def list_ollama_models(base_url: str) -> set[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {base_url}. Is `ollama serve` running?"
        ) from exc
    models = payload.get("models", [])
    return {str(item.get("name")) for item in models if item.get("name")}


def assert_ollama_model_available(model: str, base_url: str) -> None:
    models = list_ollama_models(base_url)
    if model not in models:
        raise RuntimeError(
            f"Ollama model {model!r} is not installed. Run `ollama pull {model}` and retry."
        )
