"""DeepAgents generation harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from synthetic_data_gen.backends.factory import create_langchain_chat_model
from synthetic_data_gen.generation_schema import GENERATION_BATCH_SCHEMA
from synthetic_data_gen.models import GenerationRequest
from synthetic_data_gen.prompts import build_persona_system_prompt
from synthetic_data_gen.types import (
    GeneratorBackend,
    ModelName,
    OllamaBaseUrl,
    OpenAIApiKey,
    OpenAIBaseUrl,
    PersonaName,
    RawModelOutput,
)


@dataclass
class DeepAgentGenerationHarness:
    model: ModelName
    backend: GeneratorBackend
    openai_base_url: OpenAIBaseUrl
    openai_api_key: OpenAIApiKey
    ollama_base_url: OllamaBaseUrl
    temperature: float
    _agents: dict[PersonaName, Any] = field(default_factory=dict)

    def generate(self, request: GenerationRequest) -> RawModelOutput:
        agent = self._agent_for(request)
        result = agent.invoke({"messages": [{"role": "user", "content": request.prompt}]})
        structured = result.get("structured_response") if isinstance(result, dict) else None
        if structured is not None:
            return RawModelOutput(json.dumps(_to_jsonable(structured), ensure_ascii=False))
        return RawModelOutput(_last_message_content(result))

    def _agent_for(self, request: GenerationRequest) -> Any:
        persona_name = request.persona.name
        cached = self._agents.get(persona_name)
        if cached is not None:
            return cached

        from deepagents import create_deep_agent
        from langchain.agents.structured_output import ToolStrategy

        chat_model = create_langchain_chat_model(
            backend=self.backend,
            model=self.model,
            openai_base_url=self.openai_base_url,
            openai_api_key=self.openai_api_key,
            ollama_base_url=self.ollama_base_url,
            temperature=self.temperature,
        )
        agent = create_deep_agent(
            model=chat_model,
            tools=[],
            system_prompt=str(build_persona_system_prompt(request.persona)),
            response_format=ToolStrategy(
                GENERATION_BATCH_SCHEMA,
                tool_message_content="Generated schema-valid finance router prompts.",
            ),
            name=f"finance-data-generator-{persona_name.replace(' ', '-')}",
        )
        self._agents[persona_name] = agent
        return agent


def _to_jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _last_message_content(result: object) -> str:
    if not isinstance(result, dict):
        return str(result)
    messages = result.get("messages") or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content:
            return str(content)
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
    return str(result)
