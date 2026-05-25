"""Generation harness selection."""

from __future__ import annotations

from synthetic_data_gen.backends import GenerationClient, create_generation_client
from synthetic_data_gen.harnesses.base import GenerationHarness
from synthetic_data_gen.harnesses.deepagent import DeepAgentGenerationHarness
from synthetic_data_gen.harnesses.direct import DirectSchemaHarness
from synthetic_data_gen.types import (
    GenerationHarnessName,
    GeneratorBackend,
    ModelName,
    OllamaBaseUrl,
    OpenAIApiKey,
    OpenAIBaseUrl,
)

DEEPAGENT_HARNESS = GenerationHarnessName("deepagent")
DIRECT_HARNESS = GenerationHarnessName("direct")


def normalize_generation_harness(harness: GenerationHarnessName) -> GenerationHarnessName:
    if harness in {DEEPAGENT_HARNESS, DIRECT_HARNESS}:
        return harness
    raise ValueError("generation_harness must be one of: deepagent, direct")


def create_generation_harness(
    *,
    harness: GenerationHarnessName,
    backend: GeneratorBackend,
    model: ModelName,
    openai_base_url: OpenAIBaseUrl,
    openai_api_key: OpenAIApiKey,
    ollama_base_url: OllamaBaseUrl,
    temperature: float,
    client: GenerationClient | None = None,
) -> GenerationHarness:
    normalized_harness = normalize_generation_harness(harness)
    if normalized_harness == DIRECT_HARNESS:
        direct_client = client or create_generation_client(
            backend=backend,
            model=model,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            ollama_base_url=ollama_base_url,
            temperature=temperature,
        )
        return DirectSchemaHarness(direct_client)
    if client is not None:
        return DirectSchemaHarness(client)
    return DeepAgentGenerationHarness(
        model=model,
        backend=backend,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
        temperature=temperature,
    )
