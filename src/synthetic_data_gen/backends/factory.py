"""Backend selection without leaking server-specific details into the builder."""

from __future__ import annotations

from synthetic_data_gen.backends.base import GenerationClient
from synthetic_data_gen.backends.ollama import OllamaClient, assert_ollama_model_available
from synthetic_data_gen.backends.openai_compatible import (
    OpenAICompatibleClient,
    assert_openai_model_available,
)
from synthetic_data_gen.types import (
    GeneratorBackend,
    ModelName,
    ModelServerUrl,
    OllamaBaseUrl,
    OpenAIApiKey,
    OpenAIBaseUrl,
)

VLLM_BACKEND = GeneratorBackend("vllm")
OLLAMA_BACKEND = GeneratorBackend("ollama")
_OPENAI_ALIAS = GeneratorBackend("openai")


def normalize_generator_backend(backend: GeneratorBackend) -> GeneratorBackend:
    if backend == _OPENAI_ALIAS:
        return VLLM_BACKEND
    if backend in {VLLM_BACKEND, OLLAMA_BACKEND}:
        return backend
    raise ValueError("generator_backend must be one of: vllm, ollama")


def generation_server_url(
    *,
    backend: GeneratorBackend,
    openai_base_url: OpenAIBaseUrl,
    ollama_base_url: OllamaBaseUrl,
) -> ModelServerUrl:
    normalized_backend = normalize_generator_backend(backend)
    if normalized_backend == OLLAMA_BACKEND:
        return ModelServerUrl(str(ollama_base_url))
    return ModelServerUrl(str(openai_base_url))


def create_generation_client(
    *,
    backend: GeneratorBackend,
    model: ModelName,
    openai_base_url: OpenAIBaseUrl,
    openai_api_key: OpenAIApiKey,
    ollama_base_url: OllamaBaseUrl,
    temperature: float,
) -> GenerationClient:
    normalized_backend = normalize_generator_backend(backend)
    if normalized_backend == OLLAMA_BACKEND:
        return OllamaClient(
            model=model,
            base_url=ollama_base_url,
            temperature=temperature,
        )
    return OpenAICompatibleClient(
        model=model,
        base_url=openai_base_url,
        api_key=openai_api_key,
        temperature=temperature,
    )


def assert_generation_backend_available(
    *,
    backend: GeneratorBackend,
    model: ModelName,
    openai_base_url: OpenAIBaseUrl,
    openai_api_key: OpenAIApiKey,
    ollama_base_url: OllamaBaseUrl,
) -> None:
    normalized_backend = normalize_generator_backend(backend)
    if normalized_backend == OLLAMA_BACKEND:
        assert_ollama_model_available(model=model, base_url=ollama_base_url)
        return
    assert_openai_model_available(
        model=model,
        base_url=openai_base_url,
        api_key=openai_api_key,
    )
