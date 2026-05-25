"""Model backend boundaries."""

from synthetic_data_gen.backends.base import GenerationClient
from synthetic_data_gen.backends.factory import (
    OLLAMA_BACKEND,
    VLLM_BACKEND,
    assert_generation_backend_available,
    create_generation_client,
    create_langchain_chat_model,
    generation_server_url,
    normalize_generator_backend,
)
from synthetic_data_gen.backends.openai_compatible import openai_api_key_from_env

__all__ = [
    "GenerationClient",
    "OLLAMA_BACKEND",
    "VLLM_BACKEND",
    "assert_generation_backend_available",
    "create_generation_client",
    "create_langchain_chat_model",
    "generation_server_url",
    "normalize_generator_backend",
    "openai_api_key_from_env",
]
