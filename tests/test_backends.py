from __future__ import annotations

from synthetic_data_gen.backends import generation_server_url, normalize_generator_backend
from synthetic_data_gen.types import GeneratorBackend, OllamaBaseUrl, OpenAIBaseUrl


def test_normalize_openai_alias_to_vllm() -> None:
    assert normalize_generator_backend(GeneratorBackend("openai")) == "vllm"


def test_generation_server_url_uses_selected_backend_base_url() -> None:
    assert (
        generation_server_url(
            backend=GeneratorBackend("ollama"),
            openai_base_url=OpenAIBaseUrl("http://localhost:8000/v1"),
            ollama_base_url=OllamaBaseUrl("http://localhost:11434"),
        )
        == "http://localhost:11434"
    )
