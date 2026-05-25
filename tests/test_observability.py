from __future__ import annotations

import os

from synthetic_data_gen.observability import configure_langsmith
from synthetic_data_gen.types import ProjectName


def test_configure_langsmith_skips_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    status = configure_langsmith(ProjectName("finance-router-data-gen"))

    assert status == "disabled:missing_api_key"
    assert os.getenv("LANGSMITH_TRACING") == "false"
    assert os.getenv("LANGCHAIN_TRACING_V2") == "false"
