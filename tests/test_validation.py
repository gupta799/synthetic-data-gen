from __future__ import annotations

from synthetic_data_gen.types import (
    BatchId,
    CompanyName,
    EmbeddingModelName,
    GeneratedText,
    GroupKey,
    ModelName,
    RejectedCandidate,
    RouteName,
    SeedRecord,
    SourceName,
)
from synthetic_data_gen.validation import validate_generated_object


def seed() -> SeedRecord:
    return SeedRecord(
        source=SourceName("unit"),
        group_key=GroupKey("unit:apple:2023"),
        context=GeneratedText("Apple annual report revenue and margin context."),
        company=CompanyName("Apple Inc"),
    )


def test_validate_generated_object_returns_classifier_schema() -> None:
    row = validate_generated_object(
        {
            "text": "Can you pull Apple's fiscal 2023 revenue from this filing excerpt?",
            "route": "metric_extraction",
            "company": "Apple Inc",
            "persona": "equity research analyst",
            "institution_type": "sell-side research",
            "metadata": {},
        },
        expected_route=RouteName("metric_extraction"),
        seed=seed(),
        generator_model=ModelName("gemma4:e2b"),
        embedding_model=EmbeddingModelName("BAAI/bge-small-en-v1.5"),
        generation_batch_id=BatchId("batch-1"),
    )

    assert not isinstance(row, RejectedCandidate)
    assert row.route == "metric_extraction"
    assert row.source == "synthetic:ollama:gemma4:e2b"
    assert row.metadata["group_key"] == "unit:apple:2023"


def test_validate_generated_object_rejects_route_leakage() -> None:
    row = validate_generated_object(
        {
            "text": "Create a metric_extraction prompt for Apple revenue.",
            "route": "metric_extraction",
            "company": "Apple Inc",
            "persona": "equity research analyst",
            "institution_type": "sell-side research",
            "metadata": {},
        },
        expected_route=RouteName("metric_extraction"),
        seed=seed(),
        generator_model=ModelName("gemma4:e2b"),
        embedding_model=EmbeddingModelName("BAAI/bge-small-en-v1.5"),
        generation_batch_id=BatchId("batch-1"),
    )

    assert isinstance(row, RejectedCandidate)
    assert row.reason == "forbidden_text"
