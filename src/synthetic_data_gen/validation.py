"""Validation and conversion of model outputs into classifier rows."""

from __future__ import annotations

import re
from collections import Counter

from synthetic_data_gen.labels import LABELS, validate_route
from synthetic_data_gen.types import (
    BatchId,
    CompanyName,
    EmbeddingModelName,
    GeneratedText,
    InstitutionType,
    JsonObject,
    Metadata,
    ModelName,
    PersonaName,
    RejectedCandidate,
    RejectionReason,
    RouteName,
    RouterExample,
    SeedRecord,
    SourceName,
    normalize_text,
    stable_id,
)

FORBIDDEN_TEXT_PATTERNS = (
    "as an ai",
    "synthetic data",
    "training data",
    "classifier",
    "route",
    "label",
    "metric_extraction",
    "metric extraction",
    "filing_summarization",
    "filing summarization",
    "financial_qa",
    "financial qa",
    "financial_reasoning",
    "financial reasoning",
    "comparative_analysis",
    "comparative analysis",
)

ANSWER_MARKERS = (
    "the answer is",
    "here is the",
    "below is",
    "i calculated",
    "i found",
)


def reject(reason: RejectionReason, raw_text: object, **metadata: object) -> RejectedCandidate:
    return RejectedCandidate(
        reason=reason,
        raw_text=normalize_text(raw_text),
        metadata={key: normalize_text(value) for key, value in metadata.items()},
    )


def validate_generated_object(
    payload: JsonObject,
    *,
    expected_route: RouteName,
    seed: SeedRecord,
    generator_model: ModelName,
    embedding_model: EmbeddingModelName,
    generation_batch_id: BatchId,
) -> RouterExample | RejectedCandidate:
    raw_text = GeneratedText(str(payload.get("text") or ""))
    text = normalize_text(raw_text)
    try:
        route = validate_route(payload.get("route") or "")
    except ValueError:
        return reject(
            RejectionReason("invalid_route"),
            raw_text,
            payload=payload,
            expected_route=expected_route,
        )
    if route != expected_route:
        return reject(
            RejectionReason("wrong_route"),
            raw_text,
            route=route,
            expected_route=expected_route,
        )

    lowered = text.lower()
    for pattern in FORBIDDEN_TEXT_PATTERNS:
        if pattern in lowered:
            return reject(RejectionReason("forbidden_text"), raw_text, route=route, pattern=pattern)
    for marker in ANSWER_MARKERS:
        if marker in lowered:
            return reject(
                RejectionReason("answer_like_output"),
                raw_text,
                route=route,
                marker=marker,
            )
    if re.match(r"^\s*[-*\d.]+\s+", text):
        return reject(RejectionReason("list_or_answer_format"), raw_text, route=route)
    if len(text.split()) < 8:
        return reject(RejectionReason("too_short"), raw_text, route=route)
    if len(text.split()) > 110:
        return reject(RejectionReason("too_long"), raw_text, route=route)

    persona = PersonaName(normalize_text(payload.get("persona") or ""))
    institution_type = InstitutionType(normalize_text(payload.get("institution_type") or ""))
    company_text = normalize_text(payload.get("company") or seed.company or "")
    company = CompanyName(company_text) if company_text else None
    metadata: Metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "generator_model": generator_model,
            "embedding_model": embedding_model,
            "persona": persona,
            "institution_type": institution_type,
            "seed_source": seed.source,
            "seed_group": seed.group_key,
            "group_key": seed.group_key,
            "generation_batch_id": generation_batch_id,
            "prompt_hash": stable_id(text.lower()),
        }
    )
    return RouterExample(
        id=stable_id(generator_model, route, seed.group_key, text),
        text=text,
        route=route,
        source=SourceName(f"synthetic:ollama:{generator_model}"),
        company=company,
        metadata=metadata,
    )


def assert_no_forbidden_labels(rows: list[RouterExample]) -> None:
    labels = {row.route for row in rows}
    forbidden = labels - set(LABELS)
    if forbidden:
        raise ValueError(f"Forbidden labels found: {sorted(forbidden)}")


def route_counts(rows: list[RouterExample]) -> dict[RouteName, int]:
    return dict(sorted(Counter(row.route for row in rows).items()))
