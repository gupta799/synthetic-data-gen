"""Grounding seed loading from finance datasets."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TypeAlias

from synthetic_data_gen.types import (
    CompanyName,
    DocumentType,
    FiscalPeriod,
    GeneratedText,
    GroupKey,
    SeedRecord,
    SourceName,
    normalize_text,
    stable_id,
)

FINANCEBENCH_SOURCE = SourceName("PatronusAI/financebench")
SUJET_SOURCE = SourceName("sujet-ai/Sujet-Financial-RAG-EN-Dataset")
DatasetRow: TypeAlias = Mapping[str, object]


def load_hf_dataset(
    dataset_name: SourceName,
    *args: object,
    **kwargs: object,
) -> Iterable[DatasetRow]:
    from datasets import load_dataset

    return load_dataset(str(dataset_name), *args, **kwargs)


def context_hash(text: GeneratedText) -> GroupKey:
    return GroupKey(stable_id(normalize_text(text).lower()))


def trim_context(text: object, max_chars: int = 1800) -> GeneratedText:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return GeneratedText(normalized[:max_chars].rsplit(" ", 1)[0])


def clean_company(candidate: object) -> CompanyName | None:
    cleaned = normalize_text(candidate).strip(" .,")
    if not cleaned or len(cleaned) > 80:
        return None
    lowered = cleaned.lower()
    if lowered in {"company", "the company"}:
        return None
    if lowered.startswith(("what ", "which ", "how ", "explain ", "compare ", "calculate ")):
        return None
    return CompanyName(cleaned)


def extract_company(
    question: GeneratedText,
    context: GeneratedText | None = None,
) -> CompanyName | None:
    company_suffix = r"(?:Inc\.?|Corporation|Corp\.?|Company|Co\.?|Ltd\.?|LLC|PLC|Group)"
    patterns = (
        rf"\bfor ([A-Z][A-Za-z0-9 .,&'-]{{2,80}}?{company_suffix})(?: as | with |,|\?|$)",
        rf"\bby ([A-Z][A-Za-z0-9 .,&'-]{{2,80}}?{company_suffix})(?: with |,|\?|$)",
        rf"\b([A-Z][A-Za-z0-9 .,&'-]{{2,80}}?{company_suffix})'s\b",
        r"Exact name of registrant.*?([A-Z][A-Za-z0-9 .,&'-]{3,80})",
    )
    haystacks = (question, context or GeneratedText(""))
    for haystack in haystacks:
        for pattern in patterns:
            match = re.search(pattern, haystack)
            if match:
                company = clean_company(match.group(1))
                if company:
                    return company
    return None


def iter_financebench_seeds() -> Iterable[SeedRecord]:
    rows = load_hf_dataset(FINANCEBENCH_SOURCE, split="train")
    seen: set[GroupKey] = set()
    for row in rows:
        company = row.get("company")
        doc_name = row.get("doc_name")
        doc_period = row.get("doc_period")
        question = normalize_text(row.get("question") or "")
        if question:
            group_key = GroupKey(
                f"financebench:{company}:{doc_name}:question:{row.get('financebench_id')}"
            )
            yield SeedRecord(
                source=FINANCEBENCH_SOURCE,
                group_key=group_key,
                context=question,
                company=CompanyName(str(company)) if company else None,
                document_type=DocumentType(str(doc_name)) if doc_name else None,
                period=FiscalPeriod(str(doc_period)) if doc_period else None,
                metadata={
                    "financebench_id": row.get("financebench_id"),
                    "question_type": row.get("question_type"),
                    "question_reasoning": row.get("question_reasoning"),
                },
            )
        for evidence in row.get("evidence") or []:
            text = evidence.get("evidence_text_full_page") or evidence.get("evidence_text") or ""
            text = trim_context(text)
            if len(text) < 120:
                continue
            group_hash = context_hash(text)
            if group_hash in seen:
                continue
            seen.add(group_hash)
            yield SeedRecord(
                source=FINANCEBENCH_SOURCE,
                group_key=GroupKey(f"financebench:context:{group_hash}"),
                context=text,
                company=CompanyName(str(company)) if company else None,
                document_type=DocumentType(str(doc_name)) if doc_name else None,
                period=FiscalPeriod(str(doc_period)) if doc_period else None,
                metadata={"doc_name": doc_name, "doc_period": doc_period},
            )


def iter_sujet_seeds(max_rows: int | None = None) -> Iterable[SeedRecord]:
    seen_contexts: set[GroupKey] = set()
    seen_rows = 0
    for split in ("train", "test"):
        rows = load_hf_dataset(SUJET_SOURCE, split=split, streaming=True)
        for row in rows:
            question = normalize_text(row.get("question") or "")
            context = trim_context(row.get("context") or "")
            if not question or not context:
                continue
            group_hash = context_hash(context)
            if group_hash in seen_contexts:
                continue
            seen_contexts.add(group_hash)
            yield SeedRecord(
                source=SUJET_SOURCE,
                group_key=GroupKey(f"sujet:context:{group_hash}"),
                context=GeneratedText(f"Question seed: {question}\n\nContext seed: {context}"),
                company=extract_company(question, context),
                document_type=DocumentType("filing/context"),
                metadata={"source_split": split},
            )
            seen_rows += 1
            if max_rows is not None and seen_rows >= max_rows:
                return


def load_seed_pool(max_sujet_rows: int | None = None) -> list[SeedRecord]:
    seeds = [*iter_financebench_seeds(), *iter_sujet_seeds(max_rows=max_sujet_rows)]
    if not seeds:
        raise ValueError("No grounding seeds were loaded.")
    return seeds
