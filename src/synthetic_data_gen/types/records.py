"""Dataset records and JSONL serialization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from synthetic_data_gen.types.domain import (
    CompanyName,
    DocumentType,
    ExampleId,
    FiscalPeriod,
    GeneratedText,
    GroupKey,
    PersonaName,
    RejectionReason,
    RouteName,
    SourceName,
)
from synthetic_data_gen.types.json import JsonObject, Metadata
from synthetic_data_gen.types.text import normalize_text, stable_id


@dataclass(frozen=True)
class SeedRecord:
    source: SourceName
    group_key: GroupKey
    context: GeneratedText
    company: CompanyName | None = None
    document_type: DocumentType | None = None
    period: FiscalPeriod | None = None
    metadata: Metadata = field(default_factory=dict)

    @property
    def short_context(self) -> GeneratedText:
        return GeneratedText(normalize_text(self.context)[:1800])


@dataclass(frozen=True)
class RouterExample:
    text: GeneratedText
    route: RouteName
    source: SourceName
    id: ExampleId | None = None
    company: CompanyName | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        from synthetic_data_gen.labels import validate_route

        text = normalize_text(self.text)
        if not text:
            raise ValueError("RouterExample text cannot be empty")
        route = validate_route(self.route)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "route", route)
        if self.id is None:
            object.__setattr__(
                self,
                "id",
                stable_id(self.source, route, self.company or "", text),
            )

    @property
    def group_key(self) -> GroupKey:
        value = self.metadata.get("group_key")
        return GroupKey(str(value)) if value else GroupKey(str(self.id))

    def to_json(self) -> JsonObject:
        return {
            "id": self.id,
            "text": self.text,
            "route": self.route,
            "source": self.source,
            "company": self.company,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, payload: JsonObject) -> RouterExample:
        return cls(
            id=ExampleId(str(payload["id"])) if payload.get("id") else None,
            text=GeneratedText(str(payload["text"])),
            route=RouteName(str(payload["route"])),
            source=SourceName(str(payload["source"])),
            company=CompanyName(str(payload["company"])) if payload.get("company") else None,
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RejectedCandidate:
    reason: RejectionReason
    raw_text: GeneratedText
    route: RouteName | None = None
    persona: PersonaName | None = None
    seed_group: GroupKey | None = None
    metadata: Metadata = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return {
            "reason": self.reason,
            "raw_text": self.raw_text,
            "route": self.route,
            "persona": self.persona,
            "seed_group": self.seed_group,
            "metadata": self.metadata,
        }


def write_jsonl(path: Path, rows: Iterable[RouterExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")


def write_rejected_jsonl(path: Path, rows: Iterable[RejectedCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
