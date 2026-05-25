"""Data contracts for synthetic finance router generation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthetic_data_gen.labels import validate_route

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def stable_id(*parts: object) -> str:
    payload = "\n".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class SeedRecord:
    source: str
    group_key: str
    context: str
    company: str | None = None
    document_type: str | None = None
    period: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def short_context(self) -> str:
        return normalize_text(self.context)[:1800]


@dataclass(frozen=True)
class RouterExample:
    text: str
    route: str
    source: str
    id: str | None = None
    company: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        text = normalize_text(self.text)
        if not text:
            raise ValueError("RouterExample text cannot be empty")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "route", validate_route(self.route))
        if self.id is None:
            object.__setattr__(
                self,
                "id",
                stable_id(self.source, self.route, self.company or "", text),
            )

    @property
    def group_key(self) -> str:
        value = self.metadata.get("group_key")
        return str(value) if value else str(self.id)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "route": self.route,
            "source": self.source,
            "company": self.company,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RouterExample:
        return cls(
            id=payload.get("id"),
            text=payload["text"],
            route=payload["route"],
            source=payload["source"],
            company=payload.get("company"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class RejectedCandidate:
    reason: str
    raw_text: str
    route: str | None = None
    persona: str | None = None
    seed_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
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
