"""Text normalization and stable IDs."""

from __future__ import annotations

import hashlib
import re

from synthetic_data_gen.types.domain import ExampleId, GeneratedText

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: object) -> GeneratedText:
    return GeneratedText(_WHITESPACE_RE.sub(" ", str(text)).strip())


def stable_id(*parts: object) -> ExampleId:
    payload = "\n".join("" if part is None else str(part) for part in parts)
    return ExampleId(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20])
