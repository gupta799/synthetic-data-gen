"""Parsing helpers for strict-but-recoverable model JSON output."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCED_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_payload(text: str) -> str:
    stripped = text.strip()
    fenced = _FENCED_RE.search(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        return stripped

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start >= 0 and array_end > array_start:
        return stripped[array_start : array_end + 1]

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start >= 0 and object_end > object_start:
        return stripped[object_start : object_end + 1]
    return stripped


def parse_json_objects(text: str) -> list[dict[str, Any]]:
    payload = json.loads(extract_json_payload(text))
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            payload = payload["items"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON object or JSON array")
    objects: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Expected every generated item to be an object")
        objects.append(item)
    return objects
