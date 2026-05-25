"""JSON-compatible payload types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
Metadata: TypeAlias = dict[str, JsonValue]
MetricPayload: TypeAlias = dict[str, JsonValue]
ReadonlyMetadata: TypeAlias = Mapping[str, JsonValue]
