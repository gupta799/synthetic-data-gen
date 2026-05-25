"""Structured generation schema for model backends."""

from __future__ import annotations

from synthetic_data_gen.labels import LABELS
from synthetic_data_gen.types import JsonObject

GENERATION_BATCH_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "minLength": 20,
                        "description": "One realistic finance user prompt. No answer text.",
                    },
                    "route": {"type": "string", "enum": list(LABELS)},
                    "company": {"type": "string"},
                    "persona": {"type": "string"},
                    "institution_type": {"type": "string"},
                    "metadata": {"type": "object", "additionalProperties": True},
                },
                "required": [
                    "text",
                    "route",
                    "company",
                    "persona",
                    "institution_type",
                    "metadata",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}
