"""Shared generation harness protocol."""

from __future__ import annotations

from typing import Protocol

from synthetic_data_gen.models import GenerationRequest
from synthetic_data_gen.types import ModelName, RawModelOutput


class GenerationHarness(Protocol):
    model: ModelName

    def generate(self, request: GenerationRequest) -> RawModelOutput: ...
