"""Shared generation backend protocols."""

from __future__ import annotations

from typing import Protocol

from synthetic_data_gen.types import ModelName, PromptText, RawModelOutput


class GenerationClient(Protocol):
    model: ModelName

    def generate(self, prompt: PromptText) -> RawModelOutput: ...
