"""Generation request data models."""

from __future__ import annotations

from dataclasses import dataclass

from synthetic_data_gen.personas import Persona
from synthetic_data_gen.types import BatchId, PromptText, RouteName, SeedRecord


@dataclass(frozen=True)
class GenerationRequest:
    batch_id: BatchId
    prompt: PromptText
    route: RouteName
    persona: Persona
    seed: SeedRecord
    batch_size: int
