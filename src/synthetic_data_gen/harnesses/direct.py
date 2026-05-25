"""Direct schema generation harness."""

from __future__ import annotations

from dataclasses import dataclass

from synthetic_data_gen.backends import GenerationClient
from synthetic_data_gen.models import GenerationRequest
from synthetic_data_gen.types import ModelName, RawModelOutput


@dataclass
class DirectSchemaHarness:
    client: GenerationClient

    @property
    def model(self) -> ModelName:
        return self.client.model

    def generate(self, request: GenerationRequest) -> RawModelOutput:
        return self.client.generate(request.prompt)
