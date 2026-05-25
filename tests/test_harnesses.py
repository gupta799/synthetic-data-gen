from __future__ import annotations

import json

from synthetic_data_gen.harnesses import DirectSchemaHarness
from synthetic_data_gen.models import GenerationRequest
from synthetic_data_gen.personas import PERSONAS
from synthetic_data_gen.types import (
    BatchId,
    CompanyName,
    GeneratedText,
    GroupKey,
    ModelName,
    PromptText,
    RawModelOutput,
    RouteName,
    SeedRecord,
    SourceName,
)


class FakeClient:
    model = ModelName("unit-model")

    def generate(self, prompt: PromptText) -> RawModelOutput:
        return RawModelOutput(json.dumps({"items": [{"text": prompt, "route": "financial_qa"}]}))


def test_direct_schema_harness_delegates_to_client() -> None:
    harness = DirectSchemaHarness(FakeClient())
    result = harness.generate(
        GenerationRequest(
            batch_id=BatchId("batch"),
            prompt=PromptText("What did Apple disclose about its debt maturity schedule?"),
            route=RouteName("financial_qa"),
            persona=PERSONAS[0],
            seed=SeedRecord(
                source=SourceName("unit"),
                group_key=GroupKey("unit:group"),
                context=GeneratedText("Apple debt context"),
                company=CompanyName("Apple Inc"),
            ),
            batch_size=1,
        )
    )

    assert "Apple" in result
