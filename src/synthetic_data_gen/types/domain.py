"""Narrow domain types for the synthetic data generator."""

from __future__ import annotations

from typing import NewType

BatchId = NewType("BatchId", str)
ArtifactName = NewType("ArtifactName", str)
ArtifactType = NewType("ArtifactType", str)
CliArg = NewType("CliArg", str)
CompanyName = NewType("CompanyName", str)
DocumentType = NewType("DocumentType", str)
EmbeddingModelName = NewType("EmbeddingModelName", str)
EmbeddingDevice = NewType("EmbeddingDevice", str)
EventName = NewType("EventName", str)
ExampleId = NewType("ExampleId", str)
FiscalPeriod = NewType("FiscalPeriod", str)
GeneratedText = NewType("GeneratedText", str)
GroupKey = NewType("GroupKey", str)
InstitutionType = NewType("InstitutionType", str)
ModelName = NewType("ModelName", str)
OllamaBaseUrl = NewType("OllamaBaseUrl", str)
PersonaName = NewType("PersonaName", str)
ProjectName = NewType("ProjectName", str)
PromptText = NewType("PromptText", str)
RawModelOutput = NewType("RawModelOutput", str)
RejectionReason = NewType("RejectionReason", str)
RouteName = NewType("RouteName", str)
RunName = NewType("RunName", str)
SourceName = NewType("SourceName", str)
StyleHint = NewType("StyleHint", str)


def optional_company(value: object | None) -> CompanyName | None:
    text = "" if value is None else str(value).strip()
    return CompanyName(text) if text else None
