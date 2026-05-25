"""Diversity store data models."""

from __future__ import annotations

from dataclasses import dataclass

from synthetic_data_gen.types import (
    ExampleId,
    GeneratedText,
    JsonObject,
    RejectionReason,
    RouteName,
    SimilarityScore,
)


@dataclass(frozen=True)
class DiversityPolicy:
    max_similarity: SimilarityScore = SimilarityScore(0.88)
    min_route_similarity: SimilarityScore = SimilarityScore(0.05)
    min_route_examples_for_floor: int = 25
    flush_every: int = 100


@dataclass(frozen=True)
class DiversityDecision:
    accepted: bool
    reason: RejectionReason | None
    nearest_similarity: SimilarityScore
    route_centroid_similarity: SimilarityScore
    route_count: int

    def to_metadata(self) -> JsonObject:
        return {
            "nearest_similarity": self.nearest_similarity,
            "route_centroid_similarity": self.route_centroid_similarity,
            "route_count": self.route_count,
        }


@dataclass(frozen=True)
class DiversityStoreRecord:
    id: ExampleId
    text: GeneratedText
    route: RouteName
    nearest_similarity: SimilarityScore
    route_centroid_similarity: SimilarityScore

    def to_json(self) -> JsonObject:
        return {
            "id": self.id,
            "text": self.text,
            "route": self.route,
            "nearest_similarity": self.nearest_similarity,
            "route_centroid_similarity": self.route_centroid_similarity,
        }
