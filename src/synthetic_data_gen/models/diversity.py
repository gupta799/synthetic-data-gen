"""Diversity store data models."""

from __future__ import annotations

from dataclasses import dataclass

from synthetic_data_gen.types import (
    DistanceScore,
    ExampleId,
    GeneratedText,
    JsonObject,
    RejectionReason,
    RouteName,
)


@dataclass(frozen=True)
class DiversityPolicy:
    min_neighbor_distance: DistanceScore = DistanceScore(0.08)
    max_neighbor_distance: DistanceScore = DistanceScore(0.65)
    min_route_examples_for_floor: int = 25
    flush_every: int = 100


@dataclass(frozen=True)
class DiversityDecision:
    accepted: bool
    reason: RejectionReason | None
    nearest_distance: DistanceScore
    route_count: int

    def to_metadata(self) -> JsonObject:
        return {
            "nearest_distance": self.nearest_distance,
            "route_count": self.route_count,
            "score_backend": "chroma_cosine_distance",
        }


@dataclass(frozen=True)
class DiversityStoreRecord:
    id: ExampleId
    text: GeneratedText
    route: RouteName
    nearest_distance: DistanceScore

    def to_json(self) -> JsonObject:
        return {
            "id": self.id,
            "text": self.text,
            "route": self.route,
            "nearest_distance": self.nearest_distance,
        }
