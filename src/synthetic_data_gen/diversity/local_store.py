"""Local on-disk embedding diversity store."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from synthetic_data_gen.models import DiversityDecision, DiversityPolicy, DiversityStoreRecord
from synthetic_data_gen.types import (
    GeneratedText,
    MetricPayload,
    RejectionReason,
    RouteName,
    RouterExample,
    SimilarityScore,
    stable_id,
)


class LocalDiversityStore:
    def __init__(self, path: Path, policy: DiversityPolicy, *, reset: bool = True) -> None:
        self.path = path
        self.policy = policy
        self.path.mkdir(parents=True, exist_ok=True)
        self.records_path = self.path / "records.jsonl"
        self.vectors_path = self.path / "vectors.npz"
        self.summary_path = self.path / "summary.json"
        if reset:
            self._reset_files()
        self._vectors: dict[RouteName, list[np.ndarray]] = defaultdict(list)
        self._prompt_hashes: set[str] = set()
        self._records_written = 0
        self.max_seen_similarity: list[float] = []
        self.centroid_similarity: list[float] = []
        self._records_handle = self.records_path.open("a", encoding="utf-8")

    def evaluate(
        self,
        *,
        route: RouteName,
        text: GeneratedText,
        vector: np.ndarray,
    ) -> DiversityDecision:
        prompt_hash = stable_id(text.lower())
        if prompt_hash in self._prompt_hashes:
            return DiversityDecision(
                accepted=False,
                reason=RejectionReason("duplicate_prompt"),
                nearest_similarity=SimilarityScore(1.0),
                route_centroid_similarity=SimilarityScore(1.0),
                route_count=len(self._vectors.get(route) or []),
            )

        normalized = _normalize(vector)
        nearest = self.nearest_similarity(route, normalized)
        centroid = self.route_centroid_similarity(route, normalized)
        route_count = len(self._vectors.get(route) or [])
        self.max_seen_similarity.append(float(nearest))
        self.centroid_similarity.append(float(centroid))

        if nearest >= self.policy.max_similarity:
            return DiversityDecision(
                accepted=False,
                reason=RejectionReason("low_diversity_embedding"),
                nearest_similarity=nearest,
                route_centroid_similarity=centroid,
                route_count=route_count,
            )
        if (
            route_count >= self.policy.min_route_examples_for_floor
            and centroid < self.policy.min_route_similarity
        ):
            return DiversityDecision(
                accepted=False,
                reason=RejectionReason("off_route_embedding"),
                nearest_similarity=nearest,
                route_centroid_similarity=centroid,
                route_count=route_count,
            )
        return DiversityDecision(
            accepted=True,
            reason=None,
            nearest_similarity=nearest,
            route_centroid_similarity=centroid,
            route_count=route_count,
        )

    def commit(
        self,
        *,
        row: RouterExample,
        vector: np.ndarray,
        decision: DiversityDecision,
    ) -> None:
        normalized = _normalize(vector)
        self._vectors[row.route].append(normalized)
        self._prompt_hashes.add(stable_id(row.text.lower()))
        record = DiversityStoreRecord(
            id=row.id,
            text=row.text,
            route=row.route,
            nearest_similarity=decision.nearest_similarity,
            route_centroid_similarity=decision.route_centroid_similarity,
        )
        self._records_handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        self._records_handle.flush()
        self._records_written += 1
        if self._records_written % self.policy.flush_every == 0:
            self.flush()

    def nearest_similarity(self, route: RouteName, vector: np.ndarray) -> SimilarityScore:
        vectors = self._vectors.get(route) or []
        if not vectors:
            return SimilarityScore(0.0)
        matrix = np.vstack(vectors)
        return SimilarityScore(float(np.max(matrix @ vector)))

    def route_centroid_similarity(self, route: RouteName, vector: np.ndarray) -> SimilarityScore:
        vectors = self._vectors.get(route) or []
        if not vectors:
            return SimilarityScore(0.0)
        centroid = _normalize(np.mean(np.vstack(vectors), axis=0))
        return SimilarityScore(float(centroid @ vector))

    def flush(self) -> None:
        payload = {str(route): np.vstack(vectors) for route, vectors in self._vectors.items()}
        if payload:
            np.savez(self.vectors_path, **payload)
        self.summary_path.write_text(
            json.dumps(
                {
                    "records": self._records_written,
                    "routes": {
                        str(route): len(vectors) for route, vectors in self._vectors.items()
                    },
                    "similarity": self.similarity_stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def close(self) -> None:
        self.flush()
        self._records_handle.close()

    @property
    def similarity_stats(self) -> MetricPayload:
        return {
            "nearest": _stats(self.max_seen_similarity),
            "route_centroid": _stats(self.centroid_similarity),
            "policy": {
                "max_similarity": self.policy.max_similarity,
                "min_route_similarity": self.policy.min_route_similarity,
                "min_route_examples_for_floor": self.policy.min_route_examples_for_floor,
            },
        }

    def _reset_files(self) -> None:
        for path in (self.records_path, self.vectors_path, self.summary_path):
            if path.exists():
                path.unlink()


def _normalize(vector: np.ndarray) -> np.ndarray:
    normalized = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(normalized)
    if norm == 0:
        return normalized
    return normalized / norm


def _stats(values: list[float]) -> MetricPayload:
    if not values:
        return {"count": 0, "max": 0.0, "mean": 0.0, "p95": 0.0}
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": float(len(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
    }
