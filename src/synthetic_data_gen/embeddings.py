"""Embedding-based diversity gates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode_one(self, text: str) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode_one(self, text: str) -> np.ndarray:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32)


class DiversityIndex:
    def __init__(self, threshold: float = 0.88) -> None:
        self.threshold = threshold
        self._vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        self.max_seen_similarity: list[float] = []

    def nearest_similarity(self, route: str, vector: np.ndarray) -> float:
        vectors = self._vectors.get(route) or []
        if not vectors:
            return 0.0
        matrix = np.vstack(vectors)
        similarities = matrix @ vector
        return float(np.max(similarities))

    def accept(self, route: str, vector: np.ndarray) -> tuple[bool, float]:
        nearest = self.nearest_similarity(route, vector)
        self.max_seen_similarity.append(nearest)
        if nearest >= self.threshold:
            return False, nearest
        self._vectors[route].append(vector)
        return True, nearest

    @property
    def similarity_stats(self) -> dict[str, float]:
        if not self.max_seen_similarity:
            return {"count": 0, "max": 0.0, "mean": 0.0, "p95": 0.0}
        values = np.asarray(self.max_seen_similarity, dtype=np.float32)
        return {
            "count": float(len(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.quantile(values, 0.95)),
        }


class StaticEmbedder:
    """Tiny deterministic embedder for tests."""

    def encode_one(self, text: str) -> np.ndarray:
        lowered = text.lower()
        base = np.zeros(8, dtype=np.float32)
        for index, char in enumerate(lowered.encode("utf-8")):
            base[index % len(base)] += float(char)
        norm = np.linalg.norm(base)
        if norm == 0:
            return base
        return base / norm


def pairwise_duplicates(texts: Sequence[str]) -> int:
    return len(texts) - len(set(texts))
