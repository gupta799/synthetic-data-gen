"""Embedding-based diversity gates."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

import numpy as np

from synthetic_data_gen.types import (
    EmbeddingDevice,
    EmbeddingModelName,
    GeneratedText,
    MetricPayload,
    RouteName,
)

SUPPORTED_EMBEDDING_DEVICES: tuple[EmbeddingDevice, ...] = (
    EmbeddingDevice("auto"),
    EmbeddingDevice("cpu"),
    EmbeddingDevice("cuda"),
    EmbeddingDevice("mps"),
)
DEFAULT_EMBEDDING_DEVICE = EmbeddingDevice("auto")


class Embedder(Protocol):
    def encode_one(self, text: GeneratedText) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: EmbeddingModelName,
        device: EmbeddingDevice = DEFAULT_EMBEDDING_DEVICE,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = resolve_embedding_device(device)
        self._model = SentenceTransformer(str(model_name), device=str(self.device))

    def encode_one(self, text: GeneratedText) -> np.ndarray:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32)


class DiversityIndex:
    def __init__(self, threshold: float = 0.88) -> None:
        self.threshold = threshold
        self._vectors: dict[RouteName, list[np.ndarray]] = defaultdict(list)
        self.max_seen_similarity: list[float] = []

    def nearest_similarity(self, route: RouteName, vector: np.ndarray) -> float:
        vectors = self._vectors.get(route) or []
        if not vectors:
            return 0.0
        matrix = np.vstack(vectors)
        similarities = matrix @ vector
        return float(np.max(similarities))

    def accept(self, route: RouteName, vector: np.ndarray) -> tuple[bool, float]:
        nearest = self.nearest_similarity(route, vector)
        self.max_seen_similarity.append(nearest)
        if nearest >= self.threshold:
            return False, nearest
        self._vectors[route].append(vector)
        return True, nearest

    @property
    def similarity_stats(self) -> MetricPayload:
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

    def encode_one(self, text: GeneratedText) -> np.ndarray:
        lowered = text.lower()
        base = np.zeros(8, dtype=np.float32)
        for index, char in enumerate(lowered.encode("utf-8")):
            base[index % len(base)] += float(char)
        norm = np.linalg.norm(base)
        if norm == 0:
            return base
        return base / norm


def resolve_embedding_device(requested: EmbeddingDevice) -> EmbeddingDevice:
    if requested not in SUPPORTED_EMBEDDING_DEVICES:
        choices = ", ".join(SUPPORTED_EMBEDDING_DEVICES)
        raise ValueError(f"Unknown embedding device {requested!r}; expected one of {choices}.")
    try:
        import torch
    except ImportError as exc:
        if requested == EmbeddingDevice("auto") or requested == EmbeddingDevice("cpu"):
            return EmbeddingDevice("cpu")
        raise RuntimeError("PyTorch is required for CUDA or MPS embedding devices.") from exc

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested == EmbeddingDevice("auto"):
        if cuda_available:
            return EmbeddingDevice("cuda")
        if mps_available:
            return EmbeddingDevice("mps")
        return EmbeddingDevice("cpu")
    if requested == EmbeddingDevice("cuda") and not cuda_available:
        raise RuntimeError("Embedding device 'cuda' was requested, but PyTorch cannot see CUDA.")
    if requested == EmbeddingDevice("mps") and not mps_available:
        raise RuntimeError("Embedding device 'mps' was requested, but PyTorch cannot see MPS.")
    return requested
