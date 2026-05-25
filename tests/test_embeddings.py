from __future__ import annotations

import numpy as np
import pytest

from synthetic_data_gen.diversity import LocalDiversityStore
from synthetic_data_gen.embeddings import resolve_embedding_device
from synthetic_data_gen.models import DiversityPolicy
from synthetic_data_gen.types import (
    DistanceScore,
    EmbeddingDevice,
    GeneratedText,
    RejectionReason,
    RouteName,
)


class ConstantEmbedder:
    def encode_one(self, text: GeneratedText) -> np.ndarray:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def test_local_diversity_store_rejects_low_diversity_vector(tmp_path) -> None:
    store = LocalDiversityStore(
        tmp_path / "store",
        DiversityPolicy(min_neighbor_distance=DistanceScore(0.9)),
        ConstantEmbedder(),
    )
    route = RouteName("metric_extraction")

    first = store.evaluate(route=route, text=GeneratedText("first prompt"))
    assert first.accepted is True
    assert first.nearest_distance == 1.0

    from synthetic_data_gen.types import RouterExample, SourceName

    store.commit(
        row=RouterExample(
            text=GeneratedText("first prompt"),
            route=route,
            source=SourceName("unit"),
        ),
        decision=first,
    )
    second = store.evaluate(route=route, text=GeneratedText("first prompt, please"))

    assert second.accepted is False
    assert second.reason == RejectionReason("low_diversity_embedding")
    assert second.nearest_distance == 0.0
    store.close()


def test_resolve_embedding_device_accepts_cpu() -> None:
    assert resolve_embedding_device(EmbeddingDevice("cpu")) == "cpu"


def test_resolve_embedding_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unknown embedding device"):
        resolve_embedding_device(EmbeddingDevice("tpu"))
