from __future__ import annotations

import numpy as np
import pytest

from synthetic_data_gen.diversity import LocalDiversityStore
from synthetic_data_gen.embeddings import resolve_embedding_device
from synthetic_data_gen.models import DiversityPolicy
from synthetic_data_gen.types import (
    EmbeddingDevice,
    GeneratedText,
    RejectionReason,
    RouteName,
    SimilarityScore,
)


def test_local_diversity_store_rejects_low_diversity_vector(tmp_path) -> None:
    store = LocalDiversityStore(
        tmp_path / "store",
        DiversityPolicy(max_similarity=SimilarityScore(0.9)),
    )
    vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    route = RouteName("metric_extraction")

    first = store.evaluate(route=route, text=GeneratedText("first prompt"), vector=vector)
    assert first.accepted is True
    assert first.nearest_similarity == 0.0

    from synthetic_data_gen.types import RouterExample, SourceName

    store.commit(
        row=RouterExample(
            text=GeneratedText("first prompt"),
            route=route,
            source=SourceName("unit"),
        ),
        vector=vector,
        decision=first,
    )
    second = store.evaluate(route=route, text=GeneratedText("second prompt"), vector=vector)

    assert second.accepted is False
    assert second.reason == RejectionReason("low_diversity_embedding")
    assert second.nearest_similarity == 1.0
    store.close()


def test_resolve_embedding_device_accepts_cpu() -> None:
    assert resolve_embedding_device(EmbeddingDevice("cpu")) == "cpu"


def test_resolve_embedding_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unknown embedding device"):
        resolve_embedding_device(EmbeddingDevice("tpu"))
