from __future__ import annotations

import numpy as np
import pytest

from synthetic_data_gen.embeddings import DiversityIndex, resolve_embedding_device
from synthetic_data_gen.types import EmbeddingDevice, RouteName


def test_diversity_index_rejects_low_diversity_vector() -> None:
    index = DiversityIndex(threshold=0.9)
    vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    route = RouteName("metric_extraction")

    accepted, first_similarity = index.accept(route, vector)
    second_acceptance, second_similarity = index.accept(route, vector)

    assert accepted is True
    assert first_similarity == 0.0
    assert second_acceptance is False
    assert second_similarity == 1.0


def test_resolve_embedding_device_accepts_cpu() -> None:
    assert resolve_embedding_device(EmbeddingDevice("cpu")) == "cpu"


def test_resolve_embedding_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="Unknown embedding device"):
        resolve_embedding_device(EmbeddingDevice("tpu"))
