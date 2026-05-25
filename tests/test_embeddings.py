from __future__ import annotations

import numpy as np

from synthetic_data_gen.embeddings import DiversityIndex


def test_diversity_index_rejects_near_duplicate() -> None:
    index = DiversityIndex(threshold=0.9)
    vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    accepted, first_similarity = index.accept("metric_extraction", vector)
    duplicate, second_similarity = index.accept("metric_extraction", vector)

    assert accepted is True
    assert first_similarity == 0.0
    assert duplicate is False
    assert second_similarity == 1.0
