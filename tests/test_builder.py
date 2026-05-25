from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from synthetic_data_gen.builder import BuildConfig, build_dataset
from synthetic_data_gen.labels import LABELS
from synthetic_data_gen.schema import SeedRecord


class FakeClient:
    model = "gemma4:e2b"

    def __init__(self) -> None:
        self.count = 0

    def generate(self, prompt: str) -> str:
        self.count += 1
        route = re.search(r"Target route: ([a-z_]+)", prompt).group(1)  # type: ignore[union-attr]
        persona = re.search(r"- role: ([^\n]+)", prompt).group(1)  # type: ignore[union-attr]
        institution = re.search(r"- institution type: ([^\n]+)", prompt).group(1)  # type: ignore[union-attr]
        company = re.search(r"- company: ([^\n]+)", prompt).group(1)  # type: ignore[union-attr]
        rows = [
            {
                "text": (
                    f"Please review {company} filing details for "
                    f"finance request number {self.count}-{index}."
                ),
                "route": route,
                "company": company,
                "persona": persona,
                "institution_type": institution,
                "metadata": {},
            }
            for index in range(2)
        ]
        return json.dumps(rows)


class FakeEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(64, dtype=np.float32)
        vector[sum(text.encode("utf-8")) % len(vector)] = 1.0
        return vector


def test_build_dataset_with_fake_client_writes_exact_outputs(tmp_path: Path) -> None:
    seeds = [
        SeedRecord(
            source="unit",
            group_key=f"unit:seed:{index}",
            context=f"Revenue and margin context for company {index}.",
            company=f"Company {index} Inc",
        )
        for index in range(40)
    ]
    summary = build_dataset(
        BuildConfig(
            out_dir=str(tmp_path),
            train_size=10,
            eval_size=5,
            generation_batch_size=2,
            similarity_threshold=1.1,
            max_per_seed_group=1,
            skip_model_check=True,
        ),
        client=FakeClient(),
        embedder=FakeEmbedder(),
        seeds=seeds,
    )

    assert summary["train_rows"] == 10
    assert summary["eval_rows"] == 5
    assert summary["group_leakage_count"] == 0
    assert set(summary["train_routes"]) == set(LABELS)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "eval.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "rejected.jsonl").exists()
    assert (tmp_path / "generation_events.jsonl").exists()
