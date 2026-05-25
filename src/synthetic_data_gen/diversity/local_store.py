"""LangChain/Chroma-backed local diversity store."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

from synthetic_data_gen.embeddings import Embedder
from synthetic_data_gen.models import DiversityDecision, DiversityPolicy, DiversityStoreRecord
from synthetic_data_gen.types import (
    DistanceScore,
    GeneratedText,
    Metadata,
    MetricPayload,
    RejectionReason,
    RouteName,
    RouterExample,
    stable_id,
)

if TYPE_CHECKING:
    from langchain_chroma import Chroma

SCORE_BACKEND = "chroma_cosine_distance"


class LocalDiversityStore:
    def __init__(
        self,
        path: Path,
        policy: DiversityPolicy,
        embedder: Embedder,
        *,
        reset: bool = True,
    ) -> None:
        self.path = path
        self.policy = policy
        self.path.mkdir(parents=True, exist_ok=True)
        self.records_path = self.path / "records.jsonl"
        self.summary_path = self.path / "summary.json"
        self.chroma_path = self.path / "chroma"
        if reset:
            self._reset_files()
        self._route_counts: dict[RouteName, int] = {}
        self._prompt_hashes: set[str] = set()
        self._records_written = 0
        self.nearest_distances: list[float] = []
        self._records_handle = self.records_path.open("a", encoding="utf-8")
        self._store = _create_chroma_store(self.chroma_path, _EmbedderAdapter(embedder))

    def evaluate(
        self,
        *,
        route: RouteName,
        text: GeneratedText,
    ) -> DiversityDecision:
        prompt_hash = stable_id(text.lower())
        route_count = self._route_counts.get(route, 0)
        if prompt_hash in self._prompt_hashes:
            return DiversityDecision(
                accepted=False,
                reason=RejectionReason("duplicate_prompt"),
                nearest_distance=DistanceScore(0.0),
                route_count=route_count,
            )

        nearest_distance = self.nearest_distance(route=route, text=text)
        self.nearest_distances.append(float(nearest_distance))

        if route_count > 0 and nearest_distance <= self.policy.min_neighbor_distance:
            return DiversityDecision(
                accepted=False,
                reason=RejectionReason("low_diversity_embedding"),
                nearest_distance=nearest_distance,
                route_count=route_count,
            )
        if (
            route_count >= self.policy.min_route_examples_for_floor
            and nearest_distance >= self.policy.max_neighbor_distance
        ):
            return DiversityDecision(
                accepted=False,
                reason=RejectionReason("off_route_embedding"),
                nearest_distance=nearest_distance,
                route_count=route_count,
            )
        return DiversityDecision(
            accepted=True,
            reason=None,
            nearest_distance=nearest_distance,
            route_count=route_count,
        )

    def commit(
        self,
        *,
        row: RouterExample,
        decision: DiversityDecision,
    ) -> None:
        self._store.add_texts(
            texts=[str(row.text)],
            metadatas=[self._metadata_for(row)],
            ids=[str(row.id)],
        )
        self._prompt_hashes.add(stable_id(row.text.lower()))
        self._route_counts[row.route] = self._route_counts.get(row.route, 0) + 1
        record = DiversityStoreRecord(
            id=row.id,
            text=row.text,
            route=row.route,
            nearest_distance=decision.nearest_distance,
        )
        self._records_handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        self._records_handle.flush()
        self._records_written += 1
        if self._records_written % self.policy.flush_every == 0:
            self.flush()

    def nearest_distance(self, *, route: RouteName, text: GeneratedText) -> DistanceScore:
        if self._route_counts.get(route, 0) == 0:
            return DistanceScore(1.0)
        matches = self._store.similarity_search_with_score(
            str(text),
            k=1,
            filter={"route": str(route)},
        )
        if not matches:
            return DistanceScore(1.0)
        return DistanceScore(float(matches[0][1]))

    def flush(self) -> None:
        self.summary_path.write_text(
            json.dumps(
                {
                    "backend": SCORE_BACKEND,
                    "records": self._records_written,
                    "routes": {str(route): count for route, count in self._route_counts.items()},
                    "diversity_scores": self.diversity_stats,
                    "chroma_path": str(self.chroma_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def close(self) -> None:
        self.flush()
        self._records_handle.close()

    @property
    def diversity_stats(self) -> MetricPayload:
        return {
            "nearest_distance": _stats(self.nearest_distances),
            "policy": {
                "min_neighbor_distance": self.policy.min_neighbor_distance,
                "max_neighbor_distance": self.policy.max_neighbor_distance,
                "min_route_examples_for_floor": self.policy.min_route_examples_for_floor,
                "score_backend": SCORE_BACKEND,
            },
        }

    def _metadata_for(self, row: RouterExample) -> Metadata:
        return {
            "route": row.route,
            "source": row.source,
            "company": row.company or "",
            "group_key": row.group_key,
            "prompt_hash": stable_id(row.text.lower()),
        }

    def _reset_files(self) -> None:
        for path in (self.records_path, self.summary_path):
            if path.exists():
                path.unlink()
        if self.chroma_path.exists():
            shutil.rmtree(self.chroma_path)


class _EmbedderAdapter(Embeddings):
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.encode_one(GeneratedText(text)).astype(float).tolist()


def _create_chroma_store(path: Path, embedding_function: Embeddings) -> Chroma:
    from chromadb.config import Settings
    from langchain_chroma import Chroma

    return Chroma(
        collection_name="finance_router_diversity",
        embedding_function=embedding_function,
        persist_directory=str(path),
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=Settings(anonymized_telemetry=False),
    )


def _stats(values: list[float]) -> MetricPayload:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "count": float(len(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
        "p95": float(ordered[p95_index]),
    }
