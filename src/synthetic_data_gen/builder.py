"""Synthetic finance router dataset builder."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from tqdm.auto import tqdm

from synthetic_data_gen.client import (
    DeepAgentOllamaClient,
    GenerationClient,
    assert_ollama_model_available,
)
from synthetic_data_gen.embeddings import DiversityIndex, Embedder, SentenceTransformerEmbedder
from synthetic_data_gen.labels import LABELS
from synthetic_data_gen.observability import EventLogger, WandbLogger, configure_langsmith
from synthetic_data_gen.parser import parse_json_objects
from synthetic_data_gen.personas import select_persona
from synthetic_data_gen.prompts import ROUTE_INSTRUCTIONS, build_generation_prompt
from synthetic_data_gen.seeds import load_seed_pool
from synthetic_data_gen.splitting import split_exact_by_route, validate_exact_splits
from synthetic_data_gen.types import (
    ArtifactName,
    ArtifactType,
    EmbeddingModelName,
    EventName,
    GeneratedText,
    GroupKey,
    JsonObject,
    MetricPayload,
    ModelName,
    OllamaBaseUrl,
    ProjectName,
    RejectedCandidate,
    RejectionReason,
    RouteName,
    RouterExample,
    RunName,
    SeedRecord,
    stable_id,
    write_jsonl,
    write_rejected_jsonl,
)
from synthetic_data_gen.validation import validate_generated_object


@dataclass(frozen=True)
class BuildConfig:
    out_dir: Path = Path("data/synthetic-10k")
    train_size: int = 8000
    eval_size: int = 2000
    seed: int = 7
    generator_model: ModelName = ModelName("gemma4:e2b")
    ollama_base_url: OllamaBaseUrl = OllamaBaseUrl("http://localhost:11434")
    embedding_model: EmbeddingModelName = EmbeddingModelName("BAAI/bge-small-en-v1.5")
    generation_batch_size: int = 4
    similarity_threshold: float = 0.88
    max_per_seed_group: int = 5
    max_sujet_rows: int | None = None
    max_attempts_multiplier: int = 12
    temperature: float = 0.8
    wandb_project: ProjectName | None = None
    wandb_run_name: RunName | None = None
    langsmith_project: ProjectName | None = None
    skip_model_check: bool = False


def make_generation_config(config: BuildConfig) -> JsonObject:
    payload = asdict(config)
    payload["out_dir"] = str(config.out_dir)
    payload["labels"] = list(LABELS)
    payload["route_instructions"] = ROUTE_INSTRUCTIONS
    return payload


def should_stop(counts: Counter[RouteName], per_route_goal: int) -> bool:
    return all(counts[route] >= per_route_goal for route in LABELS)


def build_summary(
    *,
    candidates: list[RouterExample],
    train: list[RouterExample],
    eval_rows: list[RouterExample],
    rejected: list[RejectedCandidate],
    config: BuildConfig,
    started: float,
    diversity: DiversityIndex,
    leakage: MetricPayload,
) -> JsonObject:
    return {
        "generator_model": config.generator_model,
        "embedding_model": config.embedding_model,
        "train_rows": len(train),
        "eval_rows": len(eval_rows),
        "total_rows": len(train) + len(eval_rows),
        "candidate_rows": len(candidates),
        "rejected_rows": len(rejected),
        "train_routes": dict(sorted(Counter(row.route for row in train).items())),
        "eval_routes": dict(sorted(Counter(row.route for row in eval_rows).items())),
        "candidate_routes": dict(sorted(Counter(row.route for row in candidates).items())),
        "personas": dict(
            sorted(Counter(row.metadata.get("persona") for row in candidates).items())
        ),
        "seed_sources": dict(
            sorted(Counter(row.metadata.get("seed_source") for row in candidates).items())
        ),
        "rejection_reasons": dict(sorted(Counter(row.reason for row in rejected).items())),
        "embedding_similarity": diversity.similarity_stats,
        "elapsed_seconds": perf_counter() - started,
        **leakage,
    }


def build_dataset(
    config: BuildConfig,
    *,
    client: GenerationClient | None = None,
    embedder: Embedder | None = None,
    seeds: list[SeedRecord] | None = None,
) -> JsonObject:
    if config.train_size % len(LABELS) != 0 or config.eval_size % len(LABELS) != 0:
        raise ValueError("train_size and eval_size must be divisible by the number of labels.")
    if config.generation_batch_size < 1:
        raise ValueError("generation_batch_size must be at least 1.")
    if client is None and not config.skip_model_check:
        assert_ollama_model_available(config.generator_model, config.ollama_base_url)

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_langsmith(config.langsmith_project)
    event_logger = EventLogger(out_dir / "generation_events.jsonl")
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        run_name=config.wandb_run_name,
        config=make_generation_config(config),
    )
    started = perf_counter()

    try:
        if client is None:
            client = DeepAgentOllamaClient(
                model=config.generator_model,
                base_url=config.ollama_base_url,
                temperature=config.temperature,
            )
        if embedder is None:
            embedder = SentenceTransformerEmbedder(config.embedding_model)
        if seeds is None:
            seeds = load_seed_pool(max_sujet_rows=config.max_sujet_rows)

        rng = random.Random(config.seed)
        per_route_goal = (config.train_size + config.eval_size) // len(LABELS)
        eval_quota = config.eval_size // len(LABELS)
        per_route_candidate_goal = max(per_route_goal + eval_quota, int(per_route_goal * 1.25))
        max_attempts = per_route_candidate_goal * config.max_attempts_multiplier * len(LABELS)
        accepted: list[RouterExample] = []
        rejected: list[RejectedCandidate] = []
        seed_group_counts: dict[GroupKey, int] = defaultdict(int)
        route_counts: Counter[RouteName] = Counter()
        diversity = DiversityIndex(threshold=config.similarity_threshold)
        attempts = 0

        progress = tqdm(total=per_route_candidate_goal * len(LABELS), desc="accepted prompts")
        while not should_stop(route_counts, per_route_candidate_goal):
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "Generation could not satisfy quotas before max attempts. "
                    f"Accepted={dict(route_counts)} rejected={len(rejected)}."
                )

            underfilled_routes = [
                route for route in LABELS if route_counts[route] < per_route_candidate_goal
            ]
            route = underfilled_routes[rng.randrange(len(underfilled_routes))]
            seed_record = seeds[rng.randrange(len(seeds))]
            if seed_group_counts[seed_record.group_key] >= config.max_per_seed_group:
                continue
            persona = select_persona(rng)
            batch_id = stable_id(config.generator_model, route, seed_record.group_key, attempts)
            prompt = build_generation_prompt(
                route=route,
                persona=persona,
                seed=seed_record,
                batch_size=config.generation_batch_size,
            )
            event_logger.log(
                EventName("generation_batch_started"),
                batch_id=batch_id,
                route=route,
                persona=persona.name,
                seed_group=seed_record.group_key,
            )

            try:
                raw = client.generate(prompt)
                objects = parse_json_objects(raw)
            except Exception as exc:
                rejected.append(
                    RejectedCandidate(
                        reason=RejectionReason("generation_or_parse_error"),
                        raw_text=GeneratedText(str(exc)),
                        route=route,
                        persona=persona.name,
                        seed_group=seed_record.group_key,
                    )
                )
                event_logger.log(
                    EventName("generation_batch_rejected"),
                    batch_id=batch_id,
                    route=route,
                    reason="generation_or_parse_error",
                    error=str(exc),
                )
                continue

            batch_accepted = 0
            for obj in objects:
                if seed_group_counts[seed_record.group_key] >= config.max_per_seed_group:
                    rejected.append(
                        RejectedCandidate(
                            reason=RejectionReason("seed_group_cap"),
                            raw_text=GeneratedText(json.dumps(obj, ensure_ascii=False)),
                            route=route,
                            persona=persona.name,
                            seed_group=seed_record.group_key,
                        )
                    )
                    continue
                result = validate_generated_object(
                    obj,
                    expected_route=route,
                    seed=seed_record,
                    generator_model=config.generator_model,
                    embedding_model=config.embedding_model,
                    generation_batch_id=batch_id,
                )
                if isinstance(result, RejectedCandidate):
                    rejected.append(result)
                    continue
                vector = embedder.encode_one(result.text)
                accepted_by_diversity, nearest = diversity.accept(result.route, vector)
                if not accepted_by_diversity:
                    rejected.append(
                        RejectedCandidate(
                            reason=RejectionReason("low_diversity_embedding"),
                            raw_text=result.text,
                            route=result.route,
                            persona=persona.name,
                            seed_group=seed_record.group_key,
                            metadata={"nearest_similarity": nearest},
                        )
                    )
                    continue

                result.metadata["nearest_similarity"] = nearest
                accepted.append(result)
                seed_group_counts[seed_record.group_key] += 1
                route_counts[result.route] += 1
                batch_accepted += 1
                progress.update(1)
                if route_counts[result.route] >= per_route_candidate_goal:
                    break

            event_logger.log(
                EventName("generation_batch_finished"),
                batch_id=batch_id,
                route=route,
                accepted=batch_accepted,
                total_route_count=route_counts[route],
                rejected_total=len(rejected),
            )
            wandb_logger.log(
                {
                    "accepted_total": len(accepted),
                    "rejected_total": len(rejected),
                    **{f"accepted/{label}": route_counts[label] for label in LABELS},
                },
                step=attempts,
            )

        progress.close()
        train, eval_rows = split_exact_by_route(
            accepted,
            train_size=config.train_size,
            eval_size=config.eval_size,
            seed=config.seed,
        )
        leakage = validate_exact_splits(
            train=train,
            eval_rows=eval_rows,
            train_size=config.train_size,
            eval_size=config.eval_size,
        )
        write_jsonl(out_dir / "train.jsonl", train)
        write_jsonl(out_dir / "eval.jsonl", eval_rows)
        write_rejected_jsonl(out_dir / "rejected.jsonl", rejected)
        (out_dir / "generation_config.json").write_text(
            json.dumps(make_generation_config(config), indent=2),
            encoding="utf-8",
        )
        summary = build_summary(
            candidates=accepted,
            train=train,
            eval_rows=eval_rows,
            rejected=rejected,
            config=config,
            started=started,
            diversity=diversity,
            leakage=leakage,
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        event_logger.log(EventName("build_finished"), summary=summary)
        wandb_logger.log_artifact(
            ArtifactName("synthetic-finance-router-data"),
            ArtifactType("dataset"),
            out_dir,
        )
        wandb_logger.finish(summary=summary)
        return summary
    finally:
        event_logger.close()
