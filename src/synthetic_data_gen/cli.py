"""CLI for synthetic finance router data generation."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from synthetic_data_gen.builder import BuildConfig, build_dataset


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    config = BuildConfig(
        out_dir=args.out,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        generator_model=args.generator_model,
        ollama_base_url=args.ollama_base_url,
        embedding_model=args.embedding_model,
        generation_batch_size=args.generation_batch_size,
        similarity_threshold=args.similarity_threshold,
        max_per_seed_group=args.max_per_seed_group,
        max_sujet_rows=args.max_sujet_rows,
        max_attempts_multiplier=args.max_attempts_multiplier,
        temperature=args.temperature,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        langsmith_project=args.langsmith_project,
        skip_model_check=args.skip_model_check,
    )
    print_json(build_dataset(config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthetic-data-gen",
        description=(
            "Generate synthetic finance router datasets with Ollama, embeddings, "
            "and observability."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build synthetic train/eval JSONL data.")
    build.add_argument("--out", default="data/synthetic-10k")
    build.add_argument("--train-size", type=int, default=8000)
    build.add_argument("--eval-size", type=int, default=2000)
    build.add_argument("--seed", type=int, default=7)
    build.add_argument("--generator-model", default="gemma4:e2b")
    build.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    build.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    build.add_argument("--generation-batch-size", type=int, default=4)
    build.add_argument("--similarity-threshold", type=float, default=0.88)
    build.add_argument("--max-per-seed-group", type=int, default=5)
    build.add_argument("--max-sujet-rows", type=int)
    build.add_argument("--max-attempts-multiplier", type=int, default=12)
    build.add_argument("--temperature", type=float, default=0.8)
    build.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT"))
    build.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME"))
    build.add_argument("--langsmith-project", default=os.getenv("LANGSMITH_PROJECT"))
    build.add_argument("--skip-model-check", action="store_true")
    build.set_defaults(func=cmd_build)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
