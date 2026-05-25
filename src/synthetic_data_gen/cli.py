"""CLI for synthetic finance router data generation."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from synthetic_data_gen.backends import VLLM_BACKEND, openai_api_key_from_env
from synthetic_data_gen.builder import BuildConfig, build_dataset
from synthetic_data_gen.harnesses import DEEPAGENT_HARNESS
from synthetic_data_gen.types import (
    CliArg,
    EmbeddingDevice,
    EmbeddingModelName,
    GenerationHarnessName,
    GeneratorBackend,
    JsonObject,
    ModelName,
    OllamaBaseUrl,
    OpenAIApiKey,
    OpenAIBaseUrl,
    ProjectName,
    RunName,
)


def print_json(payload: JsonObject) -> None:
    print(json.dumps(payload, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    config = BuildConfig(
        out_dir=Path(args.out),
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        generation_harness=GenerationHarnessName(args.generation_harness),
        generator_backend=GeneratorBackend(args.generator_backend),
        generator_model=ModelName(args.generator_model),
        openai_base_url=OpenAIBaseUrl(args.openai_base_url),
        openai_api_key=(
            OpenAIApiKey(args.openai_api_key) if args.openai_api_key else openai_api_key_from_env()
        ),
        ollama_base_url=OllamaBaseUrl(args.ollama_base_url),
        embedding_model=EmbeddingModelName(args.embedding_model),
        embedding_device=EmbeddingDevice(args.embedding_device),
        generation_batch_size=args.generation_batch_size,
        similarity_threshold=args.similarity_threshold,
        max_per_seed_group=args.max_per_seed_group,
        max_sujet_rows=args.max_sujet_rows,
        max_attempts_multiplier=args.max_attempts_multiplier,
        temperature=args.temperature,
        wandb_project=ProjectName(args.wandb_project) if args.wandb_project else None,
        wandb_run_name=RunName(args.wandb_run_name) if args.wandb_run_name else None,
        langsmith_project=ProjectName(args.langsmith_project) if args.langsmith_project else None,
        skip_model_check=args.skip_model_check,
    )
    print_json(build_dataset(config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthetic-data-gen",
        description=(
            "Generate synthetic finance router datasets with vLLM or Ollama model servers, "
            "schema output, embeddings, and observability."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build synthetic train/eval JSONL data.")
    build.add_argument("--out", default="data/synthetic-10k")
    build.add_argument("--train-size", type=int, default=8000)
    build.add_argument("--eval-size", type=int, default=2000)
    build.add_argument("--seed", type=int, default=7)
    build.add_argument(
        "--generation-harness",
        choices=("deepagent", "direct"),
        default=DEEPAGENT_HARNESS,
        help="Use DeepAgents for generation or direct schema calls for smoke tests.",
    )
    build.add_argument("--generator-backend", choices=("vllm", "ollama"), default=VLLM_BACKEND)
    build.add_argument("--generator-model", default="google/gemma-4-E4B-it")
    build.add_argument(
        "--vllm-base-url",
        dest="openai_base_url",
        default=os.getenv("VLLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL used when --generator-backend vllm.",
    )
    build.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY"),
    )
    build.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    build.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    build.add_argument(
        "--embedding-device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device for local embedding diversity checks. Auto prefers CUDA, then MPS, then CPU.",
    )
    build.add_argument("--generation-batch-size", type=int, default=1)
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


def main(argv: Sequence[CliArg] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args([str(arg) for arg in argv] if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
