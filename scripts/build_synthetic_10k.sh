#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-data/synthetic-10k}"
TRAIN_SIZE="${TRAIN_SIZE:-8000}"
EVAL_SIZE="${EVAL_SIZE:-2000}"
GENERATOR_MODEL="${GENERATOR_MODEL:-gemma4:e2b}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-finance-router-data-gen}"
LANGSMITH_PROJECT="${LANGSMITH_PROJECT:-finance-router-data-gen}"

uv sync --python 3.12

uv run synthetic-data-gen build \
  --out "${OUT}" \
  --train-size "${TRAIN_SIZE}" \
  --eval-size "${EVAL_SIZE}" \
  --generator-model "${GENERATOR_MODEL}" \
  --ollama-base-url "${OLLAMA_BASE_URL}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --embedding-device "${EMBEDDING_DEVICE}" \
  --wandb-project "${WANDB_PROJECT}" \
  --langsmith-project "${LANGSMITH_PROJECT}"
