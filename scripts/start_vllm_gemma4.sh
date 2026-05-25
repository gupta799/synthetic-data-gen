#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-google/gemma-4-E4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 0, \"audio\": 0}}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

if [[ -d "${WORKSPACE_DIR}" ]]; then
  export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/.cache/huggingface}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${WORKSPACE_DIR}/.cache/vllm}"
fi

exec vllm serve "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
