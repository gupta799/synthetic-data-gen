# Synthetic Data Gen

Generate finance-only router classification data with a local Ollama Gemma model, institution
personas, embedding diversity filters, and local/LangSmith/W&B observability.

This repo owns data generation only. The classifier repo consumes the generated JSONL files through
the `data-gen` Git submodule.

## Output Schema

Each generated row is compatible with `finance-router-classifier`:

```json
{
  "id": "...",
  "text": "Can you pull Apple's fiscal 2023 revenue from this filing excerpt?",
  "route": "metric_extraction",
  "source": "synthetic:ollama:gemma4:e2b",
  "company": "Apple Inc",
  "metadata": {}
}
```

Routes:

- `metric_extraction`
- `filing_summarization`
- `financial_qa`
- `financial_reasoning`
- `comparative_analysis`

## Project Layout

- `src/synthetic_data_gen/types/`: domain types, JSON payload types, records, and text helpers.
- `src/synthetic_data_gen/client.py`: Ollama/DeepAgents model boundary.
- `src/synthetic_data_gen/prompts.py`: baked generation prompt templates.
- `src/synthetic_data_gen/validation.py`: strict conversion from model JSON to classifier rows.
- `src/synthetic_data_gen/builder.py`: quota loop, embedding diversity gate, grouped split, artifacts.
- `src/synthetic_data_gen/observability.py`: local events plus optional W&B/LangSmith hooks.

## Setup

```bash
uv sync --python 3.12
ollama pull gemma4:e2b
```

## Build 10k Dataset

```bash
uv run synthetic-data-gen build \
  --out data/synthetic-10k \
  --train-size 8000 \
  --eval-size 2000 \
  --generator-model gemma4:e2b \
  --ollama-base-url http://localhost:11434 \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --embedding-device auto \
  --wandb-project finance-router-data-gen \
  --langsmith-project finance-router-data-gen
```

`--embedding-device auto` prefers CUDA, then MPS, then CPU. On RunPod, use
`--embedding-device cuda` if you want the local embedding diversity checks to fail loudly unless
CUDA is visible to PyTorch. Ollama acceleration is controlled by the Ollama server itself; on a GPU
RunPod image, Ollama should use CUDA when the server/runtime is configured correctly.

The build writes:

- `train.jsonl`
- `eval.jsonl`
- `summary.json`
- `rejected.jsonl`
- `generation_config.json`
- `generation_events.jsonl`

Generated data is ignored by git.

## Observability

Local observability is always on through `generation_events.jsonl`, `summary.json`, and
`rejected.jsonl`. The CLI also prints colored status logs for the generator backend, embedding
device, W&B/LangSmith status, seed counts, and final artifact paths.

External observability is opt-in:

- W&B logs generation counters, rejection counts, and the dataset artifact when `--wandb-project`
  is set.
- LangSmith tracing is enabled when `--langsmith-project` is set and your LangSmith environment is
  configured.

## Smoke Build

```bash
uv run synthetic-data-gen build \
  --train-size 50 \
  --eval-size 25 \
  --out data/smoke \
  --generator-model gemma4:e2b
```

## Use With Classifier

From `finance-router-classifier`:

```bash
uv run finance-router train \
  --device mps \
  --train data-gen/data/synthetic-10k/train.jsonl \
  --eval data-gen/data/synthetic-10k/eval.jsonl \
  --batch-size 4 \
  --max-length 768 \
  --epochs 3 \
  --out-dir models/finance-router-synthetic-10k \
  --wandb-project finance-router-classifier \
  --wandb-run-name synthetic-10k-gemma4-e2b
```
