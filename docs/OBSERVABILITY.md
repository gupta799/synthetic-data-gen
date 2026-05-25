# Generation Observability

Synthetic data generation needs visibility into both quality and throughput. This repo records that
in three layers.

## Local Artifacts

Always written:

- `generation_events.jsonl`: append-only event stream for batches, rejects, and completion.
- `accepted_candidates.jsonl`: accepted candidates as soon as they pass validation and diversity.
- `rejected.jsonl`: rejected candidate text and rejection reason.
- `summary.json`: final counts, distributions, rejection reasons, and Chroma distance stats.
- `generation_config.json`: reproducible build parameters.
- `diversity_store/`: accepted prompt records, local Chroma index, and diversity-store summary.
- colored terminal logs: backend/device status, observability status, seed counts, and output paths.

## LangSmith

When `--langsmith-project` is provided and `LANGSMITH_API_KEY` or `LANGCHAIN_API_KEY` is set, the
builder enables LangSmith tracing before constructing the selected backend client. Without an API
key, tracing is skipped and the local event stream records `disabled:missing_api_key`.

## Weights & Biases

When `--wandb-project` is provided, the builder logs:

- accepted counts by route
- total rejected count
- final dataset artifact
- final summary fields

Use `WANDB_MODE=offline` for air-gapped local runs that can sync later.
