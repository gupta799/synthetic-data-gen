# Generation Observability

Synthetic data generation needs visibility into both quality and throughput. This repo records that
in three layers.

## Local Artifacts

Always written:

- `generation_events.jsonl`: append-only event stream for batches, rejects, and completion.
- `rejected.jsonl`: rejected candidate text and rejection reason.
- `summary.json`: final counts, distributions, rejection reasons, and embedding similarity stats.
- `generation_config.json`: reproducible build parameters.

## LangSmith

When `--langsmith-project` is provided, the builder sets LangSmith tracing environment variables
before constructing the DeepAgents/Ollama client. LangChain/DeepAgents calls are then traceable in
LangSmith with the configured project.

## Weights & Biases

When `--wandb-project` is provided, the builder logs:

- accepted counts by route
- total rejected count
- final dataset artifact
- final summary fields

Use `WANDB_MODE=offline` for air-gapped local runs that can sync later.
