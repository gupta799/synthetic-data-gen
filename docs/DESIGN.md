# Design

The generator creates finance-only route-classification prompts. It does not generate answers.

## Flow

1. Load grounding seeds from FinanceBench and Sujet.
2. Select an underfilled route.
3. Select a finance institution persona with a deterministic seeded randomizer.
4. Build a persona-specific DeepAgent system prompt.
5. Ask the selected backend to generate schema-constrained JSON prompts:
   - `vllm` for GPU RunPod/OpenAI-compatible serving.
   - `ollama` for local Apple Silicon/MPS-backed serving.
6. Validate schema, route, and prompt quality.
7. Embed the prompt and ask the local diversity store whether it is acceptable.
8. Build an oversized candidate pool.
9. Split by group into exact balanced train/eval JSONL files.

## Harness Boundary

The generation harness is intentionally separate from the model backend.

- `deepagent`: production path. It uses DeepAgents with a persona-specific system prompt and a
  structured output tool schema.
- `direct`: smoke-test path. It sends the generation prompt directly to the backend schema API.

The builder does not know the details of either harness.

## Diversity Store

The local diversity store persists accepted prompt records to `diversity_store/records.jsonl` and
embedding matrices to `diversity_store/vectors.npz`. It rejects exact prompt duplicates and prompts
whose nearest same-route embedding is too similar. After enough examples exist for a route, it can
also reject prompts that are too far from that route's centroid.

## Why Grouped Splits

The same source company or filing context can create multiple valid prompts. Grouped splitting keeps
the same seed group out of both train and eval, which makes eval less leaky.

## Why Embeddings

String matching alone is not enough for synthetic data. Embedding similarity catches prompts that
are worded differently but ask nearly the same thing.
