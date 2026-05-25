# Design

The generator creates finance-only route-classification prompts. It does not generate answers.

## Flow

1. Load grounding seeds from FinanceBench and Sujet.
2. Select an underfilled route.
3. Select a finance institution persona with a deterministic seeded randomizer.
4. Ask local Gemma through Ollama and DeepAgents to generate strict JSON prompts.
5. Validate schema, route, and prompt quality.
6. Embed the prompt and reject low-diversity candidates within the same route.
7. Build an oversized candidate pool.
8. Split by group into exact balanced train/eval JSONL files.

## Why Grouped Splits

The same source company or filing context can create multiple valid prompts. Grouped splitting keeps
the same seed group out of both train and eval, which makes eval less leaky.

## Why Embeddings

String matching alone is not enough for synthetic data. Embedding similarity catches prompts that
are worded differently but ask nearly the same thing.
