# `[embedding]` — embedding model

Settings for the embedding model that converts text to vectors for semantic search.

```toml
[embedding]
provider = "local"             # "local" (fastembed) or "api" (OpenAI-compatible)
model = "all-MiniLM-L6-v2"     # model name
```text

## Provider selection

| Value | What it does |
|---|---|
| `"local"` | Embedded fastembed inference. No network, no API key. Downloads ONNX model on first run. |
| `"api"` | OpenAI-compatible HTTP API. Requires `api_key` (or env var). |

## Local provider settings

```toml
[embedding]
provider = "local"
model = "all-MiniLM-L6-v2"     # any fastembed-supported model
```text

Supported fastembed models include:

- `all-MiniLM-L6-v2` (default, 384 dimensions, ~90 MB)
- `all-mpnet-base-v2` (768 dimensions, higher quality)
- `BAAI/bge-small-en-v1.5` (384 dimensions)
- `BAAI/bge-large-en-v1.5` (1024 dimensions)
- `intfloat/multilingual-e5-large` (1024 dimensions, multilingual)

The dimension is auto-detected from the model. The local backend requires the `local-embed` (or aggregate `local`) build feature.

## API provider settings

```toml
[embedding]
provider = "api"
api_url = "https://api.openai.com/v1"
api_key = ""                   # or set LLM_MEM_EMBEDDING_API_KEY / OPENAI_API_KEY
model = "text-embedding-3-small"
```text

Common model choices:

| Model | Dimensions |
|---|---|
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| `text-embedding-ada-002` | 1536 |

> [!WARNING]
> **Dimension must match storage** —
>
> If you change the embedding model, the dimension changes. New memories use the new dimension; old memories keep their old vectors. Configure the corresponding `[vector_store.lancedb].embedding_dimension` (or VectorLite equivalent) to match.

## Tuning

```toml
batch_size = 64                # texts per API call
timeout_secs = 30              # per-call timeout
```text

`batch_size` is how many texts are bundled into a single embedding request. Larger batches are more efficient; smaller batches are more responsive.

## Combinations

You can mix LLM and embedding providers:

| LLM | Embedding | Use case |
|---|---|---|
| local | local | Default. Fully offline. |
| api | api | Cloud LLM + cloud embeddings. |
| api | local | Cloud LLM with offline embeddings (faster, cheaper). |
| local | api | Local LLM with cloud embeddings. |

The four combinations are auto-detected from the two `provider` settings.

## Next

- [`[vector_store]`](./vector-store-section.md) — where the vectors are stored
- [`[llm]`](./llm-section.md) — the LLM settings
