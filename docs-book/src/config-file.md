# Configuration

llm-mem has a single TOML config file (or no file at all — defaults work). The config has six top-level sections:

| Section | Purpose |
|---|---|
| `[llm]` | Language model (local llama.cpp or OpenAI-compatible API) |
| `[embedding]` | Embedding model (local fastembed or OpenAI-compatible API) |
| `[vector_store]` | Storage backend (LanceDB) and bank directory |
| `[memory]` | Memory behavior: thresholds, dedup, chunking, abstraction |
| `[server]` | HTTP server config (host/port) |
| `[logging]` | File logging with rotation |

## Generating a template

The fastest way to see every option is to generate a commented template:

```bash
llm-mem-mcp --generate-config > config.toml
```text

The result is a complete `config.toml` with every field commented out and explained.

## Loading order

On startup, the server:

1. Loads the config file (search order: `--config` flag → `./config.toml` → `~/.config/llm-mem/config.toml` → `/etc/llm-mem/config.toml`)
2. Applies environment variable overrides (see [Environment variables](./env-vars.md))
3. Applies CLI flag overrides (see [CLI flags](./cli-flags.md))
4. Validates the resulting config (refuses to start if invalid)

If no config file is found, **all defaults are used** and the server runs in fully-local mode with a Gemma 4 language model and an all-MiniLM embedding model.

## Minimal local config

This is the default. You don't actually need a config file, but if you want to customize `banks_dir`:

```toml
[vector_store]
banks_dir = "/var/lib/llm-mem/banks"
```text

## Minimal API config

```toml
[llm]
provider = "api"
api_key = "sk-..."              # or set LLM_MEM_LLM_API_KEY
api_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"

[embedding]
provider = "api"
api_key = "sk-..."              # or set LLM_MEM_EMBEDDING_API_KEY
api_url = "https://api.openai.com/v1"
model = "text-embedding-3-small"
```text

## Mixed config: API LLM, local embeddings

```toml
[llm]
provider = "api"
api_key = "sk-..."
model = "gpt-4o-mini"

[embedding]
provider = "local"              # keep embeddings local
model = "all-MiniLM-L6-v2"
```text

## Validation

The server validates the config at startup and refuses to start on errors. Common validations:

| Field | Constraint |
|---|---|
| `memory.similarity_threshold` | 0.0 ≤ x ≤ 1.0 |
| `memory.merge_threshold` | 0.0 ≤ x ≤ 1.0 |
| `memory.search_similarity_threshold` | 0.0 ≤ x ≤ 1.0 (if set) |
| `llm.temperature` | 0.0 ≤ x ≤ 2.0 |
| `llm.context_size` | > 0 |
| `llm.max_tokens` | > 0 |
| `llm.batch_max_tokens` | > 0 and ≤ `llm.max_tokens` |
| `embedding.batch_size` | > 0 |
| `memory.max_memories` | > 0 |
| `memory.max_content_length` | > 0 |
| `llm.context_size` | ≥ `memory.document_chunk_size / 2 + llm.max_tokens + 512` (local backend) |
| `llm.provider = "local"` | requires build feature `local-llm` or `local` |
| `embedding.provider = "local"` | requires build feature `local-embed` or `local` |
| API provider | requires `api_key` (or env var) |

If validation fails, the server prints a clear error message and exits.

## Section reference

- [`[llm]`](./llm-section.md) — language model settings
- [`[embedding]`](./embedding-section.md) — embedding model settings
- [`[vector_store]`](./vector-store-section.md) — storage backend
- [`[memory]`](./memory-section.md) — memory behavior
- [`[server]`](./server-section.md) — HTTP server
- [`[logging]`](./logging-section.md) — log output

## Environment variables

Any config field can be set via environment variable. See [Environment variables](./env-vars.md) for the full list.

## CLI flags

The `llm-mem-mcp` binary accepts flags that override the config at runtime. See [CLI flags](./cli-flags.md).

## Next

Read through the section references, or jump to the one you need.
