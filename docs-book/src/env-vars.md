# Environment variables

Any config field can be set via an environment variable. The variables are applied after the config file loads and before the server validates.

## LLM

| Variable | Maps to | Notes |
|---|---|---|
| `LLM_MEM_LLM_API_KEY` | `llm.api_key` | |
| `LLM_MEM_LLM_API_BASE_URL` | `llm.api_url` | |
| `LLM_MEM_LLM_MODEL` | `llm.model` | |
| `LLM_MEM_MODELS_DIR` | `llm.models_dir` | |
| `LLM_MEM_GPU_LAYERS` | `llm.gpu_layers` | u32 |
| `LLM_MEM_CONTEXT_SIZE` | `llm.context_size` | u32 |
| `LLM_MEM_TEMPERATURE` | `llm.temperature` | f32 |
| `LLM_MEM_MAX_TOKENS` | `llm.max_tokens` | u32 |
| `LLM_MEM_CPU_THREADS` | `llm.cpu_threads` | i32 |
| `LLM_MEM_MAX_CONCURRENT_REQUESTS` | `llm.max_concurrent_requests` | usize |

## Embedding

| Variable | Maps to | Notes |
|---|---|---|
| `LLM_MEM_EMBEDDING_API_KEY` | `embedding.api_key` | |
| `LLM_MEM_EMBEDDING_API_BASE_URL` | `embedding.api_url` | |
| `LLM_MEM_EMBEDDING_MODEL` | `embedding.model` | |

## Fallback

| Variable | Maps to | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `llm.api_key` and `embedding.api_key` | Only fills if both are still empty after the specific env vars are applied. Standard OpenAI convention. |

## Network

| Variable | Maps to | Notes |
|---|---|---|
| `HTTPS_PROXY` | outbound HTTPS proxy | Used for model downloads. Format: `http://host:port` or `http://user:pass@host:port` |
| `HTTP_PROXY` | outbound HTTP proxy | |
| `ALL_PROXY` / `all_proxy` | SOCKS proxy | |

The `llm.proxy_url` config (or `--proxy` CLI flag) overrides the env-var proxy.

## Logging

| Variable | Notes |
|---|---|
| `RUST_LOG` | tracing-subscriber filter syntax. Overrides the `logging.level` config. |
| `APPIMAGE_EXTRACT_AND_RUN` | AppImage-only. Run without FUSE by extracting to a tmpfs. |

## Abstraction pipeline

Unlike the tables above, these variables are **not** config-field overrides — they are read directly at startup. Invalid values are logged and fall back to the default.

| Variable | Default | Notes |
|---|---|---|
| `LLM_MEM_ABSTRACTION_DELAY_SECS` | `30` | Delay between abstraction pipeline cycles. |
| `LLM_MEM_ABSTRACTION_CONCURRENCY` | `3` | Concurrent abstraction tasks per cycle (`max_concurrent_tasks`). Raise for batch/benchmark workloads. |
| `LLM_MEM_BACKGROUND_LLM_CONCURRENCY` | `3` | Background LLM permits available to the abstraction pipeline. Raise alongside `LLM_MEM_ABSTRACTION_CONCURRENCY` so concurrency isn't throttled by the interactive-friendly default. |

## Usage in MCP config

Set environment variables in the MCP config's `env` block:

```json
{
  "mcp": {
    "llm-mem": {
      "type": "local",
      "command": ["/path/to/llm-mem-mcp"],
      "env": {
        "LLM_MEM_LLM_API_KEY": "sk-...",
        "LLM_MEM_MODELS_DIR": "/home/you/.cache/llm-mem/models",
        "RUST_LOG": "info,llm_mem::search=debug",
        "HTTPS_PROXY": "http://proxy.corp:8080"
      }
    }
  }
}
```

> [!TIP]
> For per-machine settings (proxy, models dir, API key), prefer the `env` block in the MCP config. For per-deployment settings (URL, model), prefer the TOML config file. Don't put secrets in the TOML if the file is committed.

## Override order

When the same field is set in multiple places, the priority is:

1. CLI flag (highest)
2. Specific env var (e.g. `LLM_MEM_LLM_API_KEY`)
3. Generic env var (e.g. `OPENAI_API_KEY`)
4. Config file value
5. Default (lowest)

So `--api-url` on the CLI beats `LLM_MEM_LLM_API_BASE_URL` which beats `[llm].api_url` in `config.toml` which beats the hardcoded default.

## Next

- [CLI flags](./cli-flags.md) — runtime overrides
- [`[llm]`](./llm-section.md), [`[embedding]`](./embedding-section.md)
