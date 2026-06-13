# Logging & debugging

llm-mem uses the `tracing` crate for structured logging. By default, events go to **stderr** (so they don't pollute the MCP JSON-RPC stream on stdout). With `logging.enabled = true`, they also go to a rotating log file.

## Log levels

| Level | What you get | When to use |
|---|---|---|
| `error` | Only errors | Quiet production |
| `warn` | Errors + warnings | Default verbosity |
| `info` | Lifecycle events, request summaries | Normal operation |
| `debug` | Per-request detail | Investigating an issue |
| `trace` | Everything, including LLM prompts | Deep debugging |

## Setting the level

### In config

```toml
[logging]
enabled = true
level = "debug"
```

### At runtime via env

The `RUST_LOG` env var overrides the config. Set it in the MCP config's `env` block:

```json
{
  "mcp": {
    "llm-mem": {
      "type": "local",
      "command": ["/path/to/llm-mem-mcp"],
      "env": {
        "RUST_LOG": "info,llama_cpp_2=warn,llm_mem::search=debug"
      }
    }
  }
}
```

`RUST_LOG` accepts the `tracing_subscriber` filter syntax:

| Syntax | Meaning |
|---|---|
| `info` | Global default = `info` |
| `llama_cpp_2=warn` | Specific crate at a level |
| `llm_mem::search=debug` | Specific module |
| `trace,hyper=info,h2=info` | Global `trace` except quiet a few noisy crates |

## Reading the log file

With `logging.enabled = true`, the log file is at `<log_directory>/llm-mem-mcp.log`. Rotation produces `.1`, `.2`, ... up to `max_files`.

```bash
# Tail the live log
tail -f llm-mem-data/logs/llm-mem-mcp.log

# Look for errors
grep ERROR llm-mem-data/logs/llm-mem-mcp.log

# Inspect a specific request lifecycle
grep "memory_id=abc-123" llm-mem-data/logs/llm-mem-mcp.log
```

The format is:

```text
2026-02-15T10:23:45.123Z INFO llm_mem::mcp: store_memory called memory_id=
2026-02-15T10:23:46.456Z INFO llm_mem::memory: stored memory memory_id=abc-123
```

## What to look for

### Server didn't start

```text
ERROR config: llm.context_size (8192) is too small for memory.document_chunk_size (2000) and llm.max_tokens (4096)
```

→ Increase `llm.context_size` in your config (or decrease the chunk size / max tokens).

### API key missing

```text
ERROR config: LLM API key is not configured. Set it in config.toml under [llm].api_key, or via env var LLM_MEM_LLM_API_KEY.
```

→ Set `LLM_MEM_LLM_API_KEY` (or `OPENAI_API_KEY` as a fallback).

### Model download failed

```text
ERROR model_downloader: failed to download gemma-4-E2B-it-Q8_0.gguf: HTTPS error: 407 Proxy Authentication Required
```

→ Set `HTTPS_PROXY` with auth, or download manually (see below).

### LLM timeout

```text
ERROR llm: completion timed out after 120s
```

→ Increase `llm.llm_timeout_secs`, switch to a smaller model, or enable GPU.

## Manual model download

If you're behind a corporate proxy, on an air-gapped machine, or just want to pre-fetch the models, download them manually:

```bash
mkdir -p llm-mem-data/models
cd llm-mem-data/models

# Language model (~2.5 GB)
curl -L -o gemma-4-E2B-it-Q8_0.gguf \
  https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf

# Embedding model — let fastembed fetch this; if blocked, download from
# https://github.com/Anush008/fastembed-rs and place in the cache dir.
```

Then disable auto-download:

```toml
[llm]
auto_download = false
```

> [!NOTE]
> **Smaller alternative** —
>
> A smaller model `smollm2-1.7b-instruct-q4_k_m.gguf` (~1 GB) is also available but requires a larger context size (set `llm.context_size = 8192` or more).

## Performance debugging

To find slow operations, set `RUST_LOG=info,llm_mem::metrics=trace` and look for:

```text
INFO llm_mem::memory: search returned 50 results in 234ms
INFO llm_mem::memory: store took 1.2s (embedding=200ms, llm_extract=900ms, write=100ms)
```

If search is slow on a large bank:

- Try `search_memory` instead of `query_memory` (less work)
- Increase `search_similarity_threshold` to filter earlier
- Consider per-project banks to keep each one small

If the LLM is the bottleneck:

- Enable GPU
- Reduce `max_concurrent_requests` to 1 to avoid contention
- Use a smaller/faster model

## Inspecting banks

```bash
# CLI
llm-mem stats --bank default
llm-mem layer-stats --bank default
llm-mem db check --all --verbose

# MCP
{"tool": "system_status"}
{"tool": "check_consistency", "arguments": {"verbose": true}}
```

## Capturing logs from the REPL

The CLI REPL captures logs in a separate buffer down to TRACE level. Dump them with the `savelog` command:

```text
> savelog debug.log
Wrote 12345 lines to debug.log
```

This is useful for diagnosing issues without restarting the CLI with `RUST_LOG=trace`.

## Next

- [Configuration](../config-file.md) — `[logging]` section
- [Environment variables](../env-vars.md) — `RUST_LOG` syntax
- [Troubleshooting](../troubleshooting.md)
