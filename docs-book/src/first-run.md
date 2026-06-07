# First run

What happens when you start `llm-mem-mcp` for the very first time.

## Step 1: Config discovery

The server searches for a config file in this order:

1. `--config <path>` CLI flag (explicit)
2. `./config.toml` (current working directory)
3. `~/.config/llm-mem/config.toml` (XDG config dir)
4. `/etc/llm-mem/config.toml` (system-wide)

If none of these exist, the server uses **all defaults** and applies any environment variable overrides (see [Environment variables](./env-vars.md)).

> [!TIP]
> **Generate a template** —
>
> To see every available setting with comments, run:
>
>bash
    llm-mem-mcp --generate-config > config.toml
```text

Edit the file, then start the server with `llm-mem-mcp --config config.toml`.
```

## Step 2: Model download (local backend)

If you configured `provider = "local"` (the default), the server downloads:

- **Language model**: `gemma-4-E2B-it-Q8_0.gguf` (~2.5 GB) from HuggingFace
- **Embedding model**: `all-MiniLM-L6-v2` (~90 MB) from fastembed's model registry

Models are stored under the directory specified by `llm.models_dir` (default: `llm-mem-data/models/`) and cached separately under `~/.cache/llm-mem/models/` (configurable via `llm.cache_dir`).

Downloads are resumable — if interrupted, just restart the server. A smaller alternative model (`smollm2-1.7b-instruct-q4_k_m.gguf`, ~1 GB) is also available but requires a larger context size.

> [!NOTE]
> **Manual model download** —
>
> If you're behind a corporate proxy or on an air-gapped machine, see [Logging & debugging](./logging-debugging.md#manual-model-download) for instructions on downloading the models manually.

## Step 3: Bank creation

The default memory bank is created automatically on first run. The bank lives at:

```text
<llm.banks_dir>/default.lance/
<llm.banks_dir>/default.sessions.db
```

Default `llm.banks_dir`: `./llm-mem-data/banks/`. See [Data directory layout](./data-directory.md) for the full schema.

## Step 4: Service startup

Once models are loaded and the bank is ready, the service:

1. Starts the abstraction pipeline (background L0→L1, L1→L2, … workers)
2. Resumes any document upload sessions that were interrupted
3. Listens on stdio for MCP requests

You'll see log output on stderr (or in the log file if `logging.enabled = true`):

```text
INFO  Starting LLM Memory MCP Server
INFO  Configuration loaded from: None
INFO  Logging to directory: ./llm-mem-data/logs
INFO  Initialized LLM client (backend: Local)
INFO  Initialized memory bank manager
INFO  Default memory bank loaded
INFO  Abstraction pipeline initialization requested
INFO  MCP server initialized successfully
```

## Step 5: Connect a client

The server is now waiting on stdio for MCP traffic. Connect it to your AI client:

- [opencode](./opencode.md)
- [VS Code Copilot](./vscode-copilot.md)
- [Troubleshooting MCP](./troubleshooting-mcp.md) — if connection fails

## Stopping the server

The server handles `SIGINT` (Ctrl+C) and `SIGTERM` gracefully:

- Stops the MCP listener
- Stops the abstraction pipeline
- Flushes any in-flight writes
- Releases the llama.cpp backend

No special shutdown command is required.

## Next

[Quickstart](./quickstart.md) — run your first query in 5 minutes.
