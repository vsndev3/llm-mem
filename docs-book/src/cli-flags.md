# CLI flags

The `llm-mem-mcp` binary accepts these flags. They override both the config file and environment variables.

```bash
llm-mem-mcp [OPTIONS]
```text

## General

| Flag | Short | Type | Default | What it does |
|---|---|---|---|---|
| `--config <PATH>` | `-c` | path | auto-discover | Path to the config file. If not given, searches `./config.toml`, `~/.config/llm-mem/config.toml`, `/etc/llm-mem/config.toml`. |
| `--agent <ID>` | | string | none | Default `agent_id` for all tool calls. Overrides the per-tool `agent_id` parameter when not set. |
| `--proxy <URL>` | | string | none | Proxy URL for model downloads. Format: `http://host:port` or `http://user:pass@host:port`. Overrides `HTTPS_PROXY`. |
| `--banks-dir <PATH>` | | path | from config | Override `vector_store.banks_dir`. |
| `--generate-config` | | bool | false | Print a commented config template to stdout and exit. |

## LLM behavior

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--no-grammar` | bool | false | Disable grammar-constrained sampling for local LLM structured output. Grammar is enabled by default. |
| `--no-structured-output` | bool | false | Disable structured output mode for API LLM. Structured output is enabled by default. |
| `--request-format <FMT>` | string | `auto` | `auto`, `rig`, or `raw`. See `[llm]` request_format. Invalid values cause the server to exit. |

## Model caching

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--no-cache-model` | bool | false | Don't cache models in `~/.cache/llm-mem/models`. Caching is enabled by default. |
| `--cache-dir <PATH>` | path | `~/.cache/llm-mem/models` | Custom cache directory. |

## Examples

```bash
# Use a specific config file
llm-mem-mcp --config /etc/llm-mem/prod.toml

# Run with raw HTTP (skip rig-core) against a self-hosted llama.cpp
llm-mem-mcp --request-format raw

# Force re-download the model (skip cache)
llm-mem-mcp --no-cache-model

# Pre-populate the cache, then run normally
llm-mem-mcp --cache-dir /mnt/fast/models

# Use a different banks directory
llm-mem-mcp --banks-dir /mnt/data/banks

# Print the config template
llm-mem-mcp --generate-config > config.toml
```text

## Standalone CLI

The `llm-mem` (no `-mcp` suffix) binary has a different flag set. It has subcommands like `search`, `list`, `stats`, `db`, etc. See [CLI overview](./cli-overview.md).

## Next

- [Environment variables](./env-vars.md)
- [Tools overview](./tools-overview.md)
