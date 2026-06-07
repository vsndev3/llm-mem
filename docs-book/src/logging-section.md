# `[logging]` — log output

File-based logging with size-based rotation. Disabled by default; logs go to stderr.

```toml
[logging]
enabled = false
log_directory = "llm-mem-data/logs"
level = "info"
max_size_mb = 1
max_files = 5
```text

| Field | Default | What it controls |
|---|---|---|
| `enabled` | `false` | If `true`, write to `llm-mem-mcp.log` in `log_directory` in addition to stderr. |
| `log_directory` | `"llm-mem-data/logs"` | Directory for the rotating log file. Created if missing. |
| `level` | `"info"` | Default level: `error`, `warn`, `info`, `debug`, `trace`. |
| `max_size_mb` | `1` | Rotate the log when it exceeds this size. |
| `max_files` | `5` | Keep this many rotated log files (e.g. `llm-mem-mcp.log.1`, `.2`, etc.). |

## Per-request verbosity

The `level` setting is the default, but the `RUST_LOG` env var (set in the MCP config's `env` block) overrides it per-process. It accepts the `tracing_subscriber` filter syntax:

```json
"env": {
  "RUST_LOG": "info,llama_cpp_2=warn,llm_mem::search=debug"
}
```text

This sets the global default to `info`, silences `llama_cpp_2` to `warn`, and turns on `debug` for the search module.

## Stderr vs file

When `logging.enabled = true`:

- The MCP JSON-RPC traffic goes to **stdout** (untouched — required by the protocol)
- All `tracing` events go to **stderr** (always) **and** to the rotating log file (when enabled)

This means the MCP client can talk to the server without log lines contaminating the protocol stream.

## Examples

Quiet (errors only):

```toml
[logging]
enabled = true
level = "error"
```text

Verbose debugging:

```toml
[logging]
enabled = true
level = "debug"
```text

Larger log files, more history:

```toml
[logging]
enabled = true
max_size_mb = 50
max_files = 10
```text

## Next

- [Logging & debugging](./logging-debugging.md) — runtime tips, manual model download
- [Environment variables](./env-vars.md)
