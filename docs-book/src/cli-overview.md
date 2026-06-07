# Standalone CLI

The `llm-mem` binary (no `-mcp` suffix) is a standalone command-line interface for working with memory banks directly, without an MCP client. It reads the same config file, talks to the same backend, and operates on the same banks as the MCP server.

```bash
llm-mem [OPTIONS] [COMMAND]
```

## When to use it

- **Local testing and debugging** — try a query without spinning up an MCP client
- **Scripts and automation** — pipe search results into other tools
- **Database administration** — exports, merges, consistency checks, renames
- **Configuration management** — generate a template config, run health checks
- **Interactive exploration** — drop into the REPL for ad-hoc work

## Modes

| Flag | Behavior |
|---|---|
| (no args) | Start the [interactive REPL](./cli-repl.md) |
| `--single` | Run a single [command](./cli-commands.md) and exit. Useful for scripts. |
| `--batch <file>` | Execute commands from a file (one per line). See [Batch mode](#batch-mode). |
| `--repl` | Force REPL mode even if subcommand is present. |
| `--config <path>` | Config file path (auto-discovered if not given). |
| `--banks-dir <path>` | Override `vector_store.banks_dir`. |

## All commands at a glance

### Document handling

- `upload` — ingest a file
- `begin-upload` / `upload-part` / `process-document` — multi-part upload for large files
- `doc-status` / `list-sessions` — check upload progress

### Browsing

- `list` — list memories with filters
- `show` — full details of a memory by ID
- `search` — text or semantic search
- `stats` / `layer-stats` / `layer-tree` — bank and pyramid statistics
- `timeline` / `timeline-graph` / `context-resume` — chronological views

### Configuration

- `generate-config` — write a config template
- `health-check` — run static + optional live config check
- `system-status` — backend health

### Database admin

- `db export` / `db import` — backup and restore
- `db merge` — merge banks
- `db check` / `db fix` — integrity check and repair
- `db rename` — atomic rename
- `db export-jsonl` / `db import` (from JSONL) — backend-independent format

### Misc

- `list-banks` — list all banks
- `list-devices` — show available embedding/LLM devices
- `export` — bank export to JSON
- `metrics` — query and cache metrics
- `clear-backoff` — reset abstraction retry timers
- `viz` — real-time TUI for processing

### REPL-only

Inside the REPL, you also get `use <bank>` (switch active bank) and `savelog` (dump log buffer). Output can be formatted as `table`, `detail`, `json`, `jsonl`, or `csv`.

## Examples

```bash
# Interactive REPL
llm-mem

# Single-command scripting
llm-mem --single search --query "JWT auth" --mode semantic --limit 5
llm-mem --single list --bank research --limit 20 --format json
llm-mem --single stats --bank default

# Database admin
llm-mem db check --all
llm-mem db fix --bank research --purge
llm-mem db export --bank default --output backups/default.db --include-sessions

# Health & config
llm-mem health-check --live
llm-mem generate-config --output /etc/llm-mem/config.toml
```

## Output formats

Most commands accept `--format` with one of:

| Value | Best for |
|---|---|
| `table` | Human reading (default for many commands) |
| `detail` | Single-memory views (`show`) |
| `json` | Programmatic consumption |
| `jsonl` | Streaming, line-by-line |
| `csv` | Spreadsheets / awk |

## Next

- [Interactive REPL](./cli-repl.md)
- [Single commands](./cli-commands.md)
- [Database management](./cli-database.md)
