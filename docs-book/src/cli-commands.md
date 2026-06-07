# Single commands

Reference for every CLI subcommand. The same commands work inside the [REPL](./cli-repl.md) (without the `--` prefix).

The convention is `llm-mem --single <command> [flags]` or just `llm-mem <command> [flags]` from the shell.

## `upload`

Ingest a file with auto-chunking and processing.

```bash
llm-mem upload --file-path /path/to/doc.md --bank project-x
llm-mem upload --file-path /path/to/code.rs --memory-type procedural --context src
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--file-path <PATH>` | path | required | File to ingest. |
| `--bank <NAME>` | string | `default` | Target bank. |
| `--process-immediately` | bool | `true` | Start background processing now. |
| `--chunk-size <N>` | integer | config | Override `document_chunk_size`. |
| `--memory-type <TYPE>` | string | `conversational` | |
| `--context <TAG>` | string[] | none | Repeatable. |
| `--format <FMT>` | enum | `table` | `table`, `json`, `jsonl`, `csv`. |

## `begin-upload` / `upload-part` / `process-document`

Three-step upload for very large files.

```bash
# Step 1: start the session
llm-mem begin-upload --file-name huge.pdf --total-size 104857600 --bank research

# Step 2: send parts (repeat for each part)
llm-mem upload-part --session-id <id> --part-index 0 --file-path part0.bin
llm-mem upload-part --session-id <id> --part-index 1 --file-path part1.bin
# ...

# Step 3: finalize
llm-mem process-document --session-id <id>
```

Use `--partial-closure` on `process-document` if you can't send all expected parts (e.g. upload was interrupted and you want to proceed with what you have).

## `doc-status` / `list-sessions`

Check upload progress.

```bash
llm-mem doc-status --session-id <id>
llm-mem list-sessions --bank research
```

## `list`

Browse memories with filters.

```bash
llm-mem list --bank default --limit 20 --memory-type factual
llm-mem list --bank research --limit 50 --format jsonl > research.jsonl
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--bank <NAME>` | string | `default` | |
| `--limit <N>` | integer | `50` | |
| `--memory-type <TYPE>` | string | none | Filter. |
| `--format <FMT>` | enum | `table` | |

## `show`

Full details of a memory.

```bash
llm-mem show --bank default --memory-id <uuid>
llm-mem show --memory-id <uuid> --format detail
```

## `search`

Text or semantic search.

```bash
llm-mem search --query "JWT auth" --mode semantic --limit 5
llm-mem search --query "TODO" --mode text --case-insensitive --show-scores
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--query <TEXT>` | string | required | |
| `--mode <MODE>` | enum | `text` | `text` (fast, no LLM) or `semantic` (embedding-based). |
| `--bank <NAME>` | string | `default` | |
| `--limit <N>` | integer | `10` | |
| `--case-insensitive` | bool | false | Text mode only. |
| `--show-scores` | bool | false | Text mode only. |
| `--threshold <F>` | float | config | Override `search_similarity_threshold`. |
| `--format <FMT>` | enum | `table` | |

## `stats` / `layer-stats` / `layer-tree`

Bank and pyramid statistics.

```bash
llm-mem stats --bank default
llm-mem layer-stats --bank default
llm-mem layer-tree --bank default --max-depth 5 --show-ids
```

`layer-tree` prints an ASCII tree of the pyramid.

## `timeline` / `timeline-graph` / `context-resume`

Chronological views.

```bash
llm-mem timeline --bank default --since 2d --granularity day
llm-mem timeline --bank default --start 2026-01-01T00:00:00Z --end 2026-02-01T00:00:00Z --granularity week
llm-mem timeline-graph --bank default --since 1w --include-semantic-edges
llm-mem context-resume --bank default --since 30d --decay-factor 4.0 --segments 5
```

See [Tools chronology](../tools-chronology.md) for the conceptual reference; the CLI exposes the same parameters.

## `generate-config`

Write a commented config template.

```bash
llm-mem generate-config --output /etc/llm-mem/config.toml
```

## `health-check`

Validate the config and (optionally) probe the live backend.

```bash
llm-mem health-check                   # static checks only
llm-mem health-check --live            # + tiny embed + completion
llm-mem health-check --live --embed-only
llm-mem health-check --live --llm-timeout-secs 60
```

This is the CLI equivalent of the [`health_check` MCP tool](../tools-introspection.md#health_check). Use it standalone when debugging config issues before connecting an MCP client.

## `system-status`

Backend health and token usage.

```bash
llm-mem system-status
llm-mem system-status --format json
```

## `list-banks`

Show all memory banks.

```bash
llm-mem list-banks
llm-mem list-banks --format json
```

## `list-devices`

Show available embedding/LLM devices (useful for GPU config).

```bash
llm-mem list-devices
```

## `export`

Export a bank to JSON (different from `db export` — this is for application-level use).

```bash
llm-mem export --bank default --output default.json --pretty
```

## `metrics`

Show accumulated query and cache metrics. Pass `--reset` to zero them after display.

```bash
llm-mem metrics
llm-mem metrics --reset
```

## `clear-backoff`

Reset abstraction retry timers. Use after fixing a transient backend issue.

```bash
llm-mem clear-backoff --bank default
llm-mem clear-backoff --bank default --layer 1
```

## `viz`

Launch a real-time TUI for visualizing document processing and abstraction pipeline activity.

```bash
llm-mem viz
```

## Next

- [Database management](./cli-database.md)
- [Tools overview](../tools-overview.md) — for the MCP equivalents
