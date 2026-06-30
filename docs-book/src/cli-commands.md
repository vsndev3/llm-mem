# Single commands

Reference for every CLI subcommand. The same commands work inside the [REPL](./cli-repl.md) (without the `--` prefix).

The convention is `llm-mem --single <command> [flags]` or just `llm-mem <command> [flags]` from the shell.

## `upload`

Ingest a file with auto-chunking and processing.

```bash
llm-mem upload --file-path /path/to/doc.md --bank project-x
llm-mem upload --file-path /path/to/code.rs --context src
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--file-path <PATH>` | path | required | File to ingest. |
| `--bank <NAME>` | string | `default` | Target bank. |
| `--process-immediately` | bool | `true` | Start background processing now. |
| `--chunk-size <N>` | integer | config | Override `document_chunk_size`. |
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
llm-mem list --bank default --limit 20
llm-mem list --bank research --limit 50 --format jsonl > research.jsonl
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--bank <NAME>` | string | `default` | |
| `--limit <N>` | integer | `50` | |
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

## `dag-export`

Export the memory bank as an interactive DAG visualization — a self-contained HTML file you can open in any browser. Nodes are memories colored by abstraction layer; edges represent abstraction, semantic, and temporal relations.

```bash
# Default: up to 200 most important nodes
llm-mem dag-export --bank default --output graph.html

# Customize node count and filtering
llm-mem dag-export --bank research --output research-graph.html --max-nodes 300 --min-importance 0.5
llm-mem dag-export --bank default --output l3-above.html --min-layer 3

# Disable certain edge types for a cleaner view
llm-mem dag-export --bank default --output no-temp.html --no-temporal
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--bank <NAME>` | string | `default` | Memory bank to export. |
| `--output <PATH>` | path | required | Output HTML file path. |
| `--max-nodes <N>` | integer | `200` | Maximum nodes in the graph. |
| `--min-importance <F>` | float | `0.0` | Minimum importance score (0.0–1.0). |
| `--semantic` / `--no-semantic` | bool | `true` | Include semantic relation edges. |
| `--temporal` / `--no-temporal` | bool | `true` | Include temporal (`happened_after`) edges. |
| `--abstraction` / `--no-abstraction` | bool | `true` | Include abstraction pyramid edges. |
| `--min-layer <N>` | integer | `-1` | Minimum layer level to include. |
| `--max-layer <N>` | integer | `99` | Maximum layer level to include. |
| `--min-relation-strength <F>` | float | `0.0` | Minimum strength for semantic edges (0.0–1.0). |

### Layered scaling

The graph automatically adjusts to the node count:

| Total bank size | Strategy |
|---|---|
| ≤100 memories | Full graph: all layers, all edges |
| 100–500 | L1+ pyramid + high-importance L0 + strong edges |
| 500–2,000 | L2+ concepts + abstraction hierarchy only |
| 2,000–10,000 | L3+L4 wisdom/concepts only |
| >10,000 | Top-N by importance, pyramid only |

### Interactivity

Open the HTML file in a browser and use:

- **Scroll/pinch** to zoom, **drag** to pan
- **Drag nodes** to reposition them
- **Hover** over a node for a content preview tooltip
- **Click** a node to open the detail panel with full metadata and edge list
- **Toggle layers** and **edge types** in the sidebar
- **Search** bar to filter nodes by label, ID, or content
- **Sliders** to adjust link distance and charge strength
- Keyboard shortcuts: `r` = reset view, `f` = fit to screen, `/` = focus search, `Esc` = close detail panel

## Next

- [Database management](./cli-database.md)
- [Tools overview](../tools-overview.md) — for the MCP equivalents
