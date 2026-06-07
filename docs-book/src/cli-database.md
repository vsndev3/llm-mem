# Database management

The `llm-mem db` subcommand handles bank-level operations: export, import, merge, check, fix, rename.

## `db export`

Export a bank to a portable `.db` file. The exported bank is ready for continued use — it's a self-contained copy.

```bash
llm-mem db export --bank default --output backups/default.db
llm-mem db export --bank research --output research-portable.db --include-sessions
```

| Flag | Type | Description |
|---|---|---|
| `--bank <NAME>` | string | Bank to export. Default: `default`. |
| `--output <PATH>` | path | Output file or directory. |
| `--include-sessions` | bool | Also copy the `.sessions.db` alongside. |

## `db export-jsonl`

Export a bank to a **streaming JSONL text file**. This format is backend-independent — a JSONL export from VectorLite can be imported into LanceDB and vice versa.

```bash
llm-mem db export-jsonl --bank default --output default.jsonl
llm-mem db export-jsonl --bank research --output research.jsonl --include-sessions
```

The JSONL file has:

- An **envelope header** with format version, app version, embedding dimension, and config snapshot
- One **memory object per line** in the body
- A **checksum footer**

This makes it the safest format for **archival** and **cross-backend migration**.

## `db import`

Import memories from a JSONL file into a bank.

```bash
llm-mem db import --bank default --input default.jsonl
llm-mem db import --bank default --input default.jsonl --dry-run    # preview only
llm-mem db import --bank default --input default.jsonl --strip-embeddings
```

| Flag | Type | Description |
|---|---|---|
| `--bank <NAME>` | string | Target bank. |
| `--input <PATH>` | path | The JSONL file. |
| `--strip-embeddings` | bool | Don't import the embedding vectors; re-embed on import. Use when the new model has a different dimension. |
| `--dry-run` | bool | Show what would be imported without modifying anything. |

The importer auto-handles dimension mismatches: if the JSONL has dimension 384 but your current model is 768, it'll strip the old vectors and mark memories for re-embedding.

## `db merge`

Merge one or more source banks (or `.db` files) into a target bank.

```bash
llm-mem db merge --source research --source notes --into combined --on-duplicate keep-newest
llm-mem db merge --source old-backup.db --into default --on-duplicate keep-first
llm-mem db merge --source a.lance --source b.lance --into merged --dry-run
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--source <NAME_OR_PATH>` | string[] | required | Repeatable. Bank names or `.db` file paths. |
| `--into <NAME>` | string | required | Target bank name (created if missing). |
| `--on-duplicate <STRATEGY>` | string | `keep-newest` | `keep-newest`, `keep-first`, or `keep-all`. |
| `--dry-run` | bool | false | Show what would be merged. |

Duplicate detection uses content hash.

## `db check`

Scan for consistency issues.

```bash
llm-mem db check --bank default
llm-mem db check --all
llm-mem db check --bank research --verbose
llm-mem db check --file external.db
```

| Flag | Type | Description |
|---|---|---|
| `--bank <NAME>` | string | Specific bank. |
| `--file <PATH>` | path | Check an external `.db` file. |
| `--all` | bool | Check every bank. |
| `--verbose` | bool | Detailed issue info. |

Detects:

- Orphaned abstractions (L1+ with no L0 source)
- Stale state (sessions stuck in `processing` past their timeout)
- Missing embeddings
- Hash mismatches
- Unreferenced forgotten memories
- Duplicate content
- Invalid layer structure

## `db fix`

Repair detected issues.

```bash
llm-mem db fix --bank default --dry-run                    # preview
llm-mem db fix --bank default                              # fix all
llm-mem db fix --bank default --fix orphaned-abstractions  # fix specific types
llm-mem db fix --bank default --no-backup                  # skip auto-backup
llm-mem db fix --bank default --purge                      # hard-delete unreferenced Forgotten
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--bank <NAME>` | string | `default` | |
| `--fix <TYPE>` | string[] | (all) | Specific issue types. Valid: `orphaned-abstractions`, `stale-states`, `missing-embeddings`, `hash-mismatches`, `unreferenced-forgotten`, `duplicate-content`, `invalid-layer-structure`. |
| `--dry-run` | bool | false | Show what would be fixed. |
| `--no-backup` | bool | false | Skip automatic backup before fixing. |
| `--purge` | bool | false | Hard-delete unreferenced Forgotten memories (vs. just marking them). |

> [!WARNING]
> `db fix` is destructive for unreferenced-forgotten memories (with `--purge`) and for invalid-layer-structure. Always run with `--dry-run` first. The default behavior (without `--no-backup`) is to take a backup automatically.

## `db rename`

Atomic rename of a bank. Moves both the memory database and the session database.

```bash
llm-mem db rename --old-name foo --new-name bar
```

## Common workflows

### Migrate a bank to a new machine

```bash
# Source
llm-mem db export --bank default --output default.db --include-sessions
# scp default.db new-host:

# Destination
llm-mem db export --bank default --output /tmp/backup.db  # safety backup
# (stop the server, place default.db in the banks_dir, restart)
```

### Cross-backend migration (e.g. VectorLite → LanceDB)

```bash
# Export as backend-independent JSONL
llm-mem db export-jsonl --bank old --output migration.jsonl

# On the new system with LanceDB configured:
llm-mem db import --bank new --input migration.jsonl
```

### Recover from corruption

```bash
llm-mem db check --all --verbose                # identify
llm-mem db fix --bank default --dry-run         # preview
llm-mem db fix --bank default                   # repair (auto-backup)
```

If `db fix` can't repair, restore from a backup created via [`backup_bank`](../tools-banks.md#backup_bank) (MCP) or `db export`.

## Next

- [Banks & backups](../tools-banks.md)
- [Operations](../data-directory.md)
