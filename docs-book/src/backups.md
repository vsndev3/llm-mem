# Backups & restore

Banks are self-contained on disk. Backing up is just copying files; restoring is putting them back. The higher-level tools add versioning, checksums, and metadata.

## Levels of backup

| Level | What it captures | When to use |
|---|---|---|
| **Raw file copy** | The `.lance/` directory as-is | Fast snapshots, machine-level backups |
| **`db export`** | A portable `.db` file with format metadata | Cross-version, cross-platform |
| **`db export-jsonl`** | A streaming text file with envelope + checksum | Long-term archival, cross-backend migration |
| **`backup_bank` (MCP)** | A versioned backup with SHA-256 + manifest | Programmatic, scripted |

For most users, **`db export-jsonl` is the safest long-term choice** because it's backend-independent and human-readable.

## Manual backup (file copy)

Stop the server, then:

```bash
# Backup everything
cp -a ./llm-mem-data /backup/llm-mem-data-$(date +%F)

# Or per-bank
cp -a ./llm-mem-data/banks/default.lance /backup/default-$(date +%F).lance
cp -a ./llm-mem-data/banks/default.sessions.db /backup/default-sessions-$(date +%F).db
```

**Stop the server first** — copying a LanceDB table mid-write can produce a corrupt backup. The `.sessions.db` SQLite file is durable, but a mid-transaction copy might lose the last few changes.

## `db export` (CLI)

Creates a portable, versioned `.db` file:

```bash
llm-mem db export --bank default --output /backup/default.db
llm-mem db export --bank research --output /backup/research.db --include-sessions
```

The `.db` file includes:

- A format version
- A snapshot of the bank's tables
- An optional `.sessions.db` (with `--include-sessions`)

To restore:

```bash
# Stop the server
# Replace the bank directory with the contents of the .db
llm-mem db import --bank default --input /backup/default.db
# Restart
```

`db import` is the inverse — it can read `.db` files exported with `db export` and load them into a target bank.

## `db export-jsonl` (CLI, recommended for archival)

Backend-independent streaming text format. The file looks like:

```jsonl
{"_format":"llm-mem-jsonl","_version":1,"_app_version":"0.1.0","_embedding_dim":384}
{"memory_id":"a1","content":"...","embedding":[...],"metadata":{...}}
{"memory_id":"b2","content":"...","embedding":[...],"metadata":{...}}
...
{"_checksum":"sha256:abc123...","_memory_count":1234}
```

To export:

```bash
llm-mem db export-jsonl --bank default --output /backup/default.jsonl
```

To import:

```bash
llm-mem db import --bank new --input /backup/default.jsonl
llm-mem db import --bank new --input /backup/default.jsonl --strip-embeddings   # if dim changed
llm-mem db import --bank new --input /backup/default.jsonl --dry-run            # preview
```

If the embedding dimension in the JSONL doesn't match the current model, the importer strips the old vectors and re-embeds on import. This makes it safe to migrate to a new embedding model.

## `backup_bank` (MCP tool)

For programmatic / AI-driven backups:

```jsonc
{
  "name": "default",
  "destination": "/home/you/llm-mem-backups"
}
```

Output:

```json
{
  "success": true,
  "data": {
    "backup_path": "/home/you/llm-mem-backups/default-2026-02-15-v3.db",
    "manifest": {
      "version": 3,
      "created_at": "2026-02-15T...",
      "memory_count": 1234,
      "sha256": "...",
      "size_bytes": 45678901
    }
  }
}
```

Versioning is automatic — the tool keeps incrementing `v1`, `v2`, `v3`, ... in the destination directory.

To restore: use `restore_bank` with `mode: "replace"` (requires `confirm: true`) or `mode: "merge"`.

## Backup retention

The `backup_bank` tool doesn't have a built-in retention policy. For long-running setups, prune old backups with a cron job:

```bash
# Keep last 30 days
find /home/you/llm-mem-backups -name "*.db" -mtime +30 -delete
```

For `db export` / `db export-jsonl` outputs, do the same — they're regular files.

## Testing your backups

A backup you haven't tested is a backup you don't have. Periodically:

1. Restore a backup to a temp directory
2. Start the server with `--banks-dir <temp dir>`
3. Run `llm-mem health-check --live`
4. Run a few queries to confirm memories are intact
5. Delete the temp directory

## Cross-backend migration

To move from VectorLite to LanceDB (or vice versa):

```bash
# Old system
llm-mem db export-jsonl --bank old-bank --output migration.jsonl

# New system (with new backend configured)
llm-mem db import --bank new-bank --input migration.jsonl
```

The JSONL format is the only one that works across backends. `.db` files are backend-specific.

## Next

- [Database management CLI](./cli-database.md)
- [GPU acceleration](./gpu-acceleration.md)
