# Data directory layout

Everything llm-mem writes to disk lives under a few well-known directories. Understanding the layout helps with backups, migrations, and debugging disk space issues.

## Top-level layout

After first run, you'll see something like this in the working directory (or wherever you pointed `LLM_MEM_MODELS_DIR` / `vector_store.banks_dir`):

```text
./llm-mem-data/
├── banks/                      # one .lance/ per memory bank
│   ├── default.lance/          # the default bank
│   ├── default.sessions.db     # document upload sessions for default
│   ├── research.lance/
│   └── research.sessions.db
├── models/                     # GGUF language model + ONNX embedding model
│   ├── gemma-4-E2B-it-Q8_0.gguf
│   └── (other model files)
└── logs/                       # when logging.enabled = true
    └── llm-mem-mcp.log
```

Plus the model cache (separate from `llm-mem-data`):

```text
~/.cache/llm-mem/models/         # downloaded HF/fastembed files (when cache_model = true)
```

## What's in a `.lance/` directory?

Each bank is a single LanceDB table. The directory is opaque to the user (it's managed by the LanceDB library) but contains:

- A `memories.lance/` subdirectory with the actual vector data
- `_versions/` for transaction history
- `_latest.manifest` and similar bookkeeping files

**Don't edit files inside a `.lance/` directory manually.** Use the `db export` / `db import` / `db check` / `db fix` tools to manipulate bank data safely.

## What's in `.sessions.db`?

A small SQLite file tracking document upload sessions:

- `session_id`
- `file_name`, `file_size`, `mime_type`
- `status` (`uploading`, `processing`, `completed`, `failed`, `cancelled`)
- Per-part metadata: `part_index`, `md5sum`, `bytes_received`
- MD5 of the assembled file
- Bank name

The sessions database is what enables **resumable uploads**: if the server crashes mid-upload, the next startup re-attaches to the incomplete session and resumes from the last received part.

## The default bank

The `default` bank is created automatically on first run, even if you never explicitly call `create_memory_bank`. To pre-create it manually:

```bash
mkdir -p /var/lib/llm-mem/banks
# (the server populates the directory on first write)
```

## Moving the data directory

To relocate the entire data tree:

1. Stop the server
2. Move the directory: `mv ./llm-mem-data /new/path/llm-mem-data`
3. Update the config or set env vars:
   - `LLM_MEM_MODELS_DIR=/new/path/llm-mem-data/models`
   - `vector_store.banks_dir=/new/path/llm-mem-data/banks`
4. Restart

Or just symlink:

```bash
ln -s /mnt/storage/llm-mem-data ./llm-mem-data
```

## Disk space

Rough size estimates:

| Component | Per memory | Notes |
|---|---|---|
| Vector (384-dim f32) | ~1.5 KB | + small overhead |
| Metadata | ~1 KB | depends on usage |
| Sessions DB | small | a few MB total |

100k memories ≈ 250 MB. 1M memories ≈ 2.5 GB.

If you need to free space:

- `cleanup_resources` (target: `models`) — reclaims the model cache
- `cleanup_resources` (target: `banks`) — deletes a specific bank
- `db export` + delete — manual cleanup with a backup
- Compact the bank: `db fix --bank default` may help if there's significant Forgotten memory accumulation

## Permissions

The server process needs read/write access to:

- `models_dir` (default `llm-mem-data/models/`)
- `banks_dir` (default `llm-mem-data/banks/`)
- `log_directory` (default `llm-mem-data/logs/`)
- The cache directory (`~/.cache/llm-mem/models/`, or whatever `cache_dir` is set to)

When running as a subprocess of an MCP client, it inherits the client's user. Make sure the working directory is writable.

## Next

- [Memory banks](./banks.md) — using banks in practice
- [Backups & restore](./backups.md)
