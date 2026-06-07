# `[vector_store]` — storage backend

Settings for the embedded vector database (LanceDB by default) and the memory bank directory.

```toml
[vector_store]
banks_dir = "llm-mem-data/banks"   # where bank databases live
store_type = "lancedb"             # "lancedb" (default) or "vectorlite" (legacy)
collection_name = "llm-memories"   # default collection name
```

## Banks directory

`banks_dir` is the parent directory for all memory banks. Each bank gets its own subdirectory:

```text
<banks_dir>/
├── default.lance/             # the default bank
├── default.sessions.db        # document upload sessions for default
├── research.lance/            # another bank
└── research.sessions.db
```

To use a different location:

```toml
[vector_store]
banks_dir = "/var/lib/llm-mem/banks"
```

The directory is created automatically on first run if it doesn't exist.

## LanceDB settings (default)

```toml
[vector_store]
store_type = "lancedb"

[vector_store.lancedb]
table_name = "memories"
database_path = "./lancedb"
embedding_dimension = 384       # MUST match the embedding model
```

`embedding_dimension` is critical: it must equal the dimension of the embedding model. Mismatched dimensions cause a runtime error.

| Embedding model | Dimension |
|---|---|
| `all-MiniLM-L6-v2` | 384 |
| `all-mpnet-base-v2` | 768 |
| `BAAI/bge-small-en-v1.5` | 384 |
| `BAAI/bge-large-en-v1.5` | 1024 |
| `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |

## VectorLite settings (legacy)

```toml
[vector_store]
store_type = "vectorlite"

[vector_store.vectorlite]
index_type = "hnsw"              # "hnsw" or "flat"
metric = "cosine"                # "cosine", "euclidean", or "dot"
persistence_path = ""            # defaults to banks_dir
embedding_dimension = 384        # optional; inferred from data if absent
```

VectorLite is the legacy backend. LanceDB is recommended for new deployments.

> [!NOTE]
> VectorLite requires the `vector-lite` build feature, which is **off by default**. Build with `--features vector-lite` to use it.

## Multiple banks

Banks are not configured in the TOML — they are created on demand by the AI calling `create_memory_bank` (or implicitly by writing to a bank name). Each bank gets its own `.lance` directory.

To pre-create banks on disk, use the CLI:

```bash
llm-mem --single list-banks
```

To restrict which banks exist, you can manage the directory directly:

```bash
mkdir -p /var/lib/llm-mem/banks
mkdir /var/lib/llm-mem/banks/research.lance
# Pre-creating an empty bank directory is allowed — the server will populate it on first write.
```

## Storage size

LanceDB stores vectors + metadata. Rough estimates:

- 384-dim f32 vector: 1.5 KB
- 100k memories × 1.5 KB vectors + 1 KB metadata ≈ 250 MB on disk
- 1M memories ≈ 2.5 GB on disk

The `.sessions.db` file is a small SQLite file tracking document upload sessions — usually a few MB.

## Next

- [`[memory]`](./memory-section.md) — what happens when you store/search
- [Data directory layout](./data-directory.md) — the full disk schema
