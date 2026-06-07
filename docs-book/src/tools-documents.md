# Document upload

Tools for ingesting whole files (markdown, code, PDFs, spreadsheets, etc.) into memory with automatic chunking and abstraction.

## `upload_document`

The main ingest tool. Takes a file path on the **server's local filesystem**, chunks it, embeds each chunk, and stores them as L0 memories. The L0→L1 worker then creates summaries in the background.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `file_path` | string | ✓ | Absolute path to the file. Must be readable by the server process. |
| `bank` | string | | |
| `memory_type` | string | | Default: `conversational`. |
| `context` | array | | Tags applied to all chunks. |
| `topics` | array | | Topics applied to all chunks. |
| `metadata` | object | | Per-file metadata (file path, line range, etc.). |
| `chunk_size` | integer | | Override `document_chunk_size`. |
| `generate_abstractions` | boolean | | If true, also trigger the L0→L1 worker after upload. Default true. |
| `max_chunk_size` | integer | | Hard cap on chunk size. |
| `describe_images` | boolean | | For images: generate AI description (requires `vision_enabled`). |
| `auto_link` | boolean | | Auto-link to similar memories. |
| `process_immediately` | boolean | | Start background processing now (vs. waiting for next worker tick). |

### Output

```json
{
  "success": true,
  "data": {
    "session_id": "uuid",
    "file_name": "architecture.md",
    "file_size": 45678,
    "chunks_stored": 23,
    "memory_ids": ["uuid-a", "uuid-b", /* ... */]
  }
}
```

### Limits

`upload_document` is best for **small to medium files** (up to a few MB). For very large files (10s of MB+), use the `ingest` tool or the multi-step `begin_store_document` / `store_document_part` / `process_document` flow.

### Format detection

The server detects format from:

1. File extension
2. MIME sniffing (`tree_magic_mini`)
3. (Optional) LLM-based detection if `llm_format_detection = true` and the above returned "unknown"

Supported formats out of the box: markdown, plain text, code (any language), JSON, JSONL, YAML, TOML, XML, CSV, TSV, PDF, DOCX, XLSX, XLS, ZIP archives, PNG/JPEG/GIF/WebP images.

## `ingest`

Alternate ingest entry point. Same as `upload_document` but with a different parameter style — preferred by the standalone CLI's `upload` command.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `file_path` | string | ✓ | |
| `bank` | string | | |
| `memory_type` | string | | |
| `context` | array | | |
| `metadata` | object | | |
| `chunk_size` | integer | | |
| `process_immediately` | boolean | | |

### Output

Same shape as `upload_document`.

## `document_status`

Check the status of a document upload session.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | | Specific session. If omitted, lists all sessions. |
| `bank` | string | | |

### Output (specific session)

```json
{
  "success": true,
  "data": {
    "session_id": "uuid",
    "file_name": "large.pdf",
    "file_size": 12345678,
    "status": "processing",        // uploading, processing, completed, failed, cancelled
    "chunks_processed": 42,
    "chunks_total": 100,
    "started_at": "...",
    "completed_at": null,
    "error": null
  }
}
```

### Output (list all)

```json
{
  "success": true,
  "data": {
    "sessions": [ /* session objects */ ]
  }
}
```

## `cancel_document`

Cancel an in-flight upload or processing session.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | ✓ | |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "message": "Session 'uuid' cancelled",
  "data": { "session_id": "uuid" }
}
```

## Multi-part upload (large files)

For very large files where `upload_document` might time out, use the multi-step flow:

1. Call `begin_store_document` (returns a `session_id`)
2. Call `store_document_part` repeatedly with each part of the file
3. Call `process_document` to finalize

These are exposed as separate MCP tools in addition to being part of the CLI. They support resumable uploads — if interrupted, the next server boot re-attaches to the existing session and resumes from the last received part.

The CLI's `begin-upload` / `upload-part` / `process-document` commands wrap these for the standalone case.

## Document storage

Documents themselves are not stored in the bank. Only the **chunks** (after parsing) and the **session state** (in `<bank>.sessions.db`) are kept. To keep the original file, store it elsewhere or use the `db export` flow to capture both the bank and the session database.

## Next

- [Introspection](./tools-introspection.md)
- [Banks & backups](./tools-banks.md)
