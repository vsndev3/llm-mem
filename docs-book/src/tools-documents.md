# Document upload

Tools for ingesting whole files (markdown, code, PDFs, spreadsheets, images, etc.) into memory with automatic chunking and abstraction.

## `upload_document`

Upload a file from the **server's local filesystem**. The server reads the file, splits it into chunks, embeds each chunk, and stores them as L0 memories. The L0→L1 worker then creates summaries in the background.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `file_path` | string | ✓ | Absolute path to the file. Must be readable by the server process. |
| `file_name` | string | | Optional name (defaults to the basename of `file_path`). |
| `mime_type` | string | | Optional MIME type (default `text/plain`). |
| `chunk_size` | integer | | Override `document_chunk_size`. |
| `process_immediately` | boolean | | Start processing after upload (default true). |
| `topics` | array | | Topics applied to all chunks. |
| `context` | array | | Context tags applied to all chunks. |
| `event_at` | string | | ISO 8601 for when the document's events occurred. Defaults to upload time. |
| `bank` | string | | |
| `user_id` / `agent_id` | string | | |

### Output

The response is not schema-constrained. A representative shape:

```json
{
  "success": true,
  "data": {
    "session_id": "uuid",
    "file_name": "architecture.md",
    "chunks_stored": 23,
    "memory_ids": ["uuid-a", "uuid-b", "…"]
  }
}
```

### Limits

`upload_document` is best for **small to medium files** (up to a few MB). For very large files (10s of MB+) or files you already have in memory, use the [`ingest`](#ingest) tool, or the CLI's multi-part upload flow (`begin-upload` / `upload-part` / `process-document`, see [CLI commands](./cli-commands.md)).

## `ingest`

Universal decomposition entry point. Takes **raw content** (a string, or base64 for binary) and automatically detects the format, decomposes it into semantic L0 chunks, creates structural relations between chunks, and optionally auto-links to existing memories.

Unlike `upload_document`, this does not read a file from disk — you pass the content directly. For binary formats (PDF, DOCX, images), base64-encode the content and set `content_encoding: "base64"`.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `content` | string | ✓ | Raw content. For binary formats, base64-encoded. |
| `content_encoding` | string | | `"base64"` for binary formats. |
| `format_hint` | string | | Optional hint: `markdown`, `json`, `yaml`, `toml`, `text`. |
| `file_name` | string | | For extension-based detection and provenance. |
| `bank` | string | | |
| `auto_link` | boolean | | Auto-link chunks to existing memories (default true). |
| `generate_abstractions` | boolean | | Generate L1+ interpretations (default true). |
| `max_chunk_size` | integer | | Max characters per L0 chunk (default 2000). |
| `metadata` | object | | Metadata attached to all chunks. |
| `source` | string | | Override the auto-derived source (e.g. a URL or DOI). |
| `describe_images` | boolean | | Generate AI descriptions for images (default true; requires a vision-capable LLM). |

### Output

```json
{
  "success": true,
  "message": "…",
  "data": {
    "status": "success",
    "session_id": "uuid",
    "format": "markdown",
    "detected_mime": "text/markdown",
    "byte_size": 45678,
    "l0_chunks": [
      { "id": "chunk-0", "memory_id": "uuid", "node_type": "text", "content_preview": "…", "char_count": 1980, "order": 0 }
    ],
    "relations": [
      { "source_chunk_id": "chunk-0", "target_chunk_id": "chunk-1", "relation": "next_chunk", "strength": 1.0 }
    ],
    "issues": [],
    "warnings": [],
    "format_hints_available": ["markdown", "text"]
  }
}
```

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
    "status": "processing",
    "chunks_processed": 42,
    "chunks_total": 100,
    "started_at": "...",
    "completed_at": null,
    "error": null
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

## Document storage

Documents themselves are not stored in the bank. Only the **chunks** (after parsing) and the **session state** are kept. To keep the original file, store it elsewhere or use the `db export` flow to capture both the bank and the session database.

## Next

- [Introspection](./tools-introspection.md)
- [Banks & backups](./tools-banks.md)
