# Storing memories

Tools for writing new memories to a bank. There are four "store" variants plus document ingestion (covered under [Document upload](./tools-documents.md)).

## `add_content_memory`

Add **raw content** to memory without any AI transformation. The content is stored and embedded exactly as-is, preserving all original phrases, keywords, and structure.

**Use this when**:

- You need exact-phrase searchability (e.g. finding "vegan chili" later)
- Storing conversation logs, documents, or code snippets where original text matters
- You want predictable semantic search based on the actual content, not AI interpretations

For AI-processed structured facts, use [`add_intuitive_memory`](#add_intuitive_memory) instead.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `content` | string | ✓ | The fact or piece of information. Should be concise and atomic. |
| `memory_type` | enum | | `conversational` (default), `procedural`, `factual`, `semantic`, `episodic`, `personal`. |
| `metadata` | object | | Key-value pairs (source file, page, timestamp, author). |
| `user_id` | string | | Only if multiple users share a bank. |
| `agent_id` | string | | Agent ID. Inherited from CLI if unset. |
| `topics` | array of string | | Subjects within the content. |
| `context` | array of string | | Broad topic tags. |
| `relations` | array of object | | `{relation, target}` pairs to other memories or entities. |
| `bank` | string | | Defaults to `"default"`. |
| `auto_link` | boolean | | Whether to auto-create `references` relations to similar memories. Server default if unset. |
| `event_at` | string | | ISO 8601 datetime — *when the event actually happened*. |
| `source` | string | | Free-form provenance (file name, URL, book title). |
| `force` | boolean | | Skip near-duplicate / contradiction checks. |

### Output

```json
{
  "success": true,
  "message": "Memory stored",
  "data": {
    "memory_id": "uuid",
    "user_id": "...",
    "agent_id": "..."
  }
}
```text

## `add_intuitive_memory`

Add memories with **AI-powered extraction and structuring**. The LLM analyzes your content, extracts key facts, organizes them into atomic insights, and generates searchable keywords.

**Use this when**:

- You want structured, reasoning-ready memories
- You need condensed insights from long conversations or documents
- You want automatic keyword extraction for hybrid search

The original text is **transformed** by the LLM. For verbatim storage, use `add_content_memory`.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `messages` | array of object | ✓ | Conversation messages: `{role, content, name?}`. |
| `memory_type` | enum | | Type for extracted facts. Defaults to `conversational`. |
| `metadata` | object | | Attached to all extracted memories. |
| `user_id` | string | | |
| `agent_id` | string | | |
| `bank` | string | | |
| `source_memory_id` | string | | Link extracted memories to a source. Auto-creates `derived_from` relation. |
| `event_at` | string | | ISO 8601 — applied to all extracted memories. |

### Output

Same shape as `add_content_memory` — `data.memory_id` references the new memory, plus internal metadata about the extraction.

## `store_memories`

Store multiple content memories in a single call. Much faster than calling `add_content_memory` repeatedly for bulk ingestion.

**Rules**:

- `items` cannot be empty
- Each item must have non-empty `content`
- Invalid items cause the **entire batch to be rejected** before any storage

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `items` | array of object | ✓ | Each: `{content, memory_type?, topics?, context?, relations?, metadata?, event_at?, source?}`. |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "message": "Stored 5 memories",
  "data": {
    "results": [ /* per-item results */ ],
    "total": 5
  }
}
```text

## Which to use?

| Scenario | Tool |
|---|---|
| *"Remember this exact sentence"* | `add_content_memory` |
| *"Extract the key facts from this conversation"* | `add_intuitive_memory` |
| *"Ingest these 100 notes at once"* | `store_memories` |
| *"Store this PDF / markdown file"* | [`upload_document`](./tools-documents.md#upload_document) |
| *"Ingest this file, please chunk it and extract structure"* | [`ingest`](./tools-documents.md#ingest) |

## Quality-control flags

`add_content_memory` accepts a `force` boolean:

- `force: false` (default) — the server checks for near-duplicates and contradictions. On detection, the call is **blocked** and the error describes the conflict.
- `force: true` — bypass the checks. Use only when you're certain the content should be stored despite the warning.

The same `force` flag exists on `add_intuitive_memory` and `store_memories` items.

> [!WARNING]
> Setting `force: true` defeats the contradiction-detection safety net. Use it sparingly and only after reviewing what the conflict actually is.

## Next

- [Finding memories](./tools-finding.md)
- [Chronological queries](./tools-chronology.md)
