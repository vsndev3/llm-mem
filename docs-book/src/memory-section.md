# `[memory]` — memory behavior

Settings for how memories are stored, deduplicated, chunked, and abstracted.

```toml
[memory]
max_memories = 10000
similarity_threshold = 0.65
search_similarity_threshold = 0.2
max_search_results = 50
```

## Capacity

| Field | Default | What it controls |
|---|---|---|
| `max_memories` | `10000` | Soft cap on memories per bank. Used for log warnings and admission control. |
| `max_search_results` | `50` | Default upper bound on results per query. Tools can override per-call. |
| `max_list_limit` | `10000` | Hard ceiling for `list_memories` with no explicit limit. Prevents accidental full-table scans. |
| `memory_ttl_hours` | `None` | Time-to-live in hours. `None` = no expiry. `Some(0)` = no expiry (same as `None`). |
| `max_content_length` | `32768` | Reject content longer than this (bytes). |
| `max_cascade_fanout` | `5000` | Max abstraction dependents to load at once during cascade deletion. Prevents OOM. |
| `raw_content_scan_limit` | `5000` | Max raw content candidates during keyword/text search fallback. |
| `max_total_candidates` | `10000` | Max in-memory candidates during pyramid assembly before final ranking. |

## Similarity thresholds

| Field | Default | What it controls |
|---|---|---|
| `similarity_threshold` | `0.65` | Deduplication threshold. When a new memory's embedding is closer than this to an existing one, dedup logic kicks in. |
| `merge_threshold` | `0.75` | Higher threshold to merge two memories into one (vs. keep both). |
| `search_similarity_threshold` | `Some(0.2)` | Reject search results below this. Lower = more results. |
| `near_duplicate_threshold` | `0.92` | When a new memory's embedding similarity to an existing memory exceeds this, a warning is logged. `0.0` disables. (Quality-control) |
| `auto_link_threshold` | `0.75` | Similarity threshold for auto-creating `references` relations between new and existing memories. `0.0` disables auto-linking. |
| `auto_link_max_relations` | `10` | Max auto-link relations per stored memory. |

## Enhancement

| Field | Default | What it controls |
|---|---|---|
| `auto_enhance` | `true` | Automatically enrich new memories with LLM-generated metadata (entities, keywords, importance). Costs LLM tokens. |
| `deduplicate` | `true` | Skip storing near-duplicates. |
| `auto_summary_threshold` | `32768` | When L0 content accumulates past this size, trigger the L0→L1 summarization worker. |

## Document processing

| Field | Default | What it controls |
|---|---|---|
| `document_chunk_size` | `2000` | Chunk size in characters for document ingestion. `0` = no chunking. |
| `chunk_threshold_chars` | `1000` | When a single L0 memory exceeds this, split into sub-chunks for embedding. `0` = disable. |
| `chunk_size_chars` | `1000` | Target sub-chunk size in characters. |
| `chunk_overlap_chars` | `100` | Overlap between consecutive sub-chunks. |

> [!WARNING]
> `chunk_threshold_chars > chunk_size_chars` will log a warning at startup — it means some content past the threshold can still slip through unsplit if it's under `chunk_threshold_chars`.

## LLM-assisted parsing (off by default)

These cost LLM tokens but help with malformed or unknown content.

| Field | Default | What it controls |
|---|---|---|
| `use_llm_query_classification` | `false` | Use LLM for query intent classification (vs. keyword heuristic). |
| `llm_format_detection` | `false` | When content-based format detection returns "unknown", ask the LLM to identify. Sends first 4 KB. |
| `llm_fallback_parsing` | `false` | When structured parsers fail, ask the LLM to extract summary, entities, and content type. |

## Cost & quality

| Field | Default | What it controls |
|---|---|---|
| `session_token_budget` | `0` | Per-session LLM token budget. `0` = unlimited. When exceeded, expensive ops (enhance, classify, abstract) are skipped. |
| `dry_run` | `false` | Log what would be done without executing LLM calls. Useful for cost estimation. |
| `contradiction_detection` | `false` | When storing new memories, ask the LLM to check the top-3 semantically similar memories for factual contradictions. |
| `access_decay_hours` | `168` | Time decay half-life (hours) for access-frequency boosting. Recent accesses boost score; older decay. `0` disables. (168 = 1 week) |

## Next

- [`[server]`](./server-section.md)
- [`[logging]`](./logging-section.md)
