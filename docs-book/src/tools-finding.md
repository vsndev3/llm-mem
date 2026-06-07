# Finding memories

Tools for retrieving stored memories. Four tools cover different access patterns.

## `query_memory`

The most powerful search tool. Supports **pyramid search**, **graph traversal**, **hybrid keyword/semantic mode**, and **context filtering**.

### Modes

| Mode | Behavior |
|---|---|
| `"balanced"` (default) | Distribute results proportionally across all layers |
| `"bottom_heavy"` | Prefer L0 (raw) results |
| `"top_heavy"` | Prefer L3+ (concept) results |
| `"dynamic"` | Auto-distribute based on query intent |
| `"graph_traversal"` | Start from a seed memory, follow relations up to N hops |

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✓ (unless using `context_ids`) | The search query. |
| `mode` | string | | One of: `balanced`, `bottom_heavy`, `top_heavy`, `dynamic`, `graph_traversal`. |
| `bank` | string | | |
| `max_results` | integer | | Default 50. |
| `min_similarity` | float | | Per-call override of `search_similarity_threshold`. |
| `memory_types` | array of string | | Filter by type. |
| `context_tags` | array of string | | Filter by `context` tag. |
| `event_after` | string | | ISO 8601 — only memories with `event_at` after this. |
| `event_before` | string | | ISO 8601 — only memories with `event_at` before this. |
| `created_after` / `created_before` | string | | Filter by storage time. |
| `keyword_weight` | float | | Hybrid mode: weight of keyword vs. semantic (0.0-1.0). |
| `context_ids` | array of string | | Restrict to this set of memory IDs (graph traversal seed). |
| `max_depth` | integer | | Graph traversal: max hops (1-5). |
| `include_layers` | array of integer | | Restrict to specific layer levels. |

### Output

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "memory_id": "uuid",
        "content": "...",
        "memory_type": "factual",
        "layer": 0,
        "similarity": 0.87,
        "context": ["project-x", "auth"],
        "topics": ["jwt"],
        "metadata": { /* ... */ },
        "event_at": "2026-02-15T10:00:00Z",
        "created_at": "2026-02-15T11:23:45Z"
      }
    ],
    "total": 12
  }
}
```

## `search_memory`

A **simplified** search with sensible defaults. Internally converts to a `query_memory` with `mode: "balanced"` and `keyword_weight: 0.2`. Use this when you don't need the full pyramid/graph power.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✓ | |
| `bank` | string | | |
| `max_results` | integer | | |
| `memory_types` | array of string | | |
| `context_tags` | array of string | | |

### Output

Same shape as `query_memory`.

## `list_memories`

Browse by filter — no semantic similarity, just structured filters. Useful for "show me everything I stored yesterday" or "list all episodic memories".

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `bank` | string | | |
| `limit` | integer | | Default 50, hard cap `max_list_limit`. |
| `offset` | integer | | For pagination. |
| `memory_types` | array of string | | |
| `context_tags` | array of string | | |
| `topics` | array of string | | |
| `event_after` / `event_before` | string | | ISO 8601. |
| `created_after` / `created_before` | string | | ISO 8601. |
| `min_layer` / `max_layer` | integer | | Filter by layer. |
| `include_forgotten` | boolean | | Include soft-deleted (default false). |

### Output

```json
{
  "success": true,
  "data": {
    "results": [ /* memory objects without similarity */ ],
    "total": 1234,
    "limit": 50,
    "offset": 0
  }
}
```

## `get_memory`

Look up a specific memory by ID. Returns the full memory object.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `memory_id` | string | ✓ | |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "data": {
    "memory_id": "uuid",
    "content": "...",
    "memory_type": "factual",
    "layer": 0,
    "context": ["..."],
    "topics": ["..."],
    "relations": [ /* {relation, target} */ ],
    "metadata": { /* ... */ },
    "event_at": "...",
    "created_at": "...",
    "updated_at": "..."
  }
}
```

## Choosing the right tool

| I want to… | Use |
|---|---|
| Find memories that match a question / topic | `query_memory` or `search_memory` |
| Browse by date / type / tag | `list_memories` |
| Get the full record for a specific memory ID | `get_memory` |
| Traverse relations from a seed memory | `query_memory` with `mode: "graph_traversal"` |
| Restrict to a specific abstraction layer | `query_memory` with `include_layers: [3]` |

## Next

- [Chronological queries](./tools-chronology.md) — timeline views
- [Navigation & relations](./tools-navigation.md) — pyramid zoom
