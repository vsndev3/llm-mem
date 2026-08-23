# Finding memories

Tools for retrieving stored memories. Four tools cover different access patterns.

## `query_memory`

The most powerful search tool. Supports **hybrid keyword/semantic search**, **pyramid allocation**, **graph traversal**, and **context filtering**.

### Important: score interpretation

Scores reflect **semantic similarity**, NOT whether the answer is present. A high score means the memory is topically related — it does not guarantee the answer exists in that memory. Always verify retrieved content against what was asked:

- Use `k >= 5` to ensure adequate recall.
- L0 = concrete source content (facts, verbatim text). The answer lives here.
- L1+ = increasingly abstract (summaries → concepts → wisdom). These help you **navigate** to relevant L0 facts but are not themselves the answer.

### Pyramid allocation (`pyramid_config`)

Pyramid search distributes result slots across abstraction layers, so a single query returns both concrete facts and abstract insights. Set `pyramid_config.mode` to one of:

| Mode | Behavior |
|---|---|
| `"bottom_heavy"` (default) | More L0 facts, fewer abstract concepts |
| `"balanced"` | Equal distribution across layers |
| `"top_heavy"` | More abstract concepts, fewer concrete facts |
| `"dynamic"` | LLM classifies query intent automatically (requires `use_llm_query_classification` config) |
| `"none"` | Skip pyramid assembly, return flat results by raw score |

`pyramid_config` also accepts `layer_weights` (per-layer weight overrides) and `per_layer_multiplier` (default 2.0).

### Graph traversal (`graph_traversal`)

Enable `graph_traversal.enabled` to replace semantic search with a pure BFS multi-hop traversal, following relations (`derived_from`, `mentions`, `knows`, …) from the top-scoring entry points:

| Field | Description |
|---|---|
| `enabled` | Turn on deep traversal (default false). |
| `max_depth` | Max hops (1-5, default 2). |
| `direction` | `"outgoing"`, `"incoming"`, or `"both"` (default). |
| `relation_types` | Restrict to specific relation types. |
| `entry_point_limit` | Max seed memories (default 5, max 10). |
| `include_paths` | Include per-result `graph_info` (distance, path, boosts). |

Without deep traversal, standard search still applies automatic 1-hop graph refinement on the top results.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✓ | The search query. |
| `k` | integer | | Max results (default 10). |
| `bank` | string | | Memory bank name (default `"default"`). |
| `topics` | array of string | | Filter by topics. |
| `context` | array of string | | Context tags for semantic scoping. |
| `keyword_only` | boolean | | Keyword-only search, no embeddings (default false). |
| `keyword_split_ratio` | float | | Fraction of results from raw keyword matching vs. semantic (0.0-1.0, default 0.2). |
| `similarity_threshold` | float | | Override similarity threshold (0.0-1.0). |
| `min_salience` | float | | Minimum salience/importance score (0-1). |
| `granularity` | string | | `"full"` (default) resolves chunk hits to the complete parent memory; `"excerpt"` returns a compact window around matched regions (session header preserved), under `excerpt_max_chars`. |
| `excerpt_max_chars` | integer | | Total char budget for excerpt-mode content (default 12000, min 1000). |
| `pyramid_config` | object | | `mode`, `layer_weights`, `per_layer_multiplier` (see above). |
| `graph_traversal` | object | | Deep traversal config (see above). |
| `created_after` / `created_before` | string | | ISO 8601 — filter by storage time. |
| `event_after` / `event_before` | string | | ISO 8601 — filter by `event_at` (or `created_at`). |
| `user_id` / `agent_id` | string | | Filter by author. |

### Output

```json
{
  "success": true,
  "count": 12,
  "memories": [
    {
      "memory_id": "uuid",
      "content": "...",
      "score": 0.87,
      "layer": 0,
      "layer_name": "raw_content",
      "context": ["project-x", "auth"],
      "topics": ["jwt"],
      "metadata": { /* ... */ },
      "event_at": "2026-02-15T10:00:00Z",
      "created_at": "2026-02-15T11:23:45Z"
    }
  ]
}
```

## `search_memory`

A **simplified** search with sensible defaults. Use this when you don't need the full pyramid/graph power.

Same score interpretation rules apply: high similarity does not guarantee the answer is present, use `k >= 5`, and L0 is where concrete facts live.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | ✓ | |
| `k` | integer | | Max results (default 10). |
| `bank` | string | | |

### Output

Same `{ success, count, memories }` shape as `query_memory`.

## `list_memories`

Browse by structured filter — no semantic similarity. Useful for "show me everything I stored yesterday".

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `limit` | integer | | Default 100, max 1000. |
| `bank` | string | | |
| `created_after` / `created_before` | string | | ISO 8601. |
| `event_after` / `event_before` | string | | ISO 8601. |
| `user_id` / `agent_id` | string | | |

### Output

```json
{
  "success": true,
  "count": 1234,
  "memories": [ /* memory objects */ ]
}
```

## `get_memory`

Look up a specific memory by ID.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `memory_id` | string | ✓ | |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "memory": {
    "memory_id": "uuid",
    "content": "...",
    "layer": 0,
    "context": ["..."],
    "topics": ["..."],
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
| Traverse relations from a seed memory | `query_memory` with `graph_traversal: { enabled: true }` |
| Favor abstract concepts over raw facts | `query_memory` with `pyramid_config: { mode: "top_heavy" }` |
| Keep retrieved content compact | `query_memory` with `granularity: "excerpt"` |

## Next

- [Chronological queries](./tools-chronology.md) — timeline views
- [Navigation & relations](./tools-navigation.md) — pyramid zoom
