# Chronological queries

Tools for time-based views of memories. Three tools cover different temporal access patterns.

All three bucket by `event_at` (the date the content refers to); if `event_at` is missing, they fall back to `created_at`.

## `get_timeline`

Returns memories bucketed by time interval. Answers "what happened in the last 2 days" or "give me a monthly summary of 2025".

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `start` | string | | ISO 8601 lower bound. Default: `end - 7d`. |
| `end` | string | | ISO 8601 upper bound. Default: now. |
| `granularity` | string | | `hour`, `day` (default), `week`, `month`, `none` (single bucket). |
| `bank` | string | | Memory bank (default `"default"`). |
| `topics` | array of string | | Filter to memories tagged with any of these topics. |
| `max_results_per_bucket` | integer | | Cap per bucket. Default 50. |
| `include_derived` | boolean | | Include L1+ abstractions (default false, L0 only). |
| `order` | string | | `asc` (default) or `desc` within each bucket. |
| `user_id` / `agent_id` | string | | Filter by author. |

### Output

```json
{
  "success": true,
  "start": "2026-02-14T00:00:00Z",
  "end": "2026-02-16T00:00:00Z",
  "granularity": "day",
  "total_count": 23,
  "bucket_count": 2,
  "buckets": [
    {
      "start": "2026-02-14T00:00:00Z",
      "end": "2026-02-15T00:00:00Z",
      "label": "2026-02-14",
      "count": 8,
      "memories": [ /* truncated to max_results_per_bucket */ ]
    }
  ]
}
```

### Examples

```json
// "What happened in the last 2 days?"
{ "end": "2026-02-15T00:00:00Z", "granularity": "day" }

// "Monthly summary of 2025"
{ "start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z", "granularity": "month" }
```

## `get_timeline_graph`

Returns a **graph** of memories: nodes (memories sorted by `event_at`) plus edges. Edges are either:

- **Auto-derived temporal**: `happened_after` — connecting memories whose events were close in time (configurable window)
- **Auto-derived co-occurrence**: `happens_within` — near-simultaneous events (`include_simultaneous`)
- **Optional semantic**: existing relation edges (`derived_from`, `mentions`, …) traversed from the relation graph (`include_semantic_edges`)

Use this tool to render a timeline as a network with D3, Graphviz, etc.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `start` / `end` | string | | Time window. |
| `granularity` | string | | `hour`, `day`, `week`, `month`, `none`. |
| `bank` | string | | |
| `topics` | array of string | | Filter by topics. |
| `max_results_per_bucket` | integer | | |
| `include_derived` | boolean | | Include L1+ abstractions. |
| `order` | string | | `asc` / `desc`. |
| `max_depth` | integer | | Semantic-relation hops from each node (default 1, max 3). |
| `relation_types` | array of string | | Whitelist of semantic relation types to follow. |
| `temporal_edge_window_secs` | integer | | Window for `happened_after` edges (default 86400 = 1 day). |
| `include_simultaneous` | boolean | | Also derive `happens_within` edges for near-simultaneous events. |
| `simultaneous_window_secs` | integer | | Window for `happens_within` (default 60). |
| `include_semantic_edges` | boolean | | Include semantic-relation edges (default true). |
| `user_id` / `agent_id` | string | | |

### Output

```json
{
  "success": true,
  "start": "...",
  "end": "...",
  "granularity": "day",
  "stats": {
    "node_count": 47,
    "edge_count": 92,
    "temporal_edge_count": 80,
    "semantic_edge_count": 12
  },
  "nodes": [
    { "id": "uuid", "event_at": "...", "layer": 0, "bucket": "2026-02-14", "memory": {} }
  ],
  "edges": [
    { "source": "uuid-a", "target": "uuid-b", "type": "happened_after", "delta_secs": 3600 }
  ]
}
```

## `get_context_resume`

Progressive precision: the most recent window returns L0 memories at full detail; progressively older windows return higher-layer abstractions (L1 summaries, L2 links, L3 concepts). Produces an exponential decay curve where precision peaks at the current time. Use this as the first call when resuming a session.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `end` | string | | ISO 8601 end of window. Default: now. |
| `lookback` | string | | Total lookback from `end`, e.g. `30d` (default), `12h`, `1w`. |
| `decay_factor` | float | | Each segment is ~N× the previous one's duration. Default 2.0. |
| `segments` | integer | | Number of precision tiers (1-10). Default 5. |
| `max_per_segment` | integer | | Cap memories per segment. Default 20. |
| `bank` | string | | |
| `topics` | array of string | | Filter by topics. |
| `user_id` / `agent_id` | string | | |

### Output

```json
{
  "success": true,
  "start": "...",
  "end": "...",
  "total_lookback_secs": 2592000,
  "decay_factor": 2.0,
  "segment_count": 5,
  "total_memories": 87,
  "segments": [
    {
      "label": "1d 06-03 — 06-04",
      "layer": 0,
      "start": "...",
      "end": "...",
      "duration_secs": 86400,
      "count": 12,
      "memories": [ /* L0 full detail */ ]
    }
  ]
}
```

Segment 0 (most recent) queries L0, segment 1 queries L1, and so on — newer segments have full detail, older ones are progressively compressed.

## When to use which

| I want to… | Use |
|---|---|
| Quick "what happened recently" list | `get_timeline` |
| Visualize memories as a temporal graph | `get_timeline_graph` |
| Get a "context resume" of a long time period, in bounded size | `get_context_resume` |
| Group by hour for fine-grained analysis | `get_timeline` with `granularity: "hour"` |
| See causal/temporal relations between memories | `get_timeline_graph` with `include_semantic_edges: true` |

## Next

- [Navigation & relations](./tools-navigation.md)
- [Abstraction pipeline](./tools-abstraction.md)
