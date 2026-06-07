# Chronological queries

Tools for time-based views of memories. Three tools cover different temporal access patterns.

## `get_timeline`

Returns memories bucketed by time interval. Answers "what happened in the last 2 days" or "give me a monthly summary of 2025".

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `bank` | string | | |
| `start` | string | | ISO 8601 lower bound (inclusive). |
| `end` | string | | ISO 8601 upper bound (exclusive). |
| `since` | string | | Relative window: `12h`, `2d`, `1w`, `3mo`. Easier than computing `start`/`end`. |
| `granularity` | string | | `hour`, `day`, `week`, `month`, `none` (flat). |
| `include_derived` | boolean | | If true, include L1+ abstractions in the timeline (uses their derived `event_at` range). Default false. |
| `max_per_bucket` | integer | | Cap per bucket. Default 50. |
| `memory_types` | array | | Filter by type. |
| `context_tags` | array | | Filter by context. |
| `event_field` | string | | `event_at` (default) or `created_at`. Use `created_at` to ask "what was stored recently". |

### Output

```json
{
  "success": true,
  "data": {
    "buckets": [
      {
        "start": "2026-02-14T00:00:00Z",
        "end":   "2026-02-15T00:00:00Z",
        "label": "2026-02-14",
        "count": 8,
        "memories": [ /* truncated to max_per_bucket */ ]
      }
    ],
    "total": 23
  }
}
```text

### Examples

```json
// "What happened in the last 2 days?"
{ "since": "2d", "granularity": "day" }

// "Monthly summary of 2025"
{ "start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z", "granularity": "month" }

// "What was stored this week, regardless of event time?"
{ "since": "1w", "granularity": "day", "event_field": "created_at" }
```text

## `get_timeline_graph`

Returns a **graph** of memories: nodes (memories sorted by `event_at`) plus edges. Edges are either:

- **Auto-derived temporal**: `happened_after` — connecting memories whose events were close in time (configurable window)
- **Auto-derived co-occurrence**: `happens_within` — same bucket
- **Optional semantic**: user-defined `relations` if `include_semantic_edges: true`

This is the tool to use for graph rendering with D3, Graphviz, etc.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `bank` | string | | |
| `start` / `end` / `since` | string | | Time window. |
| `granularity` | string | | `hour`, `day`, `week`, `month`, `none`. |
| `include_derived` | boolean | | Include L1+ abstractions. |
| `max_per_bucket` | integer | | |
| `max_depth` | integer | | Max semantic-relation hops from each timeline node. Default 1. |
| `temporal_window_secs` | integer | | Window for auto `happened_after` edges. Default 86400 (1 day). |
| `include_semantic_edges` | boolean | | Add user-defined `relations` as edges. Default true. |
| `memory_types` / `context_tags` | array | | Filters. |

### Output

```json
{
  "success": true,
  "data": {
    "nodes": [
      { "memory_id": "uuid", "content": "...", "event_at": "..." }
    ],
    "edges": [
      { "source": "uuid-a", "target": "uuid-b", "relation": "happened_after" }
    ],
    "stats": { "node_count": 47, "edge_count": 92 }
  }
}
```text

## `get_context_resume`

Progressive precision: recent L0 memories at full detail, older L1+ summaries compressed exponentially. This is the tool to use when you want "everything from the last month" but bounded in size.

The result is a series of **segments** where each segment is `decay_factor` × older than the previous. Default `decay_factor = 4.0`, `segments = 5`.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `bank` | string | | |
| `since` | string | | Lookback window. E.g. `30d`, `12h`, `1w`. |
| `end` | string | | End of window. Defaults to now. |
| `decay_factor` | float | | Each segment is N× older than the previous. Default 4.0. |
| `segments` | integer | | Number of precision segments. Default 5. |
| `max_per_segment` | integer | | Cap memories per segment. Default 20. |

### Output

```json
{
  "success": true,
  "data": {
    "segments": [
      {
        "label": "Last 12h",
        "time_range": { "start": "...", "end": "..." },
        "memories": [ /* L0 full detail */ ]
      },
      {
        "label": "12h - 2d",
        "memories": [ /* L0-L1 */ ]
      },
      {
        "label": "2d - 8d",
        "memories": [ /* L1 summaries */ ]
      }
    ]
  }
}
```text

The newest segment has full L0 detail; older segments have progressively more L1+ summaries.

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
