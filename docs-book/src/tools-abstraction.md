# Abstraction pipeline

Tools for controlling the background workers that build higher layers of the pyramid from lower-layer content.

## `start_abstraction_pipeline`

Start (or resume) the background workers. The pipeline auto-starts on server boot, so this is mainly useful after `stop_abstraction_pipeline`.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `bank` | string | | If omitted, starts the pipeline for all banks. |

### Output

```json
{
  "success": true,
  "message": "Abstraction pipeline started",
  "data": { "banks_started": ["default", "research"] }
}
```

## `stop_abstraction_pipeline`

Pause the background workers. Useful before bulk operations or for debugging.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `bank` | string | | If omitted, stops the pipeline for all banks. |

### Output

Standard success envelope.

> [!NOTE]
> `stop_abstraction_pipeline` does not cancel an in-flight LLM call — it just stops scheduling new work. Requests already in progress will complete.

## `trigger_abstraction`

Force the abstraction workers to consider a specific memory. Without this, the worker only processes memories that meet certain thresholds (e.g. the L0→L1 worker only fires after `auto_summary_threshold` bytes of L0 content accumulates).

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `memory_id` | string | ✓ | |
| `target_layer` | integer | | The layer to build up to. Default = current top + 1. |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "data": {
    "memory_id": "uuid",
    "target_layer": 2,
    "abstractions_created": 3
  }
}
```

## `create_abstraction`

Lower-level entry point for manual layer creation. Usually you want `trigger_abstraction` instead, which figures out the right thing to do.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `source_memory_ids` | array of string | ✓ | Memories to synthesize. |
| `target_layer` | integer | ✓ | The new layer to create. |
| `relation` | string | | Relation from new abstraction to sources. Default: `summary_of` (L1), `emerges_from` (L2), `emerges_from` (L3+). |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "data": {
    "new_memory_id": "uuid",
    "source_memory_ids": ["uuid-a", "uuid-b"],
    "target_layer": 2
  }
}
```

## What the pipeline actually does

The pipeline is a set of background workers, one per layer transition:

| Worker | Trigger | What it produces |
|---|---|---|
| **L0 → L1** | When total L0 content exceeds `auto_summary_threshold` bytes | L1 summaries grouped by topic. `relation: "summary_of"` from summary to sources. |
| **L1 → L2** | When multiple L1 summaries share themes | L2 connections with `relation: "related_to"`, `references`, etc. |
| **L2 → L3** | When L2 clusters form | L3 concepts with `relation: "emerges_from"`. |
| **L3 → L4+** | When L3 concepts span domains | L4+ insights / mental models. |

Each worker:

1. Selects candidate source memories
2. Asks the LLM to synthesize
3. Stores the new abstraction
4. Creates the `derived_from` / `emerges_from` / etc. relations
5. Notifies the next-layer worker (which may decide to act)

Workers respect `session_token_budget` and circuit-breaker around the LLM. Failed abstractions get retried with exponential backoff; persistent failures get logged and skipped (use `clear-backoff` from the CLI to force retry).

## How to use it

In normal operation, you don't. The pipeline runs and gradually builds higher layers as content accumulates.

You intervene when:

- **You just stored something important and want insights now** → `trigger_abstraction`
- **You want to see the architecture before doing bulk ingestion** → `stop_abstraction_pipeline`, do the ingest, `start_abstraction_pipeline`
- **The pipeline is failing repeatedly** → `clear-backoff` (CLI), then `trigger_abstraction` on a small set to debug

## Next

- [Banks & backups](./tools-banks.md)
- [Document upload](./tools-documents.md)
