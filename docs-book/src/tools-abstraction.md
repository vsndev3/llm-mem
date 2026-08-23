# Abstraction pipeline

Tools for controlling the background workers that build higher layers of the pyramid from lower-layer content.

## `start_abstraction_pipeline`

Start (or resume) the background workers. The pipeline auto-starts on server boot, so this is mainly useful after `stop_abstraction_pipeline`.

### Input

None.

### Output

```json
{
  "success": true,
  "message": "Abstraction pipeline started"
}
```

## `stop_abstraction_pipeline`

Pause the background workers. Useful before bulk operations or for debugging.

### Input

None.

### Output

```json
{
  "success": true,
  "message": "Abstraction pipeline stopped"
}
```

> [!NOTE]
> `stop_abstraction_pipeline` does not cancel an in-flight LLM call — it just stops scheduling new work. Requests already in progress will complete.

## `trigger_abstraction`

Run one-shot abstraction processing now (unlike the pipeline, this runs once and does not start background workers). The pipeline must already be running.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `target_layer` | integer | | `1` = L0→L1 (summaries), `2` = L1→L2, `3` = L2→L3, `0` = all. Default 1. |

### Output

```json
{
  "success": true,
  "l0_to_l1_created": 3,
  "l1_to_l2_created": 1,
  "l2_to_l3_created": 0,
  "errors": []
}
```

## `create_abstraction`

Create a manual abstraction (L1/L2/L3) from specific source memory IDs, with your own content.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `content` | string | ✓ | The abstraction content (summary, synthesis, or concept). |
| `source_ids` | array of string | ✓ | Source memories this abstraction derives from. |
| `target_layer` | integer | ✓ | Target layer (1=structural, 2=semantic, 3=concept). Must be higher than all source layers. |
| `relation_type` | string | | Relation to sources. Defaults by layer: `summary_of` (L1), `synthesizes` (L2), `abstracts_to_concept` (L3+). |
| `user_id` / `agent_id` | string | | |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "message": "Abstraction created",
  "data": {
    "memory_id": "uuid",
    "target_layer": 2,
    "relation_type": "synthesizes",
    "source_count": 3,
    "reverse_relation": "synthesized_by",
    "reverse_created": true
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
- **You want a specific abstraction on demand** → `create_abstraction` with explicit source IDs and content
- **The pipeline is failing repeatedly** → `clear-backoff` (CLI), then `trigger_abstraction` to debug

## Next

- [Banks & backups](./tools-banks.md)
- [Document upload](./tools-documents.md)
