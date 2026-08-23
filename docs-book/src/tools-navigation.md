# Navigation & relations

Tools for moving through the memory pyramid and managing relations between memories.

## `navigate_memory`

Move up and down the abstraction hierarchy from a seed memory. `zoom_out` returns higher-layer (more abstract) memories derived from it; `zoom_in` returns lower-layer (more detailed) source memories it was abstracted from; `both` returns both directions.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `memory_id` | string | ✓ | The seed memory. |
| `direction` | string | | `zoom_in`, `zoom_out`, or `both` (default). |
| `levels` | integer | | Levels to traverse for `zoom_in` (1-5, default 1). |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "source_memory_id": "uuid",
  "source_layer": 1,
  "zoom_in": [ /* lower-layer source memories */ ],
  "zoom_out": [ /* higher-layer abstractions */ ]
}
```

### Examples

```json
// From a JWT note, find the auth-architecture concept built on top of it
{ "memory_id": "uuid-of-jwt-note", "direction": "zoom_out" }

// From a concept memory, find all source evidence
{ "memory_id": "uuid-of-concept", "direction": "zoom_in", "levels": 3 }
```

> [!NOTE]
> To restrict a *semantic search* to a specific layer, use `query_memory` with `pyramid_config` (e.g. `{ "mode": "top_heavy" }` or `layer_weights`), not `navigate_memory`.

## `update_memory`

Update an existing memory's content and/or append relations by ID.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `memory_id` | string | ✓ | |
| `content` | string | | New content. Re-embeds the memory. |
| `relations` | array | | Relations to append, as `[{ "relation": "...", "target": "..." }]`. |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "message": "Memory updated"
}
```

> [!WARNING]
> Updating `content` re-embeds the memory. This is an LLM-free operation (uses the same embedding model), but it does cost time and changes the semantic identity. Use it for typo fixes, not wholesale rewrites.

## `force_link`

Manually create a relation between two existing memories. The reverse relation is created automatically (e.g. `references` → `referenced_by`).

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `source_id` | string | ✓ | Source memory ID (the "from" side). |
| `relation` | string | ✓ | Relation type: `references`, `contradicts`, `supports`, `depends_on`, `part_of`, `extends`, `similar_to`, `summary_of`, `synthesizes`. |
| `target_id` | string | ✓ | Target memory ID (must be an existing memory). |
| `strength` | number | | Relation strength 0.0-1.0 (default 1.0). |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "message": "Relation created",
  "data": {
    "source_id": "uuid",
    "relation": "depends_on",
    "target_id": "uuid",
    "reverse_relation": "depended_on_by",
    "reverse_created": true
  }
}
```

> [!NOTE]
> `source_id` and `target_id` must both be valid UUIDs of existing memories. Hierarchical relations (`summary_of`, `part_of`, `synthesizes`) require the source to be at a higher layer than the target. Duplicate links are rejected.

### Suggested relation predicates

| Category | Predicates |
|---|---|
| **Structural** | `chunk_of`, `summary_of`, `part_of`, `next_chunk`, `previous_chunk` |
| **Semantic** | `references`, `extends`, `supersedes`, `depends_on`, `implements`, `configures` |
| **Layered** | `derived_from`, `emerges_from`, `instance_of`, `broader_than` |
| **Provenance** | `authored_by`, `cited_by` |
| **Contradiction** | `contradicts` |

## `remove_relation`

Remove a specific relation from a memory. The reverse relation on the target is removed automatically.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `memory_id` | string | ✓ | Memory to remove the relation from. |
| `relation_type` | string | ✓ | The relation type to remove (e.g. `references`). |
| `target_id` | string | ✓ | The target memory ID in the relation. |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "message": "Relation removed",
  "data": {
    "memory_id": "uuid",
    "removed_relation": "references",
    "removed_target": "uuid",
    "reverse_relation": "referenced_by",
    "reverse_cleaned": true
  }
}
```

## Relations vs auto-linking

When auto-linking is enabled (via `auto_link_threshold`), the server automatically creates `references` relations to semantically similar existing memories. `force_link` is for cases where:

- You know of a relation the auto-linker missed
- You want a non-semantic relation (e.g. `next_chunk`, `part_of`)
- You want to control the relation strength explicitly

## Next

- [Abstraction pipeline](./tools-abstraction.md)
- [Banks & backups](./tools-banks.md)
