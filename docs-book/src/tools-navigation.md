# Navigation & relations

Tools for moving through the memory pyramid and managing relations between memories.

## `navigate_memory`

Move up and down the abstraction pyramid.

### Modes

| Mode | Behavior |
|---|---|
| `zoom_out` | From a specific memory, find the L1+ abstractions that were built from it. |
| `zoom_in` | From an L1+ memory, find the L0 source memories it was synthesized from. |
| `search_at_layer` | Restrict a semantic search to a specific layer. |

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `mode` | string | ✓ | `zoom_out`, `zoom_in`, `search_at_layer`. |
| `memory_id` | string | (zoom modes) | The seed memory. |
| `query` | string | (search mode) | The search query. |
| `layer` | integer | (search mode) | Target layer (0-4+). |
| `bank` | string | | |
| `max_depth` | integer | | How many levels to traverse. Default 3. |
| `max_results` | integer | | Default 20. |

### Output

```json
{
  "success": true,
  "data": {
    "direction": "zoom_out",
    "results": [ /* memory objects, ordered by depth */ ]
  }
}
```

### Examples

```json
// From a JWT note, find the auth architecture concept built on top of it
{ "mode": "zoom_out", "memory_id": "uuid-of-jwt-note", "max_depth": 4 }

// From a concept memory, find all source evidence
{ "mode": "zoom_in", "memory_id": "uuid-of-concept", "max_depth": 3 }

// Search only at the concept level
{ "mode": "search_at_layer", "query": "database scaling", "layer": 3 }
```

## `update_memory`

Edit an existing memory's content, type, topics, context, relations, or metadata.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `memory_id` | string | ✓ | |
| `bank` | string | | |
| `content` | string | | New content. Re-embeds the memory. |
| `memory_type` | string | | |
| `topics` | array | | Replace the topic list. |
| `context` | array | | Replace the context tags. |
| `relations` | array | | Replace the relations list. |
| `metadata` | object | | Merge or replace (see `replace_metadata` flag). |
| `replace_metadata` | boolean | | If true, replace the entire metadata object. If false (default), merge. |
| `event_at` | string | | Update event time. |
| `source` | string | | Update source provenance. |

### Output

Standard success envelope; `data.memory_id` confirms the update.

> [!WARNING]
> Updating `content` re-embeds the memory. This is an LLM-free operation (uses the same embedding model), but it does cost time and changes the semantic identity. Use it for typo fixes, not wholesale rewrites.

## `force_link`

Manually create a relation between two memories (or between a memory and a free-form entity name).

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `source_id` | string | ✓ | Source memory ID. |
| `target` | string | ✓ | Target memory ID or entity name. |
| `relation` | string | ✓ | Relation predicate. See the list below. |
| `bank` | string | | |

### Output

```json
{
  "success": true,
  "data": {
    "source_id": "uuid",
    "target": "uuid-or-name",
    "relation": "depends_on"
  }
}
```

### Suggested relation predicates

| Category | Predicates |
|---|---|
| **Structural** | `chunk_of`, `summary_of`, `part_of`, `next_chunk`, `previous_chunk` |
| **Semantic** | `related_to`, `references`, `extends`, `supersedes`, `depends_on`, `implements`, `configures` |
| **Layered** | `derived_from`, `emerges_from`, `instance_of`, `broader_than` |
| **Provenance** | `authored_by`, `cited_by` |
| **Contradiction** | `contradicts` |

These are conventions, not enforced. Any string works as `relation`, but using the vocabulary above makes graph traversal more meaningful.

## `remove_relation`

Delete a relation from a memory. (Note: this does not delete the target memory, just the edge from the source.)

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `source_id` | string | ✓ | |
| `target` | string | ✓ | |
| `relation` | string | ✓ | The exact relation to remove. |
| `bank` | string | | |

### Output

Standard success envelope.

> [!NOTE]
> If the same `target` is linked with multiple relations (e.g. both `related_to` and `references`), this only removes the one you specify.

## Relations vs auto-linking

When `auto_link: true` (or by default, with `auto_link_threshold` set), the server automatically creates `references` relations to semantically similar existing memories. `force_link` is for cases where:

- You know of a relation the auto-linker missed
- You want a non-semantic relation (e.g. `next_chunk`, `part_of`)
- You want to link a memory to a free-form entity name, not another memory

## Next

- [Abstraction pipeline](./tools-abstraction.md)
- [Banks & backups](./tools-banks.md)
