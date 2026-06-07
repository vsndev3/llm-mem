# MCP tools reference

llm-mem-mcp exposes **32 tools** to MCP clients. They're grouped by purpose here.

## The 32 tools at a glance

### Storing

- [`add_content_memory`](./tools-storing.md#add_content_memory) — raw verbatim text
- [`add_intuitive_memory`](./tools-storing.md#add_intuitive_memory) — AI-extracted structured facts
- [`store_memories`](./tools-storing.md#store_memories) — batch store
- [`upload_document`](./tools-documents.md#upload_document) — small documents in one shot
- [`ingest`](./tools-documents.md#ingest) — alternate ingest entry point

### Finding

- [`query_memory`](./tools-finding.md#query_memory) — full pyramid + graph search
- [`search_memory`](./tools-finding.md#search_memory) — simple semantic search
- [`list_memories`](./tools-finding.md#list_memories) — filter-based browse
- [`get_memory`](./tools-finding.md#get_memory) — by ID

### Chronology

- [`get_timeline`](./tools-chronology.md#get_timeline) — bucketed list
- [`get_timeline_graph`](./tools-chronology.md#get_timeline_graph) — nodes + edges
- [`get_context_resume`](./tools-chronology.md#get_context_resume) — progressive precision

### Navigation & relations

- [`navigate_memory`](./tools-navigation.md#navigate_memory) — zoom in/out across pyramid
- [`update_memory`](./tools-navigation.md#update_memory) — edit content or relations
- [`force_link`](./tools-navigation.md#force_link) — create a relation
- [`remove_relation`](./tools-navigation.md#remove_relation) — delete a relation

### Abstraction

- [`start_abstraction_pipeline`](./tools-abstraction.md#start_abstraction_pipeline) — start background workers
- [`stop_abstraction_pipeline`](./tools-abstraction.md#stop_abstraction_pipeline) — pause workers
- [`trigger_abstraction`](./tools-abstraction.md#trigger_abstraction) — force one specific abstraction
- [`create_abstraction`](./tools-abstraction.md#create_abstraction) — manual layer creation

### Banks

- [`list_memory_banks`](./tools-banks.md#list_memory_banks)
- [`create_memory_bank`](./tools-banks.md#create_memory_bank)
- [`rename_memory_bank`](./tools-banks.md#rename_memory_bank)
- [`backup_bank`](./tools-banks.md#backup_bank)
- [`restore_bank`](./tools-banks.md#restore_bank)
- [`cleanup_resources`](./tools-banks.md#cleanup_resources) — delete models or banks

### Documents

- [`upload_document`](./tools-documents.md#upload_document)
- [`ingest`](./tools-documents.md#ingest)
- [`document_status`](./tools-documents.md#document_status)
- [`cancel_document`](./tools-documents.md#cancel_document)

### Introspection

- [`system_status`](./tools-introspection.md#system_status) — health + token stats
- [`health_check`](./tools-introspection.md#health_check) — config + live backend probe
- [`check_consistency`](./tools-introspection.md#check_consistency) — DB integrity
- [`help`](./tools-introspection.md#help) — built-in tool guidance

## How parameters work

Every tool takes a JSON object of parameters. Optional parameters can be omitted. Parameters marked as accepting a `bank` field default to the `"default"` bank if not provided.

### The common `bank` parameter

Most tools accept an optional `bank` string. If omitted, the tool operates on the `"default"` bank.

```json
{ "query": "JWT auth", "bank": "research" }
```text

### Agent and user IDs

| Field | Used by | Purpose |
|---|---|---|
| `agent_id` | store/find tools | The agent storing or querying. Set once via `--agent` CLI flag and inherited by all calls. |
| `user_id` | store tools | Optional. Only needed when multiple users share a bank. |

### Common response shape

Tools return a JSON object with at least:

```json
{
  "success": true,
  "message": "human-readable status",
  "data": { /* tool-specific */ },
  "error": null
}
```text

On failure, `success` is `false` and `error` contains a human-readable description.

## Reading the per-tool pages

Each category page has a one-paragraph description of the tool, the input schema (as a table), and the output schema. For the canonical schema, see `src/operations/tools.rs` in the source.

## Next

- [Storing memories](./tools-storing.md)
- [Finding memories](./tools-finding.md)
