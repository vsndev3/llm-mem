# Introspection

Tools for inspecting the server's health, configuration, and database integrity.

## `system_status`

> [!TIP]
> **Call this first** —
>
> The recommended best practice is to call `system_status` at the start of every session. It returns backend info, model availability, and token usage stats so you know what you're working with.

### Input

None.

### Output

```json
{
  "success": true,
  "data": {
    "backend": "local",                  // "local", "api", or mixed
    "state": "ready",                    // "ready" / "loading" / "error"
    "llm_model": "gemma-4-E2B-it-Q8_0.gguf",
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_available": true,
    "embedding_available": true,
    "total_llm_calls": 42,
    "total_embedding_calls": 128,
    "total_prompt_tokens": 56789,
    "total_completion_tokens": 12345,
    "details": { /* backend-specific extra info */ }
  }
}
```

`details` is opaque per backend. Local backend includes things like GPU layer count, model size, llama.cpp backend info. API backend includes things like last API call duration, error rate.

## `health_check`

Run a full end-to-end check. **Static** checks (config validity, build features, API key presence, banks dir writability) are always free. **Live** checks (`live: true`) also issue a tiny `embed("ping")` and `complete("...pong")` against the backend to confirm it actually responds. Live checks consume a few API tokens per call.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `live` | boolean | | Default `false`. If `true`, run live backend probes. |
| `embed_only` | boolean | | Live mode only: only probe the embedding endpoint. |
| `llm_only` | boolean | | Live mode only: only probe the LLM endpoint. |
| `embed_timeout_secs` | integer | | Default 15. |
| `llm_timeout_secs` | integer | | Default 30. |

### Output

```json
{
  "success": true,
  "data": {
    "healthy": true,
    "backend": "local",
    "llm_model": "...",
    "embedding_model": "...",
    "live_run": true,
    "checks": [
      {
        "name": "config_valid",
        "category": "config",
        "status": "pass",
        "detail": "All required fields present",
        "duration_ms": 1
      },
      {
        "name": "build_features_match",
        "category": "build",
        "status": "pass",
        "detail": "local-llm and local-embed features enabled",
        "duration_ms": 0
      },
      {
        "name": "live_embed",
        "category": "live",
        "status": "pass",
        "detail": "Embedded 'ping' to dim 384 in 234ms",
        "duration_ms": 234
      },
      {
        "name": "live_llm",
        "category": "live",
        "status": "pass",
        "detail": "Completed '...pong' in 1.2s",
        "duration_ms": 1203
      }
    ]
  }
}
```

`healthy: false` means at least one check failed. Inspect the `checks` array for which one and the `detail` field for the reason.

This tool is also available as a CLI subcommand: `llm-mem health-check [--live]`.

## `check_consistency`

Scan one or all banks for integrity issues:

- Orphaned abstractions (L1+ with no L0 source)
- Stale state (sessions stuck in `processing` past their timeout)
- Missing embeddings
- Hash mismatches (content changed since embedding was generated)
- Unreferenced forgotten memories
- Duplicate content
- Invalid layer structure

The companion CLI command is `llm-mem db check [--all] [--verbose]`.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `bank` | string | | Specific bank. If omitted, checks all. |
| `verbose` | boolean | | Include detailed issue info. Default false. |

### Output

```json
{
  "success": true,
  "data": {
    "issues": [
      {
        "type": "orphaned-abstraction",
        "memory_id": "uuid",
        "severity": "warn",
        "detail": "L2 memory references missing L0 source"
      }
    ],
    "issue_count": 1,
    "bank": "default"
  }
}
```

To fix issues, use the CLI: `llm-mem db fix [--dry-run] [--purge]`.

## `help`

Returns the embedded usage guide that the MCP server exposes to AI clients. Includes best practices, the layered memory architecture explanation, memory type vocabulary, and tips.

### Input

None.

### Output

```json
{
  "success": true,
  "data": {
    "guide": {
      "overview": "...",
      "layered_memory_architecture": { /* L0 - L4+ */ },
      "domain_patterns": { /* codebase, documents, web, conversations */ },
      "banks_and_user_id": { /* when to use banks vs user_id */ },
      "memory_types": { /* vocabulary */ },
      "critical_guidelines": [ /* ... */ ],
      "tips": [ /* ... */ ]
    }
  }
}
```

This is the same content an AI client sees when it calls the MCP `prompts/list` endpoint. The AI is expected to read this and adapt its tool-use strategy accordingly.

## Debugging flow

1. `system_status` — is the server even ready?
2. `health_check` (with `live: true`) — is the backend actually responsive?
3. `check_consistency` — is the database intact?
4. Check the [log file](./logging-debugging.md) for errors
5. Run the same tool manually from the CLI to isolate MCP from application issues

## Next

- [Banks & backups](./tools-banks.md)
- [Logging & debugging](../logging-debugging.md)
