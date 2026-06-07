# Memory banks

Banks are isolated memory stores. Each bank is a separate database file with its own memories, sessions, and statistics. Memories in one bank are invisible to another.

## Why use multiple banks?

| Use case | Strategy |
|---|---|
| Per-project | One bank per codebase, switch via the `bank` parameter or `--bank` flag |
| Per-topic | One bank for cooking, another for finance, another for travel |
| Per-environment | One bank for dev, one for production data |
| Per-user | If you share a machine, separate banks per user |
| Per-purpose | Short-term context (auto-cleaned) vs. long-term knowledge (persistent) |

The **`default` bank is always available** — it's created on first run. Use additional banks when you want hard isolation.

## Naming

Bank names are free-form strings but follow these rules:

- Allowed: letters, digits, `-`, `_`
- Avoid spaces (use `-` or `_`)
- The name `default` is reserved
- Case-sensitive

Good names: `project-x`, `research-2026`, `meeting-notes`, `personal`.

## Creating banks

Banks are created automatically on first write to a new name. The server handles this transparently:

```jsonc
// MCP call to add_content_memory
{
  "content": "Postgres over MongoDB for analytics",
  "bank": "project-x"     // <-- creates project-x if it doesn't exist
}
```text

You can also pre-create with `create_memory_bank` (MCP) or `mkdir + populate` on disk.

## Listing banks

```bash
# CLI
llm-mem list-banks
```text

```json
// MCP: list_memory_banks
{}
```text

Output:

```json
{
  "banks": [
    { "name": "default", "memory_count": 1234, "size_bytes": 45678901 },
    { "name": "research", "memory_count": 567, "size_bytes": 12345678 }
  ],
  "count": 2
}
```text

## Switching the active bank

**MCP**: pass `bank: "<name>"` on every call. There's no global "active bank" concept at the protocol level.

**CLI REPL**: `use <bank>` switches the active bank for the session.

**CLI single-command**: `--bank <name>` per command.

## Isolation guarantees

- Cross-bank searches are impossible (a tool can only see one bank at a time)
- Cross-bank relations don't exist
- The abstraction pipeline operates per-bank (L0→L1 in bank A doesn't see content from bank B)
- Backups and exports are per-bank

## Soft grouping within a bank

For softer grouping without full isolation, use `context` tags and `topics`:

- `context` — broad topic tags (e.g. `["project-x", "auth", "jwt"]`)
- `topics` — specific subjects within the content

You can then filter by context/topic in queries:

```jsonc
// MCP: query_memory
{
  "query": "JWT auth",
  "context_tags": ["project-x"]
}
```text

## Bank vs user_id

`user_id` is a per-memory filter within a bank. It's optional and only useful when multiple users share a single bank.

In most cases:

- **Single user per bank**: omit `user_id`
- **Multiple users per bank**: set `user_id` to differentiate

If two users share a bank and you don't filter by `user_id`, each will see the other's memories. Use a separate bank if you want hard isolation per user.

## Lifecycle

| Operation | Tool | CLI |
|---|---|---|
| Create | `create_memory_bank` | (auto, or `mkdir`) |
| List | `list_memory_banks` | `list-banks` |
| Rename | `rename_memory_bank` | `db rename` |
| Backup | `backup_bank` | `db export` |
| Restore | `restore_bank` | `db import` |
| Delete | `cleanup_resources` (target: "banks") | `rm -rf` the bank directory |

## Next

- [Backups & restore](./backups.md)
- [Data directory layout](./data-directory.md)
