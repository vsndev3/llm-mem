# Banks & backups

Tools for managing multiple isolated memory banks, backing them up, and cleaning up resources.

## `list_memory_banks`

List all memory banks in the configured `banks_dir`.

### Input

None.

### Output

```json
{
  "success": true,
  "data": {
    "banks": [
      {
        "name": "default",
        "memory_count": 1234,
        "size_bytes": 45678901,
        "created_at": "..."
      },
      {
        "name": "research",
        "memory_count": 567,
        "size_bytes": 12345678,
        "created_at": "..."
      }
    ],
    "count": 2,
    "banks_dir": "/var/lib/llm-mem/banks"
  }
}
```text

## `create_memory_bank`

Create a new bank. The bank is also created implicitly on the first write to a new name, so this is mostly for documentation or to pre-create a known structure.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | ✓ | Bank name. Allowed chars: letters, digits, `-`, `_`. |
| `description` | string | | Human-readable description. |

### Output

```json
{
  "success": true,
  "message": "Memory bank 'research' ready",
  "data": {
    "bank": {
      "name": "research",
      "memory_count": 0,
      "size_bytes": 0
    }
  }
}
```text

## `rename_memory_bank`

Atomically rename a bank. Moves both the memory database and the session database.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `old_name` | string | ✓ | |
| `new_name` | string | ✓ | |

### Output

```json
{
  "success": true,
  "message": "Bank renamed from 'foo' to 'bar'",
  "data": { "old_name": "foo", "new_name": "bar" }
}
```text

> [!WARNING]
> If any AI client is currently using the old name, it'll see a "bank not found" error after the rename. Update the client config (or recreate the bank) before continuing.

## `backup_bank`

Create a versioned backup of a bank. Backups are SHA-256 verified and timestamped.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | | Bank to back up. Default: `default`. |
| `destination` | string | | Directory for the backup. Default: `~/llm-mem-backups/`. |

### Output

```json
{
  "success": true,
  "message": "Bank 'default' backed up successfully (v3)",
  "data": {
    "backup_path": "/home/you/llm-mem-backups/default-2026-02-15-v3.db",
    "manifest": {
      "version": 3,
      "created_at": "2026-02-15T...",
      "memory_count": 1234,
      "sha256": "...",
      "size_bytes": 45678901
    }
  }
}
```text

## `restore_bank`

Restore a bank from a backup file. Two modes:

- `replace` (default) — overwrite the current bank. **Requires `confirm: true`** as a safety check.
- `merge` — import the backup into the existing bank. Duplicates are skipped by content hash.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | | Bank to restore into. Default: `default`. |
| `source` | string | ✓ | Path to the backup `.db` file. |
| `mode` | string | | `replace` (default) or `merge`. |
| `confirm` | boolean | | Required: `true` for `replace` mode. |

### Output

```json
{
  "success": true,
  "message": "Bank 'default' restored from backup (replace mode)",
  "data": {
    "restored_path": "/var/lib/llm-mem/banks/default.lance",
    "source": "/path/to/backup.db"
  }
}
```text

For `merge` mode, the response also includes `imported`, `skipped_duplicates`, and `total_after_merge`.

## `cleanup_resources`

Delete models or memory banks. Two `target` values:

- `"models"` — delete the entire `models_dir` (then re-create it empty). The next run will re-download.
- `"banks"` — delete a specific bank by name. Requires the literal confirmation phrase.

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `target` | string | | `"models"` (default) or `"banks"`. |
| `name` | string | (for `target: "banks"`) | Bank to delete. |
| `confirm` | boolean or string | | For `target: "models"`: `confirm: true`. For `target: "banks"`: `confirm: "I confirm this data will be permanently lost"`. |

### Output

```json
{
  "success": true,
  "message": "Models directory cleaned up successfully. ..."
}
```text

> [!DANGER]
> **Destructive** —
>
> `cleanup_resources` with `target: "banks"` permanently deletes the bank's database. There is no undo. The confirmation phrase is a safety net — never bypass it.

## Backup strategy recommendations

| Use case | Recommendation |
|---|---|
| Personal use, low churn | Weekly backup, keep last 4. |
| Multi-project, active | Daily backup, keep last 7. |
| Production / shared | Continuous backup, off-site storage, restore tested. |

Backups are stored as `.db` files plus a manifest. They include the bank database but **not** the documents in the upload session state. To capture document sessions too, use `db export --include-sessions` (CLI).

## Next

- [Document upload](./tools-documents.md)
- [Introspection](./tools-introspection.md)
