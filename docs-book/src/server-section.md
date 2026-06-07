# `[server]` — HTTP server

The MCP transport is always stdio. The `[server]` section is reserved for an optional HTTP/management interface that isn't currently used by the MCP server.

```toml
[server]
host = "0.0.0.0"
port = 8000
```text

| Field | Default | What it controls |
|---|---|---|
| `host` | `"0.0.0.0"` | Bind address (currently informational). |
| `port` | `8000` | Bind port (currently informational). |

> [!NOTE]
> This section is reserved for future use. The MCP server only speaks stdio. Standalone use of the `[server]` section is not yet supported.

For practical server management, see [Operations](./data-directory.md) and [Logging & debugging](./logging-debugging.md).
