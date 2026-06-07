# MCP clients overview

llm-mem is an MCP server. It speaks the [Model Context Protocol](https://modelcontextprotocol.io/) over **stdio** (standard input/output) and registers 32 tools that the AI assistant calls.

Any MCP client can connect to it. The configuration is a JSON object that tells the client how to launch the server process.

## General shape

Every MCP client config has the same essential fields:

```json
{
  "<mcp-key>": {
    "type": "local",
    "command": "/absolute/path/to/llm-mem-mcp",
    "args": [],
    "env": {}
  }
}
```

| Field | Meaning |
|---|---|
| `type: "local"` | The server runs as a local subprocess (the only kind llm-mem supports) |
| `command` | Absolute path to the `llm-mem-mcp` binary (or AppImage) |
| `args` | Command-line arguments (optional — see [CLI flags](./cli-flags.md)) |
| `env` | Environment variables to set when launching (optional) |

The `command` must be **absolute**. The client launches it as a child process and pipes JSON-RPC over its stdin/stdout.

## Confirmed clients

| Client | Status | Config key | Docs |
|---|---|---|---|
| [opencode](./opencode.md) | Recommended | `mcp.<name>` in `opencode.json` | [opencode docs](https://opencode.ai/docs/) |
| [VS Code Copilot](./vscode-copilot.md) | Supported | `servers.<name>` in `.vscode/mcp.json` | VS Code MCP docs |
| Claude Code | Compatible | `mcpServers` in `.mcp.json` | Anthropic docs |
| Cursor | Compatible | `mcpServers` in `~/.cursor/mcp.json` | Cursor docs |
| Zed | Compatible | `context_servers` in `~/.config/zed/settings.json` | Zed docs |

The configuration format varies by client. The `command` and behavior are the same.

## Choosing between AppImage and direct binary

**AppImage** (`llm-mem-mcp-x86_64.AppImage`):
- One file, works on any glibc ≥ 2.28 distro
- Bundled OpenSSL + C++ runtime
- Larger download (~150 MB) but zero install
- Best for: portable, no-system-deps deployment

**Native binary** (`llm-mem-mcp`):
- Smaller, depends on system libc/libstdc++
- Released per platform (Linux x86_64, macOS aarch64, etc.)
- Best for: when you control the runtime environment

**From-source build** (`target/release/llm-mem-mcp`):
- After `cargo build --release`
- Best for: development, custom feature combinations

Pick the binary that matches how you [installed](./installation.md) llm-mem, and use the absolute path in your MCP config.

## Environment variables to set

The `env` map in the MCP config can set any of the [environment variables](./env-vars.md) llm-mem reads. Common ones:

```json
{
  "env": {
    "LLM_MEM_MODELS_DIR": "/home/you/.cache/llm-mem/models",
    "LLM_MEM_LLM_API_KEY": "sk-...",
    "HTTPS_PROXY": "http://proxy.corp:8080",
    "RUST_LOG": "info"
  }
}
```

> [!TIP]
> **Don't put secrets in version control** —
>
> MCP config files are often committed to source repos. Use environment variables or a secret manager for API keys, not literal values in the config.

## Verifying the connection

After updating your MCP config:

1. Restart the client (opencode, VS Code, etc.) — it spawns the server as a subprocess
2. Look in the client's MCP/tool panel — you should see 32 `llm-mem-*` tools
3. Ask the AI to call `system_status` — the response confirms the server is reachable and the models are loaded

If the tools don't appear, see [Troubleshooting MCP](./troubleshooting-mcp.md).

## Next

- [opencode](./opencode.md) — the recommended client
- [VS Code Copilot](./vscode-copilot.md) — for VS Code users
- [Troubleshooting MCP](./troubleshooting-mcp.md) — common connection issues
