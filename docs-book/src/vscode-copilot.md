# VS Code Copilot

VS Code's Copilot Chat supports MCP servers via the MCP extension. Once connected, Copilot gains the 32 llm-mem tools as part of its tool palette.

## 1. Install prerequisites

- **VS Code 1.85+**
- **GitHub Copilot Chat extension**
- **MCP support** is built into VS Code 1.85+ (no extra extension needed)

## 2. Choose config location

VS Code reads MCP servers from two places:

| Location | Scope | Path |
|---|---|---|
| **Workspace** | This project only | `<workspace>/.vscode/mcp.json` |
| **User profile** | All your workspaces | Use **MCP: Open User Configuration** from the command palette |

For a single project, use workspace scope (`.vscode/mcp.json`).

For a global setup that follows you across projects, use user scope.

## 3. Add llm-mem

Create `.vscode/mcp.json` in your workspace root:

```json
{
  "servers": {
    "llm-mem": {
      "type": "stdio",
      "command": "/absolute/path/to/llm-mem-mcp",
      "args": []
    }
  }
}
```text

### AppImage variant

```json
{
  "servers": {
    "llm-mem": {
      "type": "stdio",
      "command": "/home/you/apps/llm-mem-mcp-x86_64.AppImage",
      "args": []
    }
  }
}
```text

### From-source variant

```json
{
  "servers": {
    "llm-mem": {
      "type": "stdio",
      "command": "/home/you/projects/llm-mem/target/release/llm-mem-mcp",
      "args": []
    }
  }
}
```text

### With environment variables

```json
{
  "servers": {
    "llm-mem": {
      "type": "stdio",
      "command": "/path/to/llm-mem-mcp",
      "args": [],
      "env": {
        "LLM_MEM_MODELS_DIR": "/home/you/.cache/llm-mem/models",
        "RUST_LOG": "info"
      }
    }
  }
}
```text

## 4. Reload VS Code

After saving the config, run **MCP: List Servers** from the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`). You should see `llm-mem` listed with a green dot indicating the server started successfully.

If the dot is red or yellow, see [Troubleshooting MCP](./troubleshooting-mcp.md).

## 5. Use it

Open Copilot Chat (the chat panel in the sidebar, or `Ctrl+Shift+I` / `Cmd+Shift+I`).

In agent mode, Copilot will automatically call the MCP tools when relevant. Try:

> *What llm-mem tools are available?*

Copilot will list the 32 tools. Then:

> *Remember that the AuthService uses bcrypt for password hashing, see src/auth/hasher.rs:23.*
> *Search llm-mem for any auth-related facts.*
> *Upload README.md to the default memory bank.*

## Server lifecycle

VS Code spawns the MCP server on demand when you first use a tool, and shuts it down when VS Code quits. If you edit `.vscode/mcp.json`, you need to reload the server:

1. Open command palette
2. Run **MCP: Restart Server** (or **MCP: List Servers** → click the server)

## Multiple servers

Add multiple entries under `servers`:

```json
{
  "servers": {
    "llm-mem-default": {
      "type": "stdio",
      "command": "/path/to/llm-mem-mcp",
      "args": []
    },
    "llm-mem-research": {
      "type": "stdio",
      "command": "/path/to/llm-mem-mcp",
      "args": ["--config", "/path/to/research.toml"]
    }
  }
}
```text

Each shows up as a separate tool group in Copilot.

## Troubleshooting

- **Server not found**: confirm the absolute path with `ls` or `file <path>` from a terminal
- **Permission denied on AppImage**: `chmod +x` it
- **AppImage fails to launch**: VS Code may not have FUSE access. Try `APPIMAGE_EXTRACT_AND_RUN=1` in the `env` block
- **Models not downloading**: check `~/.cache/llm-mem/models` and your network/proxy settings
- **Tools don't appear**: reload the server from the command palette

See [Troubleshooting MCP](./troubleshooting-mcp.md) for more.

## Next

- [Troubleshooting MCP](./troubleshooting-mcp.md) — common connection issues
- [Tools overview](./tools-overview.md) — the full tool surface
