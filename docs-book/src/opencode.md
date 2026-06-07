# opencode

[opencode](https://opencode.ai) is the recommended MCP client for llm-mem. It's an open-source terminal UI for AI coding assistants with first-class MCP support.

## 1. Install opencode

Follow the [opencode install instructions](https://opencode.ai/docs/). The quick version:

```bash
# macOS / Linux
curl -fsSL https://opencode.ai/install | bash

# Or via Homebrew
brew install opencode

# Or via npm
npm install -g @opencode/cli
```text

## 2. Locate your `opencode.json`

opencode reads a config from one of two places:

- **Project-local**: `<your-project>/opencode.json` — only affects this project
- **Global**: `~/.config/opencode/config.json` — affects every project

For a project-local config (most common for code work):

```bash
cd /path/to/your/project
touch opencode.json
```text

For a global config:

```bash
mkdir -p ~/.config/opencode
touch ~/.config/opencode/config.json
```text

## 3. Add llm-mem

Edit the file you created. The minimum configuration:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "llm-mem": {
      "type": "local",
      "command": ["/absolute/path/to/llm-mem-mcp"],
      "enabled": true
    }
  }
}
```text

### If you installed the AppImage

```json
{
  "mcp": {
    "llm-mem": {
      "type": "local",
      "command": ["/home/you/apps/llm-mem-mcp-x86_64.AppImage"],
      "enabled": true
    }
  }
}
```text

Make the AppImage executable first:

```bash
chmod +x /home/you/apps/llm-mem-mcp-x86_64.AppImage
```text

### If you built from source

```json
{
  "mcp": {
    "llm-mem": {
      "type": "local",
      "command": ["/home/you/projects/llm-mem/target/release/llm-mem-mcp"],
      "enabled": true
    }
  }
}
```text

### If you have multiple memory banks

You can declare multiple llm-mem instances, one per bank:

```json
{
  "mcp": {
    "llm-mem-default": {
      "type": "local",
      "command": ["/path/to/llm-mem-mcp"],
      "enabled": true
    },
    "llm-mem-research": {
      "type": "local",
      "command": ["/path/to/llm-mem-mcp"],
      "args": ["--config", "/path/to/research-config.toml"],
      "enabled": true
    }
  }
}
```text

The first one is the default; the second uses a separate config that points to a different `banks_dir`.

## 4. Add env vars (optional)

Common additions:

```json
{
  "mcp": {
    "llm-mem": {
      "type": "local",
      "command": ["/path/to/llm-mem-mcp"],
      "enabled": true,
      "env": {
        "LLM_MEM_MODELS_DIR": "/home/you/.cache/llm-mem/models",
        "RUST_LOG": "info"
      }
    }
  }
}
```text

See [Environment variables](./env-vars.md) for the full list.

## 5. Restart opencode and verify

```bash
opencode
```text

The first MCP handshake can take 10-30 seconds (model download on first run). Watch the status indicator — when it turns green, the server is ready.

In opencode, run:

```
/tools
```text

You should see 32 llm-mem tools listed. Try:

```
> Call system_status with llm-mem
```text

If the response includes `"backend": "local"` and `"llm_available": true`, the connection is working.

## 6. Use it

From here on, the AI automatically has access to the llm-mem tools. You can just talk normally:

> *Remember that the AuthService refresh tokens expire after 7 days.*
> *What did I store about authentication?*
> *Upload docs/architecture.md to memory.*
> *Show me what I stored this week.*

The AI decides which tool to call based on your request.

## Troubleshooting

If the server doesn't start or opencode can't reach it, see [Troubleshooting MCP](./troubleshooting-mcp.md).

Common issues specific to opencode:
- **Config not picked up**: confirm the path with `opencode --print-config` or check the opencode logs
- **Permission denied on AppImage**: `chmod +x` it
- **Server starts but no tools appear**: check the opencode version supports MCP (>= 0.5.x)
- **Stale config**: restart opencode after every config change — it doesn't hot-reload MCP servers

## Next

- [VS Code Copilot](./vscode-copilot.md) — alternative client
- [Tools overview](./tools-overview.md) — what the AI can do with the connected server
