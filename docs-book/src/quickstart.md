# Quickstart

A 10-minute tour: install, connect to an AI assistant, and ask it to remember something.

## 0. Prerequisites

- llm-mem installed (see [Installation](./installation.md))
- An MCP client. We'll use [opencode](./opencode.md) here, but VS Code Copilot works too.

## 1. Start the server in MCP mode

```bash
# Either run the MCP server directly, or use the AppImage multicall
llm-mem-mcp
```

The server will:
- Look for a config file (or use defaults)
- Download models on first run (~3 GB total, resumable)
- Start listening on stdio

Leave this running, or background it:

```bash
# It runs as a foreground process — your MCP client launches it on demand.
# For manual testing:
llm-mem-mcp
```

> [!NOTE]
> You don't usually run the MCP server by hand — the MCP client launches it. The above is for sanity-checking that it starts and prints its banner.

## 2. Connect opencode

In your project directory (or globally), create `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "llm-mem": {
      "type": "local",
      "command": ["/path/to/llm-mem-mcp-x86_64.AppImage"],
      "enabled": true
    }
  }
}
```

Or if you built from source with `cargo build --release`:

```json
{
  "mcp": {
    "llm-mem": {
      "type": "local",
      "command": ["/absolute/path/to/llm-mem/target/release/llm-mem-mcp"],
      "enabled": true
    }
  }
}
```

Start (or restart) opencode. It discovers the MCP server and lists the 32 llm-mem tools in its tool palette.

## 3. Tell the AI to remember something

Open opencode and chat. Try:

> *Use the llm-mem tools to remember that the AuthService in this project handles JWT token validation and refresh — entry point is `src/auth/service.rs:45-80`. The refresh tokens expire after 7 days.*

The AI will call `add_content_memory` (or `add_intuitive_memory` if it wants the LLM to extract structured facts).

## 4. Ask it back later

In a fresh conversation (or after restarting the AI):

> *Where is JWT validation handled, and what's the refresh token expiry?*

The AI will call `query_memory` or `search_memory` and return the stored fact with the source attribution.

## 5. Try semantic search

The search works by meaning, not just keywords. Try:

> *Find anything I stored about user authentication*

Even if you stored *"AuthService handles JWT token validation"*, the search will find it because the embedding captures the meaning of "user authentication".

## 6. Add a document

> *Upload the file `docs/architecture.md` to memory.*

The AI calls `upload_document`, which:
1. Chunks the file
2. Embeds each chunk
3. Stores them as L0 memories
4. Triggers the L0→L1 worker to create summaries in the background

Later queries about the document return the relevant chunks.

## 7. Look at the timeline

> *Show me what I stored in the last 2 days, grouped by day.*

The AI calls `get_timeline` with the `--since 2d` equivalent. You get a chronological list grouped by day.

## 8. Navigate the pyramid

> *Zoom out from the JWT memory — what higher-level insights were built from it?*

The AI calls `navigate_memory` with `zoom_out`. You get the L1/L2/L3 memories that were synthesized from the original L0 entry.

## What just happened

You went from zero to a working AI memory in 10 minutes. The same patterns work for:

- Storing entire codebases (one bank per repo, `upload_document` for each file)
- Research projects (one bank per topic, semantic search to find related work)
- Personal notes (one bank, multiple `event_at`-tagged entries, timeline views)
- Meeting logs (one bank, intuitive memories for each decision)

## Next

- [MCP clients overview](./mcp-clients-overview.md) — connect other clients beyond opencode
- [Tools overview](./tools-overview.md) — the full 32-tool reference
- [Configuration](./config-file.md) — customize the server
