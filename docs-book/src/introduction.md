# Introduction

llm-mem is a **memory server for AI agents**. It gives your AI assistant persistent, searchable memory that works across sessions — conversations, codebases, documents, and anything else you want it to remember.

It's a single self-contained binary written in Rust. No databases to install, no cloud services required, no API keys needed by default. Everything runs locally on your machine, and data never leaves your computer.

> [!WARNING]
> **Alpha software** —
>
> This is alpha software. Expect rough edges, breaking changes, and experimental behavior. Use it to try things out, not for anything important.

## How it works

When connected to an AI assistant via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), llm-mem exposes a set of tools the assistant can call to:

- **Store** raw text, documents, or AI-extracted facts as memories
- **Search** by meaning (semantic search) or by keyword
- **Browse** chronologically with timeline tools
- **Navigate** a layered pyramid from raw content up to abstract insights
- **Manage** isolated memory banks, backups, and document uploads

The assistant calls these tools just like any other MCP tool — no special prompting required.

## What makes it different

| Feature | llm-mem |
|---|---|
| **Local-first** | Runs entirely on your machine by default. No cloud, no accounts, no data leaves your box. |
| **Embedded vector store** | Uses [LanceDB](https://lancedb.com/) embedded. No separate database server. |
| **Layered memory** | L0 raw → L1 summaries → L2 connections → L3 concepts → L4 insights, with auto-abstraction. |
| **AI in-process** | Bundled [llama.cpp](https://github.com/ggerganov/llama.cpp) for local LLM, [fastembed](https://github.com/Anush008/fastembed-rs) for local embeddings. Zero-config. |
| **MCP-native** | Speaks MCP stdio. Drop into opencode, VS Code Copilot, or any MCP client. |
| **Standalone CLI** | Includes an interactive REPL and single-command mode for shell scripting. |

## Next steps

- New here? Start with [What is llm-mem?](./what-is-llm-mem.md) for the conceptual overview.
- Ready to install? Jump to [Installation](./installation.md).
- Want to skim the tool surface? See [Tools overview](./tools-overview.md).
