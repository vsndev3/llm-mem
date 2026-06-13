# llm-mem

**Persistent semantic memory for AI agents** — a single-binary MCP server that gives your AI assistant searchable, layered knowledge across sessions.

[![Documentation](https://img.shields.io/badge/docs-book-blue)](https://vsndev3.github.io/llm-mem/)

Runs entirely local by default: embedded vector database (LanceDB), local LLM (llama.cpp), local embeddings (fastembed). No API keys, no cloud, no setup.

```bash
cargo build --release
# or download the AppImage from Releases
```

```
src/
├── operations/       # High-level operations and MCP tool definitions
├── memory/           # Pipeline: ingestion, search, cache, extraction, scoring
├── layer/            # Abstraction pipeline and layer navigation
├── search/           # Pyramid assembler and graph traversal engine
├── llm/              # AI clients (local + API), circuit breaker, model downloader
├── lance_store.rs    # LanceDB vector store
├── memory_bank.rs    # Multi-tenant bank management
├── mcp.rs            # MCP server with embedded usage guide
├── config.rs         # Full configuration system
└── consistency.rs    # Database check/fix engine
```

---

## Documentation

Full documentation is at **[vsndev3.github.io/llm-mem](https://vsndev3.github.io/llm-mem/)** — covers the memory pyramid, installation, configuration, MCP tools, CLI, GPU acceleration, and architecture.

---

## Quick start

```bash
# Build
cargo build --release --features vulkan

# Run the MCP server
./target/release/llm-mem-mcp

# Or the interactive CLI
./target/release/llm-mem

# Connect to opencode — add to opencode.json:
{
  "mcp": {
    "llm-mem": {
      "type": "local",
      "command": ["/path/to/llm-mem-mcp"],
      "enabled": true
    }
  }
}
```

First run downloads models (~2.5 GB LLM + ~90 MB embeddings). GPU acceleration via `--features metal|vulkan|cuda`. Set `LLM_MEM_GPU_LAYERS=99` for full GPU offload. API backends (OpenAI, Anthropic, Ollama) supported via config.

```bash
# Run tests
cargo test --release --features "default,vulkan" -- --test-threads=1 --nocapture
```

---

> **Alpha software.** Expect rough edges and breaking changes. Not for production-critical data.

## Acknowledgements

The concept of layered memory for AI agents originated from [cortex-mem](https://github.com/Sopaco/cortex-mem). llm-mem has since diverged into a fundamentally different architecture:

- **Fully self-contained** — no external databases, APIs, or services required. Everything (vector store, LLM, embeddings) runs in-process.
- **Five-tier knowledge pyramid** (L0–L4) with document chunking, cross-linking, and concept/insight synthesis vs. three-tier session-based hierarchy.
- **Local-first GPU inference** via llama.cpp + fastembed with Vulkan/Metal/CUDA backends vs. API-only.

Built on `rmcp`, `lancedb`, `llama-cpp-2`, `fastembed`, `rig-core`, and `tokio`.

## License

MIT
