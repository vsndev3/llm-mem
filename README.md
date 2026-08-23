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

## Benchmarks

**[LongMemEval-S](https://arxiv.org/abs/2410.10813)** (ICLR 2025) — retrieval-only protocol: full session histories are stored in the memory bank, and answers are generated from only the retrieved memories (mean ~10KB of excerpted context), never the full context (~115K tokens).

| Configuration | Questions | Accuracy | single-session-user | multi-session |
|---|---|---|---|---|
| Flat (L0, full sessions, top-10) | 100 | 70% | 90% | 23% |
| Flat (L0, excerpt retrieval, top-25) | 100 | **73%** | 89% | 37% |
| Pyramid balanced (L0-L3) | 10 | **80%*** | 80% | — |

\* Preliminary subset. Both configurations beat the paper's NaiveRAG baseline (~63% overall).

- **Excerpt retrieval**: chunk-level matches are resolved into compact excerpts (matched windows joined with `[...]`, session headers preserved) — same recall, ~7x smaller answer context
- **Answer generation**: gemma-4-E4B-it · **Independent verification**: Qwen3.5-122B (official `evaluate_qa.py`)
- **Throughput**: flat ~45s/question; pyramid ~16min/question (adds L0→L3 abstraction pipeline)
- The 100-question subset covers the first 100 LongMemEval-S items (70 single-session-user, 30 multi-session)

Reproduce: `bash scripts/bench_longmemeval_s.sh` (auto-resumes; see `scripts/setup_benchmarks.sh` for setup).

---

> **Beta software.** Expect rough edges and breaking changes. Not for production-critical data.

## Acknowledgements

The concept of layered memory for AI agents originated from [cortex-mem](https://github.com/Sopaco/cortex-mem). llm-mem has since diverged into a fundamentally different architecture:

- **Fully self-contained** — no external databases, APIs, or services required. Everything (vector store, LLM, embeddings) runs in-process.
- **Five-tier knowledge pyramid** (L0–L4) with document chunking, cross-linking, and concept/insight synthesis vs. three-tier session-based hierarchy.
- **Local-first GPU inference** via llama.cpp + fastembed with Vulkan/Metal/CUDA backends vs. API-only.

Built on `rmcp`, `lancedb`, `llama-cpp-2`, `fastembed`, `rig-core`, and `tokio`.

## License

MIT
