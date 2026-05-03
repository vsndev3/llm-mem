# llm-mem

A memory server for AI agents — it remembers things so your AI assistant doesn't forget.

Built as a single self-contained program in Rust. No databases to install, no cloud services required, no setup headaches. Just run it and it works. Everything runs locally on your machine by default, with no API keys needed and no data leaving your computer.

> [!WARNING]  
> **This is alpha software.** Expect rough edges, breaking changes, and experimental behavior. Use it to try things out, not for anything important.

---

## What it does

When connected to an AI assistant (Claude Desktop, Cursor, or any MCP client), llm-mem gives it real, persistent memory:

- **Remember anything** — conversations, facts, notes, or entire documents
- **Find by meaning** — semantic search that matches intent, not just keywords
- **Organize automatically** — facts are extracted, categorized, scored for importance, and deduplicated
- **Build layered knowledge** — raw notes get distilled upward into summaries, connections, concepts, and insights

Under the hood, it's an embedded vector database paired with a local AI engine, all in one binary.

---

## The memory pyramid

llm-mem doesn't just dump everything into a flat pile. It organizes memories in layers, similar to how human memory works — from concrete facts up to abstract understanding:

| Layer | Name | What it is |
|-------|------|-------------|
| **L0** | Raw content | Your conversations, notes, and documents as-is |
| **L1** | Summaries | AI-generated summaries and structural outlines |
| **L2** | Connections | Cross-document links and thematic relationships |
| **L3** | Concepts | Domain principles and theories |
| **L4** | Insights | Mental models, paradigms, and high-level patterns |

Background workers automatically create higher layers over time. You can search at any layer, zoom in to see sources of a concept, or zoom out to see what abstractions were built from something you stored.

When asked *"What pattern connects these accidental discoveries?"*, flat search fails — but pyramid search across layers can synthesize the answer.

---

## What you need

- **Rust 1.85+** (2024 edition)
- A C/C++ compiler and CMake (needed for the local AI engine)

If you don't want to run AI locally, you can skip the compiler requirement and use an API backend instead (OpenAI, OpenRouter, Anthropic, Ollama, etc.).

---

## Getting started

```bash
cargo build --release
```

That's it. Three programs land in `target/release/`:

| Program | Purpose |
|---------|---------|
| `llm-mem-mcp` | MCP server for AI assistants (Claude Desktop, Cursor, etc.) |
| `llm-mem` | Standalone command-line tool with interactive mode |
| `llm-mem-inspect` | Peek inside memory banks (no AI engine needed) |

On first run, it downloads the AI models it needs: a language model (~2.5 GB) and an embedding model (~90 MB). Downloads resume if interrupted, and proxies are respected.

### Connect to Claude Desktop

Add this to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memory": {
      "command": "/path/to/target/release/llm-mem-mcp",
      "args": ["--config", "/path/to/config.toml"]
    }
  }
}
```

### Connect to Cursor

Cursor picks up MCP servers from `~/.cursor/mcp.json`. Same idea — point it at `llm-mem-mcp`.

---

## What the AI gets

Once connected, your assistant gains 24 tools for working with memory:

### Storing
| Tool | What it does |
|------|-------------|
| `add_content_memory` | Save text verbatim — conversations, notes, snippets |
| `add_intuitive_memory` | Save content and let the AI extract key facts, entities, and structure |
| `upload_document` | Ingest a small document in one shot with automatic chunking |
| `begin_store_document` / `store_document_part` / `process_document` | Upload large documents in parts |

### Finding
| Tool | What it does |
|------|-------------|
| `query_memory` | Semantic search across any layer, with pyramid traversal, graph following, and keyword mode |
| `list_memories` | Browse by type, date, or other filters |
| `get_memory` | Look up a specific memory by ID |

### Understanding
| Tool | What it does |
|------|-------------|
| `navigate_memory` | Zoom in to see sources or zoom out to see higher abstractions |
| `update_memory` | Edit content or relations of an existing memory |
| `trigger_abstraction` | Manually force knowledge building for a specific memory |
| `start_abstraction_pipeline` / `stop_abstraction_pipeline` | Control background knowledge workers |

### Managing
| Tool | What it does |
|------|-------------|
| `list_memory_banks` / `create_memory_bank` | Organize memories into isolated banks |
| `backup_bank` / `restore_bank` | Versioned backups with SHA-256 verification |
| `rename_memory_bank` | Atomic rename of a bank and its sessions |
| `cleanup_resources` | Delete models or memory banks |

### Checking
| Tool | What it does |
|------|-------------|
| `system_status` | Full health check — model availability, pipeline status, layer distribution, active sessions |
| `status_process_document` / `list_document_sessions` / `cancel_process_document` | Document upload tracking |

### Search capabilities

The `query_memory` tool is particularly powerful:

- **Pyramid search** — distribute results proportionally across abstraction layers (bottom-heavy for facts, top-heavy for insight, or dynamically)
- **Graph traversal** — follow memory relations like `derived_from` or `mentions` up to 5 hops deep
- **Hybrid mode** — combines semantic similarity with keyword matching, with adjustable weights
- **Context filtering** — search only within a subset of memories by ID

---

## Memory banks

Memory banks are like folders for your AI's brain — separate, isolated stores of memories:

- **Per-project** — one bank for each codebase or project
- **Per-topic** — separate banks for cooking, finance, travel, etc.
- **Per-purpose** — short-term context vs long-term knowledge

A `default` bank is always there. Banks are created automatically the first time you use a name. Each bank is a separate database file, so memories in one bank are invisible to another.

---

## Configuration

llm-mem works **with zero configuration** — it runs everything locally out of the box.

If you want to customize, generate a template:

```bash
./target/release/llm-mem generate-config --output config.toml
```

The config has three main sections: the AI brain (`[llm]`), the search engine (`[embedding]`), and how memories behave (`[memory]`).

### Running everything locally (default)

No config file needed. The defaults use a Gemma 4 language model via llama.cpp and an all-MiniLM embedding model via fastembed.

### Using an online API

Set the LLM and/or embedding providers to `"api"` with your API key, endpoint, and model name. Works with OpenAI, OpenRouter, Anthropic, Ollama, and any OpenAI-compatible endpoint. You can even mix: online AI with local search, or local AI with API search.

### Environment variables

Every config option can be set via environment variables instead:

| Variable | What it sets |
|----------|-------------|
| `LLM_MEM_LLM_API_KEY` | AI completions API key |
| `LLM_MEM_LLM_API_BASE_URL` | AI API endpoint |
| `LLM_MEM_LLM_MODEL` | AI model name |
| `LLM_MEM_EMBEDDING_API_KEY` | Embedding API key |
| `LLM_MEM_EMBEDDING_API_BASE_URL` | Embedding endpoint |
| `LLM_MEM_EMBEDDING_MODEL` | Embedding model |
| `LLM_MEM_GPU_LAYERS` | GPU acceleration layers |
| `LLM_MEM_CONTEXT_SIZE` | Context window size |
| `LLM_MEM_TEMPERATURE` | AI creativity level |
| `LLM_MEM_MAX_TOKENS` | Max response length |
| `LLM_MEM_CPU_THREADS` | CPU threads (0 = auto) |
| `LLM_MEM_MAX_CONCURRENT_REQUESTS` | Parallel request limit |
| `LLM_MEM_MODELS_DIR` | Where to store models |
| `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` | Network proxy |

### Full config reference

<details>
<summary>All available settings</summary>

```toml
[llm]
provider = "local"                       # "local" or "api"
# --- Local mode ---
model_file = "gemma-4-E2B-it-Q8_0.gguf"
models_dir = "llm-mem-data/models"
gpu_layers = 0
context_size = 16644
cpu_threads = 0
auto_download = true
cache_model = true
use_grammar = false
llm_timeout_secs = 120
# --- API mode ---
# api_url = "https://api.openai.com/v1"
# api_key = ""
# model = "gpt-4o-mini"
# api_dialect = "openai-chat"             # one of: openai-chat, openai-completion, anthropic, ollama-chat, ollama-completion, custom
# request_format = "auto"                 # auto, rig, or raw
# use_structured_output = true
# structured_output_retries = 2
# custom_dialect = { endpoint_path = "...", body_template = "...", response_json_pointer = "..." }
# --- Shared ---
temperature = 0.7
max_tokens = 4096
max_concurrent_requests = 1
strip_tags = ["think"]
batch_size = 10
batch_max_tokens = 3000
batch_timeout_secs = 120
batch_timeout_multiplier = 1.0

[embedding]
provider = "local"
model = "all-MiniLM-L6-v2"
# api_url = "https://api.openai.com/v1"
# api_key = ""
# batch_size = 64
# timeout_secs = 30

[memory]
max_memories = 10000
similarity_threshold = 0.65
search_similarity_threshold = 0.2
max_search_results = 50
memory_ttl_hours = 0                       # 0 = no expiry
auto_enhance = true
deduplicate = true
merge_threshold = 0.75
auto_summary_threshold = 32768
max_content_length = 32768
document_chunk_size = 2000
use_llm_query_classification = false       # LLM-based query intent classification

[vector_store]
banks_dir = "llm-mem-data/banks"
store_type = "lancedb"

[vector_store.lancedb]
table_name = "memories"
database_path = "./lancedb"
embedding_dimension = 384

[server]
host = "0.0.0.0"
port = 8000

[logging]
enabled = false
log_directory = "llm-mem-data/logs"
level = "info"
max_size_mb = 1
max_files = 5
```

</details>

---

## Command-line tool

The `llm-mem` binary works in three modes: interactive REPL, single command, and batch from a file.

```bash
./target/release/llm-mem                       # Interactive mode
./target/release/llm-mem --single search --query "vegan recipes"
./target/release/llm-mem --batch commands.txt
```

Available commands:

| Command | What it does |
|---------|-------------|
| `search` | Semantic or text search |
| `list` | Browse memories with filters |
| `show` | Full details of a memory by ID |
| `stats` / `layer-stats` / `layer-tree` | Memory bank statistics and layer hierarchy |
| `export` | Export bank to JSON |
| `upload` / `begin-upload` / `upload-part` / `process-document` | Document ingestion |
| `doc-status` / `list-sessions` | Document upload tracking |
| `list-banks` | Show all memory banks |
| `system-status` | System health check |
| `generate-config` | Create a config file with defaults |
| `viz` | Real-time TUI visualization of processing |
| `clear-backoff` | Reset abstraction retry timers |
| `db export` | Export a bank to a portable `.db` file |
| `db merge` | Merge banks or .db files into one |
| `db check` | Detect consistency issues (orphans, stale states, hash mismatches) |
| `db fix` | Repair detected issues (with automatic backup) |
| `db rename` | Atomic rename of a bank |

In interactive mode, `use <bank>` switches active banks, and `savelog` dumps the log buffer to a file. Output can be formatted as table, JSON, JSONL, or CSV.

### Inspection tool

The `llm-mem-inspect` binary reads bank databases directly — no LLM initialization needed. It supports `list`, `show`, `export`, `stats`, `search`, `layer-stats`, `layer-tree`, and `list-banks`. Useful for debugging and data exploration without loading the full AI engine.

---

## Database management

Memory banks are standalone database files. Several operations are available:

- **Export** a bank to a portable `.db` file (with optional session data). The exported bank is ready for continued use.
- **Merge** multiple banks or exported `.db` files into one, with duplicate handling options (keep newest, keep first, keep all).
- **Check** database integrity — detects orphaned abstractions, stale states, missing embeddings, hash mismatches, unreferenced deleted memories, duplicate content, and invalid layer structure.
- **Fix** consistency issues automatically, with level-aware orphan handling and optional hard-deletion of unreferenced forgotten memories.
- **Rename** a bank atomically, moving both the memory database and its associated session database together.

---

## GPU acceleration

llm-mem can use your GPU to speed up local inference. By default it runs on CPU, which works everywhere. To use GPU:

| Platform | GPU support | Build flag |
|----------|-------------|------------|
| macOS | Apple Silicon (M1/M2/M3), Intel | `cargo build --release --features metal` |
| Linux/Windows | AMD, Intel, NVIDIA | `cargo build --release --features vulkan` |
| Linux/Windows | NVIDIA only | `cargo build --release --features cuda` |

Then set `gpu_layers = 20` (or higher) in your config to control how many model layers run on GPU.

---

## Performance

When running locally, the AI model loads into memory once at startup and stays there. Memory usage scales with request size — small queries use ~4K tokens, large ones scale to 16K+. Processing defaults to one-at-a-time for stability, configurable via `max_concurrent_requests`. CPU threads are auto-detected and can be overridden.

Background tasks (abstraction workers, document processing) use a priority queue system so interactive requests never get stuck behind batch work. A circuit breaker with exponential backoff handles API failures gracefully.

---

## Using as a library

```rust
use llm_mem::{Config, MemoryManager, MemoryMcpService, MemoryBankManager};

// Simplest: MCP service with defaults (local inference, no config needed)
let service = MemoryMcpService::new().await?;

// With a config file
let service = MemoryMcpService::with_config_path("config.toml").await?;

// Memory banks for multi-tenant isolation
let bank_manager = MemoryBankManager::new(banks_dir, llm_client, vector_config, memory_config)?;
let project_bank = bank_manager.get_or_create("my-project").await?;
```

---

## Running the tests

```bash
cargo test                              # All standard tests
cargo test --lib                        # Unit tests only
cargo test --test integration_tests     # Integration tests only
cargo test -- --nocapture               # With visible output
```

An optional evaluation suite measures retrieval quality with 15 curated memories and 20 test queries:

```bash
cargo test --test evaluation -- --ignored --nocapture
cargo test --test evaluation evaluation_retrieval_accuracy -- --ignored --nocapture
cargo test --test evaluation evaluation_real_llm -- --ignored --nocapture    # requires model download
```

The functional discovery test demonstrates the full L0→L3 pipeline using 9 documents about accidental discoveries (Penicillin, Post-it Notes, Microwave oven, etc.) and compares flat vs pyramid search:

```bash
cargo test --features local --test functional_discovery functional_real_pipeline -- --nocapture
```

These are skipped during normal `cargo test` — you opt in with `--ignored`.

---

## Manual model download

If auto-download doesn't work (corporate proxy, air-gapped machine), download models manually:

```bash
mkdir -p llm-mem-data/models
curl -L -o llm-mem-data/models/gemma-4-E2B-it-Q8_0.gguf \
  https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf
```

Then set `auto_download = false` in your config. A smaller alternative model (`smollm2-1.7b-instruct-q4_k_m.gguf`, ~1 GB) is also available but requires higher context size.

---

## Logging and debugging

Control log detail with `RUST_LOG`:

```bash
RUST_LOG=debug ./target/release/llm-mem-mcp        # Detailed
RUST_LOG=trace ./target/release/llm-mem-mcp        # Everything
RUST_LOG=warn,llama_cpp_2=debug ./target/release/llm-mem-mcp  # Focus on AI engine
```

File-based logging with rotation can be enabled in config:

```toml
[logging]
enabled = true
log_directory = "llm-mem-data/logs"
level = "info"
max_size_mb = 1
max_files = 5
```

---

## Architecture

The server speaks the Model Context Protocol (MCP) over stdio to AI assistants. Memory banks are multi-tenant — each bank has its own vector database (LanceDB) and processing pipeline. The AI engine is abstracted behind a trait, with two implementations: local via llama.cpp + fastembed, or remote via any OpenAI-compatible API.

Memory processing flows through a pipeline: extraction → classification → importance scoring → deduplication → storage. Background workers progressively create abstractions (L1 summaries from L0 content, L2 connections from L1 summaries, and upward). All of this is layered with a priority queue and circuit breaker for resilience.

Key source areas:

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

Built on: `rmcp` (MCP protocol), `lancedb` (vector store), `llama-cpp-2` + `fastembed` (local AI), `rig-core` (API client), `tokio` (async runtime).

---

## Acknowledgements

Inspired by [cortex-mem](https://github.com/Sopaco/cortex-mem) by Sopaco. The memory processing pipeline was reimplemented as a single self-contained crate with LanceDB as the embedded vector backend.

## License

MIT
