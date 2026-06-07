# What you need

The requirements depend on which build you choose and which AI backend you use.

## For a default (fully-local) build

- **Rust 1.85+** (2024 edition) — only required if building from source via `cargo`
- **A C/C++ compiler and CMake** — needed to compile the bundled `llama.cpp` and `fastembed` native code
- **~3 GB of free disk space** — for the language model (~2.5 GB) and embedding model (~90 MB), downloaded on first run
- **~4 GB of RAM** — the LLM lives in memory once loaded

If you don't want to compile a local AI engine, see the [API-only build](#api-only-build) below.

## For an API-only build

- **Rust 1.85+** — only required if building from source
- **An API key** for an OpenAI-compatible endpoint (OpenAI, OpenRouter, Anthropic, Ollama, llama-server, etc.)
- **No C/C++ toolchain** — native inference is disabled in this build

## For the AppImage

- **Linux with glibc ≥ 2.28** — Ubuntu 20.04+, Debian 11+, RHEL 9+, Fedora, Arch, openSUSE
- **libfuse2** (Debian/Ubuntu) or `fuse` (Fedora) — usually pre-installed. Only needed if you don't set `APPIMAGE_EXTRACT_AND_RUN=1`
- **No compiler toolchain** — fully self-contained
- **~3 GB disk** for models on first run

## For GPU acceleration (optional but recommended)

| Platform | GPU support | Build flag |
|---|---|---|
| macOS (Apple Silicon) | M1/M2/M3 | `--features metal` |
| Linux/Windows (AMD, Intel, NVIDIA) | Vulkan | `--features vulkan` |
| Linux/Windows (NVIDIA only) | CUDA | `--features cuda` |

GPU acceleration makes the LLM substantially faster (often 5-20x) but is not required.

## For document parsing

Document parsing uses pure-Rust libraries by default. No system dependencies required for:

- Markdown, plain text, CSV, TSV, JSON, JSONL, YAML, TOML, XML
- PDF (via `lopdf`)
- DOCX, XLSX, XLS (via `calamine`)
- ZIP archives
- PNG, JPEG, GIF, WebP images (descriptions via local LLM if `vision_enabled = true`)

## MCP client requirements

The MCP server (`llm-mem-mcp`) speaks stdio and is compatible with any MCP client. Confirmed working with:

- [opencode](https://opencode.ai) (recommended for local development)
- VS Code Copilot (with the MCP extension)
- Any other MCP-compatible client (Claude Code, Cursor, etc.)

## Next

Ready to install? See [Installation](./installation.md).
