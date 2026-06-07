# Installation

There are five ways to install llm-mem, listed from least to most effort:

1. **Pre-built AppImage** (Linux) — single file, no install, no compiler
2. **Pre-built native binary** (Linux/macOS/Windows) — download from Releases
3. **`cargo install`** — Rust toolchain required
4. **Build from source** — for contributors / custom builds
5. **Docker build** — for the AppImage itself (advanced)

For most users, **option 1 or 2 is the right choice**.

## Option 1: Pre-built AppImage (Linux, recommended)

A single-file AppImage is available that works on every modern Linux distribution without glibc version issues. It bundles both binaries (`llm-mem-mcp` and `llm-mem`) in one file, dispatching by `argv[0]` (multicall pattern). Includes LanceDB, the Vulkan loader, and a bundled OpenSSL.

```bash
# 1. Download from the GitHub Releases page
#    https://github.com/vsndev3/llm-mem/releases
#    pick: llm-mem-mcp-x86_64.AppImage   (or -aarch64)

# 2. Make it executable
chmod +x llm-mem-mcp-x86_64.AppImage

# 3. Run the MCP server (default when invoked as llm-mem-mcp)
./llm-mem-mcp-x86_64.AppImage --help

# 4. (Optional) Use the CLI via a symlink
ln -s "$(pwd)/llm-mem-mcp-x86_64.AppImage" ~/bin/llm-mem
llm-mem --help
llm-mem --single search --query "vegan recipes"
```text

The first run downloads the AI models it needs:

- A language model: **Gemma 4 E2B** (`gemma-4-E2B-it-Q8_0.gguf`, ~2.5 GB)
- An embedding model: **all-MiniLM-L6-v2** (~90 MB)

Downloads resume if interrupted. HTTP/HTTPS/SOCKS proxies are respected via `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` environment variables.

If FUSE isn't available on your system (e.g. some container environments), set:

```bash
export APPIMAGE_EXTRACT_AND_RUN=1
```text

> [!NOTE]
> The AppImage downloads a separate copy of the models per invocation. To share models across runs, set `LLM_MEM_MODELS_DIR` to a stable path.

## Option 2: Pre-built native binary (Linux, macOS, Windows)

Native binaries are built for the host platform's typical glibc/libc version. They don't bundle a C++ runtime or OpenSSL, so they rely on system libraries.

| Archive | Platform | Architecture | GPU |
|---|---|---|---|
| `llm-mem-mcp-linux-x86_64.tar.gz` | Linux (Ubuntu 24.04+) | x86_64 | Vulkan |
| `llm-mem-mcp-linux-aarch64.tar.gz` | Linux ARM64 | aarch64 | Vulkan |
| `llm-mem-mcp-macos-x86_64.tar.gz` | macOS 15+ (Intel) | x86_64 | Metal |
| `llm-mem-mcp-macos-aarch64.tar.gz` | macOS 15+ (Apple Silicon) | arm64 | Metal |
| `llm-mem-mcp-windows-x86_64.zip` | Windows 10/11 | x86_64 | Vulkan |

Install:

```bash
# Linux / macOS
tar xzf llm-mem-mcp-linux-x86_64.tar.gz
sudo mv llm-mem-mcp /usr/local/bin/

# Windows (PowerShell)
Expand-Archive llm-mem-mcp-windows-x86_64.zip
move llm-mem-mcp.exe C:\Windows\System32\
```text

On first run, the same auto-download kicks in. On Windows, you may need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) for the local LLM engine.

## Option 3: `cargo install`

If you have a Rust toolchain installed and want the latest from `crates.io` (when published):

```bash
cargo install llm-mem --features local
```text

This requires a C/C++ compiler and CMake on the system. Build time is ~10-20 minutes the first time (compile llama.cpp + fastembed native code).

## Option 4: Build from source

For contributors, custom builds, or to get unreleased features:

```bash
git clone https://github.com/vsndev3/llm-mem.git
cd llm-mem

# Default build: local LLM (llama.cpp) + local embeddings (fastembed) + LanceDB
cargo build --release

# Artifacts land in target/release/:
#   llm-mem-mcp  — MCP server
#   llm-mem     — standalone CLI
```text

### Build feature combinations

| Command | What you get |
|---|---|
| `cargo build --release` | Default: local LLM + local embeddings + LanceDB (fully offline) |
| `cargo build --release --no-default-features --features lancedb` | API-only, no local AI |
| `cargo build --release --no-default-features --features lancedb,local-embed` | API LLM + local embeddings |
| `cargo build --release --features local,vulkan` | Local + Vulkan GPU |
| `cargo build --release --features local,metal` | Local + Metal GPU (macOS) |
| `cargo build --release --features local,cuda` | Local + CUDA GPU (NVIDIA) |
| `cargo build --release --no-default-features --features lancedb,vector-lite` | API + VectorLite (legacy) |

See [GPU acceleration](./gpu-acceleration.md) for details on the GPU backends.

## Option 5: Build the AppImage yourself

If you want to produce an AppImage for a custom feature combination or to test changes:

```bash
git clone https://github.com/vsndev3/llm-mem.git
cd llm-mem

# Local Docker build (auto-detects docker or podman)
scripts/build-appimage.sh

# Native build (no Docker) — requires gcc 10+, cmake, appimagetool
scripts/build-appimage.sh --no-docker

# Build for ARM64 from an x86_64 host
scripts/build-appimage.sh --arch aarch64

# Build with zsync auto-update metadata
scripts/build-appimage.sh --update-info 'gh-releases-zsync|vsndev3|llm-mem|latest|llm-mem-mcp-x86_64.AppImage.zsync'
```text

Output: `dist/llm-mem-mcp-x86_64.AppImage` (or `-aarch64`) plus a `.zsync` delta-update file.

The CI workflow `.github/workflows/appimage.yml` is **manual-trigger only** — it produces x86_64 and aarch64 artifacts in the Actions UI without auto-publishing a release.

## Verifying the install

```bash
# MCP server
llm-mem-mcp --version
llm-mem-mcp --help

# CLI
llm-mem --help
```text

Both binaries should print their version and exit. If the binary is missing or won't run, see [Troubleshooting](./troubleshooting.md).

## Next

- [First run](./first-run.md) — what happens on the very first invocation
- [Quickstart](./quickstart.md) — get from install to working memory in 10 minutes
