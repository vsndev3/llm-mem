# AppImage Distribution

Self-contained, single-file AppImage builds of `llm-mem-mcp` that work on
**every modern Linux distribution** without users having to deal with
glibc version mismatches, missing system libraries, or manual dependency
installation.

## What you get

For each supported architecture (currently `x86_64` and `aarch64`):

- **One file**: `llm-mem-mcp-<arch>.AppImage` (~80–150 MB)
- **chmod +x and run** — no extraction, no installer, no root required
- **No glibc mismatch** — built on `manylinux_2_28` (glibc 2.28, the
  oldest still-supported in modern distros). The binary uses
  forward-compatible dynamic glibc and is statically linked to the C++
  runtime, so it runs unchanged on Ubuntu 20.04+, Debian 11+, RHEL 9+,
  Fedora 35+, Arch, openSUSE Leap 15.4+, etc.
- **No libstdc++/libgcc version mismatch** — those are statically linked
  into the binary
- **No OpenSSL host dependency** — `reqwest` in `llm-mem` is configured
  with `rustls-tls`, but the `lancedb → lance-namespace → hf-hub → reqwest`
  transitive chain still links OpenSSL (Cargo feature unification
  propagates `default-tls` from `hf-hub`). The AppImage bundles
  `libssl` and `libcrypto` from the build container, and the binary has
  `DT_RUNPATH=$ORIGIN/../lib` so it loads the bundled versions
  regardless of what's installed on the host. `AppRun` also sets
  `LD_LIBRARY_PATH` to the bundled lib dir as a safety net
- **Bundled Vulkan loader** — `libvulkan.so.1` is included in the
  AppImage, so the host's Vulkan loader version doesn't matter
- **No FUSE required** — `AppRun` forces `APPIMAGE_EXTRACT_AND_RUN=1`,
  so the AppImage works in containers, restricted SSH, and other
  environments where `/dev/fuse` is unavailable
- **LanceDB works** — its prebuilt `.so` is bundled
- **GPU works** — Vulkan ICD drivers (NVIDIA, AMD, Intel) are loaded
  from the host at runtime via the bundled loader. CUDA works as long
  as the host has the NVIDIA driver user-mode libraries

## What you trade away

- The AppImage is a single executable file but it's **larger** than a
  raw binary (it bundles a SquashFS image with the loader, libraries,
  metadata, and icon)
- **First-run startup** is ~200–500 ms slower than a raw binary (the
  extract-and-run untar). For an MCP server that runs for the whole
  agent session, this is invisible
- Some file managers won't recognize the `.AppImage` MIME type out of
  the box; users have to `chmod +x` manually

## How the MCP stdio transport works through the AppImage

```
┌─────────────────┐
│  MCP client     │  Claude Code / Claude Desktop / Cursor
│  (parent)       │
└────────┬────────┘
         │ fork + execve
         │ stdin/stdout = pipes
         ▼
┌─────────────────────────────────────────┐
│  AppImage ELF (type2-runtime)           │
│  • mount()s or extracts the SquashFS    │
│  • execve()s AppRun, preserving FDs 0,1,2
└────────┬────────────────────────────────┘
         │ execve
         ▼
┌─────────────────────────────────────────┐
│  AppRun (multicall dispatcher)          │
│  • exports APPIMAGE_EXTRACT_AND_RUN=1   │
│  • sets LD_LIBRARY_PATH to bundled libs │
│  • dispatches to llm-mem-mcp or llm-mem │
│    based on argv[0] / env / --cli flag  │
└────────┬────────────────────────────────┘
         │ exec
         ▼
┌─────────────────────────────────────────┐
│  llm-mem-mcp                            │
│  • reads JSON-RPC from stdin (FD 0)     │
│  • writes JSON-RPC to stdout (FD 1)     │
└─────────────────────────────────────────┘
```

`execve()` preserves open file descriptors across the process image
replacement. This is a fundamental Unix guarantee — the kernel copies
the `files_struct` to the new process. No special handling is required
in `AppRun`; the pipe FDs simply flow through.

## Building locally

### Via Docker (recommended)

```bash
scripts/build_appimage.sh
```

This builds for the host architecture. To build for aarch64 from an
x86_64 host, you need Docker buildx with cross-arch support (or just
run on an aarch64 host).

Output goes to `dist/`:

```
dist/
├── llm-mem-mcp-x86_64.AppImage
├── llm-mem-mcp-x86_64.AppImage.zsync
└── llm-mem-mcp-x86_64.AppImage.sha256
```

To embed a GitHub release update information string (used by
`AppImageUpdate` for delta updates):

```bash
scripts/build_appimage.sh \
  --update-info 'gh-releases-zsync|vsndev3|llm-mem|latest|llm-mem-mcp-x86_64.AppImage.zsync'
```

### Directly on the host (no Docker)

Requires `gcc ≥ 10`, `cmake`, `ninja`, `Rust 1.94+`, `squashfs-tools`,
`appimagetool`.

```bash
scripts/build_appimage.sh --no-docker
```

## Building on CI

`.github/workflows/appimage.yml` provides a **manual-trigger** workflow
(no auto-publish to GitHub Releases). To produce a build:

1. GitHub UI → Actions → **Build AppImage (manual)** → Run workflow
2. Wait ~10–15 minutes (Rust toolchain download + llama.cpp compile + AppImage assembly)
3. Download the artifact from the workflow run page

The workflow produces both `x86_64` and `aarch64` AppImages in parallel
and runs a cross-glibc smoke test (Ubuntu 20.04 with glibc 2.31 and
Ubuntu 22.04 with glibc 2.35) to confirm the forward-compatibility
claim.

## Using the AppImage

The AppImage bundles **two** binaries — `llm-mem-mcp` (MCP server) and
`llm-mem` (CLI) — and `AppRun` dispatches based on how the AppImage was
invoked. This is the standard "multicall" pattern (cf. `busybox`,
`toybox`, OpenWrt's `/bin/[`).

### Default: the MCP server

```bash
chmod +x llm-mem-mcp-x86_64.AppImage
./llm-mem-mcp-x86_64.AppImage --help
```

When invoked directly (with `argv[0]` ending in `.AppImage` or matching
`llm-mem-mcp`), `AppRun` execs the MCP server binary.

### Use the CLI via a symlink

```bash
ln -s "$(pwd)/llm-mem-mcp-x86_64.AppImage" ~/bin/llm-mem
llm-mem --help
llm-mem --single search --query "vegan recipes"
llm-mem --batch commands.txt
```

`argv[0]` becomes `llm-mem` → `AppRun` execs the CLI binary. Same
AppImage, different program.

### Explicit override

```bash
LLM_MEM_BIN=cli ./llm-mem-mcp-x86_64.AppImage --help
./llm-mem-mcp-x86_64.AppImage --cli --help
LLM_MEM_BIN=mcp ./llm-mem-mcp-x86_64.AppImage --help
./llm-mem-mcp-x86_64.AppImage --mcp --help
```

### Dispatch order (priority)

1. `LLM_MEM_BIN` env var (`mcp` or `cli`)
2. `argv[0]` basename — `llm-mem` → CLI, `llm-mem-mcp*` or `*AppImage` → MCP
3. First arg `--cli` or `--mcp`
4. Default: `mcp`

## Using the AppImage with an MCP client

### Claude Code

`~/.claude.json`:

```json
{
  "mcpServers": {
    "llm-mem": {
      "command": "/home/you/Applications/llm-mem-mcp-x86_64.AppImage",
      "args": []
    }
  }
}
```

### Claude Desktop

`~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "llm-mem": {
      "command": "/home/you/Applications/llm-mem-mcp-x86_64.AppImage",
      "args": []
    }
  }
}
```

### Cursor / Continue / Windsurf / etc.

Same `command` + `args` pattern. The AppImage is a single executable
file as far as the MCP client is concerned.

## Troubleshooting

### "dlopen(): error loading libfuse.so.2" or "AppImages require FUSE to run"

You're running an older type 1 AppImage. The current build uses type 2
with `APPIMAGE_EXTRACT_AND_RUN=1` set in `AppRun`, so this should not
happen with a freshly built artifact. If you see this, check that:

1. You downloaded the latest artifact from the workflow run
2. The file is executable (`chmod +x llm-mem-mcp-x86_64.AppImage`)
3. The download was complete (`sha256sum -c *.sha256`)

### "AppImages need FUSE" in a container

The build forces `APPIMAGE_EXTRACT_AND_RUN=1` so it should work in any
environment. If you see this error, set the env var explicitly:

```bash
APPIMAGE_EXTRACT_AND_RUN=1 ./llm-mem-mcp-x86_64.AppImage
```

### "GLIBC_2.X not found" on the target host

This means the host has a glibc older than 2.28 (e.g., CentOS 7 with
glibc 2.17 or Debian 9 with glibc 2.24). The current build does not
support these ancient systems. The trade-off was chosen deliberately —
supporting glibc 2.17 would require either downgrading the C++ toolchain
(unreliable for llama.cpp's C++17/20 features) or bundling a custom
glibc (large, complex, and breaks NSS / TLS / DNS lookups).

If you genuinely need glibc 2.17 support, open an issue and we can
investigate either a `manylinux2014` build or a separate
"legacy-glibc" AppImage.

### First-run latency

The first invocation of the AppImage per `$TMPDIR` lifetime extracts
the SquashFS (~200–500 ms). Subsequent runs on the same `$TMPDIR` are
near-instant. To avoid this on a slow filesystem, set
`APPIMAGE_EXTRACT_AND_RUN=0` and install `libfuse2` (Debian/Ubuntu) or
`fuse` (Fedora/RHEL) — the FUSE mount path is faster.

### "noexec" mount

If the AppImage is on a `noexec` filesystem, `execve` will fail.
Move it to your home directory or another executable mount.

### Verifying the artifact

```bash
# Check it's a valid AppImage
file llm-mem-mcp-x86_64.AppImage
# → "ELF 64-bit LSB executable, x86-64, ... GNU/Linux ..."

# Check the dynamic dependencies
APPIMAGE_EXTRACT_AND_RUN=1 \
  llm-mem-mcp-x86_64.AppImage --version

# Inspect the contents without running
llm-mem-mcp-x86_64.AppImage --appimage-list | head -20

# Verify the SHA256
sha256sum -c llm-mem-mcp-x86_64.AppImage.sha256
```

## Architecture decisions

### Why manylinux_2_28 (not _2_17)?

`manylinux_2_17` (PEP 600) targets CentOS 7 / glibc 2.17. We chose
`manylinux_2_28` (glibc 2.28) because:

- llama.cpp master requires GCC ≥ 7 with C++17/20 features
- The prebuilt `lancedb` `.so` is built against a recent glibc
- Ubuntu 20.04 (April 2020) and RHEL 9 (May 2022) are the oldest
  currently-supported major distros on each family, both with glibc ≥ 2.28

### Why static-link libstdc++/libgcc but not glibc?

- **libstdc++ / libgcc**: forward-incompatible symbols (e.g.,
  `GLIBCXX_3.4.30`) cause runtime errors. Static linking is easy and
  adds ~2 MB to the binary.
- **glibc**: cannot be statically linked (LGPL + missing symbol
  resolution at load time). Forward-compat works for free — build on
  the minimum glibc you want to support.

### Why rustls + bundled OpenSSL (both)?

We use `rustls` in `llm-mem`'s own code (no host OpenSSL needed for our
HTTP clients), but the `lancedb → lance-namespace → hf-hub → reqwest`
transitive chain pulls in `reqwest` with `default-tls` (OpenSSL) via
Cargo feature unification — there's no way to remove that feature
short of forking `hf-hub`. The AppImage therefore bundles
`libssl.so` and `libcrypto.so` from the build container, and the
binary has `DT_RUNPATH=$ORIGIN/../lib` so it loads the bundled versions
regardless of what the host has. Size cost: ~5–8 MB.

### Why bundle a Vulkan loader but not CUDA?

- **Vulkan loader** (`libvulkan.so.1`) is a tiny, ABI-stable library
  that just loads the system ICD driver. Bundling it adds ~200 KB and
  removes one source of version mismatch.
- **CUDA runtime** (`libcudart.so`) is part of the NVIDIA driver and
  is matched to the kernel-mode driver version. Bundling it would
  require the user to install a matching driver anyway, and shipping
  our own CUDA runtime is a license and security nightmare. The
  AppImage uses the host's `libcudart.so` (and friends) via the
  system's dynamic linker.

## Files in this directory

- `AppRun` — multicall entry point script (forces extract-and-run, sets
  `LD_LIBRARY_PATH`, dispatches to `llm-mem-mcp` or `llm-mem` based on
  `argv[0]` / `LLM_MEM_BIN` / `--cli` / `--mcp`)
- `llm-mem-mcp.desktop` — freedesktop.org desktop entry (always launches the MCP server)
- `llm-mem-mcp.png` — symlink to the 256×256 icon
- `usr/bin/llm-mem-mcp` — populated by the build script (the MCP stdio server)
- `usr/bin/llm-mem` — populated by the build script (the interactive CLI)
- `usr/lib/libssl.so*`, `usr/lib/libcrypto.so*` — bundled from the manylinux build container
- `usr/lib/libvulkan.so.1` — bundled Vulkan loader (built from source)
- `usr/share/icons/hicolor/256x256/apps/llm-mem-mcp.png` — actual icon file
- `usr/share/metainfo/llm-mem-mcp.appdata.xml` — AppStream metadata (lists both binaries)
