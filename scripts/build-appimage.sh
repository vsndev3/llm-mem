#!/usr/bin/env bash
# Build a self-contained llm-mem-mcp AppImage locally.
#
# Produces:
#   dist/llm-mem-mcp-x86_64.AppImage   (or -aarch64)
#   dist/llm-mem-mcp-x86_64.AppImage.zsync  (delta-update file)
#
# Usage:
#   scripts/build-appimage.sh                       # build for the host arch
#   scripts/build-appimage.sh --arch aarch64        # build for arm64
#   scripts/build-appimage.sh --no-docker           # build directly on host
#                                                   # (requires gcc 10+, cmake,
#                                                   #  Rust 1.94+, squashfs-tools,
#                                                   #  appimagetool)
#   scripts/build-appimage.sh --update-info 'gh-releases-zsync|vsndev3|llm-mem|latest|llm-mem-mcp-x86_64.AppImage.zsync'
#
# The resulting AppImage is built on a manylinux_2_28 base, statically
# links libstdc++/libgcc, dynamically links only glibc + (bundled) Vulkan
# loader. It runs on Ubuntu 20.04+, Debian 11+, RHEL 9+, Fedora, Arch,
# openSUSE, and any other modern Linux with glibc >= 2.28.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── argument parsing ──────────────────────────────────────────────────────
TARGET_ARCH=""
USE_DOCKER=1
UPDATE_INFO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)
            TARGET_ARCH="$2"
            shift 2
            ;;
        --no-docker)
            USE_DOCKER=0
            shift
            ;;
        --update-info)
            UPDATE_INFO="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$TARGET_ARCH" ]]; then
    HOST_ARCH="$(uname -m)"
    case "$HOST_ARCH" in
        x86_64)         TARGET_ARCH=x86_64 ;;
        aarch64|arm64)  TARGET_ARCH=aarch64 ;;
        *)
            echo "Unsupported host architecture: $HOST_ARCH" >&2
            exit 1
            ;;
    esac
fi

case "$TARGET_ARCH" in
    x86_64)  DOCKER_PLATFORM=linux/amd64;  DOCKER_ARCH=amd64;       MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64  ; APPIMAGE_ARCH=x86_64  ;;
    aarch64) DOCKER_PLATFORM=linux/arm64;  DOCKER_ARCH=arm64;       MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_aarch64 ; APPIMAGE_ARCH=aarch64 ;;
    *)
        echo "Unsupported --arch value: $TARGET_ARCH (use x86_64 or aarch64)" >&2
        exit 1
        ;;
esac

mkdir -p dist

if [[ "$USE_DOCKER" -eq 1 ]]; then
    command -v docker >/dev/null 2>&1 || { echo "docker not found in PATH" >&2; exit 1; }

    echo "==> Building llm-mem-mcp AppImage ($APPIMAGE_ARCH) via Docker..."
    echo "    base image:  $MANYLINUX_IMAGE"
    echo "    update info: ${UPDATE_INFO:-(none)}"
    echo

    # --load is needed so the scratch export stage's outputs are copied
    # to the local ./dist directory via buildkit's --output type=local.
    DOCKER_BUILDKIT=1 docker buildx build \
        --platform "$DOCKER_PLATFORM" \
        --build-arg "BASE_IMAGE=$MANYLINUX_IMAGE" \
        --build-arg "TARGETARCH=$DOCKER_ARCH" \
        ${UPDATE_INFO:+--build-arg "UPDATE_INFO=$UPDATE_INFO"} \
        --file docker/appimage/Dockerfile \
        --output "type=local,dest=$REPO_ROOT/dist" \
        --progress=plain \
        "$REPO_ROOT"
else
    echo "==> Building llm-mem-mcp AppImage ($APPIMAGE_ARCH) directly on host..."
    echo "    Make sure you have: gcc >= 10, cmake, ninja, Rust 1.94+, squashfs-tools, appimagetool"
    echo

    export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-static-libstdc++ -C link-arg=-static-libgcc"
    export CXXFLAGS="${CXXFLAGS:-} -static-libstdc++ -static-libgcc"
    export LDFLAGS="${LDFLAGS:-} -static-libstdc++ -static-libgcc"

    cargo build --release \
        --bin llm-mem-mcp \
        --bin llm-mem \
        --features local --locked

    mkdir -p dist
    rm -rf AppDir
    cp -r packaging/appimage AppDir
    install -m 0755 target/release/llm-mem-mcp AppDir/usr/bin/llm-mem-mcp
    install -m 0755 target/release/llm-mem     AppDir/usr/bin/llm-mem

    ARCH="$APPIMAGE_ARCH" appimagetool \
        --comp zstd \
        ${UPDATE_INFO:+--updateinformation "$UPDATE_INFO"} \
        AppDir "dist/llm-mem-mcp-${APPIMAGE_ARCH}.AppImage"

    rm -rf AppDir
fi

echo
echo "==> Done. Artifacts:"
ls -lh dist/

# Smoke test the resulting AppImage (best effort; may fail if FUSE is
# missing and APPIMAGE_EXTRACT_AND_RUN is not honored on this system).
APPIMAGE_FILE="$(ls dist/llm-mem-mcp-${APPIMAGE_ARCH}.AppImage 2>/dev/null | head -1 || true)"
if [[ -n "$APPIMAGE_FILE" ]] && [[ -x "$APPIMAGE_FILE" ]]; then
    echo
    echo "==> Smoke test: $APPIMAGE_FILE --version"
    if APPIMAGE_EXTRACT_AND_RUN=1 "$APPIMAGE_FILE" --version 2>&1 | head -5; then
        echo "    ✓ AppImage launches and prints version"
    else
        echo "    ! AppImage launch test failed (likely missing /dev/fuse or fuse package)"
        echo "      Install libfuse2 (Debian/Ubuntu) or fuse (Fedora) and retry."
    fi
fi
