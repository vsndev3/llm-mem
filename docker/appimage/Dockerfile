# syntax=docker/dockerfile:1.7
#
# Multi-stage Dockerfile to build an llm-mem-mcp AppImage.
#
# Target: glibc 2.28 (manylinux_2_28) — runs on Ubuntu 20.04+, Debian 11+,
# RHEL 9+, Fedora 35+, Arch, openSUSE Leap 15.4+, and any other modern Linux
# distribution with a newer glibc. C++ runtime (libstdc++/libgcc_s) is
# statically linked into the binary so the host's libstdc++ version does
# not matter.
#
# Build arguments:
#   BASE_IMAGE   The manylinux base image to use. Defaults to
#                quay.io/pypa/manylinux_2_28_x86_64. Pass an aarch64 image
#                (e.g. quay.io/pypa/manylinux_2_28_aarch64) to cross- or
#                natively build the arm64 AppImage.
#   RUST_VERSION  Pinned Rust toolchain. Defaults to 1.94.1.
#   FEATURES      Cargo features passed to `cargo build --features`.
#                 Defaults to "local". Example: "local,vulkan".
#   APPIMAGETOOL_VERSION  Pinned appimagetool release. Defaults to continuous
#                         (resolved at build time via GitHub API).
#
# Build (from the project root, host arch is x86_64):
#   docker build -f docker/appimage/Dockerfile \
#     --build-arg BASE_IMAGE=quay.io/pypa/manylinux_2_28_x86_64 \
#     --tag llm-mem-appimage:local \
#     --output type=local,dest=./dist .
#
# Build for aarch64 (run on aarch64 host, or with buildx --platform):
#   docker buildx build --platform linux/arm64 \
#     --build-arg BASE_IMAGE=quay.io/pypa/manylinux_2_28_aarch64 \
#     -f docker/appimage/Dockerfile -o type=local,dest=./dist .

ARG BASE_IMAGE=quay.io/pypa/manylinux_2_28_x86_64
ARG RUST_VERSION=1.94.1

# ── Stage 1: build the binary ─────────────────────────────────────────────
FROM ${BASE_IMAGE} AS builder

ARG RUST_VERSION
ARG FEATURES=local
ARG TARGETARCH

ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:$PATH \
    # Build flags: dynamic glibc linking is preserved (glibc cannot be
    # statically linked and is forward-compatible). libstdc++/libgcc_s/
    # libgomp are *bundled* in the AppImage (see below) so the binary
    # does not need a matching GLIBCXX on the host. -Wl,-rpath embeds
    # an RPATH of $ORIGIN/../lib so the dynamic linker finds the
    # bundled copies first.
    RUSTFLAGS="-C link-arg=-Wl,-rpath,\$\$ORIGIN/../lib" \
    PKG_CONFIG_ALL_STATIC=0

# Install Rust (stable) and build dependencies. The manylinux images ship
# with devtoolset, but we use the system GCC 10 which is sufficient for
# glibc 2.28 and libstdc++ GLIBCXX_3.4.25.
RUN set -eux; \
    yum install -y \
        cmake \
        gcc \
        gcc-c++ \
        make \
        ninja-build \
        perl \
        pkgconfig \
        file \
        findutils \
        git \
        ca-certificates \
        unzip \
        wget \
        xz \
        zlib-devel \
        openssl-devel \
        patchelf \
        # Required by `bindgen` (used transitively by llama-cpp-sys-2) to
        # parse C/C++ headers and generate Rust FFI bindings. The package
        # on AlmaLinux 8 is `clang`, which ships `libclang.so`.
        clang \
    && \
    # The AlmaLinux 8 `protobuf-compiler` package ships protoc 3.5.x, which
    # is too old for lance-encoding 4.0.0's build script (it passes
    # `--experimental_allow_proto3_optional`, a flag only present in
    # protoc 3.15.0-3.19.x). Download protoc 3.19.6 directly and prepend
    # it to PATH so the lance-encoding build script finds a compatible
    # version. Newer protoc 3.20+ removed the flag, so we pin to 3.19.6.
    PROTOC_VERSION=3.19.6 && \
    case "${TARGETARCH:-amd64}" in \
        amd64|x86_64) PROTOC_ARCH=x86_64 ;; \
        arm64|aarch64) PROTOC_ARCH=aarch64 ;; \
    esac && \
    curl -fsSL -o /tmp/protoc.zip \
        "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-${PROTOC_ARCH}.zip" && \
    unzip -q -o /tmp/protoc.zip -d /opt/protoc && \
    rm /tmp/protoc.zip && \
    ln -sf /opt/protoc/bin/protoc /usr/local/bin/protoc && \
    /usr/local/bin/protoc --version && \
    curl -fsSL --proto '=https' --tlsv1.2 https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain "${RUST_VERSION}" --profile minimal --no-modify-path \
    && \
    rustc --version && cargo --version

# Pre-warm the cargo registry with the project dependencies for faster
# incremental builds and better layer caching.
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo 'fn main(){}' > src/main.rs && \
    echo '' > src/lib.rs && \
    cargo build --release --bin llm-mem-mcp --features "${FEATURES}" 2>/dev/null || true && \
    rm -rf src

# Build the actual binaries. The AppImage packages both:
#   - llm-mem-mcp: the MCP stdio server (default program)
#   - llm-mem:     the interactive CLI / batch tool
# AppRun dispatches based on argv[0] (multicall pattern).
#
# ONNX Runtime (via absl) references __libc_single_threaded, a symbol
# introduced in glibc 2.32. The manylinux_2_28 base image ships glibc
# 2.28 which lacks it, so we provide a real writable zero-initialized
# stub. Unlike --Wl,--defsym (which creates an absolute symbol at
# address 0, crashing glibc >= 2.32 when it tries to write to it), a
# compiled C variable lives in .bss where glibc can safely update it.
# Marked __attribute__((weak)) so on hosts with glibc >= 2.32 the real
# libc symbol takes precedence, avoiding conflicts.
RUN printf '__attribute__((weak)) int __libc_single_threaded = 0;\n' > /tmp/libc_stub.c && \
    gcc -c -o /tmp/libc_stub.o /tmp/libc_stub.c
COPY build.rs ./
COPY src ./src

RUN RUSTFLAGS="$RUSTFLAGS -C link-arg=/tmp/libc_stub.o" \
    cargo build --release \
        --bin llm-mem-mcp \
        --bin llm-mem \
        --features "${FEATURES}" \
        --locked

# Capture the build container's C/C++ runtime libraries for bundling.
# llama.cpp and ONNX Runtime link against libstdc++/libgcc_s/libgomp.
# We bundle the build container's exact versions so the AppImage does
# not need a matching GLIBCXX on the host. Transitive deps (fastembed
# -> hf-hub -> reqwest with default-tls) link the system OpenSSL,
# which is also bundled.
# Note: `cp -avL` dereferences symlinks (libgcc_s.so.1 on RHEL is a
# symlink to libgcc_s-8-20210514.so.1; the actual file is needed).
RUN set -eux; \
    openssl version; \
    mkdir -p /opt/bundled-libs; \
    cp -avL /usr/lib64/libssl.so*    /opt/bundled-libs/ || true; \
    cp -avL /usr/lib64/libcrypto.so* /opt/bundled-libs/ || true; \
    cp -avL /usr/lib64/libstdc++.so* /opt/bundled-libs/ || true; \
    cp -avL /usr/lib64/libgomp.so*   /opt/bundled-libs/ || true; \
    cp -avL /usr/lib64/libgcc_s.so*  /opt/bundled-libs/ || true; \
    ls -la /opt/bundled-libs/

# ── Stage 2: assemble the AppDir ──────────────────────────────────────────
FROM ${BASE_IMAGE} AS packager

# Install appimagetool. We download the latest continuous build of the
# official Go-based appimagetool from the AppImage/appimagetool repository.
ARG APPIMAGETOOL_REF=continuous
ARG TARGETARCH

# appimagetool is itself an AppImage. Inside the Docker build context
# there is no FUSE device, so we tell it (and the final packaging step
# below) to extract-and-run instead of mounting via FUSE.
ENV APPIMAGE_EXTRACT_AND_RUN=1

RUN set -eux; \
    yum install -y \
        curl \
        file \
        findutils \
        git \
        ca-certificates \
        cmake \
        gcc \
        gcc-c++ \
        make \
        ninja-build \
        patchelf \
        squashfs-tools \
        wget \
        fuse \
        fuse-libs \
        zlib \
        openssl \
        openssl-libs \
    && \
    case "${TARGETARCH}" in \
        amd64|x86_64) AIT_ARCH=x86_64 ;; \
        arm64|aarch64) AIT_ARCH=aarch64 ;; \
        *) echo "unsupported TARGETARCH=${TARGETARCH}"; exit 1 ;; \
    esac && \
    echo "Downloading appimagetool for ${AIT_ARCH}..." && \
    curl -fsSL -o /usr/local/bin/appimagetool \
        "https://github.com/AppImage/appimagetool/releases/download/${APPIMAGETOOL_REF}/appimagetool-${AIT_ARCH}.AppImage" && \
    chmod +x /usr/local/bin/appimagetool && \
    appimagetool --version

WORKDIR /work

# Copy the built binaries and the AppDir scaffolding.
COPY --from=builder /src/target/release/llm-mem-mcp ./AppDir/usr/bin/llm-mem-mcp
COPY --from=builder /src/target/release/llm-mem     ./AppDir/usr/bin/llm-mem
COPY --from=builder /opt/bundled-libs/ ./AppDir/usr/lib-bundled/
COPY packaging/appimage/ ./AppDir/
COPY README.md LICENSE ./

# Place bundled libs in the standard library path. The binary's RPATH
# ($ORIGIN/../lib) points here. Bundled: libssl, libcrypto, libstdc++,
# libgcc_s, libgomp. Bundling the C++ runtime avoids any GLIBCXX
# version mismatch on the host.
RUN set -eux; \
    mkdir -p AppDir/usr/lib; \
    for libpattern in libssl.so* libcrypto.so* libstdc++.so* libgcc_s.so* libgomp.so*; do \
        if compgen -G "AppDir/usr/lib-bundled/$libpattern" > /dev/null; then \
            mv AppDir/usr/lib-bundled/$libpattern AppDir/usr/lib/; \
        fi; \
    done; \
    rmdir AppDir/usr/lib-bundled 2>/dev/null || true; \
    ls -la AppDir/usr/lib/

# Normalize permissions inside the AppDir.
RUN chmod 0755 \
        AppDir/AppRun \
        AppDir/usr/bin/llm-mem-mcp \
        AppDir/usr/bin/llm-mem && \
    find AppDir -type d -exec chmod 0755 {} + && \
    find AppDir -type f -exec chmod 0644 {} + && \
    chmod 0755 \
        AppDir/AppRun \
        AppDir/usr/bin/llm-mem-mcp \
        AppDir/usr/bin/llm-mem

# Verify the dynamic dependencies now that bundled libs are in place.
# Force the dynamic loader to look in AppDir/usr/lib/ via LD_LIBRARY_PATH.
# Bundled libs (libssl, libcrypto, libstdc++, libgcc_s, libgomp) must
# resolve to AppDir/usr/lib; nothing else should be a dynamic dep on a
# /lib64 path we don't ship.
RUN set -eux; \
    for BIN in llm-mem-mcp llm-mem; do \
        echo "==> Checking $BIN"; \
        LD_LIBRARY_PATH=/work/AppDir/usr/lib \
            ldd "/work/AppDir/usr/bin/$BIN" | tee /tmp/ldd-$BIN.txt; \
        if grep -E 'libssl|libcrypto|libstdc\+\+|libgcc_s|libgomp' /tmp/ldd-$BIN.txt | grep -v 'AppDir/usr/lib'; then \
            echo "ERROR ($BIN): one of the bundled libs is resolving outside AppDir/usr/lib"; \
            exit 1; \
        fi; \
    done; \
    echo '==> All bundled libs are resolved from AppDir/usr/lib.'

# Optional Vulkan loader bundling. The Vulkan loader is small (~200 KB)
# and has a stable ABI (1.3+), but the build needs the Vulkan-Headers
# sibling repo and adds significant build time. We skip it by default
# and rely on the host's libvulkan.so.1. The AppImage falls back to
# CPU-only inference if the host has no Vulkan loader installed.
# To bundle Vulkan, set ENABLE_VULKAN_BUNDLE=1 in the build args.
ARG ENABLE_VULKAN_BUNDLE=0

# Build the AppImage.
ARG VERSION=dev
ARG CHANNEL=nightly
ARG UPDATE_INFO=""

RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64|x86_64) APPIMAGE_ARCH=x86_64 ;; \
        arm64|aarch64) APPIMAGE_ARCH=aarch64 ;; \
    esac && \
    OUT="llm-mem-mcp-${APPIMAGE_ARCH}.AppImage" && \
    if [ -n "${UPDATE_INFO}" ]; then \
        UPDATE_ARG="--updateinformation ${UPDATE_INFO}"; \
    else \
        UPDATE_ARG=""; \
    fi && \
    cd /work && \
    ARCH="${APPIMAGE_ARCH}" appimagetool \
        --comp zstd \
        ${UPDATE_ARG} \
        AppDir "/work/${OUT}" && \
    ls -la /work/"${OUT}" && \
    file /work/"${OUT}" && \
    # Sanity check: extract metadata and confirm it can be inspected.
    "/work/${OUT}" --appimage-version 2>&1 | head -5

# ── Stage 3: minimal export image ─────────────────────────────────────────
FROM scratch AS export
COPY --from=packager /work/*.AppImage /
