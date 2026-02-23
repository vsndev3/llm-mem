#!/bin/bash
# Test CI build locally using Docker

set -e

echo "=== Building CI test Docker image ==="
docker build -f Dockerfile.ci-test -t llm-mem-ci .

echo ""
echo "=== Running CI build test ==="
docker run --rm \
    -v "$(pwd):/workspace" \
    -w /workspace \
    llm-mem-ci \
    cargo build --release --bin llm-mem-mcp

echo ""
echo "=== Build successful! ==="
echo "Binary location: target/release/llm-mem-mcp"
ls -lh target/release/llm-mem-mcp
