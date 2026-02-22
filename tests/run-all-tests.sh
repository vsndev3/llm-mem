#!/usr/bin/env bash
# ─── llm-mem full test runner ───────────────────────────────────────────
# Runs all tests with output directed to a work directory.
# No files are written outside this directory.
#
# Usage:
#   ./tests/run-all-tests.sh                          # unit + integration (fast)
#   ./tests/run-all-tests.sh --all                    # + evaluation (slow, downloads models)
#   WORK_DIR=/tmp/llm-mem-test ./tests/run-all-tests.sh  # custom work dir
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK_DIR="${WORK_DIR:-$SRC_DIR/test-output}"
mkdir -p "$WORK_DIR"

LOG_FILE="$WORK_DIR/test-results.log"

# ── Redirect all temp/data files into this directory ────────────────────
export CARGO_TARGET_DIR="$WORK_DIR/targets"
export TMPDIR="$WORK_DIR/tmp"
mkdir -p "$TMPDIR"

# Explicitly set models directory to be inside the test directory
export LLM_MEM_MODELS_DIR="$WORK_DIR/llm-mem-models"
mkdir -p "$LLM_MEM_MODELS_DIR"

# Timestamp
echo "════════════════════════════════════════════════════════════════" | tee "$LOG_FILE"
echo "  llm-mem test run — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "  Source:  $SRC_DIR" | tee -a "$LOG_FILE"
echo "  Target:  $CARGO_TARGET_DIR" | tee -a "$LOG_FILE"
echo "  Models:  $LLM_MEM_MODELS_DIR" | tee -a "$LOG_FILE"
echo "  TempDir: $TMPDIR" | tee -a "$LOG_FILE"
echo "════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

cd "$SRC_DIR"

FAILED=0

# ── 1. Unit tests ──────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "▶ [1/4] Unit tests (cargo test --lib)" | tee -a "$LOG_FILE"
echo "────────────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
if cargo test --lib 2>&1 | tee -a "$LOG_FILE"; then
    echo "✓ Unit tests PASSED" | tee -a "$LOG_FILE"
else
    echo "✗ Unit tests FAILED" | tee -a "$LOG_FILE"
    FAILED=$((FAILED + 1))
fi

# ── 2. Integration tests ──────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "▶ [2/4] Integration tests (cargo test --test integration_tests)" | tee -a "$LOG_FILE"
echo "────────────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
if cargo test --test integration_tests 2>&1 | tee -a "$LOG_FILE"; then
    echo "✓ Integration tests PASSED" | tee -a "$LOG_FILE"
else
    echo "✗ Integration tests FAILED" | tee -a "$LOG_FILE"
    FAILED=$((FAILED + 1))
fi

# ── 3. Binary crate tests ─────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "▶ [3/4] Binary crate tests (cargo test --bin llm-mem-mcp)" | tee -a "$LOG_FILE"
echo "────────────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
if cargo test --bin llm-mem-mcp 2>&1 | tee -a "$LOG_FILE"; then
    echo "✓ Binary crate tests PASSED" | tee -a "$LOG_FILE"
else
    echo "✗ Binary crate tests FAILED" | tee -a "$LOG_FILE"
    FAILED=$((FAILED + 1))
fi

# ── 4. Evaluation tests (optional, slow) ──────────────────────────────
if [[ "${1:-}" == "--all" ]]; then
    echo "" | tee -a "$LOG_FILE"
    echo "▶ [4/4] Evaluation tests (cargo test --test evaluation -- --ignored --nocapture)" | tee -a "$LOG_FILE"
    echo "  ⚠  First run downloads embedding model (~90 MB) + LLM model (~1.1 GB)" | tee -a "$LOG_FILE"
    echo "────────────────────────────────────────────────────────────────" | tee -a "$LOG_FILE"
    if cargo test --test evaluation -- --ignored --nocapture --test-threads=1 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ Evaluation tests PASSED" | tee -a "$LOG_FILE"
    else
        echo "✗ Evaluation tests FAILED" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
else
    echo "" | tee -a "$LOG_FILE"
    echo "▷ [4/4] Evaluation tests SKIPPED (pass --all to include)" | tee -a "$LOG_FILE"
fi

# ── Summary ────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
if [[ $FAILED -eq 0 ]]; then
    echo "  ✓ ALL TEST SUITES PASSED" | tee -a "$LOG_FILE"
else
    echo "  ✗ $FAILED TEST SUITE(S) FAILED" | tee -a "$LOG_FILE"
fi
echo "  Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

exit $FAILED
