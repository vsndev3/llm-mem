#!/usr/bin/env bash
set -euo pipefail

# README benchmark — AUTO-RESUMING version.
#
#   flat (100 questions) + pyramid_balanced (10 questions) on LongMemEval-S,
#   then independent verification + accuracy summary.
#
# Resume behavior:
#   - Each variant checkpoints after EVERY question.
#   - Re-running this script continues any variant from its last checkpoint.
#   - Partial output from an interrupted resume is merged before continuing.
#   - Completed variants are skipped entirely.
#
# Judge separation:
#   - Answer generation: gemma-4-E4B-it-GGUF (system-under-test generator)
#   - Verification:      EVAL_MODEL (stronger, independent)   default: Qwen3.5-122B-A10B-GGUF
#
# Usage:
#   bash scripts/bench_longmemeval_s.sh             # run/resume everything
#   bash scripts/bench_longmemeval_s.sh --dry-run   # show plan only
#   EVAL_MODEL=gemma-4-31B-it-GGUF bash scripts/bench_longmemeval_s.sh
#
# Endpoint configuration (all optional):
#   LLM_MEM_LLM_BASE_URL   answer-generation endpoint (default http://localhost:8080/v1)
#   LLM_MEM_LLM_API_KEY    its API key                    (default: $OPENAI_API_KEY, else "local")
#   LLM_MEM_ANSWER_MODEL   generator model name           (default: gemma-4-E4B-it-GGUF)
#
# Requirements:
#   - An OpenAI-compatible endpoint serving ANSWER_MODEL; EVAL_MODEL available
#     for hot-swap or served remotely (OpenAI-compatible)
#   - datasets in benchmark-data/, .venv, target/release/llm-mem-mcp

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$ROOT/.venv/bin/python"
DATASET="$ROOT/benchmark-data/longmemeval/longmemeval_s_cleaned.json"
RUNNER="$ROOT/benchmark/longmemeval_runner.py"
BANKS_DIR="$ROOT/llm-mem-data/banks"

export LLM_MEM_ABSTRACTION_DELAY_SECS=2
export LLM_MEM_ABSTRACTION_CONCURRENCY=8
export LLM_MEM_BACKGROUND_LLM_CONCURRENCY=8
# Excerpt-granularity retrieval: read by benchmark/longmemeval_runner.py and
# forwarded as query params.
export LLM_MEM_QUERY_GRANULARITY=excerpt
export LLM_MEM_QUERY_K=25
export LLM_MEM_EXCERPT_MAX_CHARS=18000

BASE_URL="${LLM_MEM_LLM_BASE_URL:-http://localhost:8080/v1}"
API_KEY="${LLM_MEM_LLM_API_KEY:-${OPENAI_API_KEY:-local}}"
ANSWER_MODEL="${LLM_MEM_ANSWER_MODEL:-gemma-4-E4B-it-GGUF}"
EVAL_MODEL="${EVAL_MODEL:-Qwen3.5-122B-A10B-GGUF}"

FLAT_LIMIT=100
PYRAMID_LIMIT=10

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

echo -e "${BLUE}=== README benchmark (auto-resume) ===${NC}"
echo "Answer generation : $ANSWER_MODEL @ $BASE_URL"
echo "Verification      : $EVAL_MODEL @ $BASE_URL (independent)"
echo ""

# ── Phase 0: safety checks ───────────────────────────────────────────────────
if pgrep -f "benchmark/longmemeval_runner.py" > /dev/null 2>&1; then
    echo -e "${YELLOW}Killing stale benchmark runner/mcp processes...${NC}"
    pkill -9 -f "benchmark/longmemeval_runner.py" 2>/dev/null || true
    pkill -9 -x llm-mem-mcp 2>/dev/null || true
    sleep 2
fi

if ! curl -s --max-time 10 -H "Authorization: Bearer $API_KEY" "$BASE_URL/models" > /dev/null 2>&1; then
    echo -e "${RED}FATAL: llama-server at $BASE_URL is not reachable.${NC}"
    exit 1
fi
echo -e "${GREEN}llama-server reachable${NC}"

AVAIL_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt 25 ]; then
    echo -e "${RED}FATAL: only ${AVAIL_GB}G free on $(df --output=target "$ROOT" | tail -1) — need ~20G for pyramid banks.${NC}"
    du -sh "$BANKS_DIR" "$ROOT/target" 2>/dev/null
    exit 1
fi
echo -e "${GREEN}Disk OK (${AVAIL_GB}G free)${NC}"

results_count() {  # results_count <file>
    [ -f "$1" ] && wc -l < "$1" || echo 0
}

# ── Run + resume one variant ─────────────────────────────────────────────────
run_variant() {
    local name="$1" config="$2" mode="$3" limit="$4" out="$5"
    local done_n before
    done_n=$(results_count "$out")

    if [ "$done_n" -gt "$limit" ]; then
        echo -e "${RED}WARNING: $out has $done_n > $limit lines — rename or delete it to re-run.$NC"
        return 0
    fi
    if [ "$done_n" -eq "$limit" ]; then
        echo -e "${GREEN}$name: already complete ($done_n/$limit) — skipping${NC}"
        return 0
    fi

    # Fresh banks for this variant (avoids cross-run bank contamination;
    # bank names are position-indexed and reset on resume)
    if [ "$DRY_RUN" = false ]; then
        rm -rf "$BANKS_DIR"/bm_lme_* 2>/dev/null || true
    fi

    while [ "$done_n" -lt "$limit" ]; do
        local remaining=$((limit - done_n))
        local resume_out="$out.resume"
        echo ""
        echo -e "${BLUE}════════ $name: $done_n/$limit done — resuming from question $((done_n + 1)) ($remaining to go) ════════${NC}"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] runner --pyramid-mode $mode --start-from $done_n --limit $remaining"
            return 0
        fi

        before=$done_n
        if ! LLM_MEM_CONFIG_PATH="$ROOT/$config" "$PYTHON" "$RUNNER" \
            --dataset "$DATASET" \
            --output "$resume_out" \
            --pyramid-mode "$mode" \
            --judge-model "$ANSWER_MODEL" \
            --judge-base-url "$BASE_URL" \
            --judge-api-key "$API_KEY" \
            --start-from "$done_n" \
            --limit "$remaining"; then
            echo -e "${YELLOW}Runner exited non-zero — merging whatever was checkpointed${NC}"
        fi

        if [ -f "$resume_out" ]; then
            cat "$resume_out" >> "$out"
            rm -f "$resume_out"
        fi
        done_n=$(results_count "$out")

        if [ "$done_n" -le "$before" ]; then
            echo -e "${RED}FATAL: no progress (still $done_n/$limit). Check llama-server, disk, logs in benchmark/logs/.${NC}"
            exit 1
        fi
        echo -e "${GREEN}$name progress: $done_n/$limit${NC}"
    done
}

run_variant "flat" \
    "benchmark/config_flat_api.toml" \
    "none" \
    "$FLAT_LIMIT" \
    "$ROOT/benchmark/output/longmemeval_s_flat.jsonl"

run_variant "pyramid_balanced" \
    "benchmark/config_pyramid_api.toml" \
    "balanced" \
    "$PYRAMID_LIMIT" \
    "$ROOT/benchmark/output/longmemeval_s_pyramid_balanced.jsonl"

# ── Independent verification (official LongMemEval eval) ─────────────────────
if [ "$DRY_RUN" = false ]; then
    echo ""
    echo -e "${BLUE}════════ Official eval (independent judge: $EVAL_MODEL) ════════${NC}"
    EVAL_DIR="$ROOT/benchmark-deps/longmemeval/src/evaluation"
    for hyp in \
        "$ROOT/benchmark/output/longmemeval_s_flat.jsonl" \
        "$ROOT/benchmark/output/longmemeval_s_pyramid_balanced.jsonl"; do
        echo ""
        echo "── $(basename "$hyp") ──"
        (cd "$EVAL_DIR" && \
            OPENAI_API_KEY="$API_KEY" OPENAI_BASE_URL="$BASE_URL" \
            "$PYTHON" evaluate_qa.py "$EVAL_MODEL" \
            "$(realpath --relative-to="$EVAL_DIR" "$hyp")" \
            "$(realpath --relative-to="$EVAL_DIR" "$DATASET")" 2>&1 | tail -10)
    done
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}════════ Summary (verified by $EVAL_MODEL) ════════${NC}"
"$PYTHON" - "$EVAL_MODEL" "$ROOT" <<'EOF'
import json
import sys
from pathlib import Path

eval_model = sys.argv[1]
out = Path(sys.argv[2]) / "benchmark" / "output"
for name, path in [
    ("flat", out / "longmemeval_s_flat.jsonl"),
    ("pyramid_balanced", out / "longmemeval_s_pyramid_balanced.jsonl"),
]:
    eval_path = path.with_suffix(path.suffix + f".eval-results-{eval_model}")
    if not eval_path.exists():
        print(f"{name:22s} — no eval results")
        continue
    rows = [json.loads(l) for l in eval_path.open()]
    n = len(rows)
    correct = sum(1 for r in rows if r.get("autoeval_label", {}).get("label"))
    errors = sum(1 for r in rows if r.get("error"))
    acc = correct / n if n else 0
    print(f"{name:22s} accuracy={acc:.1%} ({correct}/{n})  errored={errors}")
    by_type = {}
    for r in rows:
        t = r.get("task_type", "?")
        by_type.setdefault(t, []).append(bool(r.get("autoeval_label", {}).get("label")))
    for t, labels in sorted(by_type.items()):
        print(f"    {t:26s} {sum(labels)}/{len(labels)}")
EOF

echo ""
echo -e "${GREEN}Done. Results: benchmark/output/*.jsonl.eval-results-* | Logs: benchmark/logs/${NC}"
