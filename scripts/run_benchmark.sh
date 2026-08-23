#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$ROOT/.venv"

BENCHMARK_DIR="$ROOT/benchmark"
DATA_DIR="$ROOT/benchmark-data"
DEPS_DIR="$ROOT/benchmark-deps"
OUTPUT_DIR="$BENCHMARK_DIR/output"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Runs llm-mem benchmarks across multiple pyramid variants and compares results.

Options:
  --variant VARIANT    flat | pyramid_bottom_heavy | pyramid_balanced |
                       pyramid_top_heavy | all (default: all)
  --limit N            Max questions (default: all)
  --max-sessions N     Max sessions per question (default: all)
  --judge-model MODEL  Judge LLM (default: from env LLM_MEM_JUDGE_MODEL)
  --judge-key KEY      API key
  --judge-url URL      API base URL
  --skip-build         Skip cargo build
  --skip-judge-validation Skip judge LLM validation
  --dry-run            Print commands without executing
  --validate-judge-only  Only validate judge LLM, no benchmark

Examples:
  $0                                          # Run all 4 variants, compare
  $0 --variant flat --limit 10                # Quick flat baseline test
  $0 --variant pyramid_balanced --limit 50    # Test balanced pyramid
EOF
    exit 0
}

VARIANT="all"
LIMIT=0
MAX_SESSIONS=0
CONFIG_OVERRIDE=""
JUDGE_MODEL="${LLM_MEM_JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_API_KEY="${LLM_MEM_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
JUDGE_BASE_URL="${LLM_MEM_JUDGE_BASE_URL:-https://api.openai.com/v1}"
SKIP_BUILD=false
DRY_RUN=false
VALIDATE_JUDGE_ONLY=false
SKIP_JUDGE_VALIDATION=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)  VARIANT="${2,,}"; shift 2 ;;
        --limit)    LIMIT="$2"; shift 2 ;;
        --max-sessions) MAX_SESSIONS="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --judge-key)   JUDGE_API_KEY="$2"; shift 2 ;;
        --judge-url)   JUDGE_BASE_URL="$2"; shift 2 ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --skip-judge-validation) SKIP_JUDGE_VALIDATION=true; shift ;;
        --validate-judge-only) VALIDATE_JUDGE_ONLY=true; shift ;;
        --config) CONFIG_OVERRIDE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

DATASET="$DATA_DIR/longmemeval/longmemeval_s_cleaned.json"
EVAL_SCRIPT="$DEPS_DIR/longmemeval/src/evaluation/evaluate_qa.py"
METRICS_SCRIPT="$DEPS_DIR/longmemeval/src/evaluation/print_qa_metrics.py"
GROUND_TRUTH="$DATA_DIR/longmemeval/longmemeval_s_cleaned.json"
RUNNER="$BENCHMARK_DIR/longmemeval_runner.py"
PYTHON="$VENV_DIR/bin/python"

# Variant definitions: name → (config_file, pyramid_mode)
declare -A VARIANT_CONFIGS
VARIANT_CONFIGS[flat]="config_flat.toml none"
VARIANT_CONFIGS[pyramid_bottom_heavy]="config_pyramid.toml bottom_heavy"
VARIANT_CONFIGS[pyramid_balanced]="config_pyramid.toml balanced"
VARIANT_CONFIGS[pyramid_top_heavy]="config_pyramid.toml top_heavy"

if [ "$VARIANT" = "all" ]; then
    VARIANTS=(flat pyramid_bottom_heavy pyramid_balanced pyramid_top_heavy)
else
    if [ -z "${VARIANT_CONFIGS[$VARIANT]:-}" ]; then
        echo -e "${RED}Unknown variant: $VARIANT${NC}"
        echo "Valid: flat, pyramid_bottom_heavy, pyramid_balanced, pyramid_top_heavy, all"
        exit 1
    fi
    VARIANTS=("$VARIANT")
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   llm-mem Benchmark — Multi-Variant${NC}"
echo -e "${BLUE}============================================${NC}"
echo "Variants  : ${VARIANTS[*]}"
echo "Judge     : $JUDGE_MODEL ($JUDGE_BASE_URL)"
echo "Limit     : $LIMIT (0=all)"
echo ""

# ── Phase 0: Cleanup ─────────────────────────────────────────────────────────
echo -e "${YELLOW}[0/5] Cleaning up environment...${NC}"

if [ "$DRY_RUN" = false ]; then
    pkill -9 -f "llm-mem-mcp" 2>/dev/null || true
    sleep 1

    rm -rf "$ROOT/llm-mem-data"/banks/* 2>/dev/null || true
    rm -rf "$ROOT/llm-mem-data"/default* 2>/dev/null || true
    rm -rf "$ROOT/lancedb"/* 2>/dev/null || true
    rm -rf "$BENCHMARK_DIR/logs"/* 2>/dev/null || true
fi
echo -e "${GREEN}  Cleanup done${NC}"
echo ""

# ── Phase 1: Prerequisites + Judge validation ─────────────────────────────────
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

MISSING=()

if [ ! -f "$VENV_DIR/bin/python" ] && [ ! -f "$VENV_DIR/bin/python3" ]; then
    MISSING+=(".venv/ — run: bash scripts/setup_benchmarks.sh")
fi
if [ ! -f "$DATASET" ]; then
    MISSING+=("$DATASET — run: bash scripts/setup_benchmarks.sh")
fi
if [ -z "$JUDGE_API_KEY" ]; then
    MISSING+=("Judge API key — set LLM_MEM_JUDGE_API_KEY or OPENAI_API_KEY")
fi
if [ ${#MISSING[@]} -gt 0 ]; then
    echo -e "${RED}MISSING:${NC}"
    for m in "${MISSING[@]}"; do echo "  - $m"; done
    exit 1
fi

echo -e "${GREEN}  Prerequisites OK${NC}"

if [ "$SKIP_JUDGE_VALIDATION" = true ]; then
    echo ""
    echo -e "  ${YELLOW}Judge validation SKIPPED${NC}"
else
    echo ""
    echo -e "${YELLOW}     Validating judge LLM...${NC}"
    if [ "$DRY_RUN" = false ]; then
        if "$PYTHON" "$BENCHMARK_DIR/validate_judge.py" \
            --model "$JUDGE_MODEL" \
            --base-url "$JUDGE_BASE_URL" \
            --api-key "$JUDGE_API_KEY" 2>&1 | tail -8; then
            echo -e "  ${GREEN}Judge LLM OK${NC}"
        else
            echo -e "  ${RED}Judge LLM failed. Fix before continuing.${NC}"
            exit 1
        fi
    fi
fi

if [ "$VALIDATE_JUDGE_ONLY" = true ]; then
    echo ""
    echo -e "${GREEN}Validation complete.${NC}"
    exit 0
fi

echo ""

# ── Phase 2: Build ───────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/5] Building llm-mem-mcp...${NC}"
if [ "$SKIP_BUILD" = true ]; then
    echo "  Skipped (--skip-build)"
elif [ "$DRY_RUN" = false ]; then
    cargo build --release --features "default,vulkan" --bin llm-mem-mcp 2>&1 | tail -3
    echo -e "${GREEN}  Build complete${NC}"
fi
echo ""

# ── Phase 3: Run all variants ────────────────────────────────────────────────
echo -e "${YELLOW}[3/5] Running ${#VARIANTS[@]} variant(s)...${NC}"

declare -A VARIANT_RESULTS
TOTAL_START=$(date +%s)

for v in "${VARIANTS[@]}"; do
    read -r cfg mode <<< "${VARIANT_CONFIGS[$v]}"
    output="$OUTPUT_DIR/longmemeval_s_${v}.jsonl"

    echo ""
    echo -e "${BLUE}── $v ────────────────────────────────────${NC}"
    echo "  Config:       $cfg"
    echo "  Pyramid mode: $mode"
    echo "  Output:       $output"

    RUN_CMD=(
        "$PYTHON" "$RUNNER"
        --dataset "$DATASET"
        --output "$output"
        --pyramid-mode "$mode"
        --judge-model "$JUDGE_MODEL"
        --judge-base-url "$JUDGE_BASE_URL"
        --judge-api-key "$JUDGE_API_KEY"
    )
    if [ "$LIMIT" -gt 0 ]; then
        RUN_CMD+=(--limit "$LIMIT")
    fi
    if [ "$MAX_SESSIONS" -gt 0 ]; then
        RUN_CMD+=(--max-sessions "$MAX_SESSIONS")
    fi

    # Set the right config via env (allow --config override)
    if [ -n "$CONFIG_OVERRIDE" ]; then
        export LLM_MEM_CONFIG_PATH="$CONFIG_OVERRIDE"
    else
        export LLM_MEM_CONFIG_PATH="$BENCHMARK_DIR/$cfg"
    fi

    if [ "$DRY_RUN" = false ]; then
        v_start=$(date +%s)
        "${RUN_CMD[@]}"
        v_end=$(date +%s)
        v_dur=$((v_end - v_start))
        echo "  Duration: ${v_dur}s"
    else
        echo "  [DRY RUN] ${RUN_CMD[*]}"
    fi

    # Cleanup between variants — skipped to allow post-run inspection.
    # To re-enable bank deletion, uncomment the lines below.
    if [ "$DRY_RUN" = false ]; then
        pkill -9 -f "llm-mem-mcp" 2>/dev/null || true
        sleep 1
        # rm -rf "$ROOT/llm-mem-data"/banks/* 2>/dev/null || true
        # rm -rf "$ROOT/llm-mem-data"/default* 2>/dev/null || true
        # rm -rf "$ROOT/lancedb"/* 2>/dev/null || true
    fi
done

TOTAL_END=$(date +%s)
TOTAL_DUR=$((TOTAL_END - TOTAL_START))
echo ""
echo -e "${GREEN}  All variants complete in ${TOTAL_DUR}s${NC}"
echo ""

# ── Phase 4: Quick stats comparison ──────────────────────────────────────────
echo -e "${YELLOW}[4/5] Retrieval stats comparison...${NC}"
echo ""

if [ "$DRY_RUN" = false ]; then
    printf "%-25s %8s %8s %10s %10s %10s %10s\n" \
        "VARIANT" "QUESTIONS" "RETRIEVED" "L0" "L1" "L2" "L3"

    for v in "${VARIANTS[@]}"; do
        output="$OUTPUT_DIR/longmemeval_s_${v}.jsonl"
        if [ -f "$output" ]; then
            qc=$(wc -l < "$output")
            # Average retrieved and layer counts
            "$PYTHON" -c "
import json, sys
total=ret=0; l0=l1=l2=l3=0
for line in open(sys.argv[1]):
    r=json.loads(line)
    total+=1; ret+=r.get('num_retrieved',0)
    ly=r.get('layers',{})
    l0+=ly.get('L0',ly.get('raw_content',0))
    l1+=ly.get('L1',ly.get('structural',0))
    l2+=ly.get('L2',ly.get('semantic',0))
    l3+=ly.get('L3',ly.get('concept',0))
if total:
    print(f'AVG: ret={ret/total:.1f} L0={l0/total:.1f} L1={l1/total:.1f} L2={l2/total:.1f} L3={l3/total:.1f}')
else:
    print('AVG: no data')
" "$output"
            # Print formatted row
            stats=$("$PYTHON" -c "
import json, sys
total=ret=0; l0=l1=l2=l3=0
for line in open(sys.argv[1]):
    r=json.loads(line)
    total+=1; ret+=r.get('num_retrieved',0)
    ly=r.get('layers',{})
    l0+=ly.get('L0',ly.get('raw_content',0))
    l1+=ly.get('L1',ly.get('structural',0))
    l2+=ly.get('L2',ly.get('semantic',0))
    l3+=ly.get('L3',ly.get('concept',0))
if total:
    print(f'{total} {ret/total:.1f} {l0/total:.1f} {l1/total:.1f} {l2/total:.1f} {l3/total:.1f}')
" "$output")
            read -r _qc _ret _l0 _l1 _l2 _l3 <<< "$stats"
            printf "%-25s %8s %8s %10s %10s %10s %10s\n" \
                "$v" "$_qc" "$_ret" "$_l0" "$_l1" "$_l2" "$_l3"
        fi
    done
fi

echo ""

# ── Phase 5: Official evaluation ─────────────────────────────────────────────
echo -e "${YELLOW}[5/5] Official LongMemEval evaluation...${NC}"

if [ -f "$EVAL_SCRIPT" ]; then
    EVAL_DIR="$(dirname "$EVAL_SCRIPT")"
    for v in "${VARIANTS[@]}"; do
        output="$OUTPUT_DIR/longmemeval_s_${v}.jsonl"
        if [ ! -f "$output" ]; then
            echo "  Skip $v: no results file"
            continue
        fi
        echo ""
        echo -e "  ${BLUE}── $v ──${NC}"
        rel_out="$(realpath --relative-to="$EVAL_DIR" "$output" 2>/dev/null || echo "$output")"
        rel_gt="$(realpath --relative-to="$EVAL_DIR" "$GROUND_TRUTH" 2>/dev/null || echo "$GROUND_TRUTH")"
        if [ "$DRY_RUN" = false ]; then
            cd "$EVAL_DIR"
            OPENAI_API_KEY="${JUDGE_API_KEY}" OPENAI_BASE_URL="${JUDGE_BASE_URL}" \
            "$PYTHON" evaluate_qa.py "$JUDGE_MODEL" "$rel_out" "$rel_gt" 2>&1 | tail -5
            OPENAI_API_KEY="${JUDGE_API_KEY}" OPENAI_BASE_URL="${JUDGE_BASE_URL}" \
            "$PYTHON" print_qa_metrics.py "$JUDGE_MODEL" "$rel_out" "$rel_gt" 2>&1 | grep -E "(Accuracy|Recall|Precision|F1|score|Rate)" | head -10
            cd "$ROOT"
        fi
    done
else
    echo -e "  ${YELLOW}SKIP${NC} — eval scripts not found (clone LongMemEval in benchmark-deps)"
fi

echo ""
echo -e "${GREEN}Done.${NC}"
echo "Results: $OUTPUT_DIR/longmemeval_s_*.jsonl"
echo "Logs:    $BENCHMARK_DIR/logs/"
