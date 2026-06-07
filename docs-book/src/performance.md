# Performance

A reference for tuning llm-mem for your workload.

## First, measure

Don't tune blind. Run:

```bash
llm-mem health-check --live
llm-mem system-status
llm-mem metrics
```

These give you the backend details and per-call timings. With `RUST_LOG=debug,llm_mem::metrics=trace`, you get per-request durations in the log.

## Identify the bottleneck

Performance bottlenecks fall into a few categories:

### 1. LLM inference is slow

**Symptoms**: store operations take seconds; abstraction pipeline crawls; first query takes 30s.

**Mitigations** (try in order):

1. **Enable GPU** — biggest single win, often 5-20x. See [GPU acceleration](./gpu-acceleration.md).
2. **Use a smaller model** — `gemma-4-E2B` is ~2.5 GB; if you don't need that quality, try `smollm2-1.7b-instruct-q4_k_m` (~1 GB).
3. **Reduce `max_concurrent_requests`** — default is 1 (serial), which is safest. If you have GPU headroom, raise to 2-4 for parallelism.
4. **Use API backend** — if you have a fast API provider, an API call can be faster than local CPU inference.

### 2. Embedding is slow

**Symptoms**: searches and stores take a long time on the embedding step.

**Mitigations**:

1. **Local fastembed is fast by default** — should be 1000+ texts/sec on CPU. If it's slow, your CPU is loaded by other things.
2. **API embeddings** — if you have API budget, use `provider = "api"` for embeddings.
3. **Increase `batch_size`** — `embedding.batch_size = 256` (default 64) is more efficient for large batch operations.

### 3. Vector search is slow

**Symptoms**: `query_memory` and `search_memory` are slow on a large bank.

**Mitigations**:

1. **Use smaller banks** — split into per-project banks. Vector search is roughly linear in the bank's size.
2. **Use `search_memory` instead of `query_memory`** — the former has fewer features and is faster.
3. **Increase `search_similarity_threshold`** — filters out low-similarity candidates early.
4. **Restrict to a specific layer** — `include_layers: [0]` skips the pyramid expansion.
5. **Lower `max_search_results`** — fewer results to rank and return.

### 4. Abstraction pipeline is slow

**Symptoms**: L1+ memories take a long time to appear; background worker is the bottleneck.

**Mitigations**:

1. **Tune `auto_summary_threshold`** — the L0→L1 worker only fires past this. Higher = less frequent but cheaper per run.
2. **Pause the pipeline during bulk operations** — `stop_abstraction_pipeline` → bulk ingest → `start_abstraction_pipeline`. Avoids contention with the LLM.
3. **Set `session_token_budget`** — caps the LLM tokens spent per session, preventing runaway costs.
4. **Disable `auto_enhance`** — if you don't need LLM-generated metadata on every store, set `auto_enhance = false` to skip the LLM call.

### 5. Database operations are slow

**Symptoms**: `db check`, `db fix`, large exports take a long time.

**Mitigations**:

1. **Use `db export-jsonl` for archival** — streaming write, doesn't materialize the whole bank in memory.
2. **For very large banks, use per-bank operations** — don't `db check --all` on 50 banks at once.
3. **Disk I/O** — make sure the bank directory is on a fast SSD, not a network mount.

## Configuration tuning guide

### For a fast local setup

```toml
[llm]
provider = "local"
gpu_layers = 30               # all layers on GPU
max_concurrent_requests = 2   # parallel inference

[memory]
auto_enhance = true
auto_summary_threshold = 16384
session_token_budget = 100000
```

### For a low-resource setup

```toml
[llm]
provider = "local"
gpu_layers = 0
cpu_threads = 4
max_concurrent_requests = 1
llm_timeout_secs = 180

[memory]
auto_enhance = false
deduplicate = true
auto_summary_threshold = 65536
```

### For an API-only setup

```toml
[llm]
provider = "api"
model = "gpt-4o-mini"
max_concurrent_requests = 4
use_structured_output = true
llm_timeout_secs = 60

[embedding]
provider = "api"
model = "text-embedding-3-small"
batch_size = 100

[memory]
auto_enhance = true
```

### For very large banks (>1M memories)

```toml
[memory]
raw_content_scan_limit = 1000
max_total_candidates = 5000
max_list_limit = 1000
```

And consider per-project banks instead of one giant bank.

## When to scale

| Bank size | Recommendation |
|---|---|
| < 10k memories | Default config is fine |
| 10k-100k | Tune `max_total_candidates` and `raw_content_scan_limit` |
| 100k-1M | Consider per-project banks, LanceDB only (not VectorLite) |
| > 1M | Specialized tuning + faster hardware; consider a dedicated vector DB instead |

## Next

- [GPU acceleration](./gpu-acceleration.md)
- [Configuration](./config-file.md)
- [Troubleshooting](./troubleshooting.md)
