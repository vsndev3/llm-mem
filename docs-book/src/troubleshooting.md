# Troubleshooting

Common issues, organized by symptom.

## "Server failed to start"

### `Failed to load config from <path>`

The config file exists but has invalid TOML or fails validation. Check the message — it tells you which field is wrong.

Common culprits:

- Typo in a key name (TOML doesn't error on unknown keys by default; but serde does for known keys with `deny_unknown_fields`)
- A numeric field is a string (`max_memories = "10000"` should be `10000`)
- An enum value is wrong (`provider = "OPENAI"` should be `provider = "api"`)

### `No API key configured`

You're using `provider = "api"` but didn't set the key. Fix:

```bash
# In the MCP config's env block
"env": {
  "LLM_MEM_LLM_API_KEY": "sk-..."
}

# Or in the TOML config
[llm]
api_key = "sk-..."

# Or as a generic fallback
"env": { "OPENAI_API_KEY": "sk-..." }
```

### `llm.context_size is too small`

`context_size` must be larger than `memory.document_chunk_size / 2 + llm.max_tokens + 512`. The default `16644` works for the defaults, but if you've raised `document_chunk_size` or `max_tokens`, raise `context_size` too.

### `llm.provider = "local" but this build does not have the 'local-llm' feature enabled`

You built with `--no-default-features` and then didn't include `local-llm` (or the aggregate `local`). Rebuild:

```bash
cargo build --release --features local
```

## "Models not downloading"

### `Failed to download: 407 Proxy Authentication Required`

The proxy needs auth. Set `HTTPS_PROXY=http://user:pass@proxy.host:port` in the MCP env block, or in the `[llm].proxy_url` config (which overrides the env var).

### `Failed to download: 404 Not Found`

The HuggingFace URL has changed or the model was renamed. Check the [llm-mem releases](https://github.com/vsndev3/llm-mem/releases) for the latest model filenames, or download manually (see [Logging & debugging](./logging-debugging.md#manual-model-download)).

### `Failed to download: connection timeout`

Firewall or DNS. Verify you can reach `huggingface.co` from the server's environment. If you can't, download the model on a different machine and copy it in.

## "MCP client can't connect"

See [Troubleshooting MCP](./troubleshooting-mcp.md) for the full flow. The short version:

- Absolute path in the `command` field (no `~`, no relative paths)
- AppImage needs `chmod +x`
- Try `APPIMAGE_EXTRACT_AND_RUN=1` if FUSE is unavailable
- Restart the client after every config change
- Check the client's MCP panel for the server status

## "Tool calls fail with errors"

### `LLM timeout after 120 seconds`

The LLM is taking too long. Mitigations:

```toml
[llm]
llm_timeout_secs = 300     # give it more time
gpu_layers = 20            # use the GPU
```

Or switch to a smaller model.

### `Bank not found: 'foo'`

You passed `bank: "foo"` but no such bank exists. Either:

- Create it first with `create_memory_bank` (MCP) or just write to it (auto-create)
- Check the spelling — bank names are case-sensitive

### `Near-duplicate detected`

The new memory's embedding is very close to an existing one. By default, the server **blocks the store** and returns this error. Either:

- Pass `force: true` to store anyway
- Edit the new content to be different from the existing memory
- Re-read what the existing memory says and decide if the new content is actually redundant

### `Contradiction detected`

`memory.contradiction_detection = true` is enabled and the new memory contradicts an existing one. Same options as above.

### `Bank deletion requires user confirmation`

You called `cleanup_resources` with `target: "banks"` but didn't pass the exact confirmation phrase. This is intentional — pass:

```json
{ "target": "banks", "name": "foo", "confirm": "I confirm this data will be permanently lost" }
```

## "Performance is bad"

| Symptom | Likely cause | Fix |
|---|---|---|
| First query takes 30+ seconds | Model warmup | Wait. Subsequent queries are fast. |
| All queries take 5+ seconds | CPU-only inference on large model | Enable GPU (see [GPU acceleration](./gpu-acceleration.md)) |
| Bank has 500k+ memories, queries are slow | Vector search scales with bank size | Use smaller banks per project, increase `search_similarity_threshold` |
| API backend, every query is slow | Round-trip latency | Use a faster provider or local backend |
| High memory usage | Large model + large context | Reduce `context_size` or use a smaller model |
| High disk usage | Many banks, large embeddings | Run `db prune --all --older-than-days 0` to reclaim old LanceDB versions; run `db check` to find Forgotten memories; clean up unused banks |

## "Database corruption"

```bash
llm-mem db check --all --verbose
llm-mem db fix --bank default --dry-run    # preview
llm-mem db fix --bank default              # repair (auto-backup)
```

If `db fix` can't repair, restore from backup:

```bash
llm-mem db export --bank default --output /tmp/before.db   # safety
llm-mem db import --bank default --input /backup/latest.db
```

If even that fails, the bank is genuinely corrupted — delete the `.lance/` directory and re-create (data loss; restore from your latest JSONL export if you have one).

## "I get a different error message than the docs"

llm-mem is beta. Error messages are not stable. Open an issue at https://github.com/vsndev3/llm-mem/issues with:

1. The exact error message
2. The log file excerpt (`logging.level = "debug"`)
3. Your `config.toml` (with secrets redacted)
4. The version (`llm-mem-mcp --version`)

## Next

- [Performance](./performance.md)
- [Troubleshooting MCP](./troubleshooting-mcp.md)
- [Logging & debugging](./logging-debugging.md)
