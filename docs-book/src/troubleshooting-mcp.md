# Troubleshooting MCP

Common issues when connecting an MCP client to llm-mem-mcp.

## Server doesn't start

**Symptom**: Client shows the server as "failed" or "not connected". Log file (if `logging.enabled = true`) shows an error.

**Check**:

1. **Absolute path**: `command` must be an absolute path, not a relative one or a `~`-expanded one.
2. **Executable bit**: AppImage needs `chmod +x`. Native binaries usually don't.
3. **Run it manually**: try `./llm-mem-mcp --version` from a terminal. If that fails, the binary itself is broken.
4. **Missing libraries**: on Linux, run `ldd /path/to/llm-mem-mcp` to check for missing `.so` files. On macOS, use `otool -L`. On Windows, use Dependencies (https://lucasg.github.io/Dependencies/).
5. **AppImage + no FUSE**: add `"env": {"APPIMAGE_EXTRACT_AND_RUN": "1"}` to the MCP config.

## Server starts but no tools appear

**Check**:

1. **MCP version mismatch**: confirm your client supports MCP. opencode 0.5+, VS Code 1.85+, Claude Code 1.0+.
2. **Restart the client**: MCP server lists are loaded at startup, not hot-reloaded.
3. **Check stderr**: the server logs to stderr. With `logging.enabled = true`, also check the log file. Look for JSON-RPC errors.
4. **Try a different transport**: llm-mem only supports stdio. If your client is trying HTTP/SSE, it won't work.

## Server starts, tools appear, but calls fail

**Check**:

1. **`system_status` first**: ask the AI to call it. The response tells you if models are loaded, what backend is active, and whether the LLM/embedding are reachable.
2. **Wait for model download**: on first run, the server downloads ~3 GB of models. Until that's done, most tools return errors. Watch the log file for "model loaded" messages.
3. **API key issues**: if using the API backend, verify the key is set via `LLM_MEM_LLM_API_KEY` (or in the config). The server validates at startup and will refuse to start without it.
4. **Bank creation error**: if the `banks_dir` isn't writable, the server fails to create the default bank. Check filesystem permissions and the `banks_dir` setting.

## Performance is slow

**Check**:

1. **GPU not used**: see [GPU acceleration](./gpu-acceleration.md). CPU-only inference is 5-20x slower.
2. **First query is slow**: model loading + warmup takes 5-30 seconds. Subsequent queries are fast.
3. **Concurrent requests**: default `max_concurrent_requests = 1` for stability. Bump it if you have a beefy machine.
4. **`max_concurrent_requests` for API backend**: set to `0` (unlimited) if you have API rate-limit headroom.
5. **Large bank**: queries get slower as the bank grows past ~100k memories. Consider per-project banks.

## Logs

The server logs to stderr. With `logging.enabled = true`, it also writes to a rotating log file under `log_directory` (default `llm-mem-data/logs/`).

To get more detail, set the `RUST_LOG` env var in the MCP config's `env` block:

```json
"env": {
  "RUST_LOG": "info,llama_cpp_2=warn"
}
```

Useful levels:

| Value | What you get |
|---|---|
| `error` | Only errors |
| `warn` | Errors + warnings |
| `info` (default) | Lifecycle events, request summaries |
| `debug` | Per-request detail |
| `trace` | Everything, including the LLM's prompts |

See [Logging & debugging](./logging-debugging.md) for more.

## Common error messages

### `Failed to resolve memory bank: ...`

The bank name passed to the tool doesn't exist and can't be created. Check spelling. Banks are created on first write, not on first read.

### `LLM timeout after 120 seconds`

The LLM is taking too long. Either:

- Increase `llm.llm_timeout_secs` in the config
- Switch to a smaller/faster model
- Enable GPU acceleration
- If using an API, the upstream is slow — try a different model or provider

### `No API key configured`

You set `provider = "api"` but didn't provide a key. Set `LLM_MEM_LLM_API_KEY` (or in the config), or change `provider = "local"`.

### `Configuration Error: llm.context_size is too small`

`llm.context_size` must be larger than `memory.document_chunk_size / 2 + llm.max_tokens + 512`. Increase `context_size` or decrease `document_chunk_size`.

### `Bank deletion requires user confirmation`

You called `cleanup_resources` with `target: "banks"` but didn't pass the exact confirmation phrase. This is a safety check — pass `confirm: "I confirm this data will be permanently lost"`.

## Still stuck?

1. Run `llm-mem-mcp --generate-config` to confirm the binary works
2. Run `llm-mem health-check --live` (the CLI binary) to validate the config end-to-end
3. Check the GitHub issue tracker: https://github.com/vsndev3/llm-mem/issues
4. Include the log output and your config (with secrets redacted) when asking for help

## Next

- [Tools overview](./tools-overview.md)
- [Configuration](./config-file.md)
