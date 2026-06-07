# `[llm]` — language model

Settings for the language model used for extraction, summarization, classification, and query answering.

```toml
[llm]
provider = "local"               # "local" (llama.cpp) or "api" (OpenAI-compatible)
```text

## Provider selection

| Value | What it does |
|---|---|
| `"local"` | Embedded llama.cpp inference. No network, no API key. Downloads a GGUF model on first run. |
| `"api"` | OpenAI-compatible HTTP API. Requires `api_key` (or env var). |

You can mix: see [Embedding section](./embedding-section.md) for the combinations.

## Local provider settings

```toml
[llm]
provider = "local"
model_file = "gemma-4-E2B-it-Q8_0.gguf"   # GGUF filename (auto-downloaded if missing)
models_dir = "llm-mem-data/models"         # where to put downloaded models
gpu_layers = 0                            # 0 = CPU only; set to 20+ for GPU
context_size = 16644                      # context window in tokens
cpu_threads = 0                           # 0 = auto-detect
auto_download = true                      # fetch from HuggingFace on first run
cache_model = true                        # cache in ~/.cache/llm-mem/models
cache_dir = ""                            # custom cache dir (default: ~/.cache/llm-mem/models)
llm_timeout_secs = 120                    # completion timeout
use_grammar = false                       # grammar-constrained sampling (faster, less flexible)
proxy_url = ""                            # proxy for downloads (overrides HTTPS_PROXY)
```text

### Vision (local)

If you have a vision-capable GGUF model + mmproj file:

```toml
vision_enabled = true
mmproj_file = ""                          # empty = auto-detect mmproj-F16.gguf for Gemma 4 E2B
vision_prompt_template = "Describe this image in detail, focusing on what would make it searchable in a personal memory system. What objects, people, text, colors, scene elements, actions, and contextual cues are visible? What domain or topic does this image relate to? Be specific and concrete."
```text

When `vision_enabled = true`, ingested images (PNG/JPEG/GIF/WebP) get an AI-generated description stored alongside the original.

## API provider settings

```toml
[llm]
provider = "api"
api_url = "https://api.openai.com/v1"
api_key = ""                              # or set LLM_MEM_LLM_API_KEY / OPENAI_API_KEY
model = "gpt-4o-mini"
```text

Common `api_url` values:

| Service | URL |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Anthropic | use `api_dialect = "anthropic"` with the Anthropic endpoint |
| Ollama | `http://localhost:11434/v1` |
| llama-server | `http://localhost:8080/v1` |
| LM Studio | `http://localhost:1234/v1` |

### Dialects

```toml
api_dialect = "openai-chat"   # one of: openai-chat, openai-completion, anthropic, ollama-chat, ollama-completion, custom
```text

`openai-chat` (default) works for any OpenAI-compatible chat completions endpoint. Use `anthropic` for Anthropic, `ollama-*` for Ollama native APIs, `custom` for fully custom request/response shapes.

### Custom dialect

```toml
api_dialect = "custom"

[llm.custom_dialect]
endpoint_path = "/v1/generate"
request_body_template = '{"model": "{{model}}", "prompt": "{{prompt}}", "max_tokens": {{max_tokens}}, "temperature": {{temperature}}}'
response_content_pointer = "/choices/0/text"
```text

Placeholders: `{{prompt}}`, `{{model}}`, `{{temperature}}`, `{{max_tokens}}`. `response_content_pointer` is a JSON pointer to the text in the response.

### Request format

```toml
request_format = "auto"        # "auto" (default), "rig", or "raw"
```text

- `auto` — try rig-core first, fall back to raw HTTP on 422 errors (most compatible)
- `rig` — always use rig-core (may 422 on strict backends)
- `raw` — always use raw HTTP with plain strings (bypasses rig-core)

### Structured output

```toml
use_structured_output = true   # JSON schema validation (API only)
structured_output_retries = 2  # retries on validation failure
```text

The server asks the API for JSON schema-validated output, retrying if the LLM returns malformed JSON. Disable for backends that don't support it.

## Generation settings (both providers)

```toml
temperature = 0.7              # 0.0 = deterministic, 2.0 = very creative
max_tokens = 4096              # max tokens per completion
max_concurrent_requests = 1    # 0 = unlimited, 1 = serial (safest)
strip_tags = ["think"]         # XML tags to strip from LLM output (e.g. "think", "reason")
```text

## Advanced / batching

```toml
batch_size = 10                # items per batch request
batch_max_tokens = 3000        # tokens per batch request (must be ≤ max_tokens)
batch_timeout_secs = 120       # base timeout for batch calls
batch_timeout_multiplier = 1.0 # timeout multiplier
```text

The server batches independent LLM calls (e.g. classifying multiple memories at once) to reduce API overhead.

## Next

- [`[embedding]`](./embedding-section.md)
- [`[memory]`](./memory-section.md)
