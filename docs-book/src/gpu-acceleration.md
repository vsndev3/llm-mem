# GPU acceleration

By default, llm-mem runs on CPU. This works everywhere but is slow for the language model. GPU acceleration is **optional but strongly recommended** if you have a compatible GPU.

## Supported platforms

| Platform | GPU | Build feature |
|---|---|---|
| macOS (Apple Silicon) | M1, M2, M3, M4 | `metal` |
| macOS (Intel) | Intel iGPU | `metal` (limited) |
| Linux (AMD, Intel, NVIDIA) | Any Vulkan-capable | `vulkan` |
| Linux (NVIDIA only) | NVIDIA | `cuda` |
| Windows (AMD, Intel, NVIDIA) | Any Vulkan-capable | `vulkan` |
| Windows (NVIDIA only) | NVIDIA | `cuda` |

Vulkan is the most portable GPU backend. CUDA is fastest for NVIDIA hardware. Metal is the only option on macOS.

## Building with GPU support

### Default (CPU only)

```bash
cargo build --release
```

### Vulkan

```bash
# Linux
cargo build --release --features local,vulkan

# Windows
cargo build --release --features local,vulkan
```

Requires the Vulkan SDK and a Vulkan-capable driver installed on the system at build time.

### CUDA (NVIDIA only)

```bash
cargo build --release --features local,cuda
```

Requires CUDA toolkit + a compatible NVIDIA driver.

### Metal (macOS)

```bash
cargo build --release --features local,metal
```

## Enabling GPU at runtime

After building with the right feature, tell the runtime how many model layers to offload:

```toml
[llm]
gpu_layers = 20     # 0 = CPU only; set to 20+ for GPU
```

`gpu_layers` means **how many transformer layers run on the GPU**. The full model has a fixed number of layers (e.g. Gemma 4 E2B has ~26 layers). Setting `gpu_layers = 20` offloads 20 of 26 layers; the rest stay on CPU.

| Value | Behavior |
|---|---|
| `0` | CPU only |
| Small (e.g. `5`) | Partial offload — useful if your VRAM is limited |
| Number of model layers (e.g. `26` for Gemma 4 E2B) | Full offload — all on GPU |
| Larger than model layers | Capped at model layer count |

If you have less VRAM than the model needs, set `gpu_layers` to fit. The server will run the offloaded layers on GPU and the rest on CPU. There's no all-or-nothing requirement.

## Verifying GPU usage

In the log file (with `logging.level = "debug"`), look for:

```text
INFO llama.cpp: loading model with 20 GPU layers
```

The `system_status` tool also reports backend details:

```json
{
  "backend": "local",
  "details": {
    "n_gpu_layers": 20,
    /* ... */
  }
}
```

For the CLI:

```bash
llm-mem list-devices    # shows available devices
llm-mem system-status   # shows current backend and layer count
```

## Performance expectations

Speedups depend on hardware. Rough numbers for Gemma 4 E2B at 16K context:

| Hardware | Approx tokens/sec |
|---|---|
| Apple M1 CPU | 5-8 |
| Apple M1 GPU (Metal) | 20-35 |
| Apple M3 Pro GPU | 40-60 |
| NVIDIA RTX 3060 (CUDA) | 30-50 |
| NVIDIA RTX 4090 (CUDA) | 100-150 |
| AMD RX 7900 XT (Vulkan) | 50-80 |
| Intel i7-12700 CPU | 4-7 |

These are approximate — actual numbers depend on prompt length, model size, and driver version. The point is: **GPU is 5-20x faster than CPU**.

## AppImage and GPU

The pre-built AppImage includes both Vulkan and CUDA backends, dynamically loaded. To use GPU:

1. Install the Vulkan loader (`libvulkan1` on Debian/Ubuntu, `vulkan-tools` on Fedora) OR the CUDA runtime
2. Set `gpu_layers` in your config
3. Make sure `/dev/dri/` is accessible (for Vulkan) or `nvidia-smi` works (for CUDA)

If the AppImage fails to find the GPU at runtime, it'll fall back to CPU automatically. Check the log file for "GPU not available" warnings.

## Building the AppImage with GPU

```bash
scripts/build-appimage.sh --update-info 'gh-releases-zsync|vsndev3|llm-mem|latest|llm-mem-mcp-x86_64.AppImage.zsync'
```

The AppImage build uses the `local` feature by default. For GPU support in the AppImage, you need a build environment with Vulkan SDK or CUDA toolkit installed. See `packaging/appimage/README.md` for details.

## Multiple GPUs

llm-mem doesn't currently support splitting a model across multiple GPUs. It also doesn't let you pin to a specific GPU — it uses the first one llama.cpp finds. For most setups this is fine.

## Next

- [Configuration](../config-file.md) — `gpu_layers` and friends
- [Logging & debugging](../logging-debugging.md)
