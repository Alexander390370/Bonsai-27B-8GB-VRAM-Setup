#   Running 27B LLM on 8GB VRAM (RTX 4060)

> Running a 27B parameter model with 64K context on an 8GB VRAM laptop — here's how I got it working.

##  Overview

This repo documents my experience deploying **Ternary Bonsai 2 27B** (a ternary-weight quantized LLM) on a resource-constrained setup: WSL2 (Ubuntu) + RTX 4060 Laptop (8GB VRAM). It covers the setup steps, gotchas I ran into, and tweaks that helped squeeze out more performance.

## Hardware & Environment

- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM)
- **OS**: Windows 11 + WSL2 (Ubuntu 22.04)
- **Model**: `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (~5.6GB)
- **Runtime**: PrismML fork of `llama.cpp` (CUDA 13.3)

## Quick Start

The model server exposes an OpenAI-compatible API:

```bash
# Set WSL library path (resolve CUDA dependencies)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$PWD:$LD_LIBRARY_PATH

# Launch server with 64K context
./llama-server -m ../Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --port 8331 \
  -ngl 99 \
  -fa on \
  -c 65536 \
  --cache-type-k q4_0 --cache-type-v q4_0
```

### Parameter Notes

| Flag | What it does |
|------|---------------|
| `-ngl 99` | Offloads all layers to GPU |
| `-fa on` | Enables Flash Attention for longer contexts |
| `-c 65536` | Sets context window to 64K |
| `--cache-type-k/v q4_0` | Quantizes KV cache to 4-bit, freeing up significant VRAM |

The KV cache quantization is the most impactful flag here — it cuts KV cache VRAM usage by roughly half, which is what makes 64K possible on an 8GB card.

## Gotchas & Fixes

**1. Missing CUDA runtime libraries**
- **Symptom**: `llama-server` fails with `error while loading shared libraries: libcudart.so.13`.
- **Fix**: Add `/usr/lib/wsl/lib` to `LD_LIBRARY_PATH` (persist in `~/.bashrc`).

**2. Client compatibility — DeepSeek Harness (dsh) fails, WSL-native agents work**
- **Symptom**: Requests from the DSH desktop app on the Windows side fail with `Connection error` or `Request timed out` against `http://127.0.0.1:8331/v1`. The same `llama-server` responds normally when the client runs inside WSL (Hermes Agent works fine).
- **Likely cause**: Not a WSL networking issue — `curl` against the same endpoint returns as expected. [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) is an agent harness, so each request carries a large system prompt plus a full tool/skill registry. That pushes the 27B model into a long prefill phase, which exceeds the client's default timeout. Lighter terminal-native agents send much smaller payloads and get a first token quickly.
- **Workaround**: Raise the timeout to 120s+ where the client allows it, or trim the tool set / context it sends. In practice, lightweight agents (Hermes, minimal CLI clients) are a much better fit for a 27B model on 8GB VRAM.
- **Status**: Not fully root-caused yet — if anyone gets DSH working against a local `llama.cpp` server, I'd love to hear how.

**3. OOM crashes under load**
- **Symptom**: Server crashes when handling long contexts or complex prompts.
- **Fix**: 8GB is tight. Close other GPU-heavy apps (LM Studio, Ollama, SD WebUI) before starting.

## Things Worth Exploring

- **MTP speculative decoding**: Try `--spec-type draft-mtp --spec-draft-n-max 2` if your model supports it.
- **Community runtime**: Check out [sudoingX/bonsai2-small-gpu](https://github.com/sudoingX/bonsai2-small-gpu) for kernel-level optimizations.

## Benchmarks (My Numbers)

- **TTFT**: ~3-5s (with 64K context)
- **Throughput**: ~20-23 t/s (sustained)
- **Concurrency**: Single client at a time

## About Me

- GitHub: [@Alexander390370](https://github.com/Alexander390370)
- I tinker with AIoT, edge AI, and embedded systems. Always learning.

---
*Feel free to open issues or PRs if you find better configs!*
