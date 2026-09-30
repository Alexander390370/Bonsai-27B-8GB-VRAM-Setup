# Running 27B LLM on 8GB VRAM (RTX 4060)

> Running a 27B parameter model with 64K context on an 8GB VRAM laptop — here's how I got it working.

## Overview

This repo documents my experience deploying **Ternary Bonsai 2 27B** on a resource-constrained setup: WSL2 (Ubuntu) + RTX 4060 Laptop (8GB VRAM). It covers the setup steps, gotchas I ran into, and tweaks that helped squeeze out more performance.

The core enabler is **ternary-weight quantization** — constraining model weights to {-1, 0, +1} — which is what allows a 27B model to fit into ~5.6GB.

## Hardware & Environment

- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM)
- **OS**: Windows 11 + WSL2 (Ubuntu 22.04)
- **Model**: `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (~5.6GB)
- **Runtime**: PrismML fork of `llama.cpp` (CUDA 13.3)

## Quick Start

Make sure you have downloaded the ternary GGUF model file before launching.

The model server exposes an OpenAI-compatible API:

```bash
# Set WSL CUDA library path (temporary for current shell; add to ~/.bashrc for permanent)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$PWD:$LD_LIBRARY_PATH

# Launch server with 64K context
./llama-server -m ../Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --port 8331 \
  -ngl 99 \
  -fa on \
  -c 65536 \
  --cache-type-k q4_0 --cache-type-v q4_0
```

> **Note**: This model requires the **PrismML fork** of `llama.cpp`. Upstream `llama.cpp` does not support the `ptq1_0` weight type — it will either fail to load the GGUF or (worse) load it silently and decode garbage. `-fa on` works fine upstream, but is required here because quantized KV cache depends on Flash Attention.

### Parameter Notes

| Flag | What it does |
|------|---------------|
| `-ngl 99` | Offloads all layers to GPU |
| `-fa on` | Enables Flash Attention for longer contexts. Uses slightly more VRAM, but improves stability at 64K. |
| `-c 65536` | Sets context window to 64K |
| `--cache-type-k/v q4_0` | Quantizes KV cache to 4-bit, reducing KV cache memory by ~72%. Quality impact is a ~7.6% perplexity increase, but in practice, generation throughput will drop when working near the 64K context ceiling. |

The KV cache quantization is the most impactful flag here — it cuts KV cache VRAM usage by roughly **72%**, which is what makes 64K possible on an 8GB card.

> **Note on quality**: Since April 2026, mainline `llama.cpp` applies Hadamard rotation to KV activations ([PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)), which greatly improves low-bit KV quality. The 7.6% perplexity figure is a pre-rotation typical value; your fork may see less degradation if it includes this PR.

### Sampling Parameters (Recommended)

If you find the output quality lacking, try setting these sampling parameters explicitly. The server defaults are reasonable, but for longer contexts, a slightly lower temperature and top-p often help.

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `temperature` | `0.7` | Lower = more deterministic. Default is `0.8`. |
| `top_p` | `0.9` | Nucleus sampling. Default is `0.95`. |
| `top_k` | No change from default | — |
| `repeat_penalty` | `1.1` | Helps avoid repetition, especially in long generations. |

You can pass these per-request via the OpenAI-compatible API, or set them as server defaults with `--temp`, `--top-p`, etc.

## Gotchas & Fixes

**1. Missing CUDA runtime libraries**
- **Symptom**: `llama-server` fails with `error while loading shared libraries: libcudart.so.13`.
- **Fix**: Add `/usr/lib/wsl/lib` to `LD_LIBRARY_PATH` (persist in `~/.bashrc`).

**2. Client compatibility — DeepSeek Harness (DSH) fails, WSL-native agents work**
- **Symptom**: Requests from the DSH desktop app on Windows fail with `Connection error` or `Request timed out` against `http://127.0.0.1:8331/v1`. However, running Hermes Agent natively inside WSL against the same endpoint works flawlessly.
- **Likely cause**: Not a WSL networking issue — `curl` against the same endpoint returns as expected. There are two independent 300-second timers at play:
    1. **Node/undici's HTTP timers**: DSH uses Node's built-in `fetch`, which has a hard 300-second timeout for both response headers and body bytes. While `llama-server` sends SSE keepalive pings every 30 seconds, these pings may not be treated as valid body bytes by the client's timeout logic in all cases.
    2. **DSH's own stream idle watchdog** (`streamIdleTimeoutMs`): This is a separate timer that also defaults to 300 seconds. Raising only one timer may not resolve the issue, as the other will still fire.
- **Workaround**: A community plugin (`dsh-fetch-timeouts`) exists to raise Node's HTTP timeouts to 30 minutes. You also need to raise DSH's `streamIdleTimeoutMs` in the provider route config. As a fallback, if you cannot modify DSH internals, use a lightweight terminal-native agent (Hermes or a minimal CLI client) — these are a much better fit for a 27B model on 8GB VRAM.
- **Status**: Root cause strongly suspected; formal error-code reproduction still pending.

**3. OOM crashes under load**
- **Symptom**: Server crashes when handling long contexts or complex prompts.
- **Fix**: 8GB is tight. Close other GPU-heavy apps (LM Studio, Ollama, SD WebUI) before starting.

## Things Worth Exploring

- **MTP speculative decoding**: The Bonsai 2 GGUF may not include the MTP head by default. Check whether your specific GGUF file has embedded MTP weights; if not, you'll need the community DFlash2 draft model instead. If your file does include it, try `--spec-type draft-mtp --spec-draft-n-max 2`.
- **Community runtime**: There is a community-optimized runtime for small-VRAM setups (search GitHub for `bonsai2-small-gpu`) that provides a 1.5x faster decode kernel and a grafted MTP head.
- **Higher-precision KV cache**: If quality is more important than context length, try `q8_0` for KV cache. It saves ~47% KV memory (vs 72% for `q4_0`) but has negligible quality loss.

## Benchmarks (My Numbers)

These are rough numbers from my setup — YMMV.

**Benchmark conditions**: Idle GPU, no other CUDA workloads running, pure generation with a short prompt (low prefill cost). For long-context prefill, TTFT will be significantly higher.

| Metric | Value |
|--------|-------|
| TTFT | ~3-5s (short prompt, 64K context window allocated) |
| Throughput (generation) | ~20-23 t/s (sustained) |
| Concurrency | Single client at a time |

## Limitations

- **Single-client only.** Running multiple concurrent requests against `--parallel 1` will queue them, and heavy clients will timeout. Use a queue gate or switch to a lighter client.
- **Large prefill increases TTFT significantly.** Feeding a multi-thousand-token system prompt + tool registry will push first-token latency into tens of seconds or minutes.
- **Quality is constrained by ternary quantization.** While near-lossless on average, ternary quantization can degrade multi-step reasoning and specialist factual knowledge. The KV cache quantization (`q4_0`) adds a further perplexity increase (see the quality note above).
- **VRAM headroom is minimal.** Even minor GPU memory drift from background processes can trigger OOM.

## About Me

- GitHub: [@Alexander390370](https://github.com/Alexander390370)
- I tinker with AIoT, edge AI, and embedded systems. Always learning.
- This configuration is laptop-specific. Desktop RTX 4060 (8GB) yields comparable numbers, though sustained power and thermal behaviour differ from laptop SKUs.

---
*Feel free to open issues or PRs if you find better configs!*
