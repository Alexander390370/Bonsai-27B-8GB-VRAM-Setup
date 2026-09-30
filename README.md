#  Running 27B LLM on 8GB VRAM (RTX 4060)

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
| `-fa on` | Enables Flash Attention for longer contexts. Uses slightly more VRAM, but improves stability at 64K. |
| `-c 65536` | Sets context window to 64K |
| `--cache-type-k/v q4_0` | Quantizes KV cache to 4-bit, reducing KV cache memory by ~72%. Quality impact is a ~7.6% perplexity increase, but in practice, generation throughput degrades at longer contexts (see Benchmarks). |

The KV cache quantization is the most impactful flag here — it cuts KV cache VRAM usage by roughly **72%**, which is what makes 64K possible on an 8GB card.

### Sampling Parameters (Recommended)

If you find the output quality lacking, try setting these sampling parameters explicitly. The server defaults are reasonable, but for longer contexts, a slightly lower temperature and top-p often help.

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `temperature` | `0.7` | Lower = more deterministic. Default is `0.8`. |
| `top_p` | `0.9` | Nucleus sampling. Default is `0.95`. |
| `top_k` | `40` | Default is `40`. |
| `repeat_penalty` | `1.1` | Helps avoid repetition, especially in long generations. |

You can pass these per-request via the OpenAI-compatible API, or set them as server defaults with `--temp`, `--top-p`, etc.

## Gotchas & Fixes

**1. Missing CUDA runtime libraries**
- **Symptom**: `llama-server` fails with `error while loading shared libraries: libcudart.so.13`.
- **Fix**: Add `/usr/lib/wsl/lib` to `LD_LIBRARY_PATH` (persist in `~/.bashrc`).

**2. Client compatibility — DeepSeek Harness (dsh) fails, WSL-native agents work**
- **Symptom**: Requests from the DSH desktop app on Windows fail with `Connection error` or `Request timed out` against `http://127.0.0.1:8331/v1`. However, running Hermes Agent natively inside WSL against the same endpoint works flawlessly.
- **Confirmed cause**: Not a WSL networking issue — `curl` against the same endpoint returns as expected. DSH uses Node's built-in `fetch`, which has a hard 300-second (5-minute) timeout for both response headers and body bytes. When a 27B model is doing a long prefill (e.g., for a large system prompt + tool registry), the server stays silent for more than 5 minutes, and DSH gives up, retrying the whole prompt. Llama.cpp's `llama-server` sends SSE keepalive pings every 30 seconds, but DSH's underlying HTTP client does not treat those pings as valid body bytes, so the timeout still fires.
- **Workaround**: Use the community plugin [`dsh-fetch-timeouts`](https://www.npmjs.com/package/dsh-fetch-timeouts) to raise DSH's HTTP timeouts to 30 minutes. Alternatively, use the [`dsh-llm-gate`](https://github.com/deepseek-ai/deepseek-harness/discussions/4995) plugin if you are also hitting concurrency issues (DSH retrying requests that are queued behind a single `--parallel 1` slot). In practice, lightweight agents (Hermes, minimal CLI clients) are a much better fit for a 27B model on 8GB VRAM.
- **Status**: Root cause identified. The `dsh-fetch-timeouts` plugin directly addresses the 5-minute hard timeout.

**3. OOM crashes under load**
- **Symptom**: Server crashes when handling long contexts or complex prompts.
- **Fix**: 8GB is tight. Close other GPU-heavy apps (LM Studio, Ollama, SD WebUI) before starting.

## Things Worth Exploring

- **MTP speculative decoding**: Try `--spec-type draft-mtp --spec-draft-n-max 2` if your model supports it.
- **Community runtime**: Check out [sudoingX/bonsai2-small-gpu](https://github.com/sudoingX/bonsai2-small-gpu) for kernel-level optimizations.
- **Higher-precision KV cache**: If quality is more important than context length, try `q8_0` for KV cache. It saves ~47% KV memory (vs 72% for `q4_0`) but has negligible quality loss.

## Benchmarks (My Numbers)

These are rough numbers from my setup — YMMV.

**Benchmark conditions**: Idle GPU, no other CUDA workloads running, pure generation with a short prompt (low prefill cost). For long-context prefill, TTFT will be significantly higher.

| Metric | Value |
|--------|-------|
| **TTFT** | ~3-5s (short prompt, 64K context window allocated) |
| **Throughput (generation)** | ~20-23 t/s (sustained) |
| **Throughput (prefill)** | Not measured — varies with prompt length |
| **Concurrency** | Single client at a time |

## Limitations

- **Single-client only.** Running multiple concurrent requests against `--parallel 1` will queue them, and heavy clients will timeout. Use a queue gate or switch to a lighter client.
- **Large prefill increases TTFT significantly.** Feeding a multi-thousand-token system prompt + tool registry will push first-token latency into tens of seconds or minutes.
- **Quality is constrained by ternary quantization.** While near-lossless on average, ternary quantization can degrade multi-step reasoning and specialist factual knowledge. The KV cache quantization (`q4_0`) adds a further ~7.6% perplexity increase.

## About Me

- GitHub: [@Alexander390370](https://github.com/Alexander390370)
- I tinker with AIoT, edge AI, and embedded systems. Always learning.

---
*Feel free to open issues or PRs if you find better configs!*
