# Running 27B LLM on 8GB VRAM (RTX 4060)

> Deploying **Ternary Bonsai 2 27B** (ternary-weight, 5.6 GB) on a laptop RTX 4060 (8 GB) via WSL2. Includes the three silent-failure traps that will eat your afternoon, and measured numbers for every context size from 16K to 64K.

## TL;DR

| Context | VRAM | Headroom | Speed | Verdict |
|---|---|---|---|---|
| 16K | 6.9 GB | 1.1 GB | 18.8 tok/s | Safe, coexists with other apps |
| 32K | 7.1 GB | 0.9 GB | 20.5 tok/s | Balanced |
| **48K** | **7.4 GB** | **0.55 GB** | **20.0 tok/s** | **Recommended** |
| 64K | 7.8 GB | 0.15 GB | 19.1 tok/s | Experimental — one background GPU blip away from OOM |

**The single most important finding: `--fit off` is required for anything above 16K.** Without it, 32K/48K/64K all hang at "Loading model" forever — the process stays alive, the port listens, and nothing ever happens.

---

## Hardware & Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop, 8188 MiB total |
| OS | Windows 11 + WSL2 (Ubuntu) |
| Model | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` — 5.6 GB, PTQ1_0 = 1.75 bpw ternary, group 128 |
| Runtime | PrismML fork of `llama.cpp`, build 10743 (`adfffbe41`) |
| Base model | Qwen family (asking "who are you" returns Qwen) |

**On the 1.75 bpw figure**: pure ternary is log₂3 ≈ 1.58 bpw. The extra ~0.17 bpw is group scaling factors and metadata. The 5.6 GB file is weights only — **KV cache is separate**, which is why context size dominates your VRAM budget.

**Why the PrismML fork is mandatory**: upstream `llama.cpp` does not implement the `ptq1_0` tensor type. It will either refuse the GGUF or — worse — load it and decode garbage. Grab a prebuilt release from the PrismML fork's releases page (`llama-prism-*` tarball).

---

## Quick Start

```bash
# --- prerequisites ---
# 1. Model file: ~/bonsai/Ternary-Bonsai-2-27B-PTQ1_0.gguf  (5.6 GB)
# 2. PrismML fork unpacked: ~/bonsai/llama-prism-<hash>/
# 3. CUDA 12 libs in an isolated env (see Trap 2 — this is the step everyone misses)

SP=~/miniconda3/envs/dsh-bonsai/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$SP/cublas/lib:$SP/cuda_runtime/lib:/usr/lib/wsl/lib:$PWD:$LD_LIBRARY_PATH"

cd ~/bonsai/llama-prism-<hash>

setsid nohup ./llama-server \
  -m ../Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --host 127.0.0.1 --port 8331 \
  -ngl 99 -fa on \
  -c 49152 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --parallel 1 \
  --fit off \
  --jinja \
  --no-warmup \
  --alias bonsai-27b \
  > ~/bonsai/run.log 2>&1 < /dev/null &
disown
```

Endpoint: `http://127.0.0.1:8331/v1` (OpenAI-compatible, no auth).

### Verify it actually worked (do this every time)

```bash
# 1. GPU is visible — if this says "(none)", you're on CPU
./llama-server --list-devices
#   expected: CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (8187 MiB, ...)

# 2. VRAM actually allocated — should be 6.9-7.8 GB, NOT ~1 GB
nvidia-smi --query-gpu=memory.used --format=csv

# 3. Model loaded, not stuck loading
curl -s http://127.0.0.1:8331/v1/models
#   expected: {"models":[{"name":"bonsai-27b",...}]}
#   if you get {"error":{"message":"Loading model","code":503}} → see Trap 1
```

---

## The Three Silent-Failure Traps

None of these produce a clear error. All three cost real time.

### Trap 1: `--fit` silently aborts when combined with `-ngl 99`

**Symptom**: server starts, port listens, but every completion request returns `503 Loading model`, forever. VRAM sits at ~500 MB.

**Log line** (only visible at higher verbosity):
```
common_fit_params: failed to fit params to free device memory:
  n_gpu_layers already set by user to 99, abort
```

**Cause**: `llama.cpp` runs an auto VRAM-fitting pass (`--fit`, default `on`). It tries to reconcile your requested config with available memory. When you pin `-ngl 99` (all layers on GPU), the fitter detects a conflict and **aborts** — it neither reduces layers nor surfaces a usable error.

**Fix**: add `--fit off`. Then `-ngl 99` is taken literally.

**Measured**:

| Context | Without `--fit off` | With `--fit off` |
|---|---|---|
| 16K | loads | loads |
| 32K | hangs forever | **6 s** |
| 48K | hangs forever | **6 s** |
| 64K | hangs forever | **2.4 s** |

> **Why this is so confusing**: 16K happens to fit under the fitter's threshold, so small configs work fine. You conclude "bigger contexts are just too big for 8 GB" — when in fact they'd fit perfectly.

### Trap 2: missing `libcublas.so.12` → `-ngl` silently falls back to CPU

**Symptom**: model loads, requests succeed, but throughput is **1.6 tok/s** instead of ~20. VRAM stays near 1 GB.

**Diagnosis**:
```bash
$ ./llama-server --list-devices
Available devices:
  (none)                              # ← should be CUDA0

$ ldd libggml-cuda.so | grep "not found"
	libcublas.so.12 => not found      # ← the culprit
	libcudart.so.12 => not found      # ← usually also missing
```

**Cause**: the CUDA backend `.so` fails to load; `llama.cpp` silently falls back to CPU with no warning at normal verbosity.

**Why the obvious fix doesn't work**: adding `/usr/lib/wsl/lib` only provides the *driver* library (`libcuda.so`). The *toolkit* libraries (`libcublas`, `libcudart`) are separate and must come from a CUDA 12 install. If your other envs have CUDA 13, they won't help — this binary is linked against CUDA 12.

**Fix** (fully isolated, touches nothing else):
```bash
conda create -n dsh-bonsai python=3.11 -y
conda activate dsh-bonsai
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
# pulls in nvidia-cuda-nvrtc-cu12 as a transitive dep
```

**Gotcha inside the gotcha**: WSL often inherits `http_proxy=http://127.0.0.1:65532` from Windows, which is **unreachable from inside WSL**. pip hangs silently. Bypass it:
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="*" no_proxy="*"
pip install --proxy "" --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
```

**One more**: `python -c "import nvidia.cublas; print(nvidia.cublas.__file__)"` returns `None` — it's a namespace package. Build the path directly, don't derive it from `__file__`.

### Trap 3: process dies when the launching shell exits

**Symptom**: server works while your terminal is open, then disappears.

**Fix**: `setsid nohup ... < /dev/null &` then `disown`. A bare `&` is not enough when the parent is `wsl -e bash -lc`.

---

## Memory Budget (measured)

Steady-state VRAM after full load, idle GPU, no other CUDA workloads, measured with `nvidia-smi`.

| Context | Model | KV cache | Total | Headroom | Speed |
|---|---|---|---|---|---|
| 16K | 5.6 GB | ~0.9 GB | 6.9 GB | 1.1 GB | 18.8 tok/s |
| 32K | 5.6 GB | ~1.2 GB | 7.1 GB | 0.9 GB | 20.5 tok/s |
| 48K | 5.6 GB | ~1.5 GB | 7.4 GB | 0.55 GB | 20.0 tok/s |
| 64K | 5.6 GB | ~1.8 GB | 7.8 GB | 0.15 GB | 19.1 tok/s |

**KV cache rule of thumb**: `q4_0` KV ≈ **30 KB per token** for this model.
`48K × 30 KB ≈ 1.4 GB`. Multiply context by 0.03 GB to plan.

**Choosing a context size**:

- **16K** — if you run other GPU apps (browser hardware acceleration, LM Studio, Ollama). 1.1 GB headroom is comfortable.
- **32K** — best speed measured (20.5 tok/s) with usable headroom. Good default for agent workloads.
- **48K** — recommended sweet spot. 0.55 GB survives normal desktop noise.
- **64K** — 0.15 GB. Works, but a single browser tab with hardware acceleration can OOM it. Treat as an experiment.

> **Note on 48K headroom**: 0.55 GB assumes browser hardware acceleration is off and no other CUDA context exists. If you're running DSH/Electron/browser with GPU compositing, drop to 32K.

---

## Parameter Reference

| Flag | What it actually does |
|---|---|
| `-ngl 99` | All layers on GPU. **Requires `--fit off`** or it silently aborts above 16K. |
| `-fa on` | Flash Attention. **Uses ~660 MB MORE VRAM** at 16K (measured: 6931 MB with `-fa on` vs 6270 MB with `-fa off`). Required when KV cache is quantized. Improves long-context stability. Do not describe it as "saving VRAM". |
| `-c 49152` | Context window. 48K is the sweet spot on 8 GB. |
| `--cache-type-k/v q4_0` | 4-bit KV cache. Saves **~72%** vs f16 (4.5 effective bits vs 16). Quality cost ~7.6% perplexity, mitigated by recent Hadamard rotation in mainline. |
| `--parallel 1` | Single slot. Additional slots split the context budget between them. |
| `--fit off` | **Mandatory.** Disables the auto-fitter that conflicts with `-ngl`. |
| `--jinja` | Enables the model's chat template (needed for tool calls). |
| `--alias bonsai-27b` | Friendly model id in the API. |
| `--no-warmup` | Skips warmup, saves a couple of seconds at start. |
| `--host 127.0.0.1` | Bind loopback only. The server has **no auth** — never use `0.0.0.0` on a shared network. |

### KV cache types supported by this build

```
f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
```

**`q2_0` is not supported** — don't waste time. `q4_0` is the floor.

### Quality vs context tradeoff

| KV type | Relative KV size | Quality |
|---|---|---|
| `f16` | 100% | baseline |
| `q8_0` | ~53% | negligible loss |
| `q4_0` | ~28% | ~7.6% perplexity increase |
| `iq4_nl` | ~28% | no measurable advantage over q4_0 here |

If you want both quality and context, **`q8_0` at 32K beats `q4_0` at 64K** — the latter has 0.15 GB headroom and degraded quality.

---

## Sampling Parameters

Server defaults are reasonable. For long-context work:

| Parameter | Recommended | Notes |
|---|---|---|
| `temperature` | `0.7` | Lower = more deterministic. Default 0.8. |
| `top_p` | `0.9` | Default 0.95. |
| `repeat_penalty` | `1.1` | Helps with long generations. |

### ⚠️ This model emits `reasoning_content`

Bonsai 2 27B is a **thinking model**. The API returns a separate `reasoning_content` field; the actual answer goes in `content`.

**If `max_tokens` is too small, the entire budget is spent thinking and `content` comes back empty.** This looks like a broken model but isn't:

```json
{
  "choices": [{
    "finish_reason": "length",
    "message": {
      "role": "assistant",
      "content": "",                                    // ← empty
      "reasoning_content": "用户要求用一句话解释死锁..."  // ← budget spent here
    }
  }]
}
```

**Rule**: **150-200 tokens minimum** for a one-sentence answer; **300+** for anything requiring reasoning. Empty content almost always means "raise max_tokens", not "the model is broken".

---

## Benchmarks

**Conditions**: idle GPU (Ollama / LM Studio / SD / browser GPU compositing all stopped), single client, `temperature=0`, 48K context unless noted.

| Metric | Value |
|---|---|
| Model load time | 2.4 s (64K) – 6 s (48K), warm page cache |
| TTFT (short prompt, ~20 tokens) | **< 1 s** |
| TTFT (3K token prompt) | **2-3 s** |
| Throughput (short prompt) | **20-23 tok/s** |
| Throughput (3K token prompt) | **14 tok/s** |
| Throughput (64K context, short prompt) | 19.1 tok/s |

**Throughput degrades with prompt length**, roughly tracking prefill cost. A 3K-token prompt costs ~30% of throughput. For agent workloads with 15-20K system prompts + tool schemas, expect **single-digit tok/s** and 10+ second TTFT — this model is not suited for that.

---

## Limitations

- **Single client only.** `--parallel 1` queues concurrent requests; a second client waits indefinitely.
- **Long prompts are expensive.** TTFT scales with prefill. A 10K-token system prompt can add 20+ seconds before the first token.
- **Ternary quantization hurts reasoning.** Basic QA, arithmetic, and code generation are fine. Multi-step logical reasoning needs much longer output budgets than the parameter count implies. The model also leaks its base identity (answers "Qwen" when asked).
- **VRAM headroom is thin.** 48K leaves 0.55 GB; 64K leaves 0.15 GB.
- **No auth, no concurrency, no batching.** Single-user tool, not a service.

---

## What I Tried That Didn't Work

| Attempt | Result |
|---|---|
| `--cache-type-k q2_0` | Unsupported cache type |
| `iq4_nl` KV at 32K | Same VRAM as q4_0, no speed gain |
| `q4_0` K + `f16` V | More VRAM, no quality gain observed |
| `--no-op-offload` | Does not fix the `--fit` conflict |
| `-fa off` for more VRAM | Frees 660 MB but KV quantization requires FA |
| Freeing Ollama VRAM first | Necessary for coexistence, but unrelated to Trap 1 |

---

## Coexisting with Ollama

Both want the same GPU. Ollama keeps models resident (`keep_alive` default 5 min), so it will silently hold 5-7 GB.

```bash
# Unload all Ollama models before starting Bonsai
curl -s http://127.0.0.1:11434/api/ps   # see what's loaded
# then for each model:
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"<name>","keep_alive":0}'
```

**They cannot run simultaneously** — a 7B model at Q4 needs ~4.7 GB, and Bonsai needs 7.4 GB. On an 8 GB card that's a hard conflict.

---

## Things Worth Exploring (unverified)

- **MTP speculative decoding**: check whether your GGUF embeds the MTP head; if yes, try `--spec-type draft-mtp --spec-draft-n-max 2`. Not tested here.
- **`--kv-mean-center`**: this build exposes it (requires `q4_0` K cache) and it targets q4_0 K quality. Not benchmarked.
- **Community runtime**: [`sudoingX/bonsai2-small-gpu`](https://github.com/sudoingX/bonsai2-small-gpu) claims a 1.5x decode kernel and grafted MTP head for low-VRAM setups. Not verified.

---

## Related Work

- **[sd-forge-8gb-vram-setup](https://github.com/Alexander390370/sd-forge-8gb-vram-setup)** — Stable Diffusion on the same 8GB card
- **[OmniForge-Data-Annotation](https://github.com/Alexander390370/OmniForge-Data-Annotation)** — full-modal data annotation
- **[esp32-edge-ai-security](https://github.com/Alexander390370/esp32-edge-ai-security)** — edge AI on the hardware side

## About Me

- GitHub: [@Alexander390370](https://github.com/Alexander390370)
- I tinker with AIoT, edge AI, and embedded systems.
- Laptop-specific. Desktop RTX 4060 (8GB) yields comparable numbers; sustained power and thermal behaviour differ.

---

*All numbers measured on the hardware above with `nvidia-smi` and `llama-server` timing logs. If you find a better config, open an issue or PR.*

---

## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.