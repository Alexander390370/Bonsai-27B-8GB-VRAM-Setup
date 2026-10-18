# Running Qwen3.5 4B Locally on 8GB VRAM (Windows + WSL2)

> Two 4B models, 64K context, ~4.3 GB VRAM. Includes the fix for the "thinking loop never ends" bug that makes abliterated Qwen3.5 models unusable.

**Tested on**: RTX 4060 Laptop (8 GB), Windows 11 + WSL2 Ubuntu, October 2026.

---

## TL;DR

| Model | Size | Context | VRAM | Speed | Thinking | Use for |
|---|---|---|---|---|---|---|
| **Qwopus3.5-4B-Coder-MTP** | 2.59 GB | 64K | 4.3 GB | 54-60 tok/s | natural stop | agents, tool calls |
| **Qwen3.5-4B-abliterated-agent** | 2.71 GB | 64K | 4.3 GB | 47-52 tok/s | **budget 400** | uncensored chat |

**Key finding**: abliterated Qwen3.5 models have a broken thinking loop. The fix is `--reasoning-budget 400` + `--reasoning-budget-message`, not `--reasoning off`.

---

## Why 4B and not 7B/8B

The hard constraint is the context window, not the parameter count.

| Model class | 64K KV cache (q4_0) | Total VRAM | Fits 8 GB? |
|---|---|---|---|
| 4B | ~2.0 GB | **4.3 GB** | ✅ comfortable |
| 7B/8B | ~4.0 GB | **7.3 GB** | ⚠️ 0.7 GB left, unstable |

Also worth knowing: **Qwen2.5's architecture caps at 32K**. You cannot get 64K from it no matter what you pass. You need a Qwen3.5 (or Llama 3.1) base.

---

## Model 1: Qwopus3.5-4B-Coder-MTP

A Qwen3.5-4B fine-tune by Jackrong, trained specifically for agent workflows (Trace Inversion + agent trajectory data + curriculum SFT).

**Author's benchmarks** (self-reported, `benchlocal` framework, 15-question samples — treat as indicative, not authoritative):

| Test | Score |
|---|---|
| ToolCall | 100/100 |
| InstructFollow | 93/100 |
| BugFind | 71/100 |
| HermesAgent | 64/100 |

**MTP (Multi-Token Prediction)** is baked into training — claims 1.4-2.2x decode speedup. Requires a llama.cpp build with `--spec-type draft-mtp` support. **Ollama cannot use it.**

**My measurements**:

| Test | Result | Time |
|---|---|---|
| Chinese Q&A | ✅ correct | 6-8 s |
| Tool call (`get_weather`) | ✅ `{"city":"北京"}` | 1.1 s |
| Math trap (17 sheep) | ✅ 9 | — |
| Code (palindrome) | ✅ correct | — |
| Speed | 54-60 tok/s | — |

**Caveat**: it's a thinking model. `max_tokens` below ~500 returns an **empty** `content` field because the budget is consumed by `reasoning_content`. Give it 2000.

---

## Model 2: Qwen3.5-4B-abliterated-agent

An abliterated (refusal-direction-removed) variant, also agent-tuned.

**The bug**: the thinking loop does not terminate.

```
max_tokens=3000  ->  completion_tokens=3000, content="", finish_reason=length
reasoning_content: 10616 characters, still going
```

The model keeps second-guessing itself forever:

> "Wait, I need to be careful. If I output '我明白了', it is 4 characters... Wait, is there any chance... Okay, final decision... Wait, I'm overthinking..."

This is **not** an Ollama bug and **not** a prompt problem. Abliteration damages the thinking-termination logic.

### What does NOT work

| Attempt | Result |
|---|---|
| `--reasoning off` | ✅ content appears, but **reasoning ability is gone** — math answers wrong |
| `--reasoning-budget 800` alone | ⚠️ thinking stops at 2154 chars, but content is still thinking text |
| Bigger `max_tokens` | ❌ just burns more tokens on the loop |
| `/no_think` in prompt | ❌ ignored |

### What DOES work

```bash
--reasoning-budget 400 \
--reasoning-budget-message "Stop thinking and give the final answer now."
```

The budget caps thinking, then the injected message forces the model to exit the thinking phase and produce real content.

### Tuning the budget

| Budget | Thinking | Time | Math correct? |
|---|---|---|---|
| 800 | 2184 chars | 13.5 s | ✅ |
| **400** | **1155-1530 chars** | **6.5 s** | ✅ |
| 200 | 565 chars | 3.7 s | ❌ answered 8 instead of 9 |

**400 is the sweet spot**: same speed as the normal model, reasoning still intact.

---

## Setup

### 1. Get a llama.cpp build with MTP + reasoning-budget support

You need a build that exposes `--reasoning-budget` and `--spec-type draft-mtp`. The PrismML fork (build 10743 used here) has both. Check yours:

```bash
./llama-server --help | grep -E "reasoning-budget|spec-type"
```

Ollama does **not** expose these flags — you must use `llama-server` directly.

### 2. CUDA 12 libraries

The CUDA backend needs `libcublas.so.12` and `libcudart.so.12`. If your distro ships CUDA 13, install the 12.x libs into an isolated conda env:

```bash
conda create -n dsh-bonsai python=3.11 -y
conda activate dsh-bonsai
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
```

**Watch out**: WSL often inherits `http_proxy=http://127.0.0.1:65532` from Windows, which is unreachable inside WSL. pip hangs silently. Bypass it:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="*" no_proxy="*"
```

### 3. Download the models

HF's CDN (`cas-bridge.xethub.hf.co`) needs the proxy. Plain `curl` without `--proxy` returns 0 bytes.

```bash
# Use the proxy that works on your machine
export https_proxy=http://127.0.0.1:65532

wget -c --tries=20 --timeout=60 --waitretry=3 \
  "https://hf-mirror.com/Jackrong/Qwopus3.5-4B-Coder-MTP-GGUF/resolve/main/Qwopus3.5-4B-Coder-MTP-Q4_K_M.gguf"

wget -c --tries=20 --timeout=60 --waitretry=3 \
  "https://hf-mirror.com/AlexWortega/qwen3.5-4b-abliterated-agent-20260515/resolve/main/gguf/qwen3.5-4b-abl.Q4_K_M.gguf"
```

**Note the filename case**: `Q4_K_M` is uppercase. `q4_K_M` returns 404.

Downloads drop often (TLS resets). Wrap them in a retry loop — `wget -c` resumes correctly.

### 4. Launch

```bash
# Normal model
./llama-server \
  -m Qwopus3.5-4B-Coder-MTP-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8400 \
  -ngl 99 -fa on -c 65536 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --parallel 1 --fit off --jinja --no-warmup \
  --alias qwopus4b

# Abliterated model (note the two reasoning flags)
./llama-server \
  -m qwen3.5-4b-abliterated-agent-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8401 \
  -ngl 99 -fa on -c 65536 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --parallel 1 --fit off --jinja --no-warmup \
  --reasoning-budget 400 \
  --reasoning-budget-message "Stop thinking and give the final answer now." \
  --alias ablit4b
```

Only **one** model at a time — 8 GB can't hold both (4.3 + 4.3 > 7.4 GB usable).

---

## Parameter notes

| Flag | Why |
|---|---|
| `-ngl 99` | All layers on GPU. **Requires `--fit off`** — otherwise the auto-fitter aborts and the model never loads (see previous write-up). |
| `-fa on` | Required for quantized KV cache. Uses ~660 MB more VRAM than `-fa off`, contrary to popular belief. |
| `-c 65536` | 64K context. Above this, KV cache pushes past 8 GB. |
| `--cache-type-k/v q4_0` | Halves KV cache vs q8_0. Quality cost is small with modern Hadamard rotation. |
| `--fit off` | Mandatory with `-ngl 99`. |
| `--jinja` | Enables the chat template (needed for tool calls). |

### VRAM budget (measured)

```
Model weights (Q4_K_M)     2.8 GB
KV cache 64K (q4_0)        2.0 GB
Compute buffer             0.5 GB
────────────────────────────────
Total                      5.3 GB
Measured (nvidia-smi)      4.3 GB  (KV not fully allocated at load)
Headroom                   3.5 GB  ✅
```

---

## Connecting to DeepSeek Harness

Both models are OpenAI-compatible. Two providers:

```yaml
llamacpp:
  displayName: llama.cpp (local)
  api: openai-completions
  baseURL: http://127.0.0.1:8400/v1
  apiKeyEnv: LLAMACPP_API_KEY
  compat:
    supportsDeveloperRole: false
  models:
    - id: qwopus4b
      name: Qwopus3.5-4B-Coder-MTP | 4B - 64K - tool calls
      contextWindow: 65536
      maxTokens: 8192

llamacpp-ablit:
  displayName: llama.cpp (local, uncensored)
  api: openai-completions
  baseURL: http://127.0.0.1:8401/v1
  apiKeyEnv: LLAMACPP_API_KEY
  compat:
    supportsDeveloperRole: false
  models:
    - id: ablit4b
      name: Qwen3.5-4B-abliterated | uncensored - 64K
      contextWindow: 65536
      maxTokens: 8192
```

The API key is not checked by llama-server, but DSH requires the env var to exist.

---

## What to expect

**Both models are genuinely usable at 64K context.** That's the headline — 64K with 4B on 8 GB is comfortable, not marginal.

**The abliterated model is not a downgrade.** After the budget fix, its speed matches the normal model (6.5 s vs 6-8 s) and its reasoning is intact. The only difference is whether it refuses sensitive topics.

**The thinking loop is the real trap.** If you try an abliterated Qwen3.5 and get empty responses, don't assume the model is broken — check `reasoning_content` length and apply the budget fix.

---

## Related

- **[Bonsai-27B-8GB-VRAM-Setup](https://github.com/Alexander390370/Bonsai-27B-8GB-VRAM-Setup)** — ternary 27B on the same card (slower, different tradeoffs)
- **[sd-forge-8gb-vram-setup](https://github.com/Alexander390370/sd-forge-8gb-vram-setup)** — Stable Diffusion on 8 GB

## About

- GitHub: [@Alexander390370](https://github.com/Alexander390370)

---

*All numbers measured on the hardware above with `nvidia-smi` and llama-server timing logs, October 2026.*
