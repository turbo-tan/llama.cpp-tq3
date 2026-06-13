# TQ3 Status

Date: 2026-03-27

## Status

Historical bring-up document.

For the current active plan, use:

- `FIX_PLAN_2026-03-28.md`

Current active conclusions that supersede parts of this file:

- `TQ3_1S` is now the active weight-quality line
- `TURBO_LAYER_ADAPTIVE=1` is the best current default candidate for 9B KV experiments
- `Q4_0_TQ` and `Q4_1_TQ` are no longer the primary execution path

This note tracks the current CUDA bring-up status for `TQ3_0` weight quantization in `llama.cpp`.

## Current State

- Dense cuBLAS fallback is correct with fp16 inputs and fp32 accumulate.
- MMQ is enabled for `TQ3_0` and has exact q8_0 tile packing.
- MMQ has a focused correctness test and microbenchmark.
- MMVQ is now wired up correctly for `TQ3_0`.
- A focused CUDA test now checks the `vec_dot_tq3_0_q8_1` contract.
- On recent NVIDIA GPUs, the current bridge MMQ/MMVQ path is slower than the dense tensor-core path.
- Current backend policy prefers cuBLAS for `TQ3_0` on NVIDIA with fp16 MMA hardware.

## Key Commits In `llama.cpp`

- `78b95c438` `Fix TQ3_0 MMQ tile packing`
- `6fe8266e7` `perf: enable and profile TQ3_0 MMQ path`
- `cdb06fe00` `perf: optimize TQ3_0 MMQ subgroup decode`
- `01e76a6cf` `perf: enable TQ3_0 MMVQ path`
- `8364b7c7f` `perf: prefer cuBLAS for TQ3_0 on NVIDIA`

## Current Benchmarks

Current same-machine GPU benchmark table:

| Model | PP tok/s | TG tok/s | Model VRAM |
|---|---:|---:|---:|
| `q4_0` | `7657.23` | `96.33` | `571.37 MiB` |
| `tq3_0` | `1798.09` | `29.96` | `455.87 MiB` |

Current interpretation:

- `tq3_0` keeps about `115.5 MiB` of model VRAM savings
- `tq3_0` is still much slower than `q4_0`
- the new backend policy is a clear improvement for TQ3 relative to the earlier MMQ/MMVQ bridge on this NVIDIA GPU

### Benchmark Matrix

Use this table for future runs so weight-format and KV-cache effects stay separated.

| Model | Weight format | TQ cache | File size | CUDA model buffer | PP tok/s | TG tok/s | Quality sanity |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.5-27B | `Q8_0` | off | `27.26 GiB` | not recorded here | pending | pending | sane |
| Qwen3.5-27B | `Q4_0` | off | pending | `14067.38 MiB` | `881.58 tok/s` on PP512 | `20.49 tok/s` short decode run | sane |
| Qwen3.5-27B | `TQ3_0` | off | `11.85 GiB` | `11164.57 MiB` | `7.25 tok/s` prompt-only short run | `14.05 tok/s` short decode run | sane on `Paris`, `4`, `cold`, `kitten` |
| Qwen3.5-27B | `Q4_0_TQ` | off | `11.70 GiB` | `11164.57 MiB` | `111.85 tok/s` on PP512 | `15.19 tok/s` short decode run | sane on `4`, `Tokyo`, `40 mph`, `Jupiter`, blue-sky answer |
| Qwen3.5-27B | `Q4_0` | on | pending | pending | pending | pending | pending |
| Qwen3.5-27B | `TQ3_0` | on | pending | pending | pending | pending | pending |
| Qwen3.5-27B | `Q4_0_TQ` | on | pending | pending | pending | pending | pending |

Notes:

- `TQ cache off` means normal KV cache.
- `TQ cache on` means the same weight format with TurboQuant KV/cache enabled.
- The current `Q4_0_TQ` CUDA path now has:
  - working TG/MMVQ path
  - working PP/MMQ path
  - `graph splits = 2`
- Earlier `Q4_0_TQ` PP512 attempts crashed with illegal CUDA memory access.
- The current remaining `Q4_0_TQ` problem is speed gap versus native `Q4_0`, not correctness.

### Q4_0_TQ PP Fix

What was fixed in the latest checkpoint:

- `Q4_0_TQ` was added to the MMQ shared-memory sizing helpers in CUDA.
- explicit MMQ template-instance wiring was added for:
  - `GGML_TYPE_TQ3_0`
  - `GGML_TYPE_Q4_0_TQ`
- a focused CUDA bridge test was added for `Q4_0_TQ -> q8_0` MMQ tile packing.

Files involved in `llama.cpp`:

- `ggml/src/ggml-cuda/mmq.cuh`
- `ggml/src/ggml-cuda/mmq.cu`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/template-instances/mmq-instance-tq3_0.cu`
- `ggml/src/ggml-cuda/template-instances/mmq-instance-q4_0_tq.cu`
- `ggml/src/ggml-cuda/template-instances/generate_cu_files.py`
- `tests/test-q4_0_tq-load-tiles.cu`

Guardrails now available:

- `test-q4_0_tq-runtime`
- `test-quantize-fns`
- direct CUDA bridge test for `Q4_0_TQ -> q8_0` tile packing

Measured outcome:

- before this fix:
  - `Q4_0_TQ` PP512 path crashed on both Qwen 0.6B and Qwen3.5-27B
  - short prompt PP sanity was only `35.51 tok/s`
- after this fix:
  - Qwen 0.6B `Q4_0_TQ` PP512: `4235.19 tok/s`
  - Qwen3.5-27B `Q4_0_TQ` PP512: `111.85 tok/s`
  - Qwen3.5-27B native `Q4_0` PP512: `881.58 tok/s`

Interpretation:

- this was a real correctness/unblocking fix for `Q4_0_TQ` PP, not just tuning
- `Q4_0_TQ` prefill now runs instead of crashing
- `Q4_0_TQ` is still about `7.9x` slower than native `Q4_0` on the 27B PP512 witness

## What Improved

- MMQ subgroup decode optimization gave a real but limited PP gain.
- MMVQ enablement was the major TG unlock inside the bridge path.
- The biggest practical gain so far came from preferring cuBLAS over the current TQ3 MMQ/MMVQ bridge on NVIDIA.
- TinyLlama sanity output remains correct on the current path:
  - `The capital of France is Paris.`

## What Was Learned

- The original dense bug was fp16 accumulation, not the TQ3 dequant itself.
- The earlier `512`-entry subgroup LUT idea was wrong.
  - An 8-value TQ3 subgroup is encoded in `24` packed bits, not `9`.
- The exact MMQ tile loader is still more expensive than q4 because it reconstructs signal structure before reusing legacy q8 kernels.
- The current TQ3 bridge architecture is correct but not competitive with q4 on recent NVIDIA GPUs.
- The practical NVIDIA execution policy today is:
  - keep compact TQ3 storage
  - use dense cuBLAS execution as the working path
  - treat native TQ3 kernels as the real phase-two speed project

## CPU Status

Current CPU behavior is poor:

- `q4_0` CPU TG: about `8.58 tok/s`
- `tq3_0` CPU TG: about `0.74 tok/s`
- TQ3_0 CPU is currently about `11x` slower on token generation

Why:

- `q4_0` already has SIMD-optimized CPU vec-dot paths
- `tq3_0` CPU still pays for scalar unpack, centroid decode, and full WHT reconstruction
- the memory savings are overwhelmed by the extra compute cost

Current conclusion:

- TQ3_0 on CPU is not a performance path today
- TQ3_0 on GPU is the practical speed path right now
- CPU TQ3 is still worth pursuing for low-VRAM users, but it should be treated as a separate project

What a serious CPU path would need:

1. fused AVX2/AVX-512/NEON vec-dot for TQ3
2. SIMD-friendly WHT implementation
3. likely a path that avoids full scalar materialization of the block
4. separate tuning for prefill and decode

## Current Risks

- PP and TG are both still far behind q4, so the remaining gap is architectural, not just local tuning.
- The current NVIDIA policy is a good fallback, not the final TQ3 speed story.
- The next real speed win requires a native TQ3 GPU path that avoids repeatedly staging through legacy q8-style execution.

## Next Steps

1. Keep the current NVIDIA cuBLAS policy as the stable path.
2. Write down the next-stage native TQ3 GPU architecture as a concrete task list.
3. Profile the current cuBLAS-heavy TQ3 path to identify where dense execution still loses relative to q4.
4. Design and implement a native TQ3 GPU path instead of repeatedly staging through q8.
5. Treat CPU TQ3 as a separate follow-on project for low-VRAM users.

## 7B Dense Server Repro (2026-03-28)

Practical server benchmark witness using `Qwen2.5-Coder-7B-Instruct` via `/v1/chat/completions` with `chat_template_kwargs.enable_thinking = false`.

Artifacts:

- `Q8_0`: `/tmp/qwen25coder7b/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf`
- `Q4_0`: `/tmp/qwen25coder7b/out/Qwen2.5-Coder-7B-Instruct-Q4_0.gguf`
- `Q4_0_TQ`: `/tmp/qwen25coder7b/out/Qwen2.5-Coder-7B-Instruct-Q4_0_TQ.gguf`

Observed matrix before fallback probe:

| Weight format | K cache | CUDA model | CUDA KV | Prompt tok/s | Decode tok/s | Quality |
|---|---|---:|---:|---:|---:|---|
| `Q4_0` | `f16` | `3928 MiB` | `448 MiB` | `1318-1541` | `85.9-91.3` | good |
| `Q4_0` | `tq3_0` | `3928 MiB` | `273 MiB` | `641-965` | `55.4-65.9` | broken |
| `Q4_0_TQ` | `f16` | `3150 MiB` | `448 MiB` | `321-473` | `59.6-68.0` | good |
| `Q4_0_TQ` | `tq3_0` | `3150 MiB` | `273 MiB` | `287-433` | `44.8-49.2` | broken |

Interpretation:

- `Q4_0_TQ` weight quantization itself is viable on this dense witness.
- The current regression is in the shared `tq3_0` KV-cache path.
- Both native `Q4_0` and `Q4_0_TQ` fail when `tq3_0` K cache is enabled, so the bug is not specific to the new weight format.

Important update after broader witness testing:

- this 7B witness is not a universal KV-cache truth source
- on larger Qwen-family witnesses, `tq3_0` KV is currently sane:
  - Qwen3.5-27B `Q3_K_M` on the current branch
  - Qwen3.5-35B-A3B `Q4_K_M` on both the current branch and the Aaryan branch with mixed offload
- therefore:
  - do not use the 7B coder witness alone to judge TurboQuant KV correctness
  - do use 27B and 35B Qwen-family witnesses for cache validation
  - treat the 7B result as a model-specific incompatibility or sensitivity until proven otherwise
