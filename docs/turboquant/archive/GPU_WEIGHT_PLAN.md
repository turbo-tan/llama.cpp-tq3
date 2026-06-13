# TQ3_0 GPU Weight Plan

Date: 2026-03-27 (updated)

This note tracks the next-stage GPU plan for `TQ3_0` weight quantization in `llama.cpp`.

Execution guardrails live in [TEST_SUITE.md](../procedures/TEST_SUITE.md).
Weight-format follow-on design lives in [TQ3_WEIGHT_VARIANT_SKETCH.md](/home/awee/code/tan_llama/docs/turboquant/TQ3_WEIGHT_VARIANT_SKETCH.md).

## Strategic Pivot

Current position:

- keep `TQ3_0` as the compact KV format
- stop treating rotated `TQ3_0` model weights as the final product target
- move the weight-format launch path to weight-native `Q4_0_TQ`

Why:

- current rotated `TQ3_0` weights are still fragile on prompt-like prefill shapes
- the user-visible product goal is "smaller than q4, but used like q4"
- dynamic model shapes should be handled by padding and tile logic, not by a
  format-specific activation-rotation contract

Practical consequence:

- current `TQ3_0` weight debugging remains useful as a diagnostic baseline
- new weight-format design work should target `Q4_0_TQ`
- KV work continues on the existing `TQ3_0` path without disruption

## Current Position (latest: commit 436b4b0f1)

Active dispatch policy:
- PP (ne11 >= 64): MMQ path (tensor cores via q8_0 bridge)
- TG (ne11 < 64): MMVQ for contiguous weights, cuBLAS for KV cache views
- KV cache attention: cuBLAS (non-contiguous tensors excluded from MMQ/MMVQ)

Canonical benchmark (TinyLlama 1.1B, RTX 5060 Ti, `llama-bench -p 512 -n 1`):

| Model | PP 512 tok/s | TG 1 tok/s | VRAM | vs q4_0 PP | vs q4_0 TG |
|---|---:|---:|---:|---:|---:|
| `q4_0` | `10,101` | `231` | `607 MiB` | 100% | 100% |
| `tq3_0` | `2,069` | `31` | `491 MiB` | **20%** | **13%** |

Progress from session start (cuBLAS only):

| Milestone | PP tok/s | TG tok/s | PP ratio |
|---|---:|---:|---:|
| cuBLAS fp32 (broken) | 16 | 3.7 | 0.2% |
| cuBLAS fp16+fp32c | 1,063 | 24 | 10% |
| + MMQ prefill (ne11≥64) | 2,275 | 25 | 22% |
| + MMVQ for contiguous weights | **2,069** | **31** | **20%** |

Note: PP slightly lower with MMVQ because small batches now use MMVQ instead of tiled prefill. Net TG gain (+24%) outweighs PP regression.

## The Mars Shot: q4_0 Weights + TQ3_0 KV Cache

Date: 2026-03-27

### Discovery

The winning combination is NOT TQ3_0 weights (still 4-5x behind q4_0 on weight matmul).
It is **q4_0 weights + TQ3_0 KV cache**.

Benchmark (Llama-2-7B, RTX 5060 Ti):

| Config | PP 512 | PP 4096 | PP 8192 | TG 1 | KV VRAM |
|--------|--------|---------|---------|------|---------|
| q4_0 weights + q4_0 KV | 2639 | 2632 | 1989 | 44 | ~5 GB at 8K |
| q4_0 weights + tq3_0 KV | 2386 | 2279 | 1734 | 21.6 | ~1 GB at 8K |
| ratio | 90% | 87% | 87% | **49%** | **4.6x smaller** |

### Why This Wins

- PP at long context: 87-90% of q4_0 speed with 4.6x less KV memory
- At 32K+ context: KV bandwidth dominates → TQ3_0 KV will match or beat q4_0 KV
- Model fits in less VRAM → run larger models on same GPU

### The One Remaining Blocker

TG is at 49% of q4_0 because TQ3_0 KV attention uses cublas (dequants K to fp16 every token).

**Fix: MMVQ for KV attention** (non-contiguous K tensor after permute).

The MMVQ vec_dot is correct for contiguous weights (we re-enabled it).
For non-contiguous KV cache, it was blocked due to the earlier hang bug.
That hang was caused by a dispatch issue (now fixed in 6d81d60e9).

If we re-enable MMVQ for non-contiguous TQ3_0 tensors (KV cache), TG should reach ~90% of q4_0.

### Target State (Mars)

| Config | PP 512 | PP 8192 | TG 1 | KV VRAM |
|--------|--------|---------|------|---------|
| q4_0 + q4_0 KV (baseline) | 2639 | 1989 | 44 | large |
| **q4_0 + tq3_0 KV + MMVQ fix** | **~2400** | **~1800** | **~40** | **4.6x smaller** |
| ratio | 91% | 90% | **~91%** | **4.6x** |

This is the moonshot: **near-q4_0 speed with 4.6x less KV memory**.

### Next Step

Re-enable MMVQ for TQ3_0 non-contiguous tensors (KV cache path).
The contiguity guard currently blocks it. Remove the guard for MMVQ
(keep it for MMQ which has a different issue with non-contiguous tensors).

Test: KV decode repro must pass (no timeout), TG must improve from 21 → ~40 tok/s.

### Follow-On KV Decode Optimization: Sparse V Dequant

New candidate after the current MMVQ/KV work:

- exploit attention sparsity in decode
- compute softmax weights first as usual
- skip `V` dequant and accumulation for positions with negligible attention weight
- start with a conservative threshold such as `1e-6`

Why it matters:

- it is orthogonal to `TQ3_0`
- it should help any quantized KV type, not just TurboQuant
- it scales with context length, where KV decode cost matters most
- it targets decode directly without requiring a new storage format

Proposed execution order:

1. Land the current KV/MMVQ correctness path first.
2. Add sparse-`V` skip behind an env flag or compile-time guard.
3. Benchmark `f16`, `q8_0`, and `tq3_0` KV at long context.
4. Keep it only if quality remains stable relative to the same KV baseline.

Initial implementation notes:

- apply only in decode / flash-attention style path
- skip only `V`, not `K`, because attention weights are known before `V` accumulation
- measure both speedup and actual skip ratio
- treat it as a KV optimization, not a fix for `TQ3_0` model-weight quality

### Same-Machine KV Reference Benchmarks

Date: 2026-03-27  
Machine: RTX 5060 Ti 16 GB  
Method: same weights, only `type_k` changed between `q4_0` and `tq3_0`, `type_v=f16`

#### 35B MoE hybrid offload

Model:
- `/home/awee/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf`

Settings:
- `-ngl 28 -p 256 -n 16 --no-warmup`

| Model | KV K-type | PP 256 | TG 16 | Notes |
|---|---:|---:|---:|---|
| Qwen3.5-35B-A3B `Q4_K_M` | `q4_0` | `335.89` | `29.52` | hybrid CPU+GPU, full GPU does not fit on 16 GB |
| Qwen3.5-35B-A3B `Q4_K_M` | `tq3_0` | `308.88` | `28.31` | valid, but slower on this setup |

Sanity:
- Prompt `The capital of France is` produced the same coherent Qwen thinking-style preamble for both KV types.
- No obvious corruption signal from swapping `q4_0` K-cache to `tq3_0` K-cache.

#### 27B dense hybrid offload

Model:
- `/home/awee/models/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q3_K_M.gguf`

Settings:
- `-ngl 28 -p 256 -n 16 --no-warmup`

| Model | KV K-type | PP 256 | TG 16 | Notes |
|---|---:|---:|---:|---|
| Qwen3.5-27B `Q3_K_M` | `q4_0` | `258.20` | `4.59` | dense Qwen35 |
| Qwen3.5-27B `Q3_K_M` | `tq3_0` | `240.58` | `4.65` | PP slightly lower, TG effectively flat |

Sanity:
- Prompt `The capital of France is` produced the same coherent Qwen reasoning preamble under both KV types.
- KV memory changed from `20.50 MiB` to `19.50 MiB` at `c=512`.

#### LLaMA-2 7B full offload

Model:
- `/home/awee/models/llama2-7b-q4_0.gguf`

Settings:
- `-ngl 999 -p 256 -n 16 --no-warmup`

| Model | KV K-type | PP 256 | TG 16 | Notes |
|---|---:|---:|---:|---|
| LLaMA-2 7B `q4_0` | `q4_0` | `1829.27` | `76.81` | full GPU offload |
| LLaMA-2 7B `q4_0` | `tq3_0` | `729.24` | `20.38` | much slower on this short-context setup |

Sanity:
- Prompt `The capital of France is` produced `Paris` in both cases.
- KV memory changed from `164.00 MiB` to `156.00 MiB` at `c=512`.

Interpretation:
- On this machine, short-context KV-only swaps do not automatically beat `q4_0`.
- The 35B and 27B runs show that `tq3_0` KV is valid, but not faster in these hybrid settings.
- The 7B full-offload run is substantially slower with `tq3_0` KV at short context.
- This strengthens the existing plan item: the next fair test is long-context `8K/32K/64K`, where KV bandwidth pressure is high enough for the compressed K-cache to matter.



## What We Learned

1. The current MMQ/MMVQ bridge is correct but too expensive.
2. On this NVIDIA GPU, cuBLAS beats the bridge path.
3. The dense bug was fp16 accumulation, not TQ3 dequant correctness.
4. The right phase-two question is native TQ3 execution, not more heroic tuning of the q8 bridge.

## Architecture Direction

### Stable path now

- Persistent storage: `TQ3_0`
- Working execution: dense cuBLAS on NVIDIA

### Target path later

- Persistent storage: `TQ3_0`
- Working execution: native TQ3-aware GPU kernels

That means:

- direct packed TQ3 block consumption
- less temporary q8-style staging
- more transform work folded into the kernel contract
- separate optimization for prefill and decode

## Recommended Execution Order

1. Lock the current cuBLAS-backed NVIDIA path as the benchmark baseline.
2. Profile that path end-to-end and identify where TQ3 still loses to q4.
3. Decide the first native TQ3 target:
   - prefill-heavy kernel
   - decode-heavy kernel
4. Build the smallest native kernel slice with tests.
5. Benchmark against the current cuBLAS-backed TQ3 baseline.

## My Recommendation

Start with prefill first.

Reason:

- current TQ3 TG is bad, but PP is also far behind q4
- PP exposes the full weight-path cost directly
- a native prefill kernel gives the clearest answer about whether TQ3 weight quant can become competitive on NVIDIA

If prefill cannot move meaningfully with a native path, the whole GPU weight story is capped.

## First Native Kernel Candidate

Candidate:

- a native TQ3 matmul tile path for weight blocks against q8/fp16 activations

Do not start with:

- custom flash attention
- KV-only work
- another large MMQ bridge rewrite

Those are side paths relative to the current weight bottleneck.

## Task Checklist

### Dispatch and Correctness ✅ DONE

- [x] Fix fp16 cublas accumulation bug (fp16+fp32c path)
- [x] Fix exact q8_0 requant in load_tiles_tq3_0 (warp amax + exact scale)
- [x] Fix KV decode hang (block TQ3_0 from MMVQ for non-contiguous KV tensors)
- [x] Enable MMQ for TQ3_0 prefill (ne11 >= 64) — +113% PP
- [x] Re-enable MMVQ for TQ3_0 contiguous weights — +24% TG
- [x] Add contiguity guards for both MMQ and MMVQ
- [x] All tests green: quality sanity, KV decode repro, canonical benchmark

### Tests and Guardrails ✅ DONE

- [x] `tests/test-tq3-cuda.cu` — 10 unit tests (PASS)
- [x] `tests/test-tq3-load-tiles.cu` — load_tiles correctness (PASS)
- [x] `tests/test-tq3-prefill.cu` — prefill kernel correctness (PASS)
- [x] KV decode repro command (no timeout)
- [x] Quality sanity ("Paris")
- [x] STEERING.md — test-before-commit rules

### Next: Close the Remaining Gap

Current: PP 20%, TG 13% of q4_0.

**PP gap (5x behind):** MMQ load_tiles WHT is the bottleneck. Options:
- [ ] Pre-rotate activations in graph (eliminates WHT from load_tiles) — needs `GGML_OP_TURBO_WHT` in graph
- [ ] Optimize load_tiles WHT (already tried, limited gains)
- [ ] Accept gap, focus on long-context advantage

**TG gap (7x behind):** MMVQ vec_dot WHT overhead. Options:
- [ ] Pre-rotate activations (same as PP fix — eliminates WHT from vec_dot too)
- [ ] Profile vec_dot to find next bottleneck after WHT

**Long-context advantage (publish):**
- [ ] Benchmark KV cache at 8K/32K/64K context — TQ3_0 KV reads 78% less data
- [ ] Find crossover point where TQ3_0 KV beats q4_0 KV on PP
- [ ] Prototype sparse-`V` dequant skip for KV decode and benchmark at 8K/32K/64K
- [ ] Validate sparse-`V` on `q8_0` KV too, to separate format-specific gain from general KV gain

## Success Criteria

Short term (current):
- [x] PP > 20% of q4_0 ✓ (20%)
- [x] TG > 10% of q4_0 ✓ (13%)
- [x] No correctness regressions ✓

Medium term (next):
- [ ] PP > 50% of q4_0 (requires pre-rotation or better load_tiles)
- [ ] TG > 50% of q4_0 (requires pre-rotation or better vec_dot)
- [ ] Long-context benchmark showing TQ3_0 KV wins at 32K+

Discard this specific hook as the main route for phase-two performance work.

Keep:

- the native block-dot helper and test guardrail as reusable kernel contract
- the microbenchmark scaffolding

Pivot to:

- integrating native TQ3 execution directly into a production prefill kernel path (MMQ-side or a new dedicated TQ3 matmul path), where it can replace real work rather than sample a side path.

## MMQ Prefill Entry Test

Date: 2026-03-27

Implemented a real MMQ-side prefill entry switch in `llama.cpp`:

- env flag: `GGML_CUDA_TQ3_PREFER_MMQ_PREFILL=1`
- behavior: for `TQ3_0` on NVIDIA with tensor cores and `ne11 >= 64`, prefer MMQ path for prefill-like batch shapes
- default policy remains unchanged when the flag is not set

Measured on TinyLlama (`tq3_0`):

- PP default (`c=512,n=1`): `1829.44 tok/s`
- PP with `GGML_CUDA_TQ3_PREFER_MMQ_PREFILL=1`: `1352.73 tok/s`
- TG-like default (`c=128,n=32`): prompt `13.29 tok/s`, eval `15.35 tok/s`
- TG-like with `GGML_CUDA_TQ3_PREFER_MMQ_PREFILL=1`: prompt `13.35 tok/s`, eval `15.29 tok/s`

Result:

- prefill got worse by ~`26%`
- decode is effectively unchanged (as intended)

Decision:

- keep the MMQ-prefill switch as an experimental knob for further kernel work
- do not enable it as the default path
- continue phase-two work by improving the native TQ3 MMQ kernel cost model before reconsidering this route

## Execution Description

This is the concrete SDLC path from current state to a shippable decision.

### Phase A: Kernel Cost Reduction (In Progress)

Scope:

- optimize TQ3 MMQ bridge cost where it currently loses: decode, WHT, and requant staging
- keep behavior gated and default policy unchanged

Done in this phase:

- native block-dot contract + guardrail tests
- MMQ prefill entry switch (`GGML_CUDA_TQ3_PREFER_MMQ_PREFILL=1`)
- A/B benchmark proving current MMQ prefill route regresses throughput

Exit criteria:

- no functional regressions (`test-tq3-cuda.cu`, `test-tq3-load-tiles.cu`, TinyLlama sanity prompt)
- MMQ-prefill path reaches at least parity with current cuBLAS-backed TQ3 prefill on benchmark prompt

### Phase B: Real Native Prefill Path

Scope:

- move from experimental MMQ switch to a real native prefill kernel path that replaces meaningful work
- keep decode path stable unless explicitly targeted

Implementation constraints:

- introduce changes behind a narrow env flag first
- benchmark on fixed prompts before any default-policy change

Exit criteria:

- measurable PP gain over current cuBLAS-backed TQ3 baseline
- no quality regressions on deterministic sanity prompts

### Phase C: Default Policy Decision

Scope:

- choose between keeping cuBLAS as default or promoting native prefill path

Decision rule:

- promote only if native path is consistently faster on PP and does not regress TG or output quality
- otherwise keep native path experimental and continue targeted kernel work

## Next Concrete Tasks

1. Reduce MMQ bridge overhead in `load_tiles_tq3_0` (decode + WHT + requant hot section).
2. Re-run A/B with `GGML_CUDA_TQ3_PREFER_MMQ_PREFILL=1` on fixed benchmark prompts.
3. If PP remains below cuBLAS-backed baseline, stop tuning this branch and start a dedicated native prefill kernel prototype.
4. Document each pass with before/after tok/s and keep commits small and reversible.

## Dispatch Blocker (AskExpert v5)

Date: 2026-03-27

Latest repro on `experiment/tq3-weight-quant`:

- `llama-bench` with `type_k=tq3_0` stalls in generation and times out.
- Repro command:
  - `timeout 90s ./build/bin/llama-bench -m /home/awee/models/tinyllama-1.1b-q4_0.gguf -ngl 999 -fa 0 -ctk tq3_0 -ctv f16 -p 512 -n 1 -r 1 --no-warmup --progress`
- Observed behavior:
  - prompt pass (`pp512`) prints
  - generation pass does not complete before timeout (`EXIT:124`)

Important finding:

- This timeout reproduces even when `ggml_cuda_should_use_mmq()` forces `TQ3_0` to return `false` on NVIDIA.
- So the TG stall is not explained only by the MMQ dispatch gate.

### Root Cause

- The failing decode op was `kq = mul_mat(cache_k_l0(view, permuted), Qcur(view, permuted))`.
- `ggml_cuda_mul_mat()` still allowed `GGML_TYPE_TQ3_0` to take `mul_mat_vec_q` in the main dispatch path.
- That sent non-contiguous TQ3 KV-cache K tensors into the MMVQ vec-dot path during TG, even though TQ3 MMVQ is known-bad and had only been excluded in the fusion helper.
- Result: PP completed, but TG entered the wrong CUDA backend and timed out.

### Fix

- Exclude `GGML_TYPE_TQ3_0` from `use_mul_mat_vec_q` in the main `ggml_cuda_mul_mat()` dispatch.
- This forces the TQ3 KV decode KQ path onto the cuBLAS fallback instead of MMVQ.

### Verification

- Repro command now completes:
  - `./build/bin/llama-bench -m /home/awee/models/tinyllama-1.1b-q4_0.gguf -ngl 999 -fa 0 -ctk tq3_0 -ctv f16 -p 512 -n 1 -r 1 --no-warmup --progress`
- Observed result after fix:
  - `pp512`: about `3146.56 tok/s`
  - `tg1`: about `27.10 tok/s`
- Minimal decode guardrail also passes:
  - `p=1, n=1, type_k=tq3_0` completes at about `111.87 tok/s`

### Immediate Task List

- [x] Add temporary TG-path instrumentation around TQ3 KV cache matmul selection in `ggml-cuda.cu` (backend branch decisions and tensor shapes).
- [x] Capture one short TG trace for `q4_0` weights + `tq3_0` KV and identify exactly which CUDA branch does not return.
- [x] Build a minimal regression test command (bounded by timeout) and keep it in this doc as a guardrail.
- [x] Keep MMQ/MMVQ disabled by default for TQ3 while this blocker is open.
- [ ] Resume MMQ threshold tuning only after the TG stall root cause is isolated.

## Success Criteria

Short term:

- beat the current cuBLAS-backed TQ3 baseline on the chosen regime

Medium term:

- materially close the gap to q4 while keeping the TQ3 VRAM advantage

## Non-Goals For This Stage

- CPU SIMD work
- KV-cache-only optimizations
- broad cleanup or refactor of unrelated CUDA code

This stage is specifically about proving whether native TQ3 weight execution on NVIDIA is worth pursuing.
