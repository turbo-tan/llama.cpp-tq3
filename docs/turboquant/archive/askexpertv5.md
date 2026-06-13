# TQ3_0 CUDA — Expert Help v5: MMQ Dispatch + TG Hang

## Current State

Branch: `experiment/tq3-weight-quant` on `charpdev/llama.cpp`, commit `d41a70fb4`

### Canonical Benchmark (TinyLlama 1.1B, RTX 5060 Ti)

| Model | PP 512 tok/s | TG 32 tok/s | VRAM |
|-------|------------|------------|------|
| q4_0 | 15,092 | 394 | 607 MiB |
| tq3_0 (current) | 1,146 | 35 | 491 MiB |
| ratio | 7.6% | 8.9% | 81% |

### KV Cache Benchmark (Llama-2-7B, q4_0 weights + varying KV type)

| ctx | q4_0 KV | tq3_0 KV (cublas) | tq3_0 KV (MMQ) |
|-----|---------|-----------------|----------------|
| 512 | 3,920 | 2,646 | **3,320** |
| 2048 | 3,160 | 1,648 | **2,364** |
| 8192 | 1,997 | 688 | **1,170** |

MMQ for KV cache is **much better** than cublas (85% of q4_0 at 512 ctx).

## The Problem

When we re-enable MMQ for TQ3_0 (by removing the blanket `return false` in `should_use_mmq`), **TG hangs**. The model loads but never generates tokens.

### Root Cause Analysis

With MMQ re-enabled:
- PP (ne11 > 8): `use_mul_mat_vec_q=false` → `use_mul_mat_q=true` → MMQ ✅ fast
- TG (ne11 = 1): `use_mul_mat_vec_q=false` (MMVQ excluded, broken vec_dot) → `use_mul_mat_q=true` → MMQ for batch=1

MMQ for batch=1 is either hanging or extremely slow (MMQ is designed for large batches, not single-token generation).

### What We Need

A dispatch rule that:
1. Uses MMQ for PP (ne11 > some threshold) — gives 85% of q4_0 for KV cache
2. Uses cublas for TG (ne11 = 1) — correct and ~35 tok/s
3. Does NOT use MMVQ (vec_dot is broken, excluded)

### Attempted Fix (incomplete, interrupted)

```c
// In should_use_mmq for TQ3_0:
if (type == GGML_TYPE_TQ3_0 && ne11 <= 1) {
    return false;  // cublas for TG
}
// else: MMQ for PP
```

But this was interrupted before testing. The question is: what's the right threshold? And does MMQ for TQ3_0 work correctly for PP (not just KV cache, but also weight matmul)?

## Specific Questions

1. **MMQ dispatch threshold**: What `ne11` threshold should we use to switch between MMQ (PP) and cublas (TG) for TQ3_0? The expert's existing gate uses `ne11 >= 64` for the prefill override. Should we use the same?

2. **Weight matmul vs KV cache**: We proved MMQ works for KV cache attention (non-contiguous K tensor). Does it also work correctly for weight matmul (contiguous weight tensor)? Earlier tests showed weight MMQ produced garbage — is that still the case with the current exact q8_0 requant in load_tiles?

3. **MMA prototype results**: We tried 3 fused decode→MMA approaches. All were slower than the scalar tiled kernel (0.229 ms). The decode cost (0.144 ms) dominates. The expert's recommendation was "if gain < 20%, pivot to long-context positioning." We're at 0% gain from MMA. Should we stop MMA work and focus on:
   - Getting MMQ working for both PP and TG (dispatch fix)
   - Publishing the long-context KV cache advantage

## Current Dirty State

Uncommitted changes in:
- `ggml/src/ggml-cuda/mmq.cu` — partial MMQ re-enable (broken, TG hangs)
- `ggml/src/ggml-cuda/mmq.cuh` — WHT restored in load_tiles (correct)
- `ggml/src/ggml-cuda/ggml-cuda.cu` — rotation removed, MMVQ excluded

## Recommended Next Step

Fix the dispatch: `should_use_mmq` returns true for TQ3_0 when `ne11 >= 2` (PP), false when `ne11 = 1` (TG → cublas). Then verify:
1. TG correctness ("Paris")
2. PP speed (target: 3,320 tok/s for KV cache at 512 ctx)
3. Weight matmul correctness (tq3_0 weights + tq3_0 KV)

## Files

- `ggml/src/ggml-cuda/mmq.cu:270` — `should_use_mmq` for TQ3_0
- `ggml/src/ggml-cuda/ggml-cuda.cu:2361` — `use_mul_mat_vec_q` exclusion
- `ggml/src/ggml-cuda/mmq.cuh:3218` — `load_tiles_tq3_0` (exact q8_0 requant + WHT)
- `tests/test-tq3-load-tiles.cu` — load_tiles unit test (PASS)
- `tests/test-tq3-cuda.cu` — 10 unit tests (all PASS)
