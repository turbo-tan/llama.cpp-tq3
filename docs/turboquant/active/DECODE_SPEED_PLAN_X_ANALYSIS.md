# Plan X Analysis — Decode Speed KV Cache Enhancements

**Date**: 2026-04-19
**Branch**: `experiment/decode-speed-plan-x` in private llama.cpp
**Source**: `plan_x` (5 enhancements from TurboQuant+ papers)

---

## Summary

Two of five plan_x enhancements have been implemented and compile-verified on `experiment/decode-speed-plan-x`. A critical error in plan_x's Enhancement 1 (Block 32→128) was identified: it is **NOT a one-line change** — it requires a new block type, not a `QK_TQ3_0` redefinition.

---

## Enhancement 1: Block Size 32→128 — CORRECTION NEEDED

**plan_x claim**: "Change `QK_TQ3_0` from 32 to 128" — one-line free win.

**Reality**: This breaks the codebase in multiple ways and requires a **new type** instead.

### Why QK_TQ3_0 Cannot Simply Be Changed

1. **MMQ `static_assert(WARP_SIZE == QK_TQ3_0)`** — `ggml/src/ggml-cuda/mmq.cuh` line ~3310. WARP_SIZE is 32 on all NVIDIA GPUs. Changing QK_TQ3_0 to 128 triggers a compile-time assertion failure.

2. **MMQ tile loading** — `mmq_type_traits_tq3_0::nl = QK_TQ3_0 / WARP_SIZE = 128/32 = 4`. But the tile loader iterates `for (int i = 0; i < nl; i++)` and each iteration does a `dequantize_tq3_0` that reads `QK_TQ3_0` elements. So it would read 128×4=512 elements per warp — wrong. The `nl` variable represents the number of sub-blocks per WARP_SIZE iteration, and the inner `dequantize_tq3_0` expects 32 elements.

3. **Shared memory sizing** — MMQ allocates shared memory proportional to `QK_TQ3_0 * nwarps * ...`. Doubling QK would double shared memory usage, potentially exceeding per-SM limits.

4. **FA kernel `vec_dot_fattn_vec_KQ_tq3_0`** — This function iterates over `nl` sub-blocks and reads `QK_TQ3_0/nl` elements per iteration. Changing QK changes the loop count and data layout assumptions.

### Correct Approach: New Type `GGML_TYPE_TQ3_0_B128`

Create a new block type `block_tq3_0_b128` with `QK_TQ3_0_B128 = 128`, alongside the existing `block_tq3_0` with `QK_TQ3_0 = 32`. This:
- Preserves backward compatibility with existing TQ3_0 quantized models
- Allows MMQ to keep its warp-aligned 32-element path for weight matmul
- Only uses the 128-block type for KV cache (FA decode path) where the bandwidth savings matter
- Requires new FA template instantiation but no MMQ changes

### Estimated Real Effort

- **Medium** (4-8 hours), not "Small (1-2 hours)" as plan_x claimed
- New block struct, new ggml_type enum, new quantize/dequantize functions, new FA kernel template, new CLI flags for KV type selection

---

## Enhancement 2: Turbo4 4-bit PolarQuant — NOT YET IMPLEMENTED

**plan_x claim**: Medium effort, biggest quality improvement.

**Status**: Not started. Requires new block type `block_turbo4_0` (66 bytes/128 elements = 4.125 bpw). Would replace TQ3_0 for V cache, offering 4.5x smaller quality gap vs q8_0.

**Notes**:
- Nibble packing `(qs[j/2] >> ((j%2)*4)) & 0xF` is simpler than TQ3_0's 3-bit byte-spanning
- No QJL — proven harmful by 3 independent groups
- Needs head_dim-dependent centroid tables for both hd=128 and hd=256
- Should be built on top of Enhancement 1 (block-128 layout), since Turbo4 also uses 128-element blocks

---

## Enhancement 3: Sparse V Dequant — ✅ IMPLEMENTED

**plan_x claim**: +22.8% decode at 32K context, zero PPL cost.

**Implementation**: Done in `ggml/src/ggml-cuda/fattn-vec.cuh` on branch `experiment/decode-speed-plan-x`.

### Design Decisions (deviations from plan_x)

1. **if-gate instead of continue** — Plan_x says "skip V dequant for below-threshold positions". Initial implementation used `continue` in the `for (k0)` loop. Code review identified this as a CUDA barrier-safety risk (if `__syncthreads()` were ever added to the loop body). Changed to `if (!all_negligible_v) { ... }` wrapping, which is the standard safe CUDA pattern.

2. **Unnormalized vs normalized threshold** — The threshold is compared against `exp(KQ - KQ_max)` (unnormalized softmax weights), not final `softmax(KQ)` probabilities. An unnormalized weight of 1e-6 may normalize to >1e-3 after dividing by the sum. Documented clearly in comments. Recommended practical threshold: 1e-4 to 1e-5 (not 1e-6 as the Metal paper suggests, since Metal used normalized weights).

3. **`__constant__` memory + env var** — Threshold stored in GPU constant memory (one 4-byte copy), uploaded once via `std::once_flag` + `std::call_once`. Env var `GGML_FATTN_SPARSE_V_THRESHOLD` controls it. Default 0.0 = disabled.

4. **Thread-safe initialization** — `std::once_flag` instead of `static bool` (code review catch).

5. **Diagnostic log** — `fprintf(stderr, ...)` on first call to confirm threshold value.

### Files Changed
- `ggml/src/ggml-cuda/fattn-vec.cuh`: Sparse V skip logic in both half2 and float code paths + threshold infrastructure

### Pending Validation
- **Must bench at 4K, 8K, 16K, 32K** to confirm scaling (requires GPU)
- **Must verify PPL identical** at threshold=1e-4 (requires GPU)
- **Warp divergence analysis** — at short context, the skip may cause divergence that hurts perf. Need to measure.

---

## Bonus Enhancement: TQ3 nwarps Whitelist — ✅ IMPLEMENTED

**Not in plan_x** but related to decode speed for TQ3 types on AMD GPUs.

### What
Added `GGML_TYPE_TQ3_0`, `TQ3_1S`, `TQ3_4S` to the RDNA4/RDNA3 nwarps=8 whitelist in `mmvq.cu`. This enables multi-warp MMVQ decode for TQ3 types on AMD RDNA3/RDNA4 GPUs.

### Why
The `calc_nwarps()` function in MMVQ defaults to `nwarps=1` for unknown types. TQ3 types have the same vector-dot structure as IQ4_NL/IQ4_XS (which already use nwarps=8), so they should be in the same whitelist.

- **RDNA4**: All TQ3 types (TQ3_0, TQ3_1S, TQ3_4S) → nwarps=8
- **RDNA3**: TQ3_0, TQ3_4S only → nwarps=8 (TQ3_1S excluded as it's a niche type)

### Files Changed
- `ggml/src/ggml-cuda/mmvq.cu`: Added TQ3 case labels to nwarps=8 branches

### Expected Impact
0-20% decode speed improvement on AMD RDNA3/RDNA4 GPUs for TQ3-quantized models. No impact on NVIDIA.

---

## Enhancement 4: Boundary V / Layer-Aware — PARTIALLY EXISTS

Already implemented as `TURBO_LAYER_ADAPTIVE` env var. Needs validation on Qwen3.5-27B and proper CLI flag exposure. Not touched in this branch.

---

## Enhancement 5: Asymmetric K/V with Turbo4 — FUTURE

Depends on Enhancement 2 (turbo4) and cross-type FA kernel support. Not started.

---

## Revised Implementation Order

plan_x order: 1→2→3→4→5
Corrected order: **3→2→1→4→5**

Rationale:
- Enhancement 3 (sparse V dequant) is **already done** and gives the biggest speed win at long context
- Enhancement 2 (turbo4) gives the biggest quality improvement
- Enhancement 1 (block-128) needs a new type, should be done alongside turbo4 since they share the 128-block layout
- Enhancement 4 (boundary V) is already partially done
- Enhancement 5 (asymmetric) is future work

---

## Build Status

- ✅ Compiles clean on `experiment/decode-speed-plan-x` (no GPU testing yet)
- ⏳ GPU occupied by training — benchmark validation pending
