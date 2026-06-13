# TQ3_4S PP Speed: Fused Kernel Design

## Current State (2026-03-31)

| Path | PP tok/s | Notes |
|------|----------|-------|
| fp16 cuBLAS (current) | 315 | 78.8% time in dequant kernel |
| int8 MMQ (broken) | 105 | Too many warp shuffles |
| Q3_K_S MMQ | 689 | Target |

## Root Cause Analysis

nsys profile shows:
- `dequantize_block_tq3_4s<__half>`: 78.8% of GPU time
- CUTLASS h16816 GEMM: 9.4% of GPU time

The dequant kernel is the bottleneck. The GEMM itself is fast.

The existing int8 MMQ kernel (`load_tiles_tq3_4s`) is slower than cuBLAS because
it uses 24 warp shuffles per block (6 per group × 4 groups) to pack int8 values.

## Design: fp16 MMQ Tile Loader

### Why fp16 instead of int8

The RTX 5060 Ti (sm_120) has fp16 tensor cores (h16816: 16×8×16 MMA).
The existing MMA path in mmq.cuh already uses fp16 for the activation (y) tiles.
We can write fp16 to the weight (x) tiles too, using fp16 MMA.

This eliminates the int8 packing step: instead of 4 shuffles to pack 4 int8→int32,
each lane writes its own fp16 value directly. Only 2 shuffles needed per group.

### load_tiles_tq3_4s_fp16 (new function)

```
Per group (8 lanes):
  Lane 0: read E3M5 scale → bit-manipulation decode → broadcast (1 shfl)
  Lane 0: read 3 bytes packed bits → broadcast (1 shfl)
  Each lane: extract 3-bit idx, compute centroids[idx] * scale → __half
  Each lane: write __half to x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + blk*QI8_0 + lane]
  Lane 0: write 1.0f to x_df (scale already baked into x_qs values)

Total: 2 shuffles per group × 4 groups = 8 shuffles (vs 24 in int8 version)
```

### vec_dot_tq3_4s_fp16_mma (new function)

Reads fp16 from x_qs, uses `wmma::fragment<matrix_a, 16, 16, 16, __half>`.
Standard h16816 MMA accumulates to fp32.

The existing `vec_dot_q8_0_q8_1_mma` reads int8 packed as int32.
We need a variant that reads fp16 directly.

### mmq_type_traits update

```cpp
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_TQ3_4S> {
    static constexpr int              vdr          = VDR_Q8_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_tq3_4s_fp16<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_tq3_4s_fp16_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>; // fallback
};
```

### Expected Performance

- 3x fewer shuffles → tile loading ~3x faster
- fp16 MMA same speed as current CUTLASS h16816
- Expected PP: 400-600 tok/s (vs 315 cuBLAS, 689 Q3_K_S target)

## Updated Analysis (2026-03-31)

### Why MMQ Can't Beat fp16 cuBLAS for TQ3_4S

After studying Q4_0_TQ (same 3-bit packed format), the shuffle pattern is inherent:
- 6 shuffles per group × 4 groups = 24 shuffles per block
- This is the same for ALL 3-bit packed formats (TQ3_4S, Q4_0_TQ)
- The fp16 cuBLAS path (315 tok/s) already beats MMQ (105 tok/s)
- No MMQ optimization can overcome this — the shuffles are unavoidable for 3-bit packing

### The Only Real Path: Marlin-Style Fused Kernel

Eliminate the separate dequant kernel entirely. Each GEMM thread block:
1. Loads its tile of TQ3_4S blocks (16 bytes each) from global memory
2. Dequants in registers (no global memory write)
3. Feeds fp16 values directly to tensor core MMA
4. Accumulates in fp32 registers

This eliminates the 78.8% dequant overhead. Expected: 3-4x PP speedup.

**Complexity: HIGH** — requires writing a custom CUDA GEMM kernel from scratch.
This is expert-level work (charpdev territory). Not a quick fix.

### What We Can Do Now

The fp16 cuBLAS path (315 tok/s) is the best achievable without a custom kernel.
Focus: document the design clearly for the expert, then move on.



1. Add `load_tiles_tq3_4s_fp16` to mmq.cuh
   - Same structure as `load_tiles_tq3_4s` but writes fp16 to x_qs
   - Use bit-manipulation E3M5 decode (not ldexpf)
   - x_df = 1.0f (scale baked into x_qs)

2. Add `vec_dot_tq3_4s_fp16_mma` to mmq.cuh
   - Read fp16 from x_qs using `__half2` loads
   - Use `wmma::mma_sync` with fp16 A fragment
   - Accumulate to fp32

3. Update `mmq_type_traits<TQ3_4S>` to use new functions

4. Enable MMQ for TQ3_4S in ggml-cuda.cu (remove from tq3_1s_mmq_ok exclusion)

5. Test: bench → PPL → server

## Files to Modify

- `ggml/src/ggml-cuda/mmq.cuh`: new tile loader + vec_dot + type traits
- `ggml/src/ggml-cuda/ggml-cuda.cu`: enable MMQ for TQ3_4S

## Risk

Medium. The MMA infrastructure exists. The main risk is getting the
tile layout right for the fp16 MMA instruction. Need to verify:
- x_qs stride matches what vec_dot expects
- fp16 values are in the right order for the MMA fragment
- No bank conflicts in shared memory access

Test after each step. Never commit without running bench + PPL.
