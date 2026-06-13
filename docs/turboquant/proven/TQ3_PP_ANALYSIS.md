# TQ3_4S Prompt Processing (PP) Speed Analysis

**Date**: 2026-04-10
**GPU**: RTX 5060 Ti 16GB (SM120 Blackwell)
**Model**: Qwen3.5-27B TQ3_4S
**Baseline**: PP 360 tok/s (TQ3_4S) vs PP ~950 tok/s (Q3_K_S) — 2.6x gap
**Current status**: tile-loader rewrite remains the core win; exact reruns now place
default-KV TQ3_4S in the high-600s, with the first revalidated `700+` witness coming
from asymmetric KV (`K=q4_0, V=tq3_0`) — see [TQ3_PP_BREAKTHROUGH.md](TQ3_PP_BREAKTHROUGH.md)

## 1. Dispatch Path Discovery

TQ3_4S PP uses **MMQ** (not cuBLAS as initially suspected).

### Evidence

`nsys` profiling of PP512 on Qwen3.5-27B TQ3_4S:

| Kernel | Time % | Time (ms) | Instances |
|---|---|---|---|
| `mul_mat_q<TQ3_4S, 128>` | **94.6%** | 3690 | 397 |
| `tq3_rotate_act_kernel` | 1.4% | 55 | 495 |
| `quantize_mmq_q8_1` | 0.2% | 8 | 493 |
| cutlass HGEMM | 0.03% | 1.2 | 16 |

The dispatch path depends on batch size:
- `ne11 >= 64`: `ggml_cuda_should_use_mmq()` returns true → MMQ via `ggml_cuda_op_mul_mat`
- `ne11 < 64`: falls through to cuBLAS (all optimized path flags are false)
- `ne11 == 1`: MMVQ (token generation)

The `tq3_1s_mmq_ok` flag blocks TQ3_4S from the **direct** MMQ path
(`ggml_cuda_mul_mat_q`), but the **split/multi-GPU** MMQ path
(`ggml_cuda_op_mul_mat` with `ggml_cuda_op_mul_mat_q` callback) is used instead.

### Key finding: rotation is NOT the bottleneck

The WHT rotation (`tq3_rotate_act_kernel`) is only 1.4% of PP time.
The quantize step is 0.2%. Together they're <2%.

**The MMQ kernel itself is 94.6% of PP time.**

### What Q3_K_S does (for comparison)

Q3_K_S passes all dispatch checks → uses `ggml_cuda_mul_mat_q()` (direct MMQ):

1. **Quantize activations** f32 → q8_1 (one kernel)
2. **MMQ kernel** reads Q3_K_S tiles directly via `load_tiles_q3_K` — no dequant

## 2. Root Cause: 8x Block Count Mismatch

The MMQ tile loader iterates **per block**. TQ3_4S has 8x more blocks than Q3_K_S
for the same number of elements:

| Type | Elements/block | Bytes/block | Blocks per 5120-element row |
|---|---|---|---|
| Q3_K_S | 256 | 110 | 20 |
| TQ3_4S | 32 | 16 | 160 |

The 32-element block size is dictated by the WHT group size — it cannot be changed
without changing the quantization algorithm.

The tile loader starves the MMA pipeline: it takes 8x longer to load tiles for
TQ3_4S than Q3_K_S, while the MMA accumulation is the same speed for both.

### Measured: TQ3_4S MMQ is 8.2x slower than Q3_K_S MMQ

| Model | MMQ kernel time (PP512) | PP tok/s |
|---|---|---|
| TQ3_4S | 3690 ms | 360 |
| Q3_K_S | 451 ms | ~950 |

Ratio: 3690/451 = 8.2x ≈ 8x block count ratio.

### Tile loader optimization attempts (Path D)

Two tile loader rewrites were tested:
1. **Shared memory pack** (replace 4 gather shuffles with smem write+read): neutral (361 tok/s)
2. **8-lane direct pack** (each lane packs 4 elements, zero shuffles): slightly worse (353 tok/s)

Neither helped because the bottleneck is the **iteration count** (8x more blocks),
not the per-iteration cost. The tile loader code is already efficient enough —
it just runs 8x more often.

## 3. Why MMQ Was Disabled for TQ3_4S

When MMQ is enabled for TQ3_4S (removing the `tq3_1s_mmq_ok` block):
- **PP drops to 125 tok/s** (measured)
- 3x slower than cuBLAS (360 tok/s)

The MMQ path for TQ3_4S requires:
1. `cudaMemcpyAsync` src1 → temp buffer (bandwidth)
2. `ggml_cuda_tq3_rotate_act` WHT rotation kernel (compute + bandwidth)
3. `quantize_mmq_q8_1_cuda` quantize rotated activations (compute + bandwidth)
4. MMQ kernel with `load_tiles_tq3_4s` tile loader

Steps 1-3 add ~3 kernel launches + temp buffer bandwidth per matmul.
The tile loader also does per-subgroup work (unpack 3-bit, lookup centroids, shuffle).

The rotation overhead makes MMQ uncompetitive with cuBLAS for TQ3_4S.

## 4. Why Fused Rotate+Quantize Didn't Help

We tried fusing steps 1-3 into a single kernel using warp shuffles for the WHT butterfly.
Result: PP 348 tok/s (slightly worse than 360 baseline).

This was applied to the **MMVQ path** (which handles TG, not PP).
PP goes through cuBLAS, so the fused kernel was never called for PP.

Even in the MMVQ path, the fused kernel added 5 butterfly shuffle steps per element
to the quantize kernel, which offset the saved kernel launch overhead.

## 5. Viable Optimization Paths

### Path D: Optimize tile loader — TESTED, NO GAIN

Two rewrites tested. Neither helped because the bottleneck is 8x iteration count,
not per-iteration cost. **Dead end.**

### Path E: Process multiple TQ3_4S blocks per tile-loader iteration

The tile loader currently processes 1 block (32 elements) per warp per iteration.
If it processed 8 blocks (256 elements) per iteration — matching Q3_K_S's granularity —
the iteration count would match and the pipeline stall would disappear.

This requires restructuring the tile loader to:
1. Read 8 consecutive TQ3_4S blocks (128 bytes) per warp
2. Decode all 256 elements in parallel across the 32 lanes (8 elements per lane)
3. Pack into the same q8_0-like tile format

Each lane would handle 8 elements across 8 blocks. The scale decode and centroid
lookup would be done 8x per lane instead of 1x. This is more work per lane but
eliminates the 8x iteration overhead.

**Estimated gain: up to 8x on tile loading → potentially 2-3x on total PP.**
The MMA accumulation is unchanged, so the gain depends on how much tile loading
currently dominates vs MMA.

### Path F: Accept the gap, optimize elsewhere

The 32-element block size is fundamental to TQ3_4S. Without changing the format,
the MMQ tile loader will always iterate 8x more than Q3_K_S.

**Redirect effort to TG or other features.**

## 6. Recommended Next Steps

1. **Profile the cuBLAS path** with `nsys`/`ncu` to measure exact time split:
   dequant kernel vs fp16 convert vs HGEMM vs overhead.
   This tells us the theoretical maximum improvement from optimizing dequant.

2. **Profile the MMQ path** (when enabled) to understand why it's 3x slower:
   is it the rotation, the tile loader, or the MMA accumulation?

3. **Try Path D** (optimize dequant kernel) — lowest risk, modest gain.

4. **Investigate Path E** (custom GEMM) — highest potential but highest effort.

## 7. Failed Approaches (Do Not Repeat)

| Approach | Result | Why |
|---|---|---|
| Fused rotate+quantize kernel (warp shuffles) | -3% (348 vs 360) | Shuffle overhead ≈ saved launch overhead. Also: rotation is only 1.4% of PP time |
| Rotation cache (reuse q8_1 across matmuls) | N/A | Rotation is only 1.4% of PP time — caching it saves almost nothing |
| Tile loader: smem pack (replace 4 shuffles) | neutral (361) | Bottleneck is iteration count (8x blocks), not per-iteration cost |
| Tile loader: 8-lane direct pack | -2% (353) | 24 idle lanes reduce parallelism |
| Enable direct MMQ path for TQ3_4S | -65% (125 vs 360) | Tested by removing tq3_1s_mmq_ok block — same MMQ kernel, different dispatch |
| All 25 moonshot attempts | See archive/superseded/TQ3_MOONSHOT_MASTER_LOG.md | Various approaches, none successful |

## 8. Key Architectural Facts

```
TQ3_4S PP dispatch chain (ne11 >= 64):
  ggml_cuda_mul_mat()
    → tq3_1s_mmq_ok = false (blocks direct MMQ path)
    → use_mul_mat_q = false
    → falls through to: ggml_cuda_op_mul_mat(ctx, ..., ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda)
      → ggml_cuda_should_use_mmq() returns true for TQ3_4S when ne11 >= 64
      → rotate src1 (memcpy + WHT kernel) — 1.4% of time
      → quantize src1 → q8_1_mmq — 0.2% of time
      → mul_mat_q<TQ3_4S> MMQ kernel with load_tiles_tq3_4s — 94.6% of time

TQ3_4S small-batch dispatch (ne11 < 64):
  → all optimized paths blocked → cuBLAS fallback
  → dequantize_block_tq3_4s (includes inverse WHT) → fp16
  → cublasGemmEx HGEMM

TQ3_4S TG dispatch (ne11 = 1):
  → use_mul_mat_vec_q_direct = true
  → ggml_cuda_mul_mat_vec_q()
    → rotate + quantize src1
    → mul_mat_vec_q with vec_dot_tq3_4s_q8_1 (dp4a)

Q3_K_S PP dispatch:
  → use_mul_mat_q = true (direct MMQ path)
  → ggml_cuda_mul_mat_q()
    → quantize src1 → q8_1_mmq
    → MMQ kernel with load_tiles_q3_K (256 elements/block, 8x fewer iterations)
```

The fundamental asymmetry: Q3_K_S has 256-element super-blocks that match the MMQ
tile loader's granularity. TQ3_4S has 32-element blocks (WHT group size), requiring
8x more tile-loader iterations per row. This is an inherent cost of the TQ3 format.

## 9. Path E Attempt: Multi-Block Tile Loader

Attempted restructuring the tile loader so each warp handles an entire 64-element
tile row (2 TQ3_4S blocks) instead of 1 block. This would match Q3_K_S's warp
utilization (1 warp per row vs 8 warps per row).

### Result: broke MMQ dispatch

The restructured tile loader compiled and produced correct PPL (9.8972), but
nsys profiling revealed the MMQ kernel was no longer being called — the dispatch
fell through to cuBLAS (dequant + HGEMM).

The MMQ framework's MMA accumulation, stream-K fixup, and tile indexing all
depend on the `blocks_per_tile_x_row` warp-to-block mapping. Changing the tile
loader alone breaks the contract with the rest of the kernel.

A proper fix requires rewriting the entire MMQ kernel specialization for TQ3_4S:
tile loader + MMA accumulation + stream-K fixup. This is a multi-week effort
with high risk of subtle correctness bugs.

### Conclusion

The 2.6x PP gap (360 vs 950 tok/s) is an inherent cost of TQ3_4S's 32-element
block size. The MMQ framework is designed for 256-element super-blocks (Q3_K_S,
Q4_K, etc.) and TQ3_4S's small blocks cause 8x worse warp utilization.

**Recommended: accept the PP gap and focus effort on other tracks** (upstream sync,
Gemma4 support, fine-tuning, community). TQ3_4S's value proposition is quality-per-bit
and TG speed (dp4a), not PP throughput.

## 10. Why MMQ Produces Wrong PPL for TQ3_4S

The MMQ tile loader approximates TQ3_4S's float centroids as int8 values:

| Centroid (exact) | int8 level | level × scale | Relative error |
|---|---|---|---|
| -1.9967 | -127 | -2.1519 | 7.8% |
| -1.2914 | -79 | -1.3386 | 3.7% |
| -0.7403 | -45 | -0.7625 | 3.0% |
| -0.2475 | -14 | -0.2372 | 4.2% |
| 0.2301 | 14 | 0.2372 | 3.1% |
| 0.7252 | 45 | 0.7625 | 5.1% |
| 1.2775 | 79 | 1.3386 | 4.8% |
| 1.9889 | 127 | 2.1519 | 8.2% |

This 3-8% per-element error compounds across 5120-dim matmul → PPL 170 (vs 9.9 correct).

Even with optimized int8 levels (0.41% max error), PPL is still 132 — unusable.

The TG (vec_dot) path works because BOTH weight and activation use the same int8
representation — errors cancel in the dot product. In MMQ PP, activations are
quantized from rotated floats (different quantization), so errors don't cancel.

### Implication for PP optimization

MMQ cannot be used for TQ3_4S PP without either:
1. **Exact float centroids** in the tile loader → requires float MMA (slower than dp4a)
2. **A new quantization format** where centroids are exact powers of 2 (int8-friendly)

The cuBLAS path (dequant → fp16 → HGEMM) remains the only correct PP path.
Its bottleneck is the dequant kernel (67% of time) which includes inverse WHT.
