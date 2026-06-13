# TQ3 vs Q3_K_S: Investigation Results (2026-03-31)

## The Gap

| Format | bpw | PPL | Size |
|--------|-----|-----|------|
| Q3_K_S | 3.44 | 6.96 | 11.4 GB |
| TQ3_4S | 4.00 | 7.05 | 12.9 GB |

Q3_K_S uses 16% fewer bits AND gets 0.09 better PPL.

## What We Tested

### 1. imatrix weighting — NO EFFECT
PPL 7.049 with imatrix vs 7.047 without.
WHT makes all coefficients equally important: H[k,j]^2 = 1/32 for all k,j.
Block-level importance weighting is theoretically possible but needs
variable block sizes (CUDA alignment nightmare).

### 2. 64-element blocks — MARGINAL (1% RMSE)
Larger WHT captures slightly more correlation but not enough to matter.
Not worth the implementation complexity.

### 3. Variable-rate bit allocation per GROUP — 10% RMSE
Trying all 35 valid {0,1,2,3,4}-bit patterns for 4 groups gives 10% RMSE gain.
But WHT equalizes group energies, so zeroing any group is catastrophic (2.5x worse).
The gain comes from [4,3,3,2] and [4,4,2,2] patterns.
Problem: encoding the pattern costs bits, and variable packing kills CUDA.

### 4. Variable-rate per COEFFICIENT — 40% RMSE
Zero 4 smallest, promote 12 to 4-bit, keep 16 at 3-bit = 40% RMSE gain.
But the mask (which coefficients get which rate) costs ~30 bits.
Portfolio approach fails: patterns are nearly unique (0.04% coverage with top 16).

### 5. Centroid optimization — ALREADY DONE (3% left)
Code centroids are near-optimal for the WHT coefficient distribution.
Only 3% MSE improvement possible from better centroids.

## Why Q3_K_S Wins: The Fundamental Issue

Q3_K_S has two advantages TQ3 cannot replicate within the WHT framework:

1. **Importance weighting**: Q3_K_S uses imatrix to give important weights
   more precision. WHT destroys importance structure — all coefficients
   become equally important. This is inherent to any orthogonal transform.

2. **Asymmetric quantization**: Q3_K_S stores per-sub-block minimums,
   allowing the quantization range to shift. TQ3's centroids are fixed
   relative to zero. The shifts in TQ3_4SE help slightly (PPL 7.02 vs 7.05)
   but cost 0.5 bpw.

## Remaining Paths

### A. Block-level MILP allocation (promising, hard)
Mix TQ3 (3-bit) and TQ4 (4-bit) blocks within a tensor.
Use imatrix to decide which blocks get promoted.
At 3.5 bpw average: half blocks at 3-bit, half at 4-bit.
**Problem**: variable block sizes break CUDA alignment.
**Solution**: super-blocks with fixed total size and internal mask.

### B. Hybrid: WHT decorrelation + direct-domain quantization
Use WHT to identify the energy distribution, then quantize in
the original domain with importance-aware precision.
Essentially: use WHT as an analysis tool, not as the quantization basis.

### C. Abandon WHT for weight quantization
Keep WHT for KV cache (where it works well at 3.5 bpw).
For weights, use direct-domain quantization with TQ3's block structure
but K-quant-style importance weighting.

### D. Accept the quality ceiling, pivot to speed
TQ3_4S at 4.0 bpw / 12.9 GB is a valid format:
- Beats Q4_0 on size (12.9 vs 14.4 GB) at equal quality
- 1.5 GB smaller than IQ4_XS at only 0.24 PPL worse
- Fits 16 GB GPU with 8k context
- The WHT approach works, it just can't beat K-quants at same bpw
- **CHOSEN PATH**: focus on speed to justify the format

### Failed experiments (2026-03-31)
| Experiment | RMSE gain | PPL result | Why it failed |
|---|---|---|---|
| imatrix weighting | 0% | 7.049 (no change) | WHT equalizes importance |
| 64-element blocks | 1% | Not tested | Too small to matter |
| Variable-rate [4,3,3,2] | 6.9% | 7.24 (worse!) | 2-bit too coarse for real weights |
| Centroid optimization | 3% | Not tested | Already near-optimal |

### Speed optimization plan

**Task 1: Custom fused dequant+GEMM kernel (closes 2.1x PP gap)**

The dequant→fp16→cuBLAS path reads 12.9 GB quantized + writes 12.9 GB fp16 = 25.8 GB
bandwidth. Q3_K_S MMQ reads 11.4 GB with no intermediate write. Ratio ≈ 2.3x = the gap.

The existing MMQ kernel (load_tiles_tq3_4s in mmq.cuh) converts to q8_0 in shared memory
but is SLOWER (105 tok/s) because WHT inverse in the tile loader is too expensive.

Need a fused kernel that:
1. Loads TQ3_4S blocks directly into registers
2. Decodes E3M5 scales + unpacks 3-bit indices
3. Applies WHT inverse butterfly via warp shuffles
4. Feeds fp16 values directly to tensor core MMA instructions
5. Never writes intermediate buffer to global memory

This is NOT standard MMQ — it's a custom tensor core kernel.

**Task 2: Improve TG decode speed (closes 1.4x TG gap)**

Current MMVQ vec_dot_tq3_4s_q8_1 in vecdotq.cuh works but is unoptimized.
TG is memory-bandwidth bound (single token → single row of weight matrix).
TQ3_4S reads 12.9 GB vs Q3_K_S 11.4 GB = 1.13x more data.
Actual gap is 1.4x → 0.27x overhead from WHT decode.

Optimizations:
- Preload 4 E3M5 scales into registers (avoid repeated global reads)
- Use __ldg() for qs[] reads (read-only cache)
- Reduce WHT shuffle stages (combine with centroid multiply)
- Shared memory for centroid table (8 floats, fits in 32 bytes)

### CUDA dispatch flags (ggml-cuda.cu)

These flags control which code path TQ3_4S takes. Document here so we
don't lose track of what each flag does and where TQ3_4S is listed.

**Line ~1385: use_fp16 exclusion**
```
&& src0->type != GGML_TYPE_TQ3_4SE && src0->type != GGML_TYPE_TQ3_4SV
```
TQ3_4S is NOT excluded → uses fp16 dequant + tensor core cuBLAS. ✅ FAST (327 tok/s)
TQ3_4SE/TQ3_4SV ARE excluded → fall through to fp32 fallback.

**Line ~1388: fp32 dequant fallback**
```
if (src0->type == GGML_TYPE_TQ3_0 || src0->type == GGML_TYPE_TQ3_1S || src0->type == GGML_TYPE_TQ3_1S_AP1)
```
TQ3_4S is NOT in this list → does not use fp32 fallback. ✅

**Line ~2320, ~2371: tq3_vec_prefill_ok**
TQ3_4S IS excluded → MMVQ only used for single-token (TG), not prefill. ✅

**Line ~2385: tq3_1s_mmq_ok**
```
&& src0->type != GGML_TYPE_TQ3_4S && src0->type != GGML_TYPE_TQ3_4SE && src0->type != GGML_TYPE_TQ3_4SV
```
TQ3_4S IS excluded → MMQ disabled. ✅ (MMQ is slower at 105 tok/s)

**Line ~258 mmq.cuh: mmq_get_mma_tile_x_k**
TQ3_4S returns MMQ_MMA_TILE_X_K_Q8_0. Fixed (was missing → crash).

### Current speed (27B, RTX 5060 Ti 16GB)

| | Q3_K_S | TQ3_4S | Gap | Theoretical limit |
|---|---|---|---|---|
| PP 512 | 689 | 327 | 2.1x | 1.13x (bandwidth ratio) |
| TG 10 | 20.7 | 14.6 | 1.4x | 1.13x (bandwidth ratio) |

Both gaps are ~2x the theoretical bandwidth limit, meaning ~50% of time
is compute overhead (WHT decode, centroid lookup, scale decode).
