# TQ3_4S Prompt Processing (PP) Speed Improvement Plan

> **SUPERSEDED** by [TQ3_PP_ANALYSIS.md](TQ3_PP_ANALYSIS.md) — the ideas below were
> based on the incorrect assumption that PP uses MMQ. It actually uses cuBLAS with
> fp16 dequant. See the analysis doc for the correct dispatch path and viable options.

## Current State (2026-04-10)

| Model | Quant | PP 2048 | TG 128 | Size |
|---|---|---|---|---|
| Qwen3.5-27B | TQ3_4S | 360 tok/s | 23.3 tok/s | 12.91 GiB |
| Qwen3.5-27B | Q3_K_S | ~950 tok/s | 26.3 tok/s | 11.44 GiB |

PP gap: **2.6x slower** than Q3_K_S. TG gap: only 13%.

## Root Cause

TQ3_4S prefill has 3 extra steps Q3_K_S doesn't:
1. `cudaMemcpyAsync` — copy activations to temp buffer
2. `turbo_wht_kernel` — WHT rotation on temp buffer
3. `quantize_row_q8_1_cuda` — quantize rotated floats to q8_1

Then the standard MMQ tile loader runs on the q8_1 result.

## Previously Tried (DO NOT REPEAT)

| Approach | Result | Why it failed |
|---|---|---|
| Native scalar prefill kernel | 32x slower | Scalar ops can't compete with MMQ's tensor core path |
| Fused rotation+quantize kernel (attempt 8) | +1.4% TG only | Marginal — the rotation itself is fast, the overhead is kernel launches + temp buffer |
| KV pre-rotation via dense matmul | Overhead cancelled savings | Extra matmul cost ≥ rotation savings |
| Marlin-style fused tensor core kernel | Abandoned | WHT 32-elem blocks don't align with 16x16 MMA tiles |
| All 25 moonshot attempts | See TQ3_MOONSHOT_MASTER_LOG.md | Various reasons |

## New Ideas (NOT YET TRIED)

### Idea 1: Fuse rotation INTO MMQ tile loader (HIGH PRIORITY)

**Different from attempt 8** (which fused rotate+quantize as a separate kernel).

This fuses the rotation into `load_tiles_tq3_4s()` itself — the MMQ tile loader
reads float activations directly, rotates a 32-element group in shared memory
using the WHT butterfly, quantizes to int8 in registers, and stores into the
MMQ tile. Zero extra kernels, zero temp buffers.

- Eliminates: memcpy kernel, WHT kernel, quantize kernel (3 launches → 0)
- Eliminates: temp float buffer allocation + bandwidth
- Risk: tile loader becomes more complex, may hurt occupancy
- Estimated gain: +30-50% PP (based on 3 kernel launches being ~30% of PP time)

### Idea 2: dp4a in MMQ accumulation (MEDIUM PRIORITY)

Same technique as the successful MMVQ dp4a vec_dot (+43% TG), but applied to
the MMQ prefill path. Replace float centroid multiplies in the MMQ dot product
with int8 dp4a ops.

- The MMQ path already uses dp4a for Q3_K_S — we just need TQ3_4S-specific
  int8 centroid packing in the tile loader
- Risk: low — proven technique from MMVQ
- Estimated gain: +10-20% PP (compute reduction in inner loop)

### Idea 3: MMA (tensor cores) for WHT rotation (LOW PRIORITY)

The WHT is mathematically a matrix multiply by a Hadamard matrix H_32.
On SM120 Blackwell, MMA can do 16x16 matrix multiplies in hardware.

- Split H_32 into two 16x16 blocks and use MMA
- Or use H_128/H_256 directly if head_dim matches MMA tile size
- Risk: high — Hadamard matrix is dense ±1, MMA expects specific formats
- Previously noted as problematic (marlin_kernel_postmortem.md)
- Estimated gain: unknown, likely small since WHT is already O(n log n)

### Idea 4: Persistent kernel / graph capture (LOW PRIORITY)

Use CUDA graphs to capture the entire rotate+quantize+MMQ sequence per layer,
eliminating per-kernel launch overhead (~5-10μs per launch × 3 kernels × 48 layers
= ~720μs per token, which at 360 tok/s is ~26% of PP time).

- Risk: CUDA graphs are fragile with dynamic shapes
- llama.cpp already has graph support (`USE_GRAPHS=1`) — check if TQ3 path is captured
- Estimated gain: +10-25% PP if launches are the bottleneck

## Priority Order

1. **Idea 1** — fuse rotation into tile loader (biggest gain, eliminates root cause)
2. **Idea 2** — dp4a in MMQ (proven technique, moderate gain)
3. **Idea 4** — CUDA graph capture (low effort to check)
4. **Idea 3** — MMA for WHT (high risk, likely low reward)

## Success Criteria

- PP ≥ 600 tok/s on 27B (closing gap to within 1.6x of Q3_K_S)
- PPL unchanged (bit-identical)
- TG unchanged or improved
