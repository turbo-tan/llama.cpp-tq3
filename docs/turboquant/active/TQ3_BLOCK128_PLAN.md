# TQ3_0 KV Cache Enhancement Plan — CUDA Port from TurboQuant+

**Date**: 2026-04-19
**Repo**: Private `llama.cpp` (charpdev/t_llama.cpp)
**Source**: https://github.com/TheTom/turboquant_plus/tree/main/docs/papers
**Priority**: Ordered by effort/impact ratio

---

## Enhancement 1: Block Size 32→128 (FREE WIN)

**Paper**: `block-size-experiment.md`
**Effort**: Small (1-2 hours)
**Impact**: 12% smaller KV cache, 0-7% faster decode

### What
Change `QK_TQ3_0` from 32 to 128. All 4 blocks in a WHT rotation group store the same norm — we're wasting 6 bytes per group on duplicates.

### Changes
1. `ggml/src/ggml-common.h`: `#define QK_TQ3_0 128`
2. `ggml/src/ggml-cuda/fattn-common.cuh`: Update `nl` iteration in `vec_dot_fattn_vec_KQ_tq3_0`
3. `ggml/src/ggml-cuda/fattn-vec.cuh`: Update template instantiation if hardcoded
4. Verify all other files use `QK_TQ3_0` symbolically (should just work)

### Validation
- PPL must be identical to 4 decimal places (block32 vs block128)
- Bench decode at 512, 4096, 8192 context
- 10-question server smoke test

### Risk
- FA kernel may have hardcoded `nl=4` (head_dim/32). Now `nl=1`.
- Shared memory sizing in FA kernels may assume 32-element blocks.

---

## Enhancement 2: Turbo4 Resurrection — 4-bit PolarQuant V Cache (BIG WIN)

**Paper**: `turbo4-resurrection.md`
**Effort**: Medium (4-8 hours)
**Impact**: PPL +0.23% vs q8_0 (vs turbo3's +1.06%), matching decode speed

### What
Replace current 3-bit TQ3_0 with 4-bit PolarQuant using 16 optimal centroids. Drop QJL entirely (proven harmful for attention). Simple nibble packing (2 indices per byte).

### Why This Matters
- TQ3_0 (3.5 bpw): PPL +1.06% vs q8_0
- Turbo4 (4.25 bpw): PPL +0.23% vs q8_0 — **4.5x smaller quality gap**
- Beats q4_0 quality at better compression (3.76x vs 3.56x)
- Fixes the head_dim=128 quality gap (Qwen3.5-27B uses hd=256, but future models may use hd=128)

### Changes
1. New block struct `block_turbo4_0`:
   ```c
   typedef struct {
       ggml_half d;           // 2B norm
       uint8_t qs[64];        // 128 × 4-bit nibble-packed indices
   } block_turbo4_0;          // 66 bytes per 128 elements
   ```
2. 16-centroid codebook for N(0, 1/√128):
   ```c
   static const float turbo4_centroids[16] = {
       -0.1739, -0.1172, -0.0895, -0.0688, -0.0513, -0.0356, -0.0210, -0.0069,
        0.0069,  0.0210,  0.0356,  0.0513,  0.0688,  0.0895,  0.1172,  0.1739
   };
   ```
3. Quantize: WHT rotate → find nearest centroid (16 bins) → nibble pack
4. Dequantize: nibble unpack → centroid lookup → norm scale
5. FA kernel: direct nibble extraction (no byte-spanning)
6. Register as new ggml_type at high enum (e.g., 207) or reuse existing turbo4 slot

### Key Insight from Paper
- **No QJL** — the 1-bit error correction makes quality worse (proven by 3 independent groups)
- **Nibble packing** — `(qs[j/2] >> ((j%2)*4)) & 0xF` — simpler than TQ3_0's 3-bit byte-spanning
- **Block size 128** — one block per WHT group, same as Enhancement 1

### Validation
- PPL vs q8_0 baseline (expect +0.2-0.3%)
- PPL vs TQ3_0 (expect significant improvement)
- Decode speed (expect matching or faster than TQ3_0 — nibble unpack is simpler)
- NIAH at 4K, 8K

### Risk
- New ggml_type enum slot needed (use 207+ range)
- FA kernel needs new template instantiation for turbo4 V type
- Need head_dim-dependent centroid tables if supporting both hd=128 and hd=256

---

## Enhancement 3: Sparse V Dequant (MEDIUM WIN, LONG CONTEXT)

**Paper**: `sparse-v-dequant.md`
**Effort**: Medium (3-5 hours)
**Impact**: +22.8% decode at 32K context, zero PPL cost

### What
Skip V dequantization for positions where softmax attention weight < threshold. At long context, 90%+ of weights are negligible — we're dequantizing values that contribute nothing.

### Why
- Decode is memory-bandwidth bound at long context
- Attention weights are computed BEFORE V accumulation in FA
- Skipping negligible V positions reduces memory reads proportionally
- Benefit scales with context length (bigger win at 16K+ than 4K)

### Changes
1. In FA decode kernel, after computing softmax weights:
   - Compare each weight against threshold (e.g., 1e-6)
   - Skip V dequant + accumulation for below-threshold positions
2. Threshold can be env-var controlled for tuning

### Validation
- PPL must be identical (threshold should be conservative enough)
- Bench at 4K, 8K, 16K, 32K — expect increasing benefit
- NIAH (paper shows IMPROVED retrieval — 9/9 vs 7/9 baseline)

### Risk
- Warp divergence from conditional skip may hurt short-context perf
- Need to verify threshold is safe across model families
- Paper validated on Metal — CUDA warp behavior may differ

---

## Enhancement 4: Boundary V / Layer-Aware Compression (SMALL WIN)

**Paper**: `layer-aware-v-compression.md`
**Effort**: Small (already partially implemented via TURBO_LAYER_ADAPTIVE)
**Impact**: Better quality at same compression, or more compression at same quality

### What
Protect first 2 and last 2 layers with q8_0 V while compressing middle layers with turbo2/turbo3 V.

### Status
Already implemented as `TURBO_LAYER_ADAPTIVE` env var in our `llama-kv-cache.cpp`. Needs:
- Validation on Qwen3.5-27B (hybrid arch — paper warns about mis-targeting)
- Testing with turbo4 V (Enhancement 2) for middle layers
- Possibly expose as a proper CLI flag instead of env var

### Risk
- Paper warns hybrid architectures (Qwen3.5 with SSM layers) may mis-target boundaries
- Need to identify which layers actually have KV caches in hybrid models

---

## Enhancement 5: Asymmetric K/V with Turbo4 (FUTURE)

**Paper**: `asymmetric-kv-compression.md`
**Effort**: Medium-Large (needs cross-type FA kernel)
**Impact**: Best possible KV cache quality at aggressive compression

### What
Use q8_0 or q4_0 for K cache, turbo4 for V cache. Paper proves V tolerates aggressive compression while K needs precision (softmax amplifies K errors exponentially, V errors scale linearly).

### Status
We already use `q4_0-K + tq3_0-V` asymmetric. Upgrading V to turbo4 would improve quality. But needs FA kernel that handles different K and V types — currently blocked by template instantiation complexity.

### Depends On
- Enhancement 2 (turbo4 implementation)
- Cross-type FA kernel support

---

## Implementation Order

1. **Block size 128** — one-line change, free compression, do first
2. **Turbo4 4-bit** — biggest quality improvement, medium effort
3. **Sparse V dequant** — biggest speed improvement at long context
4. **Boundary V** — already partially done, validate and expose
5. **Asymmetric turbo4** — future, needs cross-type FA

## Branch Strategy

```
main (public)
  └── experiment/tq3-block128        (Enhancement 1)
        └── experiment/turbo4-4bit   (Enhancement 2, builds on block128)
              └── experiment/sparse-v (Enhancement 3)
```

Each enhancement validated independently before merging up.
