# TQ3_0 CPU SIMD Optimization Plan

## Latest Measured Reality

Fresh CPU-only TinyLlama measurements show the main issue is not a recent TQ3 regression.

Matched decode benchmark (`-c 128 -n 32`, prompt: `The capital of France is`):

| Branch | q4_0 TG | tq3_0 TG |
|---|---:|---:|
| `c10e0380b` (`feature/turboquant-aaryan`) | `17.93 tok/s` | `4.37 tok/s` |
| `8364b7c7f` (current) | `33.87 tok/s` | `4.60 tok/s` |

Interpretation:

- `tq3_0` CPU is roughly flat: `4.37 -> 4.60 tok/s`
- `q4_0` CPU improved a lot upstream: `17.93 -> 33.87 tok/s`
- the visible gap widened mostly because q4 got faster, not because TQ3 got materially worse

That means the CPU project should be framed as "build the first real TQ3 SIMD path", not "recover a lost fast path".

## Goal

Make TQ3_0 weight inference competitive with Q4_0 on CPU via AVX2/NEON-optimized vec_dot.

Current: TQ3_0 CPU is 11x slower than Q4_0 CPU (0.74 vs 8.58 tok/s on Llama-2-7B).
Target: within 2x of Q4_0 CPU speed, with 22% less memory bandwidth.

## Why It's Slow Now

The current CPU `vec_dot_tq3_0_q8_0` calls `dequantize_row_tq3_0` which:
1. Unpacks 3-bit indices via scalar bit manipulation (slow)
2. Does 32-element WHT butterfly in scalar C loops (5 stages × 32 ops = 160 ops)
3. Applies signs and scale per element (32 multiplies)
4. Then dots with Q8_0 in a separate loop (32 FMAs)

Q4_0's vec_dot does everything in ~4 AVX2 instructions per 32 elements.

Practical takeaway:

- there is no mature TQ3 CPU kernel to tune incrementally
- q4 benefits from years of SIMD path work and repack support
- TQ3 needs a deliberately fused SIMD implementation to become relevant on CPU

## Architecture

### Phase 1: Fused vec_dot (no separate dequant)

Instead of dequant → dot, fuse into one function that never materializes the full float array.

```
File: ggml/src/ggml-cpu/quants.c (or arch-specific file)
Function: ggml_vec_dot_tq3_0_q8_0_avx2()
```

My view:

- this fused path is the correct first target
- do not spend time trying to micro-optimize `dequantize_row_tq3_0()` in scalar form
- the right comparison is not "faster scalar dequant"; it is "direct SIMD block dot like q4/qk paths"

### Phase 2: SIMD WHT butterfly

The WHT butterfly `a+b, a-b` maps perfectly to SIMD:
- AVX2: `_mm256_add_ps` / `_mm256_sub_ps` on 8 floats at once
- NEON: `vaddq_f32` / `vsubq_f32` on 4 floats at once

For 32 elements with AVX2 (8-wide):
- Stage 1 (step=1): 4 × add/sub pairs on adjacent elements → shuffle within lanes
- Stage 2 (step=2): 4 × add/sub on pairs of pairs → shuffle within lanes
- Stage 3 (step=4): 4 × add/sub on groups of 4 → shuffle within lanes
- Stage 4 (step=8): cross-lane shuffle + add/sub (AVX2 `_mm256_permute2f128_ps`)
- Stage 5 (step=16): cross-register shuffle + add/sub

Total: ~20 SIMD instructions for the full 32-element WHT (vs 160 scalar ops).

### Phase 3: Optimized 3-bit unpack

Current: scalar switch statement per element.
Better: load 3 bytes (24 bits = 8 indices), unpack with bit shifts and masks in SIMD.

```
// 8 indices from 3 bytes: pack = byte0 | (byte1<<8) | (byte2<<16)
// idx[0] = pack & 7, idx[1] = (pack>>3) & 7, ..., idx[7] = (pack>>21) & 7
// AVX2: _mm256_and_si256(_mm256_srlv_epi32(pack_broadcast, shift_table), mask_7)
```

Then use the 3-bit index as a gather into the 8-element centroid table:
```
// AVX2: _mm256_i32gather_ps(centroids, indices, 4)
```

### Phase 4: Fused dot product

After WHT, instead of storing to memory and re-loading:
```
// vals[] now has dequanted floats in AVX2 registers
// q8[] loaded from Q8_0 block
// dot = _mm256_fmadd_ps(vals, _mm256_cvtepi8_epi32(q8), acc)
```

## Detailed Implementation

### File: `ggml/src/ggml-cpu/arch/x86/quants.c`

```c
void ggml_vec_dot_tq3_0_q8_0_avx2(int n, float * s, size_t bs,
    const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    const block_tq3_0 * x = vx;
    const block_q8_0  * y = vy;
    const int nb = n / QK_TQ3_0;

    __m256 acc = _mm256_setzero_ps();

    // Centroid LUT (8 values, fits in one AVX2 register)
    const __m256 centroids = _mm256_set_ps(
        2.1519f, 1.3439f, 0.7560f, 0.2451f,
        -0.2451f, -0.7560f, -1.3439f, -2.1519f);

    // Sign pattern (precomputed, 32 floats = 4 AVX2 registers)
    static const __m256 signs[4] = { /* precomputed */ };

    for (int i = 0; i < nb; i++) {
        float d_tq3 = GGML_FP16_TO_FP32(x[i].d);
        float d_q8  = GGML_FP16_TO_FP32(y[i].d);

        // Step 1: Unpack 3-bit indices → centroid values (4 groups × 8)
        __m256 v[4]; // 4 × 8 = 32 values
        for (int g = 0; g < 4; g++) {
            // Load 3 bytes, unpack 8 × 3-bit indices
            uint32_t pack = x[i].qs[g*3] | (x[i].qs[g*3+1]<<8) | (x[i].qs[g*3+2]<<16);
            __m256i idx = /* unpack with srlv + and */;
            v[g] = _mm256_i32gather_ps(centroid_array, idx, 4);
        }

        // Step 2: WHT butterfly (5 stages on 4 × __m256)
        // Stages 1-3: within each __m256 register
        // Stages 4-5: cross-register shuffles
        wht_avx2(v);  // ~20 instructions

        // Step 3: Apply 1/sqrt(32) * sign * d_tq3
        __m256 scale = _mm256_set1_ps(d_tq3 / sqrtf(32.0f));
        for (int g = 0; g < 4; g++)
            v[g] = _mm256_mul_ps(_mm256_mul_ps(v[g], signs[g]), scale);

        // Step 4: Dot with Q8_0
        __m256 d_scale = _mm256_set1_ps(d_q8);
        for (int g = 0; g < 4; g++) {
            __m256i q8_bytes = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)(y[i].qs + g*8)));
            __m256 q8_f = _mm256_cvtepi32_ps(q8_bytes);
            acc = _mm256_fmadd_ps(v[g], _mm256_mul_ps(q8_f, d_scale), acc);
        }
    }

    // Horizontal sum
    *s = hsum_avx2(acc);
}
```

### NEON version: `ggml/src/ggml-cpu/arch/arm/quants.c`

Same structure but with NEON intrinsics:
- `float32x4_t` (4-wide vs AVX2's 8-wide)
- 8 registers for 32 values instead of 4
- WHT stages 1-2 within registers, stages 3-5 cross-register
- `vfmaq_f32` for FMA

### WHT butterfly helper (AVX2)

```c
static inline void wht_avx2(__m256 v[4]) {
    // Stage 1 (step=1): adjacent pairs within each register
    for (int g = 0; g < 4; g++) {
        __m256 a = _mm256_shuffle_ps(v[g], v[g], 0xA0); // even elements
        __m256 b = _mm256_shuffle_ps(v[g], v[g], 0xF5); // odd elements
        v[g] = _mm256_addsub_ps(a, b); // or manual add/sub with blend
    }
    // Stage 2 (step=2): pairs of pairs
    // ... similar shuffle + add/sub
    // Stage 3 (step=4): groups of 4
    // ... _mm256_permute_ps + add/sub
    // Stage 4 (step=8): cross 8-element boundary
    // ... _mm256_permute2f128_ps + add/sub
    // Stage 5 (step=16): cross register pairs
    // v[0],v[2] interact; v[1],v[3] interact
    __m256 t0 = _mm256_add_ps(v[0], v[2]);
    __m256 t2 = _mm256_sub_ps(v[0], v[2]);
    __m256 t1 = _mm256_add_ps(v[1], v[3]);
    __m256 t3 = _mm256_sub_ps(v[1], v[3]);
    v[0]=t0; v[1]=t1; v[2]=t2; v[3]=t3;
}
```

## Estimated Performance

| Operation | Scalar ops | AVX2 ops | Speedup |
|-----------|-----------|----------|---------|
| 3-bit unpack | 32 shifts+masks | 4 srlv+gather | ~8x |
| WHT butterfly | 160 add/sub | ~20 add/sub | ~8x |
| Sign + scale | 64 mul | 8 mul | ~8x |
| Dot product | 32 FMA | 4 FMA | ~8x |
| **Total per block** | **~288 ops** | **~36 ops** | **~8x** |

With 8x speedup on the compute, TQ3_0 CPU should be ~1.4x slower than Q4_0 (vs 11x now).
Combined with 22% less memory bandwidth, net result: **within 1.0-1.5x of Q4_0 CPU speed**.

My caution:

- this is an optimistic upper bound, not a planning baseline
- the real first milestone should be more conservative: get TQ3 CPU within `2-3x` of q4 on decode
- once the fused AVX2 path exists, then decide whether NEON and deeper WHT tuning are justified

## Effort Estimate

| Phase | Effort | Impact |
|-------|--------|--------|
| AVX2 vec_dot | 2-3 days | 8x speedup |
| NEON vec_dot | 1-2 days | 8x speedup (Apple/ARM) |
| Unit tests | 0.5 day | correctness gate |
| Integration + benchmark | 0.5 day | end-to-end validation |
| **Total** | **4-6 days** | **TQ3_0 CPU competitive with Q4_0** |

My adjustment:

- `4-6 days` is plausible for a first AVX2 prototype plus tests
- "competitive with q4" is too strong for that first pass
- a better staging is:
  - Milestone 1: AVX2 fused vec-dot with correctness tests
  - Milestone 2: benchmark against current q4 CPU on TinyLlama and a 7B model
  - Milestone 3: only then decide whether NEON and further WHT specialization are worth it

## Dependencies

- None — pure CPU work, independent of CUDA track
- Can be done in parallel with GPU MMQ/MMVQ optimization
- File: `ggml/src/ggml-cpu/arch/x86/quants.c` (AVX2)
- File: `ggml/src/ggml-cpu/arch/arm/quants.c` (NEON)
- Registration: `ggml/src/ggml-cpu/ggml-cpu.c` (type traits)

## Recommended Execution Order

1. Add a block-level CPU reference test for `vec_dot_tq3_0_q8_0`.
2. Implement AVX2 fused decode + WHT + dot for one block size only.
3. Wire it behind feature detection without touching the scalar fallback.
4. Benchmark TinyLlama decode first.
5. If the AVX2 path clears a meaningful bar, then add NEON.

## Q4 Structural Template To Copy

The right model is not the scalar `dequantize_row_tq3_0()` flow. The right model is the x86 q4 direct vec-dot structure in [quants.c](/home/awee/code/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c#L543).

What q4 does structurally:

1. Load one quant block and one q8 block.
2. Compute the combined block scale once.
3. Decode packed integers directly into SIMD registers.
4. Dot against q8 bytes immediately.
5. Accumulate in SIMD.
6. Horizontally reduce once at the end.

That is the pattern TQ3 should follow.

What not to copy from current TQ3 scalar CPU:

- Do not call `dequantize_row_tq3_0()` from the hot vec-dot path.
- Do not materialize `float rotated[32]` and `float dequantized[32]` per block.
- Do not store intermediate WHT output to memory if it can stay in registers.

Useful nearby templates:

- q4 direct x86 vec-dot: [quants.c](/home/awee/code/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c#L543)
- tq1/tq2 x86 direct packed decode style: [quants.c](/home/awee/code/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c#L1080)
- current scalar TQ3 dequant path to replace in hot CPU dot: [ggml-quants.c](/home/awee/code/llama.cpp/ggml/src/ggml-quants.c#L2410)

## File-Level Implementation Checklist

### 1. Add the AVX2 kernel in x86 quants

File:
- [quants.c](/home/awee/code/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c)

Add:
- `ggml_vec_dot_tq3_0_q8_0()` AVX2 fast path

Structure:
- iterate over `block_tq3_0` / `block_q8_0`
- unpack four 3-byte groups into four SIMD vectors
- convert 3-bit codes to centroid values
- run the 32-element inverse WHT entirely in registers
- multiply by combined `d_tq3 * d_q8`
- dot directly with q8 values
- accumulate in `__m256`

Fallback:
- keep scalar/generic fallback for non-AVX2 builds

### 2. Add small AVX2 helpers instead of one monolithic kernel

File:
- [quants.c](/home/awee/code/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c)

Add helper functions for:
- unpack one 24-bit subgroup into 8 indices
- centroid decode for 8 values
- WHT stage helpers operating on `__m256` groups
- signed q8 load and float conversion
- horizontal sum

Reason:
- q4 works because the hot path is simple and composable
- TQ3 will be too hard to maintain if written as one giant function

### 3. Keep the scalar reference untouched

Files:
- [ggml-quants.c](/home/awee/code/llama.cpp/ggml/src/ggml-quants.c)
- [ggml-common.h](/home/awee/code/llama.cpp/ggml/src/ggml-common.h)

Do not change:
- `block_tq3_0`
- `dequantize_row_tq3_0()`
- quantization format

Reason:
- these are the correctness oracle for the SIMD path

### 4. Register the CPU vec-dot path correctly

Files to inspect:
- [ggml-cpu.c](/home/awee/code/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c)
- [ggml-cpu-impl.h](/home/awee/code/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-impl.h)
- [quants.h](/home/awee/code/llama.cpp/ggml/src/quants.h)

Ensure:
- TQ3 selects the AVX2 vec-dot when available
- non-AVX2 still goes through the existing generic path
- the q8 counterpart matches the actual activation type used on CPU

### 5. Add a block-contract unit test before tuning

Suggested new test:
- `tests/test-tq3-cpu-vecdot.cpp`

Test cases:
- randomized `block_tq3_0` + `block_q8_0`
- SIMD vec-dot vs scalar reference dot
- zero block
- extreme centroid patterns
- repeated-run determinism

Tolerance:
- tight relative error, because this should be an exact algebraic rearrangement, not an approximation

### 6. Add an end-to-end CPU benchmark guardrail

Use:
- TinyLlama `q4_0`
- TinyLlama `tq3_0`

Track:
- prompt tok/s
- decode tok/s
- ratio of `tq3_0 / q4_0`

Success bar for first AVX2 pass:
- materially above current `~4.6 tok/s` TQ3 decode
- ideally at least `10+ tok/s` on this machine before doing NEON

## My Recommended First Kernel Shape

Start with:
- one `block_tq3_0` at a time
- AVX2 only
- q8_0 counterpart only
- no attempt to support multiple q8 layouts in the first patch

Why:

- q4 succeeds because the hot path is narrow and specialized
- TQ3 needs the same discipline
- the first job is to prove the fused decode + WHT + dot structure works and is faster

## Concrete Next Coding Step

Implement this exact minimal slice first:

1. `unpack_tq3_group_8_avx2()`
2. `decode_tq3_centroids_8_avx2()`
3. `wht32_tq3_avx2(__m256 v[4])`
4. `ggml_vec_dot_tq3_0_q8_0()` AVX2 branch using those helpers
5. `tests/test-tq3-cpu-vecdot.cpp`

That is the smallest complete unit that matches the q4 design philosophy and gives a trustworthy benchmark result.

## Task Checklist

### Prep

- [ ] Confirm the CPU-side q8 counterpart for TQ3 weight matmul on the hot path (`q8_0` vs other q8 layout)
- [ ] Identify the exact registration point that selects TQ3 CPU vec-dot
- [ ] Freeze a baseline benchmark on this machine for:
  - [ ] `q4_0` TinyLlama CPU PP/TG
  - [ ] `tq3_0` TinyLlama CPU PP/TG

### Reference and Tests

- [ ] Add `tests/test-tq3-cpu-vecdot.cpp`
- [ ] Add randomized block-level reference test: SIMD vs scalar CPU result
- [ ] Add zero-block case
- [ ] Add extreme centroid-pattern case
- [ ] Add repeated-run determinism case

### AVX2 Helpers

- [ ] Add `unpack_tq3_group_8_avx2()`
- [ ] Add `decode_tq3_centroids_8_avx2()`
- [ ] Add `wht32_tq3_avx2(__m256 v[4])`
- [ ] Add q8 load / convert helper for the chosen q8 format
- [ ] Add horizontal reduction helper if needed

### First Fast Path

- [ ] Implement `ggml_vec_dot_tq3_0_q8_0()` AVX2 fast path
- [ ] Keep scalar/generic fallback unchanged
- [ ] Wire feature detection so AVX2 path is used only when available
- [ ] Verify the new path does not call `dequantize_row_tq3_0()` in the hot loop

### Integration

- [ ] Register the TQ3 CPU vec-dot path in the CPU backend/type traits
- [ ] Rebuild CPU-only binary
- [ ] Run block-level test suite
- [ ] Run TinyLlama CPU decode sanity prompt

### Benchmark Gate

- [ ] Re-run `q4_0` TinyLlama CPU PP/TG
- [ ] Re-run `tq3_0` TinyLlama CPU PP/TG
- [ ] Record absolute tok/s
- [ ] Record `tq3_0 / q4_0` ratio
- [ ] Decide whether AVX2 first pass is good enough to continue to NEON

### Stretch Follow-Ups

- [ ] Add NEON implementation
- [ ] Add larger-model CPU benchmark (7B class)
- [ ] Consider deeper WHT specialization only after AVX2 first pass is proven

## Guardrails

- Block contract test: SIMD result vs scalar CPU result on randomized blocks
- End-to-end decode sanity: TinyLlama prompt should remain semantically correct
- Performance gate:
  - track `q4_0` and `tq3_0` on the same CPU-only binary
  - report both absolute tok/s and ratio vs `q4_0`

## References

- Q4_0 AVX2 vec_dot: `ggml/src/ggml-cpu/arch/x86/quants.c` — template for SIMD structure
- WHT SIMD literature: "Fast Walsh-Hadamard Transform" — well-studied, many SIMD implementations exist
- TQ1_0/TQ2_0 have no SIMD either — this would be the first WHT-based SIMD quant in llama.cpp
