# Q4_0_TQ Speed Tuning Plan

Date: 2026-03-28

## Status

Historical / secondary plan.

Do not treat this as the active execution track.

Current canonical plan:

- `FIX_PLAN_2026-03-28.md`

Current reality:

- `Q4_0_TQ` work produced useful speed and fit lessons
- but it is no longer the main product line
- active weight-quality work is on `TQ3_1S`
- active KV work is on `TQ3_0` with adaptive policy

## Goal

Make Q4_0_TQ MMQ as fast as native Q4_0 MMQ.
Current gap: Q4_0_TQ PP ~863 tok/s vs Q4_0 PP ~3644 tok/s (24% ratio, 7B model).

## Why The Gap Exists

Q4_0 MMQ hot path:
```
load_tiles_q4_0: nibble shift (1 op/elem) → int8 tile → tensor core
```

Q4_0_TQ MMQ hot path (current):
```
load_tiles_q4_0_tq: 3-bit unpack (3 ops/elem) + centroid lookup + scale → q8 bridge → tensor core
```

The q8 bridge is the structural bottleneck. It adds:
1. Per-block warp reduce for amax (5 shuffle stages)
2. Per-element quantize (roundf + clamp)
3. Larger shared memory tile (q8_0 format vs q4_0 format)

## The Fused Rotation Insight

With pre-rotated activations (already implemented), the load_tiles WHT is gone.
But the q8 bridge remains. The fix: **fuse rotation into Q8_1 quantization** so activations
arrive pre-rotated at the MMQ kernel, and load_tiles just does centroid lookup + scale.

## Files To Touch

### 1. `ggml/src/ggml-cuda/mmq.cu` — Dispatch

**What:** Add `quantize_mmq_q8_1_tq3_cuda` as the src1 quantizer for Q4_0_TQ MMQ.
**Why:** Current path uses generic `quantize_mmq_q8_1_cuda` which doesn't rotate activations.
**Change:**
```c
// In ggml_cuda_mul_mat, for Q4_0_TQ:
ggml_cuda_op_mul_mat(ctx, src0, src1, dst,
    ggml_cuda_op_mul_mat_q,
    quantize_mmq_q8_1_tq3_cuda);  // NEW: fused rotate + quantize
```

### 2. `ggml/src/ggml-cuda/mmq.cuh` — load_tiles_q4_0_tq

**What:** Simplify `load_tiles_q4_0_tq` to use LUT-based unpack + direct scale (no q8 bridge).
**Why:** With pre-rotated activations, the centroid values ARE the correct values. No WHT, no warp reduce.
**Change:**
```cuda
// Current: 3-bit unpack → centroid → warp reduce amax → q8 requant
// New:     3-bit unpack via LUT → centroid × rms → direct int8 (fixed scale)
const float d = 2.1519f * rms / 127.0f;  // fixed scale, no warp reduce
const int8_t qval = (int8_t)roundf(centroid[idx] * rms / d);
```

### 3. `ggml/src/ggml-cuda/tq3-native.cuh` — Fused Quantize Kernel

**What:** Add `quantize_mmq_q8_1_tq3_cuda` — fuses WHT rotation + Q8_1 quantization.
**Why:** Eliminates the separate rotation kernel and temp buffer allocation.
**New kernel:**
```cuda
// One pass: load fp32 → WHT(sign*x)/sqrt(32) → Q8_1 quantize
// No intermediate buffer. Called instead of quantize_mmq_q8_1_cuda for Q4_0_TQ.
__global__ void quantize_mmq_q8_1_tq3_kernel(
    const float * __restrict__ x,
    block_q8_1 * __restrict__ y,
    int64_t ne0, int64_t ne1);
```

### 4. `ggml/src/ggml-cuda/vecdotq.cuh` — LUT for 3-bit Unpack

**What:** Add 256-entry constant memory LUT mapping byte → 8 int8 centroid values.
**Why:** Replaces 8 individual bit extractions + 8 centroid lookups with 1 table lookup per 8 elements.
**Change:**
```cuda
// 256-entry LUT: byte → 8 pre-scaled int8 centroid values
// 2KB constant memory, loaded once
__constant__ int8_t TQ3_BYTE_LUT[256][8];

// In load_tiles: 3 bytes → 3 LUT lookups → 24 int8 values
// vs current: 3 bytes → 8 bit extractions + 8 centroid lookups
```

### 5. `ggml/src/ggml-cuda/ggml-cuda.cu` — Remove Separate Rotation

**What:** Remove the `ggml_cuda_tq3_rotate_act` call from the cublas path for Q4_0_TQ.
**Why:** Rotation is now fused into `quantize_mmq_q8_1_tq3_cuda`. No separate rotation needed.
**Change:** Remove the rotation block in `ggml_cuda_op_mul_mat_cublas` for Q4_0_TQ.

## Expected Performance

| Step | Current cost | After fix | Savings |
|------|-------------|-----------|---------|
| Rotation kernel | ~0.006 ms | 0 (fused) | 100% |
| Q8_1 quantize | ~0.05 ms | ~0.05 ms (fused with rotation) | 0% |
| load_tiles WHT | 5 shuffles/block | 0 | 100% |
| load_tiles unpack | 8 ops/elem | 1 LUT lookup/8 elem | 87% |
| load_tiles warp reduce | 5 shuffles/block | 0 (fixed scale) | 100% |
| Tensor core matmul | same | same | 0% |

**Theoretical result:** load_tiles cost ≈ q4_0 load_tiles. Total MMQ ≈ q4_0 × (3.5/4.5) = **1.28x faster than q4_0**.

## Test Before Each Change (STEERING.md)

```bash
# After each file change, run:
./build/bin/test-backend-ops test -o MUL_MAT -p 'type_a=q4_0_tq,type_b=f32,m=128,n=512,k=256'

# Quality sanity
./build/bin/llama-completion -m /home/awee/models/tinyllama-1.1b-tq3_0.gguf \
    -ngl 999 -fa 0 -n 10 --no-warmup -no-cnv \
    -p "The capital of France is" --seed 42

# Benchmark (target: PP > 1000 tok/s on 7B)
./build/bin/llama-bench -m [7B_Q4_0_TQ_model] -ngl 999 -fa 1 -p 512 -n 1 -r 3
```

## Implementation Order

1. **LUT in vecdotq.cuh** — standalone, no dependencies, easy to test
2. **Simplified load_tiles in mmq.cuh** — uses LUT, test with existing tile test
3. **Fused quantize kernel in tq3-native.cuh** — new kernel, test in isolation
4. **Wire fused quantize in mmq.cu** — dispatch change, test end-to-end
5. **Remove separate rotation in ggml-cuda.cu** — cleanup, verify no regression

## Status Update (2026-03-28)

**Expert assessment:** The structural diagnosis here is correct — the remaining tax is
conversion/staging, not small math ops. However, the v0 contract cannot reach native Q4_0
territory even with all these optimizations applied.

**The v1 contract is now the main plan, not a separate experiment.**

The Q4_1_TQ / v1 path removes the hot-path scale reconstruction tax entirely by changing
the block format. The ideas in this document (fused rotation, LUT unpack) are still valid
but should be applied to the v1 contract, not the old v0 path.

**Ranking (expert):**
1. Q4_1_TQ / new contract — best legs now
2. Ideas from this doc that reduce conversion/staging — useful if they map onto v1
3. Sparse-V — later, TG only
4. Bit-serial / codebook matmul — moonshot backlog

## What NOT To Change

- KV cache path (separate track)
- Moonshot ideas (separate backlog in MOONSHOT_IDEAS.md)

## Key Invariant

The rotation must happen exactly once per activation block, before Q8_1 quantization.
After rotation, load_tiles sees pre-rotated activations and does NOT need WHT.
The fused kernel guarantees this without a separate rotation pass.
