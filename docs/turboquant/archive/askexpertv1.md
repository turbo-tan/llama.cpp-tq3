# TurboQuant TQ3_0 CUDA Implementation — Expert Help Needed

## What We're Building

CUDA GPU support for TurboQuant TQ3_0 (3.5-bit quantization) in llama.cpp. TQ3_0 uses a Walsh-Hadamard Transform (WHT) + Lloyd-Max codebook to achieve 4.6x compression vs fp16 with near-zero quality loss. It's designed for KV cache compression but we're also experimenting with weight quantization.

Based on Aaryan-Kapoor's CPU implementation (block-level WHT, Apache 2.0).
Paper: TurboQuant (arXiv 2504.19874, ICLR 2026).

Branch: `feature/turboquant-aaryan` on `charpdev/llama.cpp`

## Block Format

```c
#define QK_TQ3_0 32
typedef struct {
    ggml_half d;                    // RMS scale (2 bytes)
    uint8_t qs[QK_TQ3_0 * 3 / 8];  // 3-bit packed indices (12 bytes)
} block_tq3_0;                      // 14 bytes total = 3.5 bpv
```

## Algorithm

```
Quantize: x → RMS normalize → sign flips → WHT forward → 1/sqrt(32) → nearest centroid → pack 3-bit
Dequant:  unpack 3-bit → centroid lookup → WHT inverse → 1/sqrt(32) → undo signs → scale by RMS
```

Sign pattern: `sign[i] = ((i * 0x9E3779B9) >> 31) ? -1 : +1` (golden ratio hash)

N(0,1) Lloyd-Max centroids (8 levels for 3 bits):
```
{-2.1519, -1.3439, -0.7560, -0.2451, +0.2451, +0.7560, +1.3439, +2.1519}
```

WHT is a butterfly transform, 5 stages for block size 32:
```c
for (int step = 1; step < 32; step <<= 1)
    for (int i = 0; i < 32; i += step*2)
        for (int j = i; j < i+step; j++) {
            float a = x[j], b = x[j+step];
            x[j] = a + b; x[j+step] = a - b;
        }
```

WHT is self-inverse (up to 1/N scaling). Forward and inverse are the same butterfly, just with signs applied at different points.

## What Works

1. **CPU quantize + dequant + vec_dot** — fully correct, PPL 8.14 on Llama-2-7B
2. **CUDA quantize kernel** (`quantize_f32_tq3_0_block` in `cpy-utils.cuh`) — single-thread-per-block, matches CPU output exactly (unit tested)
3. **CUDA dequant kernel** (`dequantize_block_tq3_0` in `convert.cu`) — 32 threads per block, warp-shuffle WHT, matches CPU output exactly (unit tested)
4. **cublas fallback path for KV cache** — dequant TQ3_0 → fp32 → SGEMM. Produces correct text output ("The capital of France is Paris"). Works because KV cache tensor after `ggml_permute` is non-contiguous → forces fp32 cublas path.
5. **9 standalone CUDA unit tests** — all pass (quantize, dequant, roundtrip cosine, dot product, GPU vs CPU equivalence)

## What's Broken

### Problem 1: mmvq vec_dot (TG path) — fundamentally incompatible

The mmvq kernel in llama.cpp assigns threads to quantized blocks via:
```
kbx = tid / (qi/vdr)     // which block this thread works on
iqs = vdr * (tid % (qi/vdr))  // offset within the block
```

For q4_0: `qi=4, vdr=2` → `qi/vdr=2` → 2 threads per block, each independently processes half. No cross-thread communication needed.

**TQ3_0's WHT requires all 32 elements to interact.** We tried splitting into 4 groups of 8 elements across 4 threads, with warp shuffle for the inter-group WHT stages (steps 8 and 16). The problem:

- With `qi=4, vdr=4`: `qi/vdr=1` → every thread gets `iqs=0`. All threads process group 0 only. **Bug: groups 1-3 never processed.**
- With `qi=16, vdr=4`: `qi/vdr=4` → 4 threads per block. But the warp shuffle `__shfl_sync(0xF, vals[j], partner)` used mask `0xF` (lanes 0-3), while the 4 cooperating threads aren't necessarily in lanes 0-3. **Bug: shuffle reads from wrong threads.**
- With `qi=1, vdr=1`: `qi/vdr=1` → each thread processes full block independently (no shuffle). But `blocks_per_iter = warp_size / (qi/vdr) = 32`, and there are only 4 blocks per row (head_dim=128/32=4). Only threads 0-3 do work, threads 4-31 skip. **Bug: 7/8 of computation missing.**
- With `qi=32, vdr=1`: `qi/vdr=32` → all 32 threads in a warp share ONE block. Each computes the same full dot product. Results summed 32x. **Bug: 32x the correct value.**

**We wrote a standalone test** that proves the 4-thread cooperative WHT works when we control thread assignment:
```cuda
// This PASSES — partner_lane = (threadIdx.x & ~3) | (g ^ step_t)
int partner_lane = (threadIdx.x & ~3) | (g ^ step_t);
float other = __shfl_sync(0xFFFFFFFF, vals[j], partner_lane);
```

But we haven't found `qi/vdr` values that make the mmvq kernel assign 4 consecutive threads to each block AND keep them in predictable warp lanes.

**Current workaround:** Exclude TQ3_0 from mmvq entirely (`src0->type != GGML_TYPE_TQ3_0` in `use_mul_mat_vec_q`). Falls to cublas.

### Problem 2: mmq load_tiles (PP path) — produces wrong output

The mmq kernel dequants K to q8_0 format in shared memory via `load_tiles_tq3_0`. Our implementation:

1. Each of 32 threads in a warp unpacks one 3-bit index from the TQ3_0 block
2. Looks up the centroid value
3. Performs WHT inverse via `__shfl_xor_sync(0xFFFFFFFF, val, step)` across the warp
4. Applies sign, scale, and requantizes to int8
5. Writes to shared memory as q8_0 format
6. Reuses existing q8_0 tensor core matmul

The warp shuffle WHT should work here because all 32 threads ARE in one warp (the mmq kernel processes one block per warp). But when we enable mmq for TQ3_0, the output is wrong ("P" then garbage).

We haven't written a standalone unit test for load_tiles yet. The mmq kernel is complex (shared memory tiling, tensor cores) and hard to test in isolation.

**Current workaround:** Exclude TQ3_0 from mmq (`src0->type != GGML_TYPE_TQ3_0` in `use_mul_mat_q`). Falls to cublas.

### Problem 3: cublas fp16 path — produces garbage for weight quantization

When TQ3_0 is used for **model weights** (not KV cache), the weight tensor is contiguous, so `use_fp16 = true` → dequant to fp16 → cublasGemmEx fp16. This produces garbage output.

When we force `use_fp16 = false` for TQ3_0 (→ fp32 SGEMM), it STILL produces garbage.

But the **same dequant kernel** produces correct output when used for KV cache (non-contiguous tensor → fp32 cublas path → correct "Paris" output).

CPU path with the same quantized weight file produces correct output.

Our standalone CUDA unit tests for the dequant kernel pass at all scales (up to 4M elements).

**We don't understand why the cublas path works for KV cache but not for weights.** The dequant kernel is identical. The cublas GEMM is the same. The only difference is the tensor shape and how `ggml_cuda_op_mul_mat` iterates over dimensions.

## Current State

| Path | KV cache | Weights | Speed |
|------|----------|---------|-------|
| CPU (ngl=0) | ✅ correct | ✅ correct | slow |
| cublas fp32 | ✅ correct | ❌ garbage | PP 283, TG 61 |
| cublas fp16 | not tested | ❌ garbage | — |
| mmvq | ❌ broken | ❌ broken | TG ~94 (when it worked) |
| mmq | ❌ broken | N/A | PP ~530 (when it worked) |
| flash attn | ✅ (falls to CPU) | N/A | slow |

## Key Files

- `ggml/src/ggml-cuda/cpy-utils.cuh` — CUDA quantize kernel (`quantize_f32_tq3_0_block`)
- `ggml/src/ggml-cuda/convert.cu` — CUDA dequant kernel (`dequantize_block_tq3_0`)
- `ggml/src/ggml-cuda/vecdotq.cuh` — mmvq vec_dot (`vec_dot_tq3_0_q8_1`)
- `ggml/src/ggml-cuda/mmq.cuh` — mmq load_tiles (`load_tiles_tq3_0`)
- `ggml/src/ggml-cuda/common.cuh` — type traits (`qi=4, vdr=4, qr=2, qk=32`)
- `ggml/src/ggml-cuda/ggml-cuda.cu` — dispatch (mmvq/mmq exclusions)
- `ggml/src/ggml-quants.c` — CPU quantize/dequant (reference implementation)
- `tests/test-tq3-cuda.cu` — standalone CUDA unit tests (9 tests, all pass)

## Comparison with Other Implementations

| | Aaryan-Kapoor | TheTom | Us |
|---|---|---|---|
| CPU | ✅ | ✅ | ✅ (from Aaryan) |
| Metal (Apple) | ❌ | ✅ (q8_0 parity) | ❌ |
| CUDA (NVIDIA) | ❌ | ❌ | ✅ cublas only |
| Flash attention | ❌ | ✅ (Metal) | ❌ (falls to CPU) |

TheTom's Metal implementation uses flash attention which handles TQ3_0 natively in the attention kernel — no separate dequant + matmul. This avoids the mmvq/mmq problem entirely.

spiritbuun/llama-cpp-turboquant-cuda is just a fork of TheTom's repo with no additional CUDA work despite the name.

## Questions for Expert

1. **mmvq vec_dot:** Is there a `qi/vdr` combination that makes 4 consecutive threads share one block with predictable warp lanes? Or should we write a custom mmvq kernel for TQ3_0 that bypasses the standard dispatch?

2. **mmq load_tiles:** The warp-shuffle WHT should work (32 threads = 1 warp). What could cause wrong output? Is there a shared memory bank conflict or synchronization issue we're missing?

3. **cublas weight path:** Why does the same dequant kernel produce correct results for KV cache (non-contiguous, fp32 SGEMM) but garbage for weights (contiguous, fp16 or fp32 GEMM)? The standalone unit test passes at production scale.

4. **Alternative approach:** Should we implement a custom flash attention kernel for TQ3_0 on CUDA (like TheTom did for Metal) instead of trying to fit TQ3_0 into the existing mmvq/mmq framework?

## Hardware

- NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell, compute capability 12.0)
- Ubuntu 22.04, CUDA 12.x

## Models Tested

- Llama-2-7B Q4_0 (with TQ3_0 KV cache) — primary test model
- TinyLlama 1.1B (requantized Q4_0 → TQ3_0 weights) — weight quant test

## vLLM TurboQuant PR (lishunyang12, March 26 2026)

PR: https://github.com/vllm-project/vllm/pull/38280

Key insights from their approach:

1. **They use Triton kernels** (not CUDA C) for encode/decode — much easier to write and debug
2. **Phase 1 stores bf16** — they dequant BEFORE FlashAttention, not inside the matmul kernel. This is the same as our cublas fallback approach. They acknowledge this doesn't save memory yet.
3. **Phase 2 (planned)** will use a "dual-cache architecture": compressed uint8 persistent cache + temporary bf16 working cache decoded on-the-fly before each FlashAttention call
4. **Their results on H200**: 100% quality match at 2/3/4-bit, zero TTFT overhead, 21% throughput improvement at bs=16
5. **Blocker**: FlashAttention's C++ kernel rejects anything other than fp16/bf16/fp8. Storing packed indices requires changes to block allocator, KV cache spec, and attention metadata builder.

**Key architectural difference from our approach:**
- They dequant KV cache to bf16 BEFORE attention (like our cublas fallback)
- We tried to dequant INSIDE the matmul kernel (mmvq/mmq) which caused all the thread mapping bugs
- Their approach is simpler and correct, just slower (extra dequant pass)
- The speed comes from Phase 2 (packed storage) which they haven't implemented yet

**This validates our cublas fallback as the correct Phase 1 approach.** The mmvq/mmq optimizations are Phase 2 work that should be done after correctness is established.
