# TQ3_0 CUDA — Expert Help v6: MMVQ for Non-Contiguous KV Cache

## Current State (commit aa01fb0c0)

Branch: `experiment/tq3-weight-quant` on `charpdev/llama.cpp`

### Benchmark (Llama-2-7B, RTX 5060 Ti, q4_0 weights)

| KV type | PP 512 | PP 4096 | TG 1 | KV VRAM |
|---------|--------|---------|------|---------|
| q4_0 | 2616 | 2650 | 45 | large |
| tq3_0 | 2425 | 2277 | **22** | **4.6x smaller** |
| ratio | 93% | 86% | **49%** | |

PP is 86-93% of q4_0. **TG is only 49% because MMVQ is blocked for KV attention.**

## The Problem

MMVQ for TQ3_0 is blocked for non-contiguous tensors (KV cache K after `ggml_permute`).

The block is in `ggml/src/ggml-cuda/ggml-cuda.cu:2366`:
```c
bool use_mul_mat_vec_q = ggml_cuda_can_use_mul_mat_vec_q(...)
    && (src0->type != GGML_TYPE_TQ3_0 || ggml_is_contiguous(src0));
```

### Why It's Blocked

The MMVQ kernel (`ggml_cuda_mul_mat_vec_q`) reads `src0->data` directly without handling non-contiguous strides. For KV attention, K is a view with permuted strides after `ggml_permute(ctx0, k, 0, 2, 1, 3)`.

The cublas path (which works) goes through `ggml_cuda_op_mul_mat` which copies non-contiguous data to a contiguous buffer via `ggml_cuda_cpy_tensor_2d` before calling the kernel.

### What We Tried

Removing the contiguity guard → TG hangs (MMVQ reads wrong data from non-contiguous K).

## The Question

**How do we enable MMVQ for non-contiguous TQ3_0 KV cache tensors?**

Options:

### Option A: Copy before MMVQ (like cublas does)
In `ggml_cuda_mul_mat_vec_q`, detect non-contiguous src0 and copy to a temp buffer first.
- Pro: minimal change to vec_dot kernel
- Con: adds a copy overhead per TG step

### Option B: Stride-aware MMVQ kernel
Modify the MMVQ kernel to accept stride parameters and index correctly into non-contiguous data.
- Pro: no copy overhead
- Con: complex kernel change, affects all types

### Option C: Force contiguous copy in the dispatch
In `ggml_cuda_mul_mat`, before calling `ggml_cuda_mul_mat_vec_q` for TQ3_0, copy src0 to a contiguous buffer.
- Pro: isolated to TQ3_0, doesn't affect other types
- Con: adds copy overhead

### Option D: Use flash attention for KV cache
Flash attention handles non-contiguous KV natively. TQ3_0 is not in the fattn kernel list, so it falls back to CPU — but if we add TQ3_0 to fattn, it would work.
- Pro: clean architecture
- Con: requires adding TQ3_0 to fattn kernel (significant work)

## Expected Impact

If TG is fixed (MMVQ for KV attention):
- TG: 22 → ~40 tok/s (estimated, based on MMVQ working for contiguous weights)
- PP: unchanged (~2400 at 512 ctx)
- Combined: q4_0 weights + TQ3_0 KV at 90%+ of q4_0 speed with 4.6x less KV memory

## The Mars Shot

With MMVQ fixed for KV attention:

| Config | PP 512 | TG 1 | KV VRAM | vs q4_0 |
|--------|--------|------|---------|---------|
| q4_0 + q4_0 KV | 2616 | 45 | large | 100% |
| **q4_0 + tq3_0 KV (fixed)** | **~2400** | **~40** | **4.6x smaller** | **~90%** |

Near-q4_0 speed, 4.6x less KV memory. This is the world-changing combination.

## Test Commands

```bash
# KV decode repro (must not timeout)
timeout 90 ./build/bin/llama-bench \
    -m /home/awee/models/tinyllama-1.1b-q4_0.gguf \
    -ngl 999 -fa 0 -ctk tq3_0 -ctv f16 -p 512 -n 1 -r 1 --no-warmup

# Quality sanity
./build/bin/llama-completion \
    -m /home/awee/models/tinyllama-1.1b-tq3_0.gguf \
    -ngl 999 -fa 0 -n 10 --no-warmup -no-cnv \
    -p "The capital of France is" --seed 42

# TG benchmark (target: tg1 > 35 tok/s)
./build/bin/llama-bench \
    -m /home/awee/models/tinyllama-1.1b-q4_0.gguf \
    -ngl 999 -fa 0 -ctk tq3_0 -ctv f16 -p 512 -n 1 -r 3 --no-warmup
```

## Branch

`experiment/tq3-weight-quant` on `charpdev/llama.cpp`, commit `aa01fb0c0`
