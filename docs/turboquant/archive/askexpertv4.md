# TQ3_0 CUDA — Expert Help v4: Tensor Core Path

## Current State

All correctness issues resolved. Multiple optimization passes completed.

### Benchmark (TinyLlama 1.1B, RTX 5060 Ti 16GB)

| Model | PP tok/s | TG tok/s | VRAM |
|-------|---------|---------|------|
| q4_0 | 775 | 463 | 571 MiB |
| tq3_0 | 233 | 35 | 456 MiB |
| ratio | **30%** | **7.6%** | **80%** |

### Optimization Journey (PP tok/s)

| Step | PP | Change |
|------|-----|--------|
| cuBLAS fp32 | 16 | baseline |
| cuBLAS fp16+fp32c | 147 | +9x |
| Native prefill v1 (WHT+dot) | 204 | +39% |
| Tiled prefill (smem activation) | 233 | +14% |
| Pre-rotated + simplified load_tiles | 233 | flat |

## The Bottleneck

We profiled and confirmed: **the bottleneck is not WHT compute, not memory bandwidth — it is compute efficiency (scalar warps vs tensor cores).**

```
Memory bound at 400 GB/s: 0.006 ms
q4_0 MMQ (tensor cores):  0.041 ms  (6.8x memory bound — tensor core overhead)
tq3_0 tiled (scalar):     0.229 ms  (38x memory bound — scalar warp overhead)
```

q4_0 uses WMMA/MMA tensor cores (16×16 tiles, ~100 TFLOPS effective).
Our kernel uses scalar warps (~20 TFLOPS).
That's the 5.6x gap.

## What We've Tried

1. **Larger tile size** (TILE_N=8→32): slower due to register pressure
2. **Shared memory activation cache** (tiled prefill): 1.55x speedup
3. **Pre-rotation** (eliminate WHT from load_tiles): 1.12x on load_tiles, flat end-to-end
4. **MMQ with pre-rotated activations**: same speed as tiled prefill

## The Question

**How do we get TQ3_0 onto tensor cores?**

The challenge: tensor cores expect int8 or fp16 matrices. TQ3_0 blocks are 3-bit packed with a WHT transform. The current MMQ path dequants TQ3_0 → q8_0 in `load_tiles`, then uses the q8_0 tensor core path. But `load_tiles` is the bottleneck.

### Option A: Native TQ3_0 WMMA kernel

Design a custom WMMA kernel where:
- A matrix: TQ3_0 centroids (3-bit → fp16, 8 values per lane)
- B matrix: pre-rotated fp16 activations
- Accumulate in fp32

The centroid values are in [-2.15, +2.15] — fits fp16 perfectly.
With pre-rotated activations, the dot product is just `centroid · act_rot` — no WHT.

Key question: can we express TQ3_0 centroid lookup + dot product as a WMMA operation?

### Option B: Improve load_tiles throughput

The current load_tiles processes one TQ3_0 block per warp (32 threads).
For a 2048×2048 matmul: 2048 × 64 = 131072 blocks to process.
At 32 threads/block: 131072 warps needed.

Could we process multiple blocks per warp? Or use a different shared memory layout that better feeds the tensor core path?

### Option C: Accept the gap, optimize for long context

At long context (32K+ tokens), KV cache bandwidth dominates.
TQ3_0 KV cache is 4.6x smaller than fp16 → reads 78% less data.
At 32K context, PP drops 85% for q4_0 but only 65% for tq3_0 (less KV data).
TQ3_0 could actually BEAT q4_0 at long context even with the current kernel.

## Specific Questions

1. **WMMA feasibility**: Can we write a WMMA kernel that takes TQ3_0 blocks directly (without going through q8_0)? The centroid lookup is a 3-bit → fp16 table lookup per element. Is there a way to express this in the WMMA fragment API?

2. **load_tiles bottleneck**: The current load_tiles does: unpack 3-bit → centroid → q8_0 requant → write to shared memory. With pre-rotation, it's: unpack 3-bit → centroid → q8_0 requant (no WHT). The requant (warp reduce amax + quantize) is now the bottleneck. Can we skip the requant and feed fp16 centroids directly to the tensor core path?

3. **Long context strategy**: Should we focus on the long-context advantage instead of closing the short-context gap? At 32K tokens, TQ3_0 KV cache saves 3.2 GB vs fp16. That's the real value proposition.

## Code Locations

- `ggml/src/ggml-cuda/tq3-prefill.cuh` — tiled prefill kernel (current best PP)
- `ggml/src/ggml-cuda/mmq.cuh:3218` — load_tiles_tq3_0 (simplified, no WHT)
- `ggml/src/ggml-cuda/tq3-native.cuh` — block-dot helper + rotation kernel
- `ggml/src/ggml-cuda/ggml-cuda.cu:2392` — rotation wired into MMQ path
- `tests/test-tq3-prefill.cu` — prefill correctness test (PASS)
- `tests/test-tq3-load-tiles.cu` — load_tiles correctness test (PASS)
- `tests/test-tq3-cuda.cu` — 10 unit tests (all PASS)

## Branch

`experiment/tq3-weight-quant` on `charpdev/llama.cpp`
Latest commit: `d41a70fb4`

## Hardware

RTX 5060 Ti 16GB (Blackwell, CC 12.0, tensor cores available)
