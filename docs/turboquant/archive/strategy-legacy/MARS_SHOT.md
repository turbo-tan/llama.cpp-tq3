# TQ3_0 — The World-Changing Shot

Date: 2026-03-27

## The Vision

TQ3_0 weights + TQ3_0 KV cache = smaller model AND smaller KV cache.

- Model: 2.81 GB vs 3.56 GB (q4_0) — 21% smaller
- KV cache: 4.6x smaller than fp16
- Quality: better than Q4_0 (PPL +0.08 vs +0.47)

This is what Aaryan and TheTom don't have. q4_0 weights + TQ3_0 KV is just KV compression.
TQ3_0 weights + TQ3_0 KV is the full package.

## Current State

| Config | PP 512 | TG 1 | Model | KV (8K) |
|--------|--------|------|-------|---------|
| q4_0 + q4_0 KV | 2639 | 44 | 3.56 GB | ~5 GB |
| tq3_0 + tq3_0 KV | 443 | 5.15 | 2.81 GB | ~1 GB |
| ratio | 17% | 12% | 79% | 20% |

TQ3_0 weights are 4-5x slower. The bottleneck is the q8_0 bridge in MMQ.

## Why the Gap Exists

q4_0 MMQ: bit-shift → int8 tile → tensor core (trivial decode, full tensor core speed)
tq3_0 MMQ: 3-bit unpack → centroid → q8_0 requant → tensor core (expensive bridge)

The bridge costs:
- 3-bit unpack: 1.6x slower than nibble shift
- q8_0 requant: warp reduce + quantize (eliminated with fixed scale, but still overhead)
- Tile layout: DS4 vs D4 (q4_0 uses more efficient layout)

## The Path to World-Changing

### Option A: Fix the MMQ bridge (incremental)
- Already done: pre-rotation eliminates WHT
- Already done: fixed scale eliminates warp reduce
- Remaining: tile layout (DS4 vs D4) — change TQ3_0 to use DS4
- Expected gain: ~1.5x → PP ~35% of q4_0

### Option B: Native TQ3_0 tensor core kernel (moonshot)
- Decode TQ3_0 → fp16 fragment directly (no q8_0 bridge)
- Use WMMA/MMA with fp16 weights + fp16 activations
- We tried this: decode overhead dominates (0.21 ms vs 0.014 ms GEMM)
- The decode is memory-bound reading 1.8 MB of TQ3_0 data
- Need to fuse decode INTO the tensor core kernel (like q4_0 MMQ does)
- This is the real moonshot — requires a custom WMMA kernel that reads TQ3_0 blocks directly

### Option C: Accept weight gap, win on KV (near-term)
- q4_0 weights + TQ3_0 KV: PP 87-90% of q4_0, TG 49% (needs MMVQ fix)
- Fix MMVQ for KV attention → TG ~90% of q4_0
- This beats Aaryan and TheTom (they have no CUDA)
- Publish: "near-q4_0 speed with 4.6x less KV memory"

## The World-Changing Number

If we achieve Option B (native TQ3_0 tensor core):
- TQ3_0 weights: ~90% of q4_0 speed (same tensor cores, less data to read)
- TQ3_0 KV: 4.6x smaller
- Combined: **faster than q4_0 at long context** (less data = less bandwidth = faster)

At 32K context with TQ3_0 weights + TQ3_0 KV:
- Weight matmul: ~90% of q4_0 (22% less weight data)
- KV attention: faster than q4_0 (78% less KV data)
- Net: **TQ3_0 beats q4_0 at long context**

## Immediate Next Steps

1. Fix MMVQ for KV attention (non-contiguous tensors) → TG 49% → ~90%
2. Change TQ3_0 MMQ to use DS4 layout → PP 17% → ~25%
3. Design native TQ3_0 WMMA kernel → PP 17% → ~90%

Step 1 is one line of code. Steps 2-3 are the real work.
