# TQ3_0 CUDA — Expert Help v2: Speed Problem

## Progress Since v1

All correctness issues resolved. TQ3_0 works for both **weight quantization** and **KV cache** on CUDA:

| Model | Q8_0 | TQ3_0 | Savings | Output |
|-------|------|-------|---------|--------|
| TinyLlama 1.1B | 1.1 GB | 491 MB | 55% | ✅ correct |
| Llama-2-7B | 6.7 GB | 2.9 GB | 57% | ✅ correct |
| Qwen3.5-9B | 8.9 GB | 4.2 GB | 53% | ✅ correct |

## The Speed Problem

TQ3_0 is **~30x slower** than q4_0 because we're forced onto the cublas fp32 fallback path:

```
q4_0 path:  load q4_0 block → bit-shift to int8 → dp4a/tensor core matmul (fused)
tq3_0 path: dequant ALL weights to fp32 → allocate temp buffer → cublasSgemm
```

| Model | q4_0 TG | tq3_0 TG | Ratio |
|-------|---------|----------|-------|
| Llama-2-7B | 88 tok/s | 3.4 tok/s | 26x slower |
| Qwen3.5-9B | ~60 tok/s | 3.1 tok/s | 19x slower |

Three paths are disabled for TQ3_0:

### 1. mmvq (TG path) — disabled, expert confirmed qi=16/vdr=4 is correct

The vec_dot with 4-thread cooperative warp-shuffle WHT passes standalone tests. Expert confirmed `qi=16, vdr=4` gives `qi/vdr=4`, and 4 consecutive threads DO stay in the same warp. The shuffle uses `partner_lane = (threadIdx.x & ~3) | (g ^ step_t)`.

**But when integrated into llama.cpp, output is wrong.** We haven't isolated why. The standalone test uses raw float Q values; the real mmvq uses Q8_1 blocks. Could be a Q8_1 indexing issue.

### 2. mmq (PP path) — disabled, expert identified the bug

Expert said: the fixed scale `rms * 3.5 / 127` in `load_tiles_tq3_0` is approximate, not a proper per-block q8_0 requantization. Needs warp reduction for amax and the exact scale layout the q8 path expects. We haven't fixed this yet.

### 3. fp16 cublas — disabled, causes garbage for weights

The WHT produces values that lose precision in fp16 GEMM. Forcing fp32 SGEMM fixes correctness but is slower. Could the issue be in the fp16 dequant kernel output (values too small for fp16 range), or in cublas fp16 accumulation?

## What We Need

The fastest path to usable speed. Options:

**Option A: Fix mmvq (TG) + mmq (PP)**
- mmvq: debug why the integrated vec_dot fails (Q8_1 indexing? thread mapping in the actual mmvq kernel vs our test harness?)
- mmq: implement proper per-block q8_0 requantization with warp amax reduction
- Estimated speed: PP ~530 tok/s, TG ~94 tok/s (what we measured before correctness was verified)

**Option B: Fix fp16 cublas only**
- Diagnose why fp16 GEMM produces garbage (is it the dequant output range? accumulation precision?)
- If fixable, get ~2x speedup over fp32 cublas with zero kernel work
- Estimated speed: PP ~500 tok/s, TG ~60 tok/s

**Option C: Use flash attention for KV cache path**
- Add TQ3_0 to fattn kernel support list
- Handles KV cache attention natively (like TheTom's Metal approach)
- Doesn't help weight matmul speed

## Key Files

- `ggml/src/ggml-cuda/vecdotq.cuh:1240` — vec_dot_tq3_0_q8_1 (mmvq)
- `ggml/src/ggml-cuda/mmq.cuh:3233` — load_tiles_tq3_0 (mmq)
- `ggml/src/ggml-cuda/common.cuh:1033` — type traits (qi=16, vdr=4)
- `ggml/src/ggml-cuda/ggml-cuda.cu` — three exclusion lines forcing cublas
- `ggml/src/ggml-cuda/convert.cu` — dequant kernel (correct, unit tested)
- `tests/test-tq3-cuda.cu` — 9 passing unit tests

## Branch

`experiment/tq3-weight-quant` on `charpdev/llama.cpp`
(KV cache work on `feature/turboquant-aaryan`)

## Hardware

RTX 5060 Ti 16GB (Blackwell, CC 12.0)
