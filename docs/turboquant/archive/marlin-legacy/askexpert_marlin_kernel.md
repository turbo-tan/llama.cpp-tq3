# Ask Expert: TQ3_4S Marlin Kernel Performance

## Recent Progress (2026-04-01)

### What We Accomplished
1. ✅ Fixed TQ3_4S Marlin kernel correctness
   - Root cause: Missing warp reduction (all 32 threads were overwriting output)
   - Solution: Added `__shfl_down_sync` reduction before writing
   - Result: Output now matches cuBLAS exactly (max error 0.0007)

2. ❌ Performance optimization failed
   - Attempted to achieve 2x speedup over cuBLAS
   - Result: 5x SLOWER than cuBLAS (32s vs 6s per chunk)
   - Conclusion: Fused dequant+WHT+GEMM approach doesn't work

### Current Status
- **Correctness**: ✅ Kernel produces correct results
- **Performance**: ❌ 5.3x slower than cuBLAS baseline
- **Recommendation**: Revert to cuBLAS path (315 tok/s acceptable)

## Context
Implementing CUDA kernel for TQ3_4S quantized matrix multiplication to beat cuBLAS performance.

## Problem
TQ3_4S format requires:
1. Dequantize 3-bit values using 8 centroids
2. Apply Walsh-Hadamard Transform (WHT) - 32-element butterfly
3. Matrix multiply with input

Current cuBLAS path: Dequantize to fp16 → cuBLAS GEMM = 315 tok/s
Goal: Fused kernel = 600+ tok/s (2x speedup)

## What We Tried

### Attempt 1: Fine-grained (1 warp per output)
```cuda
Grid: (M, batch)
Block: 32 threads (1 warp)
Each warp: dequant 32 weights → WHT → dot product → 1 output
```
**Result**: 8.9M blocks for M=17408, batch=512 → GPU scheduler overload, hangs

### Attempt 2: Tile batch dimension
```cuda
Grid: (M, batch/16)
Block: 512 threads (16 warps)
```
**Result**: 557K blocks → still hangs

### Attempt 3: Tile both dimensions
```cuda
Grid: (M/4, batch/16)
Block: 512 threads
```
**Result**: 139K blocks → runs but 32s/chunk vs 6s/chunk cuBLAS (5x SLOWER)

## Core Constraint
WHT requires 32 consecutive elements processed by 1 warp:
```cuda
// WHT butterfly - must be done by full warp
for (int step = 1; step < 32; step <<= 1) {
    float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
    if (lane & step) val = other - val;
    else val = other + val;
}
```

This prevents efficient tensor core usage (needs 16×16 tiles, not 32×1).

## Questions

1. **Is fused dequant+WHT+GEMM fundamentally the wrong approach?**
   - Should we precompute WHT offline and store weights in transformed space?
   - Would this allow standard tensor core GEMM?

2. **Can WHT be reformulated for tensor cores?**
   - Is there a way to express 32-element WHT as 16×16 matrix ops?
   - Or split into smaller WHTs that tile better?

3. **What's the right granularity?**
   - Current: 1 warp processes 32 K elements → 1 output
   - Should each block process larger output tiles (64×64)?
   - How to amortize WHT cost across multiple outputs?

4. **Memory bandwidth vs compute?**
   - Is the bottleneck memory (loading quantized weights)?
   - Or compute (WHT butterfly overhead)?
   - cuBLAS loads fp16 (2x more bandwidth) but no WHT

5. **Alternative: Separate kernels?**
   - Kernel 1: Dequant + WHT → fp16 buffer
   - Kernel 2: cuBLAS GEMM on fp16
   - Would this be faster than fused approach?

## Measurements

**cuBLAS path** (baseline):
- Dequant kernel: ~X ms
- cuBLAS GEMM: ~Y ms  
- Total: 6s per chunk (315 tok/s)

**Marlin kernel** (failed):
- Fused kernel: 32s per chunk (59 tok/s)
- 5.3x slower

## Format Details

**TQ3_4S block** (32 elements):
```c
struct block_tq3_4s {
    uint8_t d[4];      // 4 scales (6-bit each)
    uint8_t qs[12];    // 32 × 3-bit indices = 96 bits
};
```

**Dequantization**:
1. Unpack 3-bit index → lookup centroid
2. Multiply by scale
3. Apply WHT (32-element butterfly)
4. Multiply by sign pattern
5. Normalize by 1/√32

## Hardware
- GPU: RTX 5060 Ti (Ampere, compute 8.0)
- VRAM: 16GB
- Typical workload: M=17408, batch=512, K=5120

## What Should We Do?

Is the Marlin-style fused kernel approach viable for TQ3_4S, or should we:
1. Stick with cuBLAS (315 tok/s is acceptable)
2. Try separate dequant + cuBLAS kernels
3. Redesign quantization format to avoid runtime WHT
4. Something else entirely?

---

## How to Reproduce / Benchmark

### Build llama.cpp with TQ3_4S support
```bash
cd /home/awee/code/llama.cpp
git checkout feature/tq3_1k-explore  # Branch with TQ3_4S kernel
cd build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
make llama-perplexity -j8
```

### Run perplexity benchmark (cuBLAS baseline)
```bash
cd /home/awee/code/llama.cpp
./build/bin/llama-perplexity \
    -m /home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
    -ngl 99 \
    -fa 1 \
    -c 2048 \
    -f wikitext-2-raw/wiki.test.raw \
    --chunks 100 \
    > /home/awee/code/tan_llama/artifacts/ppl_tq3_4s_$(date +%Y%m%d_%H%M).txt 2>&1
```

**Expected output**:
```
Final estimate: PPL = 6.7727 +/- 0.05378
prompt eval time = 583298.95 ms / 204800 tokens (2.85 ms per token, 351.11 tokens per second)
```

### Quick test (1 chunk)
```bash
./build/bin/llama-perplexity \
    -m /home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
    -ngl 99 -fa 1 -c 2048 --no-warmup \
    -f wikitext-2-raw/wiki.test.raw --chunks 1
```

**Expected**: ~6 seconds per chunk with cuBLAS

### Monitor GPU usage
```bash
# In separate terminal
nvidia-smi dmon -s u -c 100
```

**Expected**: 90-100% GPU utilization during prompt eval

### Files
- Model: `/home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf` (12.9 GB)
- Test data: `/home/awee/code/llama.cpp/wikitext-2-raw/wiki.test.raw`
- Results: `/home/awee/code/tan_llama/artifacts/ppl_tq3_4s_*.txt`

### Key Metrics
- **PPL**: Should be ~6.77 (quality)
- **PP tok/s**: ~315-350 tok/s (speed)
- **Memory**: ~13.4 GB VRAM at c=2048

### Kernel Code Locations
- Kernel: `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/tq3_4s_marlin.cu`
- Integration: `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` (search for `tq3_4s_marlin_gemm`)
- Header: `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/tq3_4s_marlin.cuh`
