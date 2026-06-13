# TQ3_4S Marlin Kernel Status (2026-04-01 09:25)

## Current State

### ✅ Correctness
The kernel produces correct results matching cuBLAS:
```
ref=[0.0658 -0.4026 0.2677 0.8080]
test=[0.0658 -0.4035 0.2677 0.8078]
```

### ❌ Performance Problem
**EXTREMELY SLOW** - kernel hangs on real workloads.

**Root cause**: Grid size explosion with large batch sizes.
- Example: M=17408, batch=512, K=5120
- Grid: (17408, 512) = **8.9 million blocks**
- Each block: 32 threads doing K/32 iterations
- Total: 285 billion thread-iterations per matmul

The kernel is called ~100 times per forward pass, making it unusable.

## Why It's Slow

Current design: **1 warp (32 threads) per output element**
- Grid: `(M, batch)`
- Block: 32 threads
- Each warp computes one Y[m,b] by:
  1. Dequantizing 32 weights (collaborative)
  2. Applying WHT (butterfly, 5 steps)
  3. Dot product with 32 X elements
  4. Warp reduction
  5. Write 1 output

**Problem**: For batch=512, we launch M×512 blocks. With M=17408, that's 8.9M blocks overwhelming the GPU scheduler.

## What Needs to Change

The kernel needs to process **multiple output elements per block**:

1. **Tile the output**: Each block computes 64×16 or 128×8 output tile
2. **Use shared memory**: Load W and X into shared memory once
3. **Reuse data**: Amortize dequant+WHT cost across multiple outputs
4. **Reduce grid size**: From millions of blocks to thousands

## Comparison

**Current (broken)**:
- Grid: (M, batch) - millions of blocks
- Each block: 1 output element
- Speed: Unusable (hangs)

**Target (WMMA-based)**:
- Grid: ((M+63)/64, (batch+15)/16) - thousands of blocks  
- Each block: 64×16 = 1024 output elements
- Speed: 2x faster than cuBLAS

## Next Steps

1. Revert to cuBLAS for now (kernel is too slow)
2. Implement proper tiled WMMA kernel
3. Benchmark against cuBLAS baseline

## Lesson Learned

Fine-grained parallelism (1 warp per element) works for small problems but doesn't scale to production batch sizes. Need coarse-grained tiling from the start.
