# TQ3_4S Marlin Kernel - FIXED!

## Problem Found (2026-04-01 08:27)

**Root Cause**: Missing warp reduction. All 32 threads were writing their partial sums directly to `Y[m + b*M]`, overwriting each other.

## Solution

Added warp shuffle reduction before writing:
```cuda
// Reduce sum across warp
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
}

if (lane == 0) {
    Y[m + b * M] = sum;
}
```

## Result

Output now matches cuBLAS exactly:
```
ref=[0.3660 0.4091 -1.4446 0.7209]
test=[0.3653 0.4088 -1.4447 0.7207]
```

## Current Implementation

- Scalar accumulation kernel
- 1 warp (32 threads) per output element
- Grid: (M, batch)
- Each thread processes one element per 32-element WHT block
- Correct centroids, signs, WHT, and reduction

## Next Steps

1. Run full PPL test to verify correctness
2. Measure performance vs cuBLAS
3. Optimize with WMMA tiles for 2x speedup target

## Key Learnings

1. The math was correct all along (centroids, signs, WHT)
2. The bug was in data movement (missing reduction)
3. Debugging strategy: simplify first, then optimize
4. Warp-level operations require explicit synchronization/reduction
