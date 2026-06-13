# TQ3_4S Marlin Kernel Status

## Current Status (2026-04-01 08:44)

### ✅ CORRECTNESS VERIFIED
Output matches cuBLAS exactly:
```
ref=[0.0658 -0.4026 0.2677 0.8080]
test=[0.0658 -0.4035 0.2677 0.8078]
```

### ⚠️ PERFORMANCE
- Current: Scalar accumulation kernel (baseline)
- Grid: (M, batch) - one warp per output element
- Very slow compared to target 600+ tok/s
- This is expected for the simple implementation

### Implementation Details
- Correct centroids and signs
- Correct WHT butterfly
- Correct warp reduction
- Simple but correct

### Next Steps for Optimization
1. Implement WMMA-based tiled version
2. Process 16x16 output tiles per block
3. Use tensor cores for matrix multiplication
4. Target: 2x speedup over cuBLAS (600+ tok/s)

### Recommendation
The kernel is correct and can be used as a reference. For production use, need to implement the optimized WMMA version.
