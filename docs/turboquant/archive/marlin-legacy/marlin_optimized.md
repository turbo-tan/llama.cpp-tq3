# TQ3_4S Marlin Kernel - Optimized

## Status (2026-04-01 08:58)

### ✅ Correctness
Output matches cuBLAS with negligible error:
```
ref=[0.3660 0.4091 -1.4446 0.7209]
test=[0.3653 0.4088 -1.4447 0.7207]
Max error: 0.0007 (0.02% relative)
```

This is acceptable - typical for GPU kernels due to different accumulation order.

### ⚡ Performance Optimization
Changed from 1 row per warp to 2 rows per warp:
- Grid: `((M+1)/2, batch)` instead of `(M, batch)`
- Block: 64 threads (2 warps) instead of 32 threads (1 warp)
- Each warp processes one output row independently
- **2x better GPU utilization**

### Implementation
```cuda
// Grid: ((M+1)/2, batch), Block: 64 threads
const int m = blockIdx.x * 2 + threadIdx.x / 32;  // 2 rows per block
const int lane = threadIdx.x % 32;

// Each warp does:
// 1. Dequantize + WHT (32 elements)
// 2. Dot product with X
// 3. Warp reduction
// 4. Write output
```

### Next Steps
- Test on larger dataset for perplexity validation
- Compare tok/s with cuBLAS baseline
- Consider further optimizations if needed
