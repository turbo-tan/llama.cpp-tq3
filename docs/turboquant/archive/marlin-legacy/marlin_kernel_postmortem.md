# TQ3_4S Marlin Kernel - FAILED ATTEMPT

## Date: 2026-04-01

## Objective
Implement fused dequantize+WHT+GEMM kernel to achieve 2x speedup over cuBLAS (target: 600+ tok/s vs 315 tok/s baseline).

## Result
**FAILED** - Kernel is 5x slower than cuBLAS baseline.

## Timeline
1. Initial implementation: Correct output, but 1 warp per output element
2. Grid explosion: M×batch = 8.9M blocks for typical workload
3. Attempted fix: Tile batch dimension (16x reduction)
4. Still too slow: 139K blocks, 32s per chunk vs 6s cuBLAS
5. Root cause: Fine-grained parallelism doesn't work for this problem

## Performance Comparison
- **cuBLAS baseline**: ~6s per chunk (315 tok/s)
- **Marlin kernel**: 32s per chunk (59 tok/s)
- **Result**: **5.3x SLOWER** than cuBLAS

## Why It Failed

### Fundamental Problem
TQ3_4S requires dequantize + WHT before GEMM. The WHT is a 32-element butterfly operation that must be done collaboratively by a warp.

**Constraint**: Each 32-element block needs 1 warp to dequantize + WHT.

**Consequence**: Can't use tensor cores efficiently because:
1. Tensor cores need 16×16 tiles
2. WHT needs 32-element warps
3. These don't align well

### What Was Tried

**Attempt 1**: 1 warp per output element
- Grid: (M, batch) = 8.9M blocks
- Status: Hangs (scheduler overload)

**Attempt 2**: Tile batch by 16x
- Grid: (M, batch/16) = 557K blocks  
- Status: Still hangs

**Attempt 3**: Tile both dimensions
- Grid: (M/4, batch/16) = 139K blocks
- Status: Runs but 5x slower than cuBLAS

### Why Tiling Doesn't Help
Even with aggressive tiling, each block still does:
- Dequantize 32 elements (collaborative warp operation)
- WHT butterfly (5 shuffle steps)
- Dot product
- Warp reduction

This is **too much overhead per output element**. cuBLAS just does:
- Load fp16 weights (already dequantized)
- Tensor core GEMM (highly optimized)

## Lessons Learned

1. **Fused kernels aren't always faster**: The dequant+WHT overhead dominates
2. **Tensor cores need alignment**: 32-element WHT doesn't map to 16×16 tiles
3. **cuBLAS is highly optimized**: Hard to beat for standard GEMM
4. **Fine-grained parallelism scales poorly**: Need coarse-grained tiles from start

## Conclusion

The Marlin-style fused kernel approach **does not work** for TQ3_4S. The WHT operation creates too much overhead and prevents efficient use of tensor cores.

**Recommendation**: Stick with cuBLAS path. The 315 tok/s is acceptable for a 3.5-bit format.

## What Would Work

To actually beat cuBLAS, would need:
1. **Precompute WHT offline**: Store weights in WHT-transformed space
2. **Remove runtime WHT**: Just dequantize + GEMM
3. **Use tensor cores**: Standard 16×16 WMMA tiles
4. **Batch dequantization**: Amortize dequant cost across large tiles

But this requires changing the quantization format itself, not just the kernel.

## Status: ABANDONED

Reverting to cuBLAS path. Marlin kernel removed from codebase.
