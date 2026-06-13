# TQ3_4S Marlin Kernel Performance Analysis

## Problem Statement
Compute: Y[M, batch] = W[M, K] @ X[K, batch]
- W is quantized as TQ3_4S (32 elements per block)
- Need to dequantize + WHT before multiply

## Current Implementation Analysis

### Grid/Block Configuration
```
Grid: (M, batch)
Block: 32 threads (1 warp)
```

### Workload per matmul
Example: M=17408, batch=512, K=5120

**Total blocks launched**: M × batch = 17,408 × 512 = **8,912,896 blocks**

**Work per block**:
- Threads: 32
- K iterations: K/32 = 5120/32 = 160
- Operations per thread per iteration:
  - Dequantize: ~10 ops
  - WHT butterfly: 5 shuffle steps × 2 ops = 10 ops
  - Multiply-add: 2 ops
  - Total: ~22 ops
- Total ops per block: 32 × 160 × 22 = **112,640 ops**

**Total operations**: 8,912,896 blocks × 112,640 ops = **1.004 trillion operations**

### GPU Limits (RTX 5060 Ti, Ampere-class)
- Max blocks in flight: ~32K (hardware scheduler limit)
- Block scheduling overhead: High when launching millions of blocks
- Warp occupancy: Low (only 1 warp per block = 3.125% of SM capacity)

### Why It's Slow
1. **Scheduler bottleneck**: 8.9M blocks >> 32K max in-flight
2. **Poor occupancy**: 1 warp/block wastes 96.875% of SM capacity
3. **No data reuse**: Each block dequantizes W independently

## Optimal Design

### Target Configuration
```
Grid: (M/64, batch/16) 
Block: 256 threads (8 warps)
Each block computes: 64×16 output tile
```

### Workload Analysis
Example: M=17408, batch=512, K=5120

**Total blocks**: ⌈17408/64⌉ × ⌈512/16⌉ = 272 × 32 = **8,704 blocks**

**Reduction**: 8,912,896 → 8,704 = **1024x fewer blocks**

### Work per block
- Output tile: 64 × 16 = 1024 elements
- Threads: 256 (8 warps)
- Each thread computes: 1024/256 = 4 output elements

**K-loop tiling**:
- Outer loop: K/32 = 160 iterations
- Per iteration:
  - Load W tile: 64×32 (shared memory)
  - Load X tile: 32×16 (shared memory)
  - Dequantize W: 64×32 elements (collaborative)
  - Compute: 64×16 outputs using 32×16 X

**Operations per block**:
- Dequant + WHT: 64 × 160 × 32 × 22 = 7.2M ops
- GEMM: 64 × 16 × 160 × 32 × 2 = 10.5M ops
- Total: ~17.7M ops per block

**Total operations**: 8,704 × 17.7M = **154 billion ops**

### Comparison

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Blocks | 8.9M | 8.7K | **1024x fewer** |
| Warps/block | 1 | 8 | **8x more** |
| Occupancy | 3% | 25% | **8x better** |
| Data reuse | None | 16x batch | **16x reuse** |
| Scheduler pressure | Extreme | Normal | **1024x less** |

## Implementation Strategy

### Phase 1: Reduce batch dimension (IMMEDIATE FIX)
```
Grid: (M, batch/16)
Block: 512 threads (16 warps)
Each warp: 1 output element
```
- Blocks: 8.9M → 557K (**16x reduction**)
- Should be usable immediately

### Phase 2: Tile both dimensions (OPTIMAL)
```
Grid: (M/64, batch/16)
Block: 256 threads (8 warps)
Shared memory: W[64×32] + X[32×16]
```
- Blocks: 8.9M → 8.7K (**1024x reduction**)
- Target performance

## Complexity Analysis

### Current: O(M × batch) blocks
- batch=512: **Unusable** (millions of blocks)
- batch=1: Usable (thousands of blocks)

### Phase 1: O(M × batch/16) blocks
- batch=512: **Usable** (hundreds of thousands)
- 16x reduction in grid size

### Phase 2: O(M/64 × batch/16) blocks  
- batch=512: **Optimal** (thousands)
- 1024x reduction in grid size
- Maximum data reuse

## Conclusion

The current kernel has **O(M × batch)** complexity in grid size, which is catastrophic for batch=512.

**Immediate fix**: Reduce batch dimension by 16x → Phase 1
**Optimal solution**: Tile both dimensions → Phase 2

Let's implement Phase 1 first to get it working, then optimize to Phase 2.
