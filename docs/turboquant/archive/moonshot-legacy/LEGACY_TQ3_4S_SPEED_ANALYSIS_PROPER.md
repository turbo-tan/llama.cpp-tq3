# TQ3_4S Speed Analysis - Proper Investigation

Date: 2026-04-01

## Current Performance Gap

**Measured speeds (27B, c=2048):**
- TQ3_4S: 284-351 tok/s
- Q3_K_S: 635-786 tok/s  
- **Gap: 2.2-2.8x slower**

## Bottleneck Analysis

### What I Got Wrong Initially

**Claimed:** "WHT overhead is the bottleneck, removing it gives 2x speedup"

**Reality from profiling:**
- WHT rotation: 0.09-0.49 ms (~1-2% of total)
- Dequant kernel: 78.8% of GPU time
- GEMM: 9.4% of GPU time

**Conclusion:** WHT is NOT the bottleneck. The dequant kernel is.

### Why TQ3_4S Dequant is Slow

From `PP_SPEED_DESIGN.md`:

**Current TQ3_4S dequant (fp16 cuBLAS path):**
```
1. Unpack 3-bit indices (bit manipulation)
2. Decode E3M5 scale (bit manipulation)  
3. Lookup centroid from table
4. Multiply centroid × scale
5. Write fp16 to output buffer
```

**Q3_K_S dequant (optimized MMQ):**
```
1. Unpack 2-bit + 1-bit indices (simpler)
2. Load fp16 scale directly (no decode)
3. Lookup from smaller table
4. Multiply + write
```

**Key differences:**
- E3M5 decode adds overhead (bit manipulation)
- 3-bit unpacking is more complex than 2+1 bit
- Centroid table is larger (8 entries vs 4)

### Why Q3_K_S is Fast

**Q3_K_S advantages:**
1. **Simpler unpacking**: 2-bit + 1-bit is easier than 3-bit
2. **No scale encoding**: fp16 scales stored directly
3. **Smaller tables**: 4-entry lookup vs 8-entry
4. **Better memory access**: Aligned reads
5. **Optimized MMQ kernel**: Uses fp16 tensor cores directly

## Real Speed Optimization Paths

### Option 1: Optimize TQ3_4S Dequant Kernel (Realistic)

**Target:** Reduce dequant from 78.8% → 40-50% of time

**Approaches:**
1. **Fused dequant+scale**: Combine E3M5 decode with centroid lookup
2. **Vectorized unpacking**: Use SIMD to unpack 4 indices at once
3. **Shared memory caching**: Cache centroids in shared memory
4. **Better memory coalescing**: Align reads/writes

**Expected gain:** 1.3-1.5x (315 → 410-470 tok/s)

### Option 2: fp16 MMQ Tile Loader (From PP_SPEED_DESIGN.md)

**Current int8 MMQ:** 24 warp shuffles per block (broken, 105 tok/s)
**Proposed fp16 MMQ:** 8 warp shuffles per block

**Changes:**
- Write fp16 directly to weight tiles (no int8 packing)
- Use h16816 tensor cores (16×8×16 MMA)
- Bake scale into fp16 values

**Expected gain:** 1.5-2x (315 → 470-630 tok/s)

### Option 3: Arithmetic Dequant (From EXL2 analysis)

**Current:** `val = centroids[idx] * scale`
**Proposed:** `val = (idx * step + offset) * scale`

**Benefits:**
- No table lookup (compute instead of memory)
- Better for GPU (ALU faster than memory)

**Challenges:**
- Need to find formula that matches centroid distribution
- May lose quality if formula doesn't fit well

**Expected gain:** 1.2-1.3x (315 → 380-410 tok/s)

### Option 4: Hybrid Format (TQ3_4S_FAST)

**Idea:** Simplify encoding to match Q3_K_S advantages

**Changes:**
- Replace E3M5 with fp16 scales (no decode overhead)
- Keep 3-bit centroids (quality advantage)
- Optimize unpacking with lookup tables

**Trade-off:** +0.5 bpw (4.0 → 4.5 bpw) but 1.5x faster

**Expected gain:** 1.4-1.6x (315 → 440-500 tok/s)

## Recommended Path

**Priority 1:** Option 2 (fp16 MMQ Tile Loader)
- Already designed in PP_SPEED_DESIGN.md
- Clear implementation path
- Targets the actual bottleneck (dequant kernel)
- Expected: 1.5-2x speedup

**Priority 2:** Option 1 (Optimize existing dequant)
- Lower risk, incremental improvement
- Can combine with Option 2
- Expected: 1.3-1.5x speedup

**Priority 3:** Option 3 (Arithmetic dequant)
- Requires quality validation
- May not fit centroid distribution well
- Expected: 1.2-1.3x speedup

## Why TQ3_5 (Precomputed WHT) Won't Work

**Problem 1: Basis consistency**
- Removing WHT inverse leaves output in rotated basis
- Breaks residual connections, LayerNorm, non-linearities
- Requires full graph rewrite (not a simple format change)

**Problem 2: Wrong bottleneck**
- WHT is only 1-2% of runtime
- Even if removed completely, gain is negligible
- Real bottleneck is dequant kernel (78.8% of time)

**Problem 3: Complexity**
- Would need to track basis state across entire graph
- Handle residuals, norms, activations in rotated domain
- Much larger than "6 hour" estimate

## Conclusion

**What I learned:**
1. Always profile before optimizing
2. Rotation overhead is tiny (1-2%), not the bottleneck
3. Dequant kernel is 78.8% of time - that's what to optimize
4. Q3_K_S is fast because of simpler encoding, not lack of WHT

**Next steps:**
1. Implement fp16 MMQ tile loader (from PP_SPEED_DESIGN.md)
2. Profile and measure actual speedup
3. If successful, combine with dequant optimizations
4. Target: 470-630 tok/s (1.5-2x improvement)
