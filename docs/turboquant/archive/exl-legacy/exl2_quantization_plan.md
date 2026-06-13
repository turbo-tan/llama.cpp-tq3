# EXL2 Quantization Plan

## Problem
Need base Qwen3.5-27B model (HuggingFace format) to quantize to EXL2.
Currently only have GGUF quantizations.

**UPDATE**: Attempted download but **disk is full** (1.7T / 1.8T used, only 5.4M free).

## Disk Space Issue
```
/dev/nvme1n1p2  1.8T  1.7T  5.4M 100% /
```

Cannot download 54 GB base model without freeing space first.

## Options

### Option 1: Download base model (~54 GB)
```bash
cd /home/awee/models
huggingface-cli download Qwen/Qwen3.5-27B --local-dir Qwen3.5-27B
```
**Time**: ~30-60 minutes depending on connection
**Space**: 54 GB

### Option 2: Use existing quantization for analysis
We already have comprehensive data:
- TQ3_4S: 12.9 GB, PPL 6.77, 315 tok/s
- IQ4_XS: 13.27 GB, PPL 6.83
- UD-Q2_K_XL: 9.76 GB, PPL 7.53

**Focus on**: Understanding EXL2's approach from code analysis

### Option 3: Test with smaller model
Download Qwen3.5-9B and quantize that for quick testing:
```bash
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/models/Qwen3.5-9B
```
**Time**: ~10 minutes
**Space**: ~18 GB

## Recommendation

**Skip EXL2 quantization** for now because:

1. **54 GB download** takes too long
2. **Quantization** takes 2-4 hours
3. **Already have code analysis** showing key differences
4. **Can't match screenshot test conditions** anyway

## What We Learned from Code

### EXL2's Speed Advantage
1. **No WHT** - Direct 3-bit packing
2. **Arithmetic dequant** - No lookup table
3. **Adaptive kernels** - Optimized per problem size
4. **Free tiling** - No 32-element constraint

### How to Apply to TQ3

**TQ3_5 Format Proposal**:
```
1. Remove runtime WHT
   - Apply WHT during quantization
   - Store WHT-transformed weights
   
2. Use arithmetic dequant
   - Replace centroid lookup with formula
   - Faster than memory access
   
3. Adaptive kernel selection
   - Different kernels for different M sizes
   - Like EXL2's unit_exl2_3a.cu
   
4. Proper tiling
   - No 32-element constraint
   - Use tensor cores efficiently
```

### Implementation Path

**Phase 1: Precompute WHT** (format change)
- Quantize: `x → WHT → quantize → store`
- Dequantize: `load → dequantize → done` (no runtime WHT)
- Benefit: Removes 5 shuffle steps overhead

**Phase 2: Arithmetic dequant** (kernel change)
- Replace: `centroids[idx] * scale`
- With: `(idx * step + offset) * scale`
- Benefit: Compute instead of memory access

**Phase 3: Adaptive kernels** (optimization)
- Implement kernel variants for M=1,2,3,4...
- Select best kernel at runtime
- Benefit: 20-30% speedup

**Phase 4: Tensor cores** (major rewrite)
- Use WMMA for GEMM
- Tile output 16×16 or 32×32
- Benefit: 2-3x speedup

## Next Steps

1. **Document EXL2 analysis** ✓ (done)
2. **Design TQ3_5 format** (precomputed WHT)
3. **Prototype arithmetic dequant** (test quality)
4. **Benchmark against TQ3_4S**
5. **If successful, implement full TQ3_5**

## Conclusion

Don't need to quantize to EXL2 - we have enough information from code analysis to design TQ3_5 format that combines:
- TQ3's quality (WHT-based quantization)
- EXL2's speed (precomputed transform, arithmetic dequant)

Target: **TQ3_5 = TQ3_4S quality + 2x EXL2-like speed**
