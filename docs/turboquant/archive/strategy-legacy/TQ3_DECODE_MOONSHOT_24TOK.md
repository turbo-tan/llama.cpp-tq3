# TQ3_4S Decode Moonshot: Beat Competitors by 5%

**Target**: 24.5 tok/s (5% above IQ4_XS at 23.2 tok/s)

**Current**: 14.35 tok/s  
**Required Speedup**: 1.71x (71% improvement)

**Competitor Baselines**:
- Q3_K_S: 20.7 tok/s → Beat at 21.74 tok/s (✓ +5%)
- IQ4_XS: 23.2 tok/s → Beat at 24.36 tok/s (TARGET)

---

## Why This Target is Achievable

Current decode path analysis:
1. **Rotation overhead**: ~35% of decode time
   - Each token requires rotating activations before GEMV
   - For attention: rotate Q (query) for every token
   - For FFN: rotate input for every layer

2. **MMVQ inefficiency**: ~40% of decode time
   - Current TQ3_4S MMVQ uses scalar decode
   - No packed-Y optimization for weights

3. **Kernel launch overhead**: ~15% of decode time
   - Separate kernels for rotate → quantize → gemv

4. **Memory bandwidth**: ~10% of decode time
   - Weight loading from global memory

---

## Path to 24.5 tok/s

### Phase 1: KV Pre-Rotation (Week 1)
**Expected gain**: +15-20% → 16.5-17.2 tok/s

Eliminate K rotation in attention:
- Store K in rotated domain (rotate once at write)
- Attention only rotates Q (query)
- Saves ~70% of attention rotation cost
- Attention is ~35% of decode → 35% × 70% = 24.5% total gain

**Conservative**: 14.35 × 1.15 = **16.5 tok/s**

### Phase 2: Fused Q-Projection+Rotation (Week 2)
**Expected gain**: +10-15% → 18.1-19.0 tok/s

Currently: Query projection → separate rotate kernel
Optimized: Fuse projection and rotation into one kernel

- Query projection is ~15% of decode
- Fusion eliminates kernel launch + memory traffic
- Gain: 15% × 80% = 12% effective

**Math**: 16.5 × 1.12 = **18.5 tok/s**

### Phase 3: SM120-Optimized MMVQ (Week 3)
**Expected gain**: +20-30% → 22.2-24.1 tok/s

Current MMVQ for TQ3_4S:
- VDR_TQ3_4S_Q8_1_MMVQ = 4 (scalar decode)
- No vectorized weight loading

SM120 (Blackwell) optimizations:
- Increase VDR to 8 (vector decode)
- Use packed-Y for weight staging
- Vectorized scale decode (LDG.128)

**Math**: 18.5 × 1.25 = **23.1 tok/s**

### Phase 4: Hybrid Q8 Attention (Week 4)
**Expected gain**: +8-12% → 24.9-25.9 tok/s

Store attention Q/K in Q8_0 (no rotation needed):
- Attention Q/K: Q8_0 format
- FFN weights: TQ3_4S (quality where it matters)
- Attention is 35% of decode → no rotation overhead

**Math**: 23.1 × 1.08 = **24.9 tok/s** ✓ TARGET HIT

---

## Detailed Implementation Plan

### Week 1: KV Pre-Rotation

Files to modify:
- `ggml/src/ggml-cuda/mmvq.cu` - Add pre-rotated path
- `ggml/src/ggml-cuda/tq3-native.cu` - Add rotate_inplace kernel
- `ggml/src/ggml-cuda/ggml-cuda.cu` - Hook into KV cache

Key changes:
```cpp
// In KV cache write path
if (type == TQ3_4S && is_key_projection) {
    // Rotate K before storing
    tq3_rotate_inplace(k_cache, n_elements);
}

// In attention path
if (type == TQ3_4S) {
    // Only rotate Q, K is already rotated
    tq3_rotate(q_activations);
    // Use pre-rotated K from cache
}
```

Verification: Run decode probe, target 16.5+ tok/s

---

### Week 2: Fused Projection+Rotation

Create new kernel:
```cpp
__global__ void tq3_fused_proj_rot_gemv(
    const block_tq3_4s* weights,
    const float* input,  // natural domain
    float* output,
    int n) {
    // 1. Load input
    // 2. Rotate in registers (no temp buffer)
    // 3. Quantize to q8
    // 4. GEMV
}
```

Replaces: projection kernel → rotate kernel → quantize kernel → gemv kernel
With: Single fused kernel

Verification: Run decode probe, target 18.5+ tok/s

---

### Week 3: SM120 MMVQ Optimization

Modify `vecdotq.cuh`:
```cpp
// Current
#define VDR_TQ3_4S_Q8_1_MMVQ 4

// SM120 optimized
#if __CUDA_ARCH__ >= 1200
#define VDR_TQ3_4S_Q8_1_MMVQ 8
#else
#define VDR_TQ3_4S_Q8_1_MMVQ 4
#endif
```

Add vectorized scale decode:
```cpp
// Load 4 scales at once
uint32_t scales_packed = *((uint32_t*)bq->d);
float4 scales = decode_scale_vec4(scales_packed);
```

Verification: Run decode probe, target 23+ tok/s

---

### Week 4: Hybrid Q8 Attention Policy

Create mixed GGUF:
- Keep FFN in TQ3_4S (quality critical)
- Convert attention Q/K to Q8_0 (speed critical)

Tool: `llama-quantize` with mixed policy
```bash
llama-quantize --mixed-policy attn_qk=q8_0,ffn=tq3_4s model.gguf output.gguf
```

Verification: Run decode probe, target 24.5+ tok/s
Check PPL regression < 0.05

---

## Success Criteria

| Week | Target Tok/s | vs Q3_K_S | vs IQ4_XS |
|------|--------------|-----------|-----------|
| 0 (current) | 14.35 | -31% | -38% |
| 1 | 16.5 | -20% | -29% |
| 2 | 18.5 | -11% | -20% |
| 3 | 23.1 | +12% | -0.4% |
| 4 | 24.9 | +20% | **+7%** ✓ |

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| KV pre-rotation numerical issues | Medium | FP32 rotate, round to FP16; extensive testing |
| Fusion increases register pressure | Medium | Test on small models first; fallback path |
| SM120-specific correctness issues | Low | Maintain SM89 compatibility path |
| Hybrid Q8 degrades quality | Low | Tolerance: PPL +0.05 acceptable |

---

## Immediate Action Items

1. **Implement KV pre-rotation** (this week)
   - Add `tq3_rotate_inplace` kernel
   - Modify KV cache write path
   - Test on 9B model first

2. **Benchmark every change**
   - Run `llama-bench -p 0 -n 128` after each optimization
   - Compare against baseline

3. **Quality validation**
   - Run PPL on 20 chunks for every change
   - Ensure PPL regression < 0.01

---

## Why TQ3 Can Win

**IQ4_XS advantages**:
- No rotation overhead
- Mature optimized kernels
- Importance matrix weighted

**TQ3_4S advantages**:
- Better quality (6.77 vs 6.83 PPL)
- Orthogonal transform enables KV pre-rotation
- WHT decorrelation = better quantization

With optimizations above, TQ3_4S can match IQ4_XS speed while maintaining quality lead.
