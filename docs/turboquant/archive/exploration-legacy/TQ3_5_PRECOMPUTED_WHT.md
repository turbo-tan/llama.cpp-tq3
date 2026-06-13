# TQ3_5: Precomputed WHT for Speed

Date: 2026-04-01

## Objective

Remove runtime WHT overhead by precomputing the transform during quantization.

**Current TQ3_4S flow:**
```
Quantize:   x → WHT → quantize → store
Dequantize: load → dequantize → WHT_inverse → output
```

**TQ3_5 flow:**
```
Quantize:   x → WHT → quantize → store (same)
Dequantize: load → dequantize → output (no WHT inverse!)
```

## Speed Target

- Current TQ3_4S: 310 tok/s (27B prefill)
- Target TQ3_5: 500-600 tok/s (1.6-2x speedup)
- Q3_K_S baseline: 689 tok/s

**Bottleneck removed:** 5 shuffle stages × 32 elements = 160 ops per block

## Implementation Plan

### Phase 1: New Format Tag (TQ3_5)

Block structure (same as TQ3_4S):
```c
typedef struct {
    ggml_half d[4];      // 4 scales (fp16)
    uint8_t qs[12];      // 32 × 3-bit indices packed
} block_tq3_5;
```

**Key difference:** Weights are stored in WHT-transformed domain

### Phase 2: Quantizer

```c
void quantize_row_tq3_5_ref(const float * x, block_tq3_5 * y, int64_t k) {
    // Apply WHT
    tq3_0_rht_forward(x, rotated);
    
    // Quantize in rotated domain (same as TQ3_4S)
    // ... per-group scale + centroid selection ...
    
    // Store rotated weights directly (no inverse)
}
```

**Cost:** One-time during quantization (~5-10 min for 27B)

### Phase 3: Dequantizer

```c
void dequantize_row_tq3_5(const block_tq3_5 * x, float * y, int64_t k) {
    // Unpack indices
    uint8_t idx[32];
    unpack_3bit(x->qs, idx);
    
    // Lookup centroids + scale
    for (int i = 0; i < 32; ++i) {
        y[i] = TQ3_0_CENTROIDS[idx[i]] * scale[i/8];
    }
    
    // NO WHT INVERSE - output is in rotated domain!
}
```

**Speedup:** Removes 5 shuffle stages (160 ops) per 32 elements

### Phase 4: Activation Rotation

Since weights are in WHT domain, activations must also be rotated:

```cuda
// Before matmul: rotate activations
x_rotated = WHT(x);

// Matmul in rotated domain
y_rotated = W_rotated @ x_rotated;

// After matmul: inverse rotate output
y = WHT_inverse(y_rotated);
```

**Key insight:** Activation rotation is already implemented for TQ3_4S!
Just need to skip the weight-side WHT inverse.

## Quality Impact

**Expected:** Minimal to none
- Same quantization algorithm
- Same centroids and scales
- Only difference: no runtime WHT inverse

**Potential issue:** Numerical precision
- TQ3_4S: fp32 WHT → quantize → fp32 WHT_inverse
- TQ3_5: fp32 WHT → quantize → (no inverse, stays in rotated domain)

The rotated domain is mathematically equivalent, so quality should be identical.

## Implementation Steps

1. **Add GGML_TYPE_TQ3_5** to ggml.h
2. **Copy TQ3_4S quantizer**, remove WHT inverse from dequant
3. **Test on small model** (Qwen3.5-9B)
4. **Benchmark speed** on 27B
5. **Verify PPL** matches TQ3_4S

## Expected Timeline

- Implementation: 2-3 hours
- Testing: 1 hour
- Benchmarking: 2 hours
- **Total: ~6 hours**

## Risks

1. **Activation rotation complexity**: Need to ensure all matmul paths handle rotated activations
2. **Numerical stability**: Staying in rotated domain might accumulate errors
3. **Compatibility**: Existing TQ3_4S models won't work with TQ3_5 runtime

## Fallback

If quality degrades:
- Keep TQ3_4S as-is
- Focus on kernel optimization instead (fused dequant+matmul)
