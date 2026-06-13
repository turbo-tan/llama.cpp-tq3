# TQ3_4S Speed Optimization Master Plan

**Date**: 2026-04-05  
**Status**: Analysis Complete, Ready for Implementation  
**Current Baseline**: 315 tok/s PP, 14.6 tok/s TG  
**Target**: 600+ tok/s PP (match Q3_K_S), 20+ tok/s TG

---

## Executive Summary

This plan consolidates 25+ attempted optimizations and identifies the highest-impact remaining opportunities. TQ3_4S achieves superior quality (PPL 6.77 vs 6.80 for Q3_K_S) but lags in speed due to:

1. **Prompt Processing**: 10x weight tile expansion in MMQ bridge (128 B → 1296 B)
2. **Token Generation**: Runtime WHT rotation overhead on each layer's activations

The optimizations below are **specifically designed for TQ3's WHT-based architecture** and leverage its unique mathematical properties.

---

## Part 1: Architecture Deep Dive - Why TQ3 Is Different

### Mathematical Foundation

```
Standard Quantization:  w · x  (direct dot product)
TQ3 Quantization:     R(w) · R(x)  where R = WHT rotation

R is orthogonal: <w, x> = <R(w), R(x)>
```

**Key Insight**: WHT is a fixed linear transform. Unlike learned transforms (QuIP#), it:
- ✅ Can be precomputed (weights rotated offline)
- ❌ Cannot be skipped for activations (computed online)
- ✅ Has O(n log n) complexity via butterfly (not O(n²))
- ❌ Requires 32-element collaborative warp operations

### Current Pipeline Bottlenecks

```
PROMPT PATH (slow):
  Activations [float] 
    → rotate (WHT) → temp buffer [float]
    → quantize → q8_1
    → expand to MMQ tile (10.1x expansion!)
    → MMA GEMM
    
DECODE PATH (slow):
  Token embedding [float]
    → layer 0: rotate → GEMM → output
    → layer 1: rotate → GEMM → output
    ...
    → layer N: rotate → GEMM → logits
```

---

## Part 2: Optimization Opportunities (Prioritized)

### Tier 1: High Impact, Verified Microbenchmarks

#### 1. Architecture-Specific Row Blocking ⚡ **HIGHEST PRIORITY**

**Problem**: SM120 (Blackwell) has different optimal row blocking than Ampere

**Evidence** (from bench_tq3_multrow_cta_sm120_20260404.txt):
```
SM120 (RTX 5060 Ti):
  2row_packedx: 1.478x speedup ✓ WINNER
  4row_packedx: 1.308x speedup (register pressure)
  
Older arch:
  4row_packedx: 1.693x speedup ✓ WINNER
```

**Solution**:
```cpp
// In mmq.cuh - template specialization
template<int arch>
struct tq3_row_blocking {
    static constexpr int value = 4;  // default
};

template<>
struct tq3_row_blocking<120> {  // SM120 Blackwell
    static constexpr int value = 2;
};

// Usage
constexpr int ROWS = tq3_row_blocking<__CUDA_ARCH__ / 10>::value;
```

**Expected Gain**: +40-70% on SM120 (current dev platform)
**Implementation Cost**: Low (template dispatch)
**Risk**: Low (already verified in microbench)

**TQ3 Benefit**: ⭐⭐⭐⭐⭐ (Directly leverages packed-x which is proven)

---

#### 2. Activation-Side Packed-X Staging (Attempt 25 Verified)

**Problem**: Weight tile loader re-decodes TQ3_4S blocks every K-iteration

**Current** (slow):
```cpp
for each K-slab:
  for each row:
    load_tiles_tq3_4s()  // re-decode from global memory every time
```

**Optimized**:
```cpp
// Stage packed4 activation blocks once in shared mem
__shared__ int packed_x[ROWS][TILE_K/4];

// Producer warps: load and pack activations
// Consumer warps: reuse packed_x across all rows
```

**Evidence**: Already verified at 343 tok/s (1.055x gain)

**Expected Gain**: +5-10% additional on top of row blocking
**Implementation Cost**: Medium (already partially implemented)
**Risk**: Low (proven in runtime)

**TQ3 Benefit**: ⭐⭐⭐⭐⭐ (Core to TQ3's packed4 micro-architecture)

---

#### 3. Scale Collapse Optimization (Attempt 21 Verified)

**Problem**: `load_tiles_tq3_4s` writes same subgroup scale twice (for each 4-value half)

**Current**:
```cpp
// 8 values per subgroup, stored as two 4-value packs
float dA0 = scale[subgroup];  // first half
float dA1 = scale[subgroup];  // second half (same value!)
sum += C0*dA0 + C1*dA1;
```

**Optimized**:
```cpp
// Collapse: (C0 + C1) * dA
sum += (C0 + C1) * dA;
```

**Evidence**: 249 tok/s vs 247 tok/s baseline (1.008x)

**Expected Gain**: +0.8% (small but free)
**Implementation Cost**: Low (one-line change)
**Risk**: None (already verified)

**TQ3 Benefit**: ⭐⭐⭐⭐☆ (TQ3-specific, leverages 4-scale-per-block layout)

---

### Tier 2: Medium Impact, Requires Implementation

#### 4. KV Cache Pre-Rotation 🎯 **BIGGEST DECODE OPPORTUNITY**

**Problem**: K/V tensors are rotated every time they're used in attention
**Reality**: Each KV entry is used `seq_len` times during generation

**Current Attention** (slow):
```cpp
for each token in cache:
    k_rotated = rotate(k_cache[token])  // every access!
    attn_score = query · k_rotated
```

**Optimized Attention**:
```cpp
// At KV computation time (once per token)
k_cache[token] = rotate(compute_k(x))

// At attention time (reuse rotated)
for each token in cache:
    attn_score = query_rotated · k_cache[token]  // already rotated!
```

**Mathematical Justification**:
```
Attention(Q, K, V) = softmax(Q·K^T / √d) · V

With rotation:
  = softmax(R(Q)·R(K)^T / √d) · R(V)  // if we rotate Q, K, V
  
But we need unrotated output, so:
  Output = R^{-1}(softmax(R(Q)·R(K)^T / √d) · R(V))
  
Actually, better: rotate Q only, use pre-rotated K:
  = softmax(R(Q)·K_cache_rotated^T / √d) · V
```

**Implementation**:
```cpp
// In llama.cpp attention computation
// File: ggml/src/ggml-cuda/attn.cu or similar

// Compute K and V (already done)
struct ggml_tensor * k = compute_k(ctx, x);
struct ggml_tensor * v = compute_v(ctx, x);

// NEW: Rotate K before caching (one-time cost)
if (ctx->model.type == TQ3_4S) {
    ggml_cuda_tq3_rotate_inplace(k);  // new kernel
    // V doesn't need rotation for standard attention
}

// Cache the rotated K
kv_cache.store(layer, head, pos, k, v);

// At attention time: rotate Q, use cached K directly
struct ggml_tensor * q_rotated = ggml_cuda_tq3_rotate(q);
struct ggml_tensor * k_cached = kv_cache.get(layer, head, pos);

// Dot product: no rotation needed for K!
scores = ggml_cuda_dot(q_rotated, k_cached);
```

**Expected Gain**:
- Decode: +30-50% (attention is 30-40% of decode time)
- Prompt: Minimal (K computed once per prompt anyway)

**Implementation Cost**: Medium (requires attention path modification)
**Risk**: Medium (requires correctness validation, changes KV cache format)

**TQ3 Benefit**: ⭐⭐⭐⭐⭐ **MASSIVE** - This is unique to TQ3's rotation-based design. Other quants (Q3_K_S, IQ4_XS) can't do this!

---

#### 5. Weight-Side Packed-Y (Corrected)

**Problem**: Similar to packed-X, but for weight loading

**Evidence** (from microbench):
```
compact_coop_packed_y_corrected_cta: 2.230 us/launch (0.992x, nearly parity)
```

The issue was numerical — scales applied in wrong order. With correction:
- Expected parity or slight win
- Combines with packed-X for additive benefit

**Implementation**:
```cpp
// In load_tiles_tq3_4s
// Stage weight scales in shared mem alongside packed indices
__shared__ float weight_scales[ROWS][TILE_K/8];
```

**Expected Gain**: +2-5% (additive with packed-X)
**Implementation Cost**: Medium (requires correctness proof)
**Risk**: Low (microbench shows path is viable)

**TQ3 Benefit**: ⭐⭐⭐⭐☆ (Reduces weight load traffic)

---

#### 6. Scale Prefetch with Double Buffering

**Problem**: Scale decode has ~4 cycle latency, stalls pipeline

**Solution**: Prefetch next scale while computing current

```cpp
float rms = decode_scale(current);
#pragma unroll
for (int k = 0; k < TILE_K; k += 8) {
    float rms_next = decode_scale(k + 8);  // prefetch
    compute_with(rms);
    rms = rms_next;  // rotate
}
```

**Expected Gain**: +3-5%
**Implementation Cost**: Low
**Risk**: Medium (register pressure increase)

**TQ3 Benefit**: ⭐⭐⭐☆☆ (Generic optimization, not TQ3-specific)

---

### Tier 3: Strategic / Architectural Changes

#### 7. Hybrid TQ3+Q8 Attention Policy

**Insight**: Attention Q/K projections have different access patterns than FFN

**Policy**:
- FFN weights: TQ3_4S (heavy compute, benefits from WHT decorrelation)
- Attention Q/K: Q8_0 (mature fast path, attention softmax smooths noise)
- Attention V/Output: TQ3_4S or Q8_0 (experiment)

**Evidence**: Q3_K_S uses mixed policy (different bits for different layers)

**Expected Gain**: +20-30% on prompt (attention becomes fast path)
**Implementation Cost**: Medium (requires format conversion, quantization support)
**Risk**: Low (Q8_0 is well-tested)

**TQ3 Benefit**: ⭐⭐⭐⭐☆ (Keeps TQ3 quality where it matters, fast path elsewhere)

---

#### 8. Native TQ3 Prompt Kernel (No MMQ Bridge)

**Current**: TQ3 → expand to MMQ tile → generic MMA
**Target**: TQ3 → native dp4a → direct accumulation

**Design** (from TQ3_NATIVE_PROMPT_KERNEL_DESIGN.md):
```cpp
// One-row native slab kernel
__global__ void tq3_native_prompt_kernel(
    const block_tq3_4s* weights,
    const float* activations,
    float* output) {
    
    // Producer: rotate + quantize activation slab to q8
    __shared__ int8_t act_q8[256];
    rotate_and_quantize(activations, act_q8);
    
    // Consumer: load TQ3 weights, decode to packed4, dp4a
    int32_t acc = 0;
    #pragma unroll
    for (int k = 0; k < 256; k += 4) {
        uint32_t packed4 = load_packed4(weights, k);
        acc = __dp4a(packed4, *(uint32_t*)&act_q8[k], acc);
    }
    
    // Apply scales and write back
    output[col] = acc * dA * dW;
}
```

**Challenge**: CTA microbench shows this is NOT automatically faster:
```
native_cta:           4.343 us/launch (0.511x of bridge!)
compact_predecode_cta: 4.377 us/launch (0.507x)
```

**Lesson**: Native isn't automatically faster. The 10x tile expansion is wasteful, but the MMQ path is highly optimized.

**Path Forward**: 
- Keep MMQ structure
- Inject TQ3-specific tile loading (packed-X, scale collapse)
- Don't try to rewrite the whole kernel

**Expected Gain**: Unknown (needs experimentation)
**Implementation Cost**: High
**Risk**: High (previous native attempts failed)

**TQ3 Benefit**: ⭐⭐☆☆☆ (Theoretical best, but hard to achieve)

---

#### 9. CUDA Graphs for Async Rotation

**Problem**: Kernel launch overhead + rotation stalls next layer

**Solution**: Capture full layer execution in CUDA Graph

```cpp
// Capture once
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaGraphCaptureModeGlobal);
rotate_kernel<<<...>>>(...);
gemm_kernel<<<...>>>(...);
cudaStreamEndCapture(stream, &graph);

// Replay many times
cudaGraphLaunch(graph, stream);
```

**Expected Gain**: +10-20% on small batches (launch overhead amortization)
**Implementation Cost**: High (requires graph-safe memory management)
**Risk**: High (llama.cpp memory model may not support graphs)

**TQ3 Benefit**: ⭐⭐⭐☆☆ (Generic, but helps hide TQ3 rotation cost)

---

#### 10. Subgroup Scale Delta Encoding (Format Change)

**Problem**: 4 scales per block (4 bytes) often correlated

**Solution**: Store 1 mean + 4 deltas:
```cpp
struct block_tq3_4s_v2 {
    uint8_t d_mean;     // block mean scale
    int8_t  d_delta[4]; // per-subgroup delta
    uint8_t qs[12];     // 3-bit indices (unchanged)
};
```

**Memory**: 5 bytes → ~2.5 bytes (estimate)

**Expected Gain**: +5-10% (memory bandwidth reduction)
**Implementation Cost**: High (requires new format, conversion tools)
**Risk**: High (quality impact unknown)

**TQ3 Benefit**: ⭐⭐⭐☆☆ (Format optimization, not unique to TQ3)

---

## Part 3: TQ3-Specific Benefit Analysis

| Optimization | Generic? | TQ3 Unique? | Impact |
|-------------|----------|-------------|--------|
| Row blocking | No | Yes (SM-specific) | High |
| Packed-X | No | Yes (3-bit specific) | High |
| Scale collapse | No | Yes (4-scale layout) | Low |
| **KV Pre-rotation** | **No** | **⭐ YES** | **MASSIVE** |
| Packed-Y | No | Yes | Medium |
| Scale prefetch | Yes | No | Low |
| Hybrid policy | Yes | Partial | Medium |
| Native kernel | No | Yes | Unknown |
| CUDA Graphs | Yes | No | Medium |
| Delta encoding | Yes | No | Low |

**Key Insight**: TQ3's WHT rotation is normally a **cost**, but KV pre-rotation turns it into an **asset** — something only TQ3 can do effectively (because rotation is fixed/orthogonal, not learned).

---

## Part 4: Implementation Roadmap

### Phase 1: Quick Wins (Week 1)

1. **Merge verified optimizations** (Attempt 25 + 21)
   - Packed-X activation staging
   - Scale collapse
   - Status: Already working, needs cleanup

2. **Architecture-specific row blocking**
   - Implement SM120 detection
   - Template dispatch for 2-row vs 4-row
   - Benchmark on RTX 5060 Ti

**Expected Result**: 350-380 tok/s prompt (from 315)

---

### Phase 2: Decode Speed (Week 2-3)

3. **KV Cache Pre-Rotation**
   - Implement `ggml_cuda_tq3_rotate_inplace`
   - Modify attention path to pre-rotate K
   - Add runtime flag: `--tq3-prerotate-kv`
   - Validate correctness on test suite
   
**Expected Result**: 18-22 tok/s decode (from 14.6)

---

### Phase 3: Polish (Week 4)

4. **Packed-Y correction**
   - Fix numerical issues from microbench
   - Port to runtime
   
5. **Hybrid Q8 attention policy**
   - Generate mixed GGUF with Q8 for attn Q/K
   - Validate quality impact

**Expected Result**: 400-450 tok/s prompt, 20-24 tok/s decode

---

### Phase 4: Moonshot (Future)

6. **Native prompt kernel v2**
   - Based on lessons from Attempt 20-25
   - Focus on minimal predecode, not full native
   
7. **CUDA Graphs**
   - Requires llama.cpp architectural changes
   - Evaluate feasibility

**Expected Result**: 500-600 tok/s prompt (match Q3_K_S)

---

## Part 5: Success Metrics

| Metric | Current | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|---------|---------|---------|---------|---------|
| PP tok/s | 315 | 370 | 370 | 425 | 550 |
| TG tok/s | 14.6 | 14.6 | 20 | 22 | 24 |
| PPL (27B) | 6.77 | 6.77 | 6.77 | 6.78* | 6.78* |
| VRAM (27B) | 12.9 GB | 12.9 GB | 12.9 GB | 13.2 GB* | 13.2 GB* |

*Slight degradation expected with hybrid policy, within acceptable range

---

## Part 6: Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Numerical issues | Maintain exact reference implementation, compare bit-for-bit |
| Quality regression | Run full PPL suite on each change |
| VRAM increase | Make optional via flags (e.g., `--tq3-prerotate-kv`) |
| SM-specific bugs | Test on multiple arch if possible (SM89, SM120) |
| Merge conflicts | Small incremental PRs, not monolithic changes |

---

## Appendix A: Verification Checklist

For each optimization:

- [ ] Microbenchmark validates speedup
- [ ] Bit-exact or near-exact (diff < 1e-5) vs reference
- [ ] PPL regression < 0.01 on 27B model
- [ ] Works on both SM89 and SM120
- [ ] Memory usage within 5% of baseline

## Appendix B: Reference Files

- `TQ3_MOONSHOT_MASTER_LOG.md` — Attempt history
- `TQ3_Q3KS_BOTTLENECK_MAP.md` — Pipeline analysis
- `TQ3_NATIVE_PROMPT_KERNEL_DESIGN.md` — Kernel design notes
- `bench_tq3_multrow_cta_sm120_20260404.txt` — SM120 benchmarks
- `mmq.cuh` — Core MMQ implementation

---

## Conclusion

TQ3_4S can achieve parity with Q3_K_S on speed while maintaining quality superiority. The path requires:

1. **Immediate**: Port verified microbenchmark wins (packed-X, row blocking)
2. **Short-term**: KV pre-rotation — TQ3's unique advantage
3. **Medium-term**: Hybrid quantization policy

The mathematical properties of WHT (orthogonality, fixed transform) that give TQ3 its quality advantage also enable optimizations impossible with learned transforms.
