# TQ3_4S Decode Speed Improvement Plan

**Date:** 2026-04-16  
**Branch:** `experiment/decode-speed`  
**GPU:** RTX 5060 Ti 16GB (SM 120a, 288 GB/s)  
**Current:** 23 tok/s (27B dense), ~107 tok/s (35B MoE)  
**Target:** 2x–4x improvement

---

## The Fundamental Limitation: Memory Bandwidth

Decode (single-token generation) is **memory-bandwidth-bound**, not compute-bound.  
Each generated token requires reading ALL model weights once.

### Theoretical Ceiling Calculation

| Model | Total Weights | Active Weights | Bytes/Token (TQ3_4S) | BW Limit (288 GB/s) | Current tok/s | Ceiling |
|-------|--------------|----------------|----------------------|---------------------|--------------|---------|
| Qwen3.5-27B dense | 27B | 27B | 13.5 GB | 21.3 tok/s | 23.0 | **1.08x** of ceiling |
| Qwen3.6-35B-A3B MoE | 35B | 3B | 1.5 GB | 192 tok/s | ~107* | **0.56x** of ceiling |
| Qwen3.5-9B dense | 9B | 9B | 4.5 GB | 64 tok/s | ~42* | ~0.66x |

\*Estimated from historical benchmarks

### Key Insight

**The 27B dense model is already AT the memory-bandwidth ceiling.** At 23 tok/s with 13.5 GB weights and 288 GB/s bandwidth, we're at 96% of theoretical maximum. **No kernel optimization can make it faster.**

The only way to get 2x-4x on a dense model is:
1. **Reduce weight size** (lower bpw → TQ2 at 2 bpw, but quality tanks)
2. **Speculative decoding** (draft model verifies multiple tokens per forward pass)
3. **Use MoE models** (only read a fraction of weights per token)

**For MoE models, there IS significant headroom.** The Qwen3.6-35B-A3B at ~107 tok/s is only at 56% of the 192 tok/s ceiling. Kernel optimizations CAN help here.

---

## Strategy A: MoE-First Optimization (2x potential on MoE)

Qwen3.6-35B-A3B MoE only activates 3B of 35B parameters per token. The TQ3_4S weight matrix reads are ~1.5 GB/token — well within bandwidth budget. The bottleneck shifts to **kernel launch overhead, MoE routing, and KV cache**.

### A1. TQ3_4S in nwarps=8 whitelist (NVIDIA GENERIC)

**Problem:** TQ3_4S is NOT in the `nwarps=8` whitelist for any architecture. On GENERIC (NVIDIA), TQ3_4S gets `nwarps=4` for `ncols_dst=1`, while Q4_0, Q8_0, etc. get `nwarps=4` as well. On RDNA4, simple types get `nwarps=8` but TQ3_4S falls to `nwarps=1`.

**Change:** Add `GGML_TYPE_TQ3_4S` (and `TQ3_1S`) to the `nwarps=4` case for GENERIC NVIDIA. Test whether `nwarps=8` works on SM120 without excessive register pressure.

**Expected gain:** 0-20% on single-token decode. May hurt on MoE multi-token (ncols_dst>1) if register pressure causes spills. **Must test empirically.**

**File:** `ggml/src/ggml-cuda/mmvq.cu` → `calc_nwarps()` function

**Risk:** Low. If `nwarps=8` causes regression, just use `nwarps=4` (same as current). Whitelist addition is purely additive.

### A2. Fuse activation WHT rotation into quantize kernel

**Problem:** TQ3 types require WHT rotation (`ggml_cuda_tq3_rotate_act`) before quantization to Q8_1. This is a separate kernel launch — adds latency on the critical decode path.

**Current flow (2 kernel launches):**
```
1. memcpy activation to temp buffer
2. WHT rotation kernel (tq3_rotate_act_kernel)
3. quantize F32→Q8_1 kernel
4. MMVQ kernel
```

**Proposed flow (1 fused kernel):**
```
1. memcpy activation to temp buffer
2. Fused WHT+quantize kernel (single launch)
3. MMVQ kernel
```

**Expected gain:** 5-10% by eliminating one kernel launch and its associated overhead (synchronization, L2 flush).

**File:** `ggml/src/ggml-cuda/mmvq.cu` → modify the TQ3 path in `ggml_cuda_mul_mat_vec_q()`

**Risk:** Medium. Fused kernels are harder to debug. Must verify PPL unchanged.

### A3. Scale-baking for MMVQ decode (adapt MMQ trick)

**Problem:** The MMVQ decode path uses `vec_dot_tq3_4s_q8_1()` which does per-subgroup E3M5 scale decode + centroid lookup + dp4a. The MMQ (prefill) path uses `load_tiles_tq3_4s()` which "bakes" all 4 subgroup scales into int8 values, then uses fast Q8_0×Q8_1 dot products.

**Idea:** Create a new MMVQ kernel that does scale baking for decode:
1. For each row, load the full TQ3_4S block
2. Decode all 4 E3M5 scales, compute `d_block`
3. Bake centroids × rms_g / d_block → int8
4. Dot the baked int8 values against Q8_1 activation using dp4a

**Why this could help:** The MMQ baking approach eliminated the per-subgroup `ldexpf()` calls and the `2.1519/127` correction factor. In MMVQ, each thread currently does `tq3_4s_ratio4s()` per subgroup — if we pre-bake once per block (4 subgroups), we reduce scale decode overhead from 4× to 1×.

**Expected gain:** 5-15% on decode. The main savings is eliminating repeated E3M5 decode and simplifying the dp4a scale path.

**File:** New `vec_dot_tq3_4s_q8_1_baked()` in `vecdotq.cuh`, new dispatch in `mmvq.cu`

**Risk:** High. This is a significant kernel rewrite. Must verify bit-exact PPL (the int8 rounding error from baking is <0.4% per element but accumulates over 5120 dimensions).

---

## Strategy B: Speculative Decoding (2x on dense, 1.5x on MoE)

This is the highest-impact approach for dense models where we're at the bandwidth ceiling.

### B1. Draft-model speculative decoding

Use a small draft model (e.g., Qwen3.5-4B TQ3_4S = 2 GB) to generate 4-8 candidate tokens, then verify them in a single forward pass of the 27B target model.

**Math:** If draft model acceptance rate is ~70% (typical for same-family models):
- 4-token speculation: expected 2.8 tokens accepted per forward pass → ~2.5x speedup
- 8-token speculation: expected 3.5 tokens accepted → ~2.0x speedup (diminishing returns)

**Implementation:** llama.cpp already has speculative decoding support (`llama-speculative`). Need to:
1. Convert Qwen3.5-4B to TQ3_4S (already exists: `Qwen_Qwen3.5-4B-TQ1_0.gguf`)
2. Launch draft + target model simultaneously
3. Wire up the speculative decode loop

**Expected gain:** 2-2.5x on 27B dense (23 → 50-58 tok/s). 1.3-1.5x on MoE (already fast).

**Risk:** Low (uses existing infrastructure). Requires 2 GB extra VRAM for draft model.

### B2. EAGLE3-style speculative decoding (weight-based)

Status: historical research direction only. Not part of the active plan while storage remains constrained.

Use the target model's own embeddings + a small MLP head to generate draft tokens. No separate model needed.

**Advantage:** No extra VRAM. Draft quality matches target model closely → higher acceptance rate.

**Implementation:** Need to:
1. Train a small draft head (2-3 layers) on the TQ3_4S model's hidden states
2. Integrate with llama.cpp speculative framework
3. The draft head runs in FP16 (tiny) and generates tokens using the same KV cache

**Expected gain:** 2-3x on dense, 1.5x on MoE. Higher acceptance rate than B1.

**Risk:** High. Requires training data and custom integration. Not a drop-in solution.

---

## Strategy C: MoE-Specific Optimizations (for Qwen3.6-35B-A3B)

The MoE model has different bottlenecks than dense models. With only 3B active parameters, the compute is light but the routing/expert-swap overhead dominates.

### C1. Expert prefetch / pipelining

While expert A is computing, prefetch expert B's weights. Overlap MoE routing with weight loads.

**Expected gain:** 10-30% on MoE decode (reduce idle time between expert dispatches).

**Risk:** Medium. Requires modifying the MoE dispatch path in ggml-cuda.

### C2. Expert weight caching in L2 / shared memory

Hot experts (used frequently) can be cached in GPU L2. The 35B model has 96 experts × ~340M params each. With top-2 routing, only 2 experts are active per token.

On RTX 5060 Ti, L2 cache is 48 MB. One expert's TQ3_4S weights are ~170 MB — too large for L2. But the **attention layers** are shared across all experts and could be cached.

**Expected gain:** 5-15% on MoE (reduces attention weight re-reads).

**Risk:** Low. L2 cache control hints are non-invasive.

### C3. Reduce KV cache footprint for MoE

MoE models have the same attention structure as dense models but with smaller active dimension. KV cache compression (TQ3_0, Q4_0) frees bandwidth for weight reads.

**Already implemented:** TQ3_0 KV cache support exists. Just needs testing on Qwen3.6.

---

## Strategy D: Long-Term Architectural Changes (4x+ potential)

### D1. TQ3_4S with 2-bit quantization for cold experts (MoE only)

In MoE models, most experts are rarely activated. Use TQ2 (2 bpw) for cold experts and TQ3_4S (4 bpw) for hot experts. Mixed-precision per-expert quantization.

**Expected gain:** 30-50% model size reduction → more fits in cache → effectively 1.3-1.5x speed improvement.

**Risk:** High. Requires per-expert quantization pipeline and dispatch logic.

### D2. Flash Decoding / split-K for single-token decode

Currently, MMVQ processes one row per warp (ncols_dst=1). Flash Decoding splits the K dimension across multiple warps, then reduces. This can increase occupancy for large models.

**Expected gain:** 10-25% on dense decode (better GPU utilization for large hidden dims).

**Risk:** Medium. Needs custom kernel for TQ3_4S split-K.

---

## Implementation Priority (Ordered by Impact × Feasibility)

| # | Strategy | Model | Expected Gain | Complexity | Risk | Priority |
|---|----------|-------|--------------|------------|------|----------|
| 1 | **B1: Draft-model speculative decoding** | 27B dense | **2-2.5x** | Low | Low | 🔴 Critical |
| 2 | **A1: TQ3_4S nwarps whitelist** | All | 0-20% | Trivial | Low | 🟡 Quick win |
| 3 | **A2: Fuse WHT+quantize** | All | 5-10% | Medium | Medium | 🟢 Worthwhile |
| 4 | **C1: Expert prefetch** | MoE | 10-30% | Medium | Medium | 🟢 MoE-specific |
| 5 | **A3: Scale-baking for decode** | All | 5-15% | High | High | 🟠 Risky |
| 6 | **C2: Expert weight caching** | MoE | 5-15% | Low | Low | 🟡 Nice-to-have |
| 7 | **D2: Flash Decoding split-K** | Dense | 10-25% | High | Medium | 🟠 Later |
| 8 | **B2: EAGLE3 draft head** | Dense | 2-3x | High | High | Historical research |
| 9 | **D1: Mixed-precision experts** | MoE | 30-50% size | High | High | 🔵 Future |

---

## Action Plan for This Branch

### Phase 1: Quick Wins (1-2 hours)
1. Add TQ3_4S + TQ3_1S to nwarps=4 whitelist for NVIDIA GENERIC
2. Rebuild, warmup benchmark, compare TG128
3. If regression, revert; if improvement, commit

### Phase 2: Fuse WHT+Quantize (2-4 hours)
1. Create fused `tq3_rotate_and_quantize_q8_1` kernel
2. Replace the two-launch path in `ggml_cuda_mul_mat_vec_q()`
3. Verify PPL unchanged (10-chunk minimum)
4. Benchmark TG128

### Phase 3: Speculative Decoding (4-8 hours)
1. Convert Qwen3.5-4B to TQ3_4S (or use existing TQ1_0 as draft)
2. Configure speculative decode with 4-token speculation
3. Benchmark 27B dense: expect 50-58 tok/s
4. Benchmark 35B MoE: expect 130-160 tok/s

### Phase 4: MoE Optimizations (if Phase 3 shows MoE headroom)
1. Profile MoE decode to find actual bottleneck (routing? expert swap? attention?)
2. Implement expert prefetch if routing dominates
3. Test TQ3_0 KV cache on Qwen3.6-35B-A3B

### Validation Gate
After each phase:
1. **PPL gate:** 10-chunk minimum, must be within 0.1 of baseline
2. **Chat quality gate:** 7/8 SOP minimum
3. **No garbage output** (mandatory)

---

## Qwen3.6-35B-A3B Model Status

Model files on disk:
- `/home/awee/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-TQ3_4S.gguf`
- `/home/awee/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf`
- `/home/awee/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_S.gguf`
- `/home/awee/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Small-TQ3_4S.gguf`

The TQ3_4S quantized version already exists and should be the primary target for MoE decode optimization.

---

## Screenshots from 2026-04-16

23 screenshots captured between 20:32–20:59 (could not OCR — no tesseract installed, GlmOcr requires API key). These likely show the user's benchmark results or reference speed numbers from other quantization methods. Will need user to describe content if critical for the plan.
