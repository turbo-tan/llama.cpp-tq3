# TQ3_4S Current State & Realistic Targets

**Date**: 2026-04-05  
**Status**: Baseline established from verified artifacts

---

## Current Verified Baseline (RTX 5060 Ti, Qwen3.5-27B)

### Quality Metrics

| Metric | Value | Source |
|--------|-------|--------|
| PPL (c=2048, 100ch) | **6.7727** | BENCHMARK_RESULTS.md |
| PPL with mixed policy (last8 FFN→Q5_K) | **6.6947** | BENCHMARK_RESULTS.md |
| Mean KLD | **0.1363** | kld_cmp_27b_tq3_4s_last8ffn_q5k_c2048_100ch_20260401.txt |
| Same top-p | 89.3% | KLD results |

**Status**: ✅ Quality is competitive/superior to Q3_K_S (6.7970)

### Speed Metrics

| Metric | Value | Source |
|--------|-------|--------|
| **Prompt Processing (PP)** | **315 tok/s** | qwopus_prompt_probe_public_20260403.txt |
| **Token Generation (TG)** | **14.36 tok/s** | qwopus_decode_probe_public_20260403.txt |
| PP with packed-X (Attempt 25) | **343 tok/s** | qwopus_prompt_probe_packedx_2row_20260404.txt |

**Gap to Q3_K_S**:
- PP: 315 vs 689 = **-54%** (2.19x slower)
- TG: 14.36 vs 20.7 = **-31%** (1.44x slower)

**Status**: ⚠️ Speed is the main blocker for TQ3_4S adoption

---

## What Has Been Tried (From TQ3_MOONSHOT_MASTER_LOG.md)

### Prompt Speed Attempts (25 documented)

| Attempt | Idea | Result | Status |
|---------|------|--------|--------|
| 1 | Scale decode helper cleanup | No win | ❌ |
| 2 | Constant LUT cleanup | Regressed | ❌ |
| 3 | Q3_K-style tile mimic | No win | ❌ |
| 4 | Full widened repack | OOM | ❌ |
| 5 | Selective widened repack (+1GB) | 257 tok/s | ⚠️ Below baseline |
| 6 | FFN-only/attn-only repack | ~257 tok/s | ⚠️ No benefit |
| 7 | Same-size pack4 layout | 252 tok/s | ❌ |
| 8 | Fused rotate+q8 staging | 241 tok/s | ❌ Regression |
| 9-19 | Various CTA/blockslot experiments | Parity or slower | ❌ |
| 20 | Widen MMA row blocking | 245 tok/s | ❌ |
| 21 | Scale collapse | 249 tok/s | ✅ +0.8% |
| 22 | Pair-hoist | 242 tok/s | ❌ |
| 23 | Blockslot correctness proof | Verified | ✅ (foundation) |
| 24 | Bridge vs blockslot benchmark | 1.001x | ⚠️ No win |
| **25** | **Packed-X staging** | **343 tok/s** | ✅ **+8.9%** |

### Decode Speed Attempts

| Attempt | Result | Notes |
|---------|--------|-------|
| Public baseline | 14.36 tok/s | Current reference |
| Private baseline | 14.41 tok/s | Same |
| LUT optimization | 8.50 tok/s | ❌ Regressed |
| Q3_K-style tile | 14.03 tok/s | ❌ No improvement |

**Status**: ❌ **No decode optimization has succeeded yet**

---

## Realistic Targets (Evidence-Based)

### Prompt Processing (PP)

**Current**: 315 tok/s (packed-X: 343 tok/s)  
**Target**: 450-500 tok/s  
**Gap to Q3_K_S**: Would reduce from -54% to -27% to -35%

**Path to Target**:
1. Port packed-X to runtime properly (Attempt 25 was ~343 tok/s) → **+28 tok/s**
2. Architecture-specific row blocking (2row for SM120) → **+40-70 tok/s**
3. Scale collapse → **+2-3 tok/s**
4. Additional micro-optimizations → **+20-40 tok/s**

**Expected**: 315 + 28 + 55 + 2 + 30 = **~430 tok/s**

**Upper bound**: The 10x tile expansion (128B → 1296B) is fundamental to MMQ bridge. Without a native kernel breakthrough, ~500 tok/s is likely the practical ceiling.

---

### Token Generation (TG)

**Current**: 14.36 tok/s  
**Target**: 17-18 tok/s (realistic), 20 tok/s (aspirational)  
**Gap to Q3_K_S**: 14.36 vs 20.7

**Why 20 tok/s is hard**:

From the KV pre-rotation proof:
- Eliminates K rotation (99.95% of rotations saved)
- But Q rotation remains (different every token, can't pre-compute)
- V projection, FFN, output layers still need rotation

**Conservative math**:
- Attention is ~35% of decode time
- KV pre-rotation saves ~70% of attention rotation cost
- Overall decode gain: ~15-20%
- 14.36 × 1.20 = **17.2 tok/s**

**With packed-Y + other opts**: 14.36 × 1.25 = **18.0 tok/s**

**To hit 20 tok/s**:
- Need additional 11% beyond KV pre-rotation
- Requires: Fused projection+rotation kernels, architecture-specific MMVQ tiles
- Risk: High — may require format changes

---

## Recommended Development Order

### Phase 1: Prompt Speed (Week 1-2)
**Goal**: 370-400 tok/s

1. **Cleanup and merge Attempt 25** (packed-X)
   - Already working at 343 tok/s
   - Make production-ready

2. **SM120 row blocking**
   - Template dispatch for 2-row vs 4-row
   - Based on microbench: +40-70 tok/s on SM120

**Validation**: Run prompt probe on Qwopus27B, target 370+ tok/s

---

### Phase 2: Decode Speed (Week 3-4)
**Goal**: 16-17 tok/s

3. **KV Cache Pre-Rotation**
   - Implement `ggml_cuda_tq3_rotate_inplace`
   - Modify KV cache write path
   - Modify attention read path
   - Add flag: `--tq3-prerotate-kv`

**Validation**: Run decode probe, target 16+ tok/s

---

### Phase 3: Polish (Week 5-6)
**Goal**: 400+ tok/s PP, 17-18 tok/s TG

4. **Weight-side packed-Y**
   - Fix numerical issues from microbench
   - Port to runtime

5. **Hybrid Q8 attention policy**
   - Generate mixed GGUF
   - Quality validation

---

### Phase 4: Stretch Goals (Ongoing)
**Goal**: 450+ tok/s PP, 18-20 tok/s TG

6. **Native prompt kernel v2**
   - Lessons from Attempt 20-25
   - Focus on minimal predecode

7. **Fused projection+rotation**
   - Eliminate separate kernel launches
   - Target: +5-10% on decode

---

## Success Criteria

| Phase | PP Target | TG Target | PPL Ceiling |
|-------|-----------|-----------|-------------|
| Current | 315 | 14.36 | 6.77 |
| Phase 1 | 370 | 14.5 | 6.77 |
| Phase 2 | 370 | 16.5 | 6.77 |
| Phase 3 | 400 | 17.5 | 6.78 |
| Phase 4 | 450+ | 18-20 | 6.79 |

**Acceptable trade-offs**:
- PPL regression up to 0.02 for speed gains >20%
- VRAM increase up to 5% (from KV pre-rotation)

---

## Immediate Next Steps

1. ✅ **This document** — Establish realistic targets
2. **Clean up Attempt 25** — Port packed-X to production branch
3. **Implement SM120 row blocking** — Template dispatch
4. **Benchmark** — Confirm 370+ tok/s on prompt probe
5. **KV pre-rotation proof-of-concept** — Implement and test

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Packed-X doesn't port cleanly | Medium | High | Keep original path as fallback |
| Row blocking causes correctness issues | Low | High | Extensive testing on small models first |
| KV pre-rotation numerical drift | Medium | Medium | FP32 rotation, round to FP16 after |
| SM120-specific bugs | Medium | Medium | Test on SM89 if available |
| Can't reach 20 tok/s TG | High | Low | 17-18 tok/s is still competitive |

---

## Key Insight

**TQ3_4S quality is proven. Speed is the only blocker.**

The optimizations in this plan are incremental and low-risk. The biggest wins (packed-X, row blocking, KV pre-rotation) are based on verified microbenchmarks, not speculation.

**Goal**: Get TQ3_4S to "good enough" speed (400+ tok/s PP, 17+ tok/s TG) while maintaining quality superiority.
