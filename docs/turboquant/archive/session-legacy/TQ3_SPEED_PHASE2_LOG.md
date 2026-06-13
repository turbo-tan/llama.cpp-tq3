# TQ3_4S Speed Phase 2 — Session Log

**Date**: 2026-04-05/06  
**Branch**: `feature/tq3-speed-phase2` (llama.cpp), `feature/tq3-speed-343` (public)  
**GPU**: RTX 5060 Ti (SM120 Blackwell)

---

## Baseline

- Build `5c10e1e68` with packed-X dp4a patch
- **PP2048: 343 tok/s** (verified in qwopus_prompt_probe_packedx_2row_20260404.txt)
- **TG128: 14.84 tok/s**
- PPL 27B (c=2048, 100ch): **6.7727** (from ppl_27b_tq3_4s_c2048_100ch_20260331_2143.txt)
- PPL 27B (c=2048, full): **6.8224** (from ppl_27b_tq3_4s_c2048_full_20260401_185504.txt)
- PPL 9B (5ch): **7.8197** (measured this session)

---

## What was tried

### 1. MMA packed-X activation staging ✅ +1.1% PP

Commit `d61a76563` — added `vec_dot_tq3_4s_q8_1_mma_packedx` which stages
activation (y) into shared memory before the j0 loop in the TURING MMA path.

**Result (27B, llama-bench r=3):**

| | PP2048 | TG128 |
|---|---|---|
| 343 baseline | 356 ± 3.5 | 15.70 ± 0.05 |
| MMA packed-X | **360 ± 0.04** | **15.82 ± 0.02** |

**PPL (9B, 5 chunks):** 7.8197 = 7.8197 (identical to baseline)

### 2. Native TQ3_4S prefill kernel ❌ 32x slower

Commit `1e4abd87c` — bypasses MMQ bridge entirely. Loads raw TQ3_4S blocks,
rotates activations with WHT, scalar dot product with warp reduce.

**Findings:**
- Scale decode bug found: TQ3_4S uses mini-float `(1 + m/32) * 2^(e-9)`,
  NOT linear `byte * constant`. Fixed with `tq3_4s_decode_scale()`.
- Standalone correctness test: PASS (max_err = 2.4e-4)
- Model PPL: valid (no NaN after scale fix) but regressed (8.63 vs 8.15 on 9B)
  due to int8 activation quantization being less precise than q8_1
- **Speed: 63.9s vs 2.0s per pass — 32x slower**
- Root cause: naive scalar warp-reduce (one lane per weight) cannot compete
  with dp4a-based MMQ. Needs cooperative predecode + dp4a inner loop.
- **Kernel disabled** (`use_tq3_4s_native = false`)

---

## PPL verification status

| Model | Test | 360 build | 343 baseline | Historical ref |
|---|---|---|---|---|
| 9B | 5 chunks | **7.8197** | **7.8197** | — |
| 9B | full pass | **9.0412** | not run | — |
| 27B | 100ch c=2048 | **TODO** | not run this session | **6.7727** |
| 27B | full c=2048 | **TODO** | not run this session | **6.8224** |

**External reference (HuggingFace):**
- Qwen3.5-9B Q4_K_M (4-bit): PPL ≈ 7.7
- Our 9B TQ3_4S (3.5-bit): PPL = 9.04 — reasonable for lower bit-width

---

## PP regression investigation

The 350 PP binary (`build-360`) was built from `5c10e1e68` with a working tree state that included uncommitted changes. A clean rebuild from the same commit gets only 236 PP.

**Root cause identified**: The 350 binary used `GGML_CUDA_FORCE_CUBLAS` path which dequantizes TQ3_4S to F32 and uses cuBLAS SGEMM. Both binaries use the same cuBLAS code, yet 350 binary gets 353 PP and current gets 238 PP with `FORCE_CUBLAS=1`. The difference is unexplained — possibly a CUDA driver cache effect or a subtle difference in the working tree state.

**Files investigated** (all ruled out as cause):
- mmvq.cu routing
- mmq.cuh MMA/dp4a functions
- mmq.cu quantizer selection
- ggml-cuda.cu routing (tq3_1s_mmq_ok, tq3_vec_prefill_ok)
- load_tiles_tq3_4s function
- CUDA graphs
- Build flags (GGML_NATIVE, GGML_CUDA_NO_PEER_COPY)

**Current clean build numbers**: PP=236, TG=24 tok/s

---

## TODO before merge to main

- [ ] Run 27B full-chunk PPL on 360 build, confirm ≈ 6.82
- [ ] Run 27B full-chunk PPL on 343 baseline, confirm ≈ 6.82
- [ ] Compare both — must be identical or within ±0.01

---

## Commits on `feature/tq3-speed-phase2`

1. `d61a76563` — perf: TQ3_4S MMA packed-X activation staging (+1.1% PP)
2. `1e4abd87c` — experiment: native TQ3_4S prefill kernel (disabled)

## Commits on public `feature/tq3-speed-343`

1. `f1a3eb354` — Checkpoint: TQ3_4S CUDA optimisations (VDR=8, scale table, packed-X, SM120 row blocking)
2. `2501722b0` — perf: TQ3_4S MMA packed-X activation staging
