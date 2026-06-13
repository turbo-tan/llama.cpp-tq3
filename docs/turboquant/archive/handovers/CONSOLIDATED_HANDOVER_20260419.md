# Consolidated Handover — Active Tracks (2026-04-20)

---

## Track A: Speculative Decode for Hybrid SSM (Qwen3.5-27B)

### Current Result

- **+28.7% avg speed** (22.7 -> 29.3 tok/s), **+60% peak** (36.3 tok/s) under relaxed gating
- Relaxed run: 10/10 correct answers, 8/10 token-exact
- Recovered exact run: **+16.7% avg speed** (21.2203 -> 24.7688 tok/s), 10/10 token-exact
- GPU shadow buffer: 0.5ms save/restore (was 26ms CPU)
- Current recovered branch: `experiment/specdecode-next-v2-20260419`
- Worktree: `/home/awee/code/worktrees/llama-specdecode-next`
- Recovery fix: default `LOSSY_NGRAM_BATCH=384`; old hardcoded `512` OOMed before the 299 MB GPU shadow allocation

### How It Works

```
1. N-gram cache lookup (4-gram → 3-gram, ~0ms)
2. If guess found + model confident → GPU shadow save (0.5ms)
3. Batch decode [cur, guess] as pp2 (63ms for 2 tokens)
4. Verify: argmax(logits[0]) == guess?
   YES → accept both, discard shadow
   NO  → GPU shadow restore (0.5ms), attn_seq_rm, re-decode cur
```

### Known Limitation

pp2 ≠ tg1+tg1 on Qwen3.5 hybrid SSM (~0.1 logit diff after 2 tokens). Confirmed across all quants. Upstream issue — not TQ3_4S specific. Drift only affects continuation phrasing, never primary answers.

### Key Files

| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/ssm-shadow.cu` | GPU D2D shadow (cudaMalloc + async copy) |
| `examples/jacobi/lossy-ngram.cpp` | Main speculation tool |
| `examples/jacobi/pp2-vs-tg1.cpp` | Divergence diagnostic |
| `src/llama-memory-recurrent.h/cpp` | Shadow API |
| `src/llama-context.cpp` | `llama_memory_attn_seq_rm` |

### Branches

- `experiment/specdecode-next-v2-20260419` — active recovered GPU-shadow branch
- `experiment/persistent-decode` — base branch for recovered work

---

## Track B: TQ3_4S Quantization

### Fixed (branch: `feature/tq3-4s-quantize-gemma4`)

Three fixes to enable quantization:
1. `ggml/src/ggml.c`: quantize dispatch + `from_float_ref`
2. `src/llama-quant.cpp`: ftype mapping + token_embd guard
3. `tools/quantize/quantize.cpp`: CLI entry

### Results

- **Qwen3.5-9B** Q8_0 → TQ3_4S: ✅ 4.35 BPW, 64.8 tok/s
- **Gemma4 26B-A4B** Q4_K_M → TQ3_4S: ✅ quantizing (background)
- **Gemma4 31B** Q8_0: download started from unsloth

---

## Track C: Decode Speed Plan X (FreeBuff Session)

### Branch: `experiment/decode-speed-plan-x`

### ✅ Implemented: Sparse V Dequant

**File**: `ggml/src/ggml-cuda/fattn-vec.cuh` (+51 lines)

- Skips V dequant for negligible attention weights
- Env var `GGML_FATTN_SPARSE_V_THRESHOLD` (default 0.0 = disabled)
- Expected: +22.8% decode at 32K context, zero PPL cost
- Thread-safe (`std::once_flag`), CUDA barrier-safe (if-gate, not continue)
- Both half2 and float paths covered
- **Status**: Build-verified, needs GPU benchmark

### ✅ Implemented: TQ3 nwarps Whitelist

**File**: `ggml/src/ggml-cuda/mmvq.cu` (+5 lines)

- RDNA4 (GFX12): TQ3_0, TQ3_1S, TQ3_4S → nwarps=8
- RDNA3 (GFX11): TQ3_0, TQ3_4S → nwarps=8
- Expected: 0–20% decode on AMD GPUs

### ❌ Corrected: Block 32→128 (plan_x was wrong)

Plan_x claimed one-line change. Actually breaks:
- `static_assert(WARP_SIZE == QK_TQ3_0)` in mmq.cuh
- MMQ tile loaders, shared memory sizing, FA kernel assumptions
- Correct approach: new type `GGML_TYPE_TQ3_0_B128` with own struct

### Remaining Plan X Items

| # | Enhancement | Status |
|---|---|---|
| 1 | Block 128 | Needs new type design |
| 2 | Turbo4 PolarQuant V | Not started |
| 3 | Sparse V Dequant | ✅ Implemented |
| 4 | Boundary V / Layer-Aware | Partially done |
| 5 | Asymmetric K/V + Turbo4 | Blocked on #2 |

---

## Active Plan

Use [ACTIVE_PROGRESS_PLAN.md](../../active/ACTIVE_PROGRESS_PLAN.md) as the single checklist-driven source of truth for current execution.

EAGLE is intentionally out of the active plan for now because it increases storage requirements without helping the current highest-priority path.

---

## Priority Next Steps

1. **Revalidate speculative decode** on current branch tip and save fresh artifacts
2. **GPU benchmark** sparse V dequant (Track C) when GPU is free
3. **Save artifact-backed conclusion** for sparse V dequant thresholds and correctness
4. **Only then** decide whether to validate boundary V or design `GGML_TYPE_TQ3_0_B128`

---

## Downloads In Progress

- Gemma4 31B Q8_0: `tail /tmp/gemma31b_download.log`
- Gemma4 26B TQ3_4S: `tail /tmp/gemma4_quant.log`

---

## Repo Layout

```
/home/awee/code/llama.cpp       — private (charpdev/t_llama.cpp)
/home/awee/code/llama.cpp-tq3   — public (stable TQ3)
/home/awee/code/tan_llama       — docs, artifacts, scripts
```

### Active Branches

| Branch | Track | Content |
|--------|-------|---------|
| `feature/tq3-4s-quantize-gemma4` | B | Quantizer fixes |
| `experiment/persistent-decode-cow-shadow` | A | GPU shadow speculation |
| `experiment/decode-speed-plan-x` | C | Sparse V + nwarps |
| `backup-all-work` | — | Full TQ3_4S CUDA kernels |
| `main` | — | TQ3_4S type + KV cache (7 ahead of public) |

---

## Track C Update (2026-04-19)

### Gemma4 TQ3_4S: 13% Slower Than Q3_K_S

| Model | Q3_K_S | TQ3_4S | Gap |
|-------|--------|--------|-----|
| PP512 | 2654 tok/s | 2100 tok/s | -21% |
| TG128 | 100 tok/s | 88 tok/s | -13% |

Root cause: NOT rotation overhead (tested: disabling rotation doesn't help).
The gap is from TQ3_4S vec_dot complexity (E3M5 decode + WHT extraction)
being heavier than Q3_K_S's simpler dequant, especially on Gemma4's small
dimensions (2816 hidden, 704 expert FFN = only 22 blocks per expert row).

Fix options:
- Mixed quantization: TQ3_4S for attention/shared FFN, Q4_K for experts
- Kernel tuning for small-K matrices (VDR/qi adjustment)
- Not a priority — Q3_K_S is fine for Gemma4 at 100 tok/s

### SuperGemma4 BF16 Conversion: In Progress

Converting from original safetensors (Jiunsong/supergemma4-26b-abliterated-multimodal).
Required 3 fixes to tensor_mapping.py + constants.py for switch_glu expert naming.
Check: `tail /tmp/convert_supergemma4.log`
Once done: quantize BF16→TQ3_4S for proper quality comparison.
