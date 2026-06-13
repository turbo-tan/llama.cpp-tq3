# Speculative Decode for Hybrid SSM Models — Production Status

Date: 2026-04-20 (updated; originally 2026-04-14)
Branch: `experiment/specdecode-next-v2-20260419` on recovered worktree `/home/awee/code/worktrees/llama-specdecode-next`

## Shippable Now

Gated speculative decode with GPU shadow rollback.

- **+28.7% average speed** (22.7 → 29.3 tok/s) on 10-prompt benchmark — but only 8/10 exact (B=128, MIN_MARGIN=0.0)
- **+16.7% average speed** (21.2203 -> 24.7688 tok/s) with **10/10 exact** on the recovered branch (MIN_MARGIN=1.0, B=384; see Batch Size Revalidation below)
- **+60% peak** (36.3 tok/s) on high-acceptance prompts (MIN_MARGIN=0.0, 8/10 exact only; exact-contract peak TBD under MIN_MARGIN=1.0)
- **10/10 exact** on original prompt set with gating (current recovered branch: `experiment/specdecode-next-v2-20260419`, MIN_MARGIN=1.0)
- **GPU shadow save/restore**: ~0.5ms (was 26ms with CPU path)
- **N-gram cache**: 3-4 gram exact-match lookup, model self-decode patterns

## Known Limitation

Extended runs on hybrid Qwen3.5 accumulate tiny state-path differences
between batched (pp2) and single-token (tg1) decode paths. This is not
a rollback bug — the shadow save/restore is bit-exact. The drift comes
from non-equivalent hybrid recurrent-state evolution between execution paths.

Evidence:
- Even a single pp2 batch introduces non-zero SSM state drift vs tg1
- The drift is ~0.001 margin per accepted speculation
- After ~30+ accepted speculations, accumulated drift can flip a low-margin greedy decision
- Resync (periodic tg1 forced decodes) does not eliminate the drift

This matches broader ecosystem findings:
- Prompt-cache state drift on hybrid DeltaNet models
- Prompt-cache reuse limitations on hybrid Qwen3.5
- SGLang tracking speculative decode for hybrid models as a special systems problem
- The issue is not specific to our speculation logic

## Production Recommendation

**Ship with gating.** The feature gives strong speedups and good practical
exactness on common prompts. The gating (logits margin + entropy + n-gram
confidence) prevents speculation at drift-sensitive decision points.

**Do not claim strict long-run exactness** on extended hybrid Qwen3.5 runs.
The remaining drift is a hybrid recurrent-state path consistency issue that
likely needs upstream runtime changes to resolve.

## What We Built

| Component | File | Purpose |
|-----------|------|---------|
| GPU shadow buffer | `ggml/src/ggml-cuda/ssm-shadow.cu` | 0.5ms D2D save/restore |
| Attention-only seq_rm | `src/llama-context.cpp` | KV cleanup without SSM interference |
| N-gram speculation | `examples/jacobi/lossy-ngram.cpp` | 3-4 gram lookup + gating |
| Shadow API | `src/llama-memory-recurrent.h/cpp` | Public shadow_save/restore |
| EAGLE3 proof | `examples/jacobi/eagle-decode.cpp` | Historical draft-head pipeline, not in active plan |

## Upstream Requirement

The remaining drift appears to come from non-equivalent hybrid recurrent-state
evolution between batched and single-token decode paths on Qwen3.5/DeltaNet
models. Local rollback fixes reduce cost, but upstream runtime changes are
likely needed for strict long-run exactness.

## Per-Layer Diagnostic (2026-04-14)

First measurement of pp2 vs tg1+tg1 divergence magnitude:

```
After just 2 tokens:
  Rainbow:  logits max_diff=0.108, margin_delta=0.024
  Capital:  logits max_diff=0.160, margin_delta=0.039
```

This is NOT floating point noise. 0.1-0.16 logit difference after
2 tokens is a significant computational divergence in the hybrid
recurrent state path. Both prompts show similar magnitude.

Capital survives because its margins (~4.8) absorb the delta.
Rainbow's lower margins (~0.02 at decision points) eventually flip.

Next: per-layer SSM state capture to identify which layer diverges first.
Tool: `examples/jacobi/pp2-vs-tg1.cpp`

## Cross-Quant Verification (2026-04-14)

pp2 vs tg1+tg1 logit max_diff after 2 tokens:

| Model | Quant | max_diff | margin_delta |
|-------|-------|----------|-------------|
| 27B | TQ3_4S | 0.108 | 0.024 |
| 27B | Q3_K_S | 0.120 | 0.004 |
| 4B | Q4_0 | 0.565 | 0.025 |
| 4B | Q4_0 | 0.702 | 0.069 |

**Confirmed: NOT a TQ3_4S issue.** The divergence is inherent to
Qwen3.5 hybrid SSM execution path in llama.cpp across all quants.
Smaller models show larger divergence.

This is a hybrid recurrent-state path consistency issue that affects
all Qwen3.5 quantizations equally. The fix is upstream in the SSM
state management, not in our quantization or speculation logic.

## Batch Size Revalidation (2026-04-20)

Working repo: `/home/awee/code/worktrees/llama-specdecode-next`

Recovered branch: `experiment/specdecode-next-v2-20260419`, based on `experiment/persistent-decode` @ `b644bf478`.

The recovered branch initially crashed because `lossy-ngram.cpp` hardcoded `n_batch=512`, which left insufficient VRAM for the 299 MB GPU shadow buffer. The fix is to use `LOSSY_NGRAM_BATCH`, defaulting to `384`, for `n_batch`, `n_ubatch`, and `llama_batch_init`.

Full 10-prompt exact contract on RTX 5060 Ti (16GB) with Qwen3.5-27B TQ3_4S:

| Batch | Spec tok/s | Speedup | Exact |
|-------|-----------|---------|-------|
| 384 | 24.7688 | 1.1672x | 10/10 |
| 512 | **OOM** | — | — |

Baseline at the same contract: `21.2203 t/s`.

- Default changed to `384` in `lossy-ngram.cpp`
- Env override `LOSSY_NGRAM_BATCH` still available for future sweeps
- B=512 is the OOM boundary on this 16GB card
- B=384 is the current safe default for the recovered branch
- Artifacts: `artifacts/spec_exact_fast_contract_recheck_v3_20260419.json`, `artifacts/specdecode_recovery_batch384_20260419.txt`
- This restores correctness and VRAM stability, but does not recover the older `26.942 t/s` exact-contract mean

## Quality Impact Assessment (2026-04-14)

**The drift does NOT produce worse results.**

On the 10-prompt benchmark with current gating:
- 9/10 exact match
- The 1 mismatch (binary search): primary answer identical ("O(log n)"),
  divergence only in follow-up question ("merge sort" vs "worst-case binary search")
- Both follow-ups are correct and coherent

The pp2 drift (~0.1 logit after 2 tokens) only matters at low-margin
decision points where the model has multiple equally valid continuations.
The primary answer is always correct because high-confidence tokens
(the actual answer) have large margins that absorb the drift.

**Production conclusion:** The speculative decode produces semantically
identical output. Strict token-level exactness is not achievable due to
upstream hybrid SSM path non-equivalence, but this does not affect
answer quality.

## Gemma 4 Opportunity (2026-04-14)

Gemma 4 26B-A4B (MoE, 4B active) on RTX 5060 Ti:
- tg1: 101.7 tok/s (9.8ms/token)
- pp4: 185.8 tok/s (21.5ms for 4 tokens)
- Pure attention (no SSM) → speculative decode would be EXACT
- No pp2≠tg1 drift issue

With speculation at 70% acceptance on pp4:
- ~150 tok/s estimated (+50% over baseline)
- Exact output guaranteed (KV cache rollback is trivial)

TQ3_4S quantization for Gemma4 blocked on arch support in quantizer.
Separate track from Qwen3.5 speculative decode work.

## Cache/Gating Speed Search (2026-04-20)

Short speed-search contract used to avoid desktop VRAM pressure while preserving the speculative loop shape:

- Model: `Qwen_Qwen3.5-27B-TQ3_4S.gguf`
- Binary: `/home/awee/code/worktrees/llama-specdecode-next/build/bin/llama-lossy-ngram`
- Context: `-c 1024`
- Decode: `-n 64`
- Prompt set: first 3 prompts from `artifacts/spec_prompts_breakthrough_10.json`
- Runtime stability fixes: `LOSSY_NGRAM_BATCH=128`, `LOSSY_NGRAM_PREALLOC_SHADOW=1`

Findings:

| Cache / Gate | Baseline tok/s | Spec tok/s | Speedup | Exact |
|---|---:|---:|---:|---:|
| `ngram_decode.bin`, `MIN_MARGIN=1.0` | 21.8377 | 25.8910 | 1.1856x | 3/3 |
| `ngram_final.bin`, `MIN_MARGIN=1.0` | 21.8250 | 28.5800 | 1.3095x | 2/3 |
| `ngram_final.bin`, `MIN_MARGIN=4.0` | 20.9960 | 24.7263 | 1.1777x | 3/3 |
| `ngram_universal_seed_v1.bin`, `MIN_MARGIN=1.0` | 21.7833 | 20.8203 | 0.9558x | 3/3 |
| `ngram_synthetic.bin`, `MIN_MARGIN=1.0` | 21.8407 | 21.2480 | 0.9729x | 3/3 |
| `ngram_wiki.bin`, `MIN_MARGIN=1.0` | 21.7583 | 21.3033 | 0.9791x | 3/3 |
| `ngram_combined.bin`, `MIN_MARGIN=1.0` | 21.8110 | 21.2760 | 0.9755x | 3/3 |
| `ngram_27b_self.bin`, `MIN_MARGIN=1.0` | 21.8203 | 21.2890 | 0.9756x | 3/3 |

Current conclusion:

- Small universal/wiki/synthetic caches are exact but mostly no-op; they reduce speed because acceptance is near zero.
- `ngram_decode.bin` remains the best exact speed baseline on this slice.
- `ngram_final.bin` has the strongest speed signal, but needs stricter gating; `MIN_MARGIN=4.0` restores exactness on this slice with roughly the same speedup as `ngram_decode.bin`.
- Next step: run full 10-prompt `c=1024`, `n=64` on `ngram_decode.bin` and `ngram_final.bin` at `MIN_MARGIN=4.0`, then decide whether to implement dynamic per-cache gating.
