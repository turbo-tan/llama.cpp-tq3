# Speculative Decode for Hybrid SSM Models — Full Handover

Date: 2026-04-14
Current progress updated: 2026-04-20
Branch: `experiment/persistent-decode` @ `36d6a9309` on `charpdev/t_llama.cpp`
Current recovered branch: `experiment/specdecode-next-v2-20260419` from `experiment/persistent-decode` @ `b644bf478`

## Summary

Built a speculative decode system for Qwen3.5-27B hybrid (SSM + attention). Best recorded speedup: **+38.7%** (MIN_MARGIN=0.0, 8/10 exact). Current recovered exact-contract configuration (MIN_MARGIN=1.0, B=384): **+16.7%** with 10/10 exact. It also recorded an EAGLE3 draft-head experiment, but that work is historical and not part of the current active plan.

## What Ships Today

### N-gram Cache + seq_cp Fork: +38.7% on 27B (MIN_MARGIN=0.0, 8/10 exact)

> **Note:** The +38.7% was with relaxed exactness (MIN_MARGIN=0.0). The current exact-contract
> recovered configuration (MIN_MARGIN=1.0, B=384) gives **+16.7%** with 10/10 exact — see the Batch Recovery section below.

```bash
# Capture decode patterns (one-time, ~2 hours)
NO_SPEC=1 NGRAM_SAVE=cache.bin llama-lossy-ngram -m MODEL -p "prompt" ...
# (chain across many prompts with NGRAM_LOAD + NGRAM_SAVE)

# Run with speculation
NGRAM_STATIC=cache.bin MIN_MARGIN=4.0 llama-lossy-ngram -m MODEL -p "prompt"
```

| Prompt | Baseline | Speculative | Delta |
|--------|----------|-------------|-------|
| Capital of France | 23.5 | 36.2 | +54% |
| Hash table | 23.5 | 33.9 | +44% |
| Reverse linked list | 23.5 | 33.4 | +42% |
| TCP vs UDP | 23.5 | 35.5 | +51% |
| Solve 2x+5=17 | 23.5 | 26.4 | +12% |
| Exercise benefits | 23.5 | 31.3 | +33% |
| Rainbow | 23.5 | 33.5 | +43% |
| SQL duplicates | 23.5 | 26.9 | +14% |
| Stack vs heap | 23.5 | 35.3 | +50% |
| Binary search | 23.5 | 33.4 | +42% |
| **AVERAGE** | **23.5** | **32.6** | **+38.7%** |

*These numbers were measured with MIN_MARGIN=0.0 (8/10 exact only). Under the current
recovered exact-contract config (MIN_MARGIN=1.0, B=384), the average speedup is +16.7% with 10/10 exact.*

All primary answers were verified correct. Strict token-exact output requires the gated exact contract, not the relaxed +38.7% run.

### Three Key Innovations

**1. seq_cp Fork (zero-cost SSM backup)**
```
Before: seq_cp(0, 1)          — metadata only, ~0ms
Accept: seq_rm(1)              — discard backup
Reject: seq_rm(0) + seq_cp(1,0) + seq_rm(1) + re-decode cur
```
Replaces shadow save/restore (150MB GPU→CPU→GPU, ~3ms). The tensor copy is deferred to GPU graph build.

**2. Exact-match 3-4 gram lookup**
Skip 1-2 grams (too ambiguous across prompts). Try 4-gram first, fall back to 3-gram. No frequency thresholds — trust the cache.

**3. Logits margin gate**
Only speculate when `margin(top1 - top2) > MIN_MARGIN`. Prevents speculation on uncertain tokens.

## What We Explored

### Approaches Tested

| Approach | Speed | Quality | Why |
|----------|-------|---------|-----|
| Lossy (no rollback) | +25-78% | ❌ Broken | SSM corruption cascades |
| Shadow save/restore | +16% | ✅ Correct | 150MB copy overhead |
| **seq_cp fork** | **+38.7%** | **✅ Correct** | **Zero-cost backup** |
| Tree decode (seq_cp branches) | -36% | ✅ Correct | 3-token batch too expensive |
| 0.8B live draft (CPU) | -24% | ✅ Correct | CPU inference too slow |
| EAGLE3 head (CPU) | -36% | ✅ Correct | CPU matmul bottleneck |

### N-gram Cache Sources

| Source | Size | Speed | Notes |
|--------|------|-------|-------|
| Wikitext-2 | 3.9MB | +7.8% | Good general, wrong lookup slot initially |
| Model self-decode (510 prompts) | 7.2MB | +38.7% | Best — exact model patterns |
| Alpaca (GPT-3.5) | 3.8MB | +5.0% | Wrong model, adds noise |
| Synthetic (hand-written) | 49KB | — | Too small alone |

**Critical fix:** Load cache into `nc_context` slot (1-4 gram lookup), not `nc_static` (2-gram only). This doubled acceptance rates.

### EAGLE3 Draft Head Pipeline (Historical)

This section is retained for recordkeeping only. It is not current execution guidance.

Full pipeline working on 5060 Ti:

```
capture-hidden → train (PyTorch) → export (GGUF) → eagle-decode
```

| Step | Tool | Output |
|------|------|--------|
| Capture hidden states | `llama-capture-hidden` | Binary file: (hidden, token) pairs |
| Train head | `train_eagle.py` | PyTorch model |
| Export GGUF | `export_eagle_gguf.py` | Standard GGUF with eagle.* tensors |
| Inference | `llama-eagle-decode` | Loads GGUF, predicts, verifies |

**Results on 4B (51K training pairs):**
- 1-step accuracy: 71.6% val (predicts same token as logits — redundant)
- Seen prompts: 92-100%
- Unseen prompts: 10-34%

**Blocker:** CPU inference for 256→248K vocab projection = 12ms/token. Kills speed on both 4B (8.7ms baseline) and 27B (44ms baseline).

### DFlash Architecture (from DDTree paper)

The correct EAGLE approach, learned from DDTree repo analysis:

1. **Predict hidden states, not tokens** — reuse model's LM head (on GPU, free)
2. **Small transformer with cross-attention** to target hidden states
3. **Predicts entire block** in one pass (not token by token)
4. Gets **6-8x speedup** on Qwen3 models

Our hidden-state head (42MB, 10.5M params) achieves cosim=0.67 with actual hidden states. But without the LM head on GPU, can't convert to tokens efficiently.

**Key insight:** NN analysis shows even perfect hidden state match gives only 46.5% token accuracy without the LM head. The LM head projection is essential and must run on GPU.

## Files

### Core Tools (llama.cpp/examples/jacobi/)

| File | Purpose |
|------|---------|
| `lossy-ngram.cpp` | **Production tool** — n-gram speculation with seq_cp fork |
| `eagle-decode.cpp` | EAGLE3 head inference from GGUF |
| `capture-hidden.cpp` | Hidden state capture for EAGLE training |
| `draft-decode.cpp` | Live 0.8B draft model experiment |
| `lossy-decode.cpp` | Original lossy decode (hand-rolled bigrams) |
| `tree-decode.cpp` | Tree decode with seq_cp branches |
| `shadow-decode.cpp` | Shadow state prototype |

### Training Scripts (/tmp/)

| File | Purpose |
|------|---------|
| `train_eagle.py` | Original token-prediction head training |
| `train_eagle2.py` | Hyperparameter sweep |
| `train_eagle3.py` | Per-prompt accuracy tracking |
| `export_eagle_gguf.py` | GGUF export using gguf library |

### Infrastructure Changes

| File | Change |
|------|--------|
| `src/llama-memory-recurrent.h/cpp` | Shadow save/restore API |
| `src/llama-context.cpp` | `llama_memory_shadow_save/restore()` public API |
| `include/llama.h` | API declarations |
| `common/ngram-cache.cpp` | Relaxed thresholds for 3-4 gram lookup |

### Cache Files (tan_llama/artifacts/ngram_caches/)

| File | Size | Source |
|------|------|--------|
| `ngram_wiki.bin` | 3.9MB | Wikitext-2 |
| `ngram_decode.bin` | 7.2MB | 510 prompts self-decode patterns |
| `ngram_synthetic.bin` | 49KB | Hand-written QA/code/math |
| `ngram_combined.bin` | 4.1MB | Wiki + synthetic + 27b-self |

### GGUF Files (/tmp/)

| File | Size | Content |
|------|------|---------|
| `eagle_head_4b.gguf` | 258MB | Token-prediction head for 4B (2560→256→248K) |
| `eagle_head_27b.gguf` | 261MB | Token-prediction head for 27B (5120→256→248K) |

## Environment Variables

| Var | Default | Purpose |
|-----|---------|---------|
| `NGRAM_STATIC` | — | Path to static n-gram cache |
| `NGRAM_LOAD` | — | Load previous dynamic cache |
| `NGRAM_SAVE` | — | Save dynamic cache (merges context + dynamic) |
| `MIN_MARGIN` | 4.0 | Logits margin threshold |
| `NO_SPEC` | — | Disable speculation |
| `DEBUG_SPEC` | — | Print per-token decisions |
| `EAGLE_HEAD` | — | Path to EAGLE3 GGUF head |
| `EAGLE_DATA` | — | Output path for hidden state capture |

## Batch Scaling (27B TQ3_4S, RTX 5060 Ti)

| Batch | tok/s | Time | Ratio vs tg1 |
|-------|-------|------|---------------|
| tg1 | 22.8 | 43.8ms | 1.00x |
| pp2 | 31.6 | 63.3ms | 1.44x |
| pp3 | 40.7 | 73.8ms | 1.68x |
| pp4 | 49.0 | 81.6ms | 1.86x |

Break-even acceptance: 44% for pp2. Our average: 89%.

### LOSSY_NGRAM_BATCH Recovery (2026-04-20)

Working repo: `/home/awee/code/worktrees/llama-specdecode-next`

The current source checkout no longer carried the `examples/jacobi` sources, so the active work was recovered from `experiment/persistent-decode` into `experiment/specdecode-next-v2-20260419`.

The first recovered run reproduced a real failure:

- `lossy-ngram.cpp` hardcoded `n_batch=512`
- the 27B model plus compute buffers left too little VRAM for the 299 MB GPU shadow buffer
- first speculative save on `reverse_linked_list` failed with `ssm-shadow: cudaMalloc(299 MB) failed: out of memory`

Fix applied:

- add `LOSSY_NGRAM_BATCH`
- default to `384`
- use the same value for `ctx_params.n_batch`, `ctx_params.n_ubatch`, and `llama_batch_init`

Full 10-prompt exact contract recheck on RTX 5060 Ti (16GB):

| LOSSY_NGRAM_BATCH | Spec tok/s | Speedup | Exact |
|-------------------|-----------|---------|-------|
| 384 | 24.7688 | 1.1672x | 10/10 |
| 512 | **OOM** | — | — |

Baseline at the same contract: `21.2203 t/s`.

- Default changed to `384` in `lossy-ngram.cpp`
- Env override `LOSSY_NGRAM_BATCH` still available for future sweeps
- B=512 is the OOM boundary on this 16GB card
- B=384 is the current safe default for the recovered branch
- This restores correctness and VRAM stability, but does not recover the older `26.942 t/s` exact-contract mean
- Artifact: `/home/awee/code/tan_llama/artifacts/spec_exact_fast_contract_recheck_v3_20260419.json`
- Artifact: `/home/awee/code/tan_llama/artifacts/specdecode_recovery_batch384_20260419.txt`

## Next Steps

### 1. Ship N-gram Approach
- Integrate `lossy-ngram` logic into `llama-server`
- Build cache generation pipeline (automated)
- Prefer a **universal static cache + runtime local cache** over per-model self-decode caches
- Keep model-self caches only as an upper-bound reference, not the product story

### 2. Historical Longer-Term Direction: Train DFlash Draft Model
- Use DDTree codebase (already supports Qwen3)
- Train on 27B hidden states (5000+ prompts, ~1 hour)
- The draft model predicts hidden states, reuses target LM head
- Expected: 4-8x speedup (DDTree paper numbers)

### 3. Historical Longer-Term Direction: Integrate Draft Head into llama.cpp
- Add draft model as CUDA graph node
- Feed predicted hidden states through existing output layer
- Use seq_cp fork for SSM rollback (our innovation)
- This combines DDTree's draft quality with our SSM rollback solution

### 4. Extend to 3-4 Token Speculation
- pp3 costs 1.68x for 3 tokens (vs 1.44x for pp2)
- With 70%+ acceptance on 2 guesses: +50-60% possible
- Needs multi-step hidden state prediction (DFlash does this)

## Key Learnings

1. **SSM rollback is solvable** — seq_cp fork is zero-cost and correct
2. **N-gram cache quality > quantity** — model's own patterns beat large external corpora
3. **3-4 grams only** — 1-2 grams are too ambiguous across prompts
4. **CPU head inference is a dead end** — any draft prediction must run on GPU
5. **Predict hidden states, not tokens** — reuse the model's LM head (DFlash insight)
6. **pp2/tg1 ratio determines profitability** — 27B (1.44x) is ideal, 4B (1.73x) is marginal
7. **Self-decode caches do not scale operationally** — the scalable path is `universal static cache + runtime local cache`, optionally augmented by a compact learned drafter later

## Updated Strategic Direction

### What We Keep

- `seq_cp` rollback for hybrid SSM correctness
- exact 3-4 gram lookup
- logits-margin gating
- runtime-local cache built from prompt + generated tokens

### What Changes

- stop treating offline model-self decode caches as the main shipped asset
- build a **small universal static cache** from mixed prompt corpora that transfer across models sharing a tokenizer family
- measure that cache first on 4B / 9B / 27B before committing to a larger draft-model project

### Scalable Layering

1. **Runtime local cache**
   - always on
   - zero distribution cost
   - captures prompt/session repetition

2. **Universal static cache**
   - built once from mixed QA / code / reasoning / formatting corpora
   - shared across a tokenizer family
   - intended to replace per-model self-cache as the default static prior

3. **Compact learned block drafter**
   - only if the universal cache saturates
   - should augment the cache, not replace it
   - should be small enough that n-gram coverage removes the need for a 2B-class DFlash sidecar

### Immediate Experiment Order

1. Build a generic static cache from existing local prompt sets and small public corpora
2. Measure transfer on qwen35 `4B`, `9B`, and `27B`
3. Compare:
   - runtime-local only
   - universal static only
   - universal static + runtime-local

## Exact 10/10 Fast Contract Recovery (2026-04-14 evening)

The old fast path was real on speed but not exact. The stable recovery was:

- keep the proven **GPU full shadow buffer**
- tighten speculation entry instead of widening rollback complexity
- separate `static` from `context/dynamic` cache lookup
- default to:
  - `4-gram only`
  - `dynamic speculation off`
  - `MIN_MARGIN=1.0`
  - `--temp 0 --top-k 1`

Result on the 10-prompt breakthrough set:

- baseline: `21.734 t/s`
- speculative: `26.942 t/s`
- speedup: `1.240x`
- exact matches: `10 / 10`

Per-prompt:

| Prompt | Baseline | Speculative | Exact |
|--------|----------|-------------|-------|
| Capital of France | 21.81 | 31.99 | ✅ |
| Hash table | 21.78 | 21.13 | ✅ |
| Reverse linked list | 21.75 | 28.48 | ✅ |
| TCP vs UDP | 21.78 | 29.46 | ✅ |
| Solve 2x+5=17 | 21.61 | 26.20 | ✅ |
| Exercise benefits | 21.65 | 29.81 | ✅ |
| Rainbow | 22.04 | 30.07 | ✅ |
| SQL duplicates | 21.75 | 21.99 | ✅ |
| Stack vs heap | 21.66 | 20.92 | ✅ |
| Binary search | 21.51 | 29.37 | ✅ |

Practical takeaway:

- `GPU shadow` solved rollback cost
- `gating` solved exactness
- `row-shadow / copy-on-write` is still experimental and was not stable enough to trust for exact-output validation
   - model-self cache upper bound
4. Only after that decide whether a learned drafter is still required

## GPU Shadow Buffer Breakthrough (2026-04-14 afternoon)

### Root Cause Chain (solved)
1. CPU shadow: 64 × `cudaMemcpy+sync` = 26ms per save — killed all gains
2. seq_cp: shared tensors — speculation overwrites backup
3. Cell 1 = rs_z (zero state) — can't use for shadow
4. Tensor ne[1]=1 even with n_seq_max>1 — buffer too small for intra-tensor copy
5. cudaStreamPerThread race — copy not visible to compute stream

### Fix: Persistent GPU Shadow Buffer
- `cudaMalloc` ~150MB dedicated shadow buffer at first speculation
- `cudaMemcpyAsync` D2D for all 64 R/S tensors + single `cudaDeviceSynchronize`
- Save: ~0.5ms (was 26ms). Restore: ~0.5ms. Total rollback: ~1ms.
- File: `ggml/src/ggml-cuda/ssm-shadow.cu`

### Results (commit b644bf478)
```
Config: MIN_MARGIN=0.0, --temp 0 --top-k 1, -c 2048 -n 128
Cache: ngram_decode.bin (7.2MB)

Average: 22.7 → 29.3 tok/s (+28.7%), 8/10 exact
Peak:    22.6 → 36.3 tok/s (+60.3%) on Capital of France
```
