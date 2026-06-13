# TriAttention Handover — Investigation State

**Date:** 2026-04-13  
**Branch:** `experiment/triattention-rework` (private repo: `charpdev/t_llama.cpp`)  
**Paper:** arXiv:2604.04921 — TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---

## What We Are Trying To Do

Implement TriAttention KV cache eviction in llama.cpp. The idea: use pre-RoPE Q/K statistics
(which are stable across positions) to score KV cache tokens by importance, then evict the
lowest-scoring ones to reduce memory and improve throughput.

Target: measurable KV reduction with <2% PPL degradation on Qwen3.5-9B and 27B TQ3_4S.

---

## Current State

### 2026-04-13 Update

The previous probe blocker is resolved in the current `/home/awee/code/llama.cpp/build` path.

What changed:

- `llama_triattention_rework_enable` is exported again from `libllama.so`
- the rolling probe links and runs against the same runtime library
- runtime eviction is now observed directly in the probe

Confirmed probe evidence:

- `accumulate_q: capture complete ...`
- `maybe_score_cache: score summary ...`
- `maybe_score_cache: eviction budget=... status=applied ...`

So the old assumption "probe never shows eviction firing" is no longer true for the current build.

### What Works
- Pre-RoPE Q capture: working correctly for qwen35-family models
- Center computation (`finalize_centers`): working, no NaN after recent fix
- Score computation (`maybe_score_cache`): produces finite scores when called from `llama-perplexity`
- `llama_triattention_rework_enable` symbol: exported from libllama.so ✅
- NaN guard: patched in `src/llama-triattention-rework.cpp` ✅

### What Is Broken / Under Investigation

The main runtime blocker has narrowed:

- eviction now fires in the probe
- finite score summaries are observed on the trusted witnesses
- the remaining work is score-model quality, not probe wiring

The current open question is:

- how to improve token scoring / selection quality beyond the current weighted-standardized path

---

## Key Files

| File | Purpose |
|------|---------|
| `src/llama-triattention-rework.cpp` | Core scorer + eviction engine (763 lines) |
| `src/llama-triattention-rework.h` | Struct definition |
| `src/llama-context.cpp` | Integration: `llama_triattention_rework_enable()`, `maybe_score_cache()` call |
| `tan_llama/tools/triatt_rolling_ppl.cpp` | Rolling PPL probe tool |
| `tan_llama/bin/triatt_rolling_ppl` | Built binary (links against build-360) |
| `docs/triattention_rework_plan.md` | Design notes |
| `docs/triattention_runtime_eviction_redesign.md` | Scorer redesign rationale |

---

## Resolved Bug

### What Was Wrong

Two concrete issues were mixed together:

1. the public runtime hook had fallen out of the shared-library export surface in one build,
   so local probes could not reliably relink against the current `libllama.so`

2. some earlier probe configurations were invalid because `triatt_tokens > prefill`,
   which means `capture_complete` never became true and scoring could not legally run

After fixing the export surface and rerunning with `triatt_tokens <= prefill`, the probe
does show runtime scoring and eviction.

Example probe evidence:

```text
accumulate_q: capture complete after 12 tokens (32 layers registered, 4 KV heads, head_dim=256)
maybe_score_cache: score summary active=12 eligible=8 scored_blocks=32 ...
maybe_score_cache: eviction budget=8 active=12 eligible=8 retain_mid=4 remove=4 ... status=applied
```

---

## Env Vars

| Var | Default | Purpose |
|-----|---------|---------|
| `LLAMA_TRIATT_EVICT_BUDGET` | 0 (disabled) | Max tokens to retain after eviction |
| `LLAMA_TRIATT_EVICT_PREFIX` | 0 | Tokens at start always protected |
| `LLAMA_TRIATT_EVICT_RECENT` | 0 | Recent tokens always protected |
| `LLAMA_TRIATT_UTILITY_SIGN` | 1 | Sign of utility score (+1 or -1) |
| `LLAMA_TRIATT_SCORE_MODEL` | 0 | 0=old scorer, 1=linear model |
| `LLAMA_TRIATT_SCORE_SIGNED_COEF` | 1.0 | Weight for signed utility |
| `LLAMA_TRIATT_SCORE_POSONLY_COEF` | 0.0 | Weight for positive-only utility |
| `LLAMA_TRIATT_SCORE_AGE_COEF` | 0.0 | Weight for token age |
| `LLAMA_TRIATT_SCORE_ABSZ_COEF` | 0.0 | Weight for mean absolute z-score |
| `LLAMA_TRIATT_SCORE_BIAS` | 0.0 | Constant bias term |
| `LLAMA_TRIATT_LAMBDA` | 0.25 | Regularization (0=pure utility sort) |
| `LLAMA_TRIATT_SEGMENTS` | 8 | Number of geometric segments |

---

## How To Run

### Build
```bash
cd /home/awee/code/llama.cpp
git checkout experiment/triattention-rework
cmake --build build-360 --target llama-perplexity -j$(nproc)

# Build probe
cd /home/awee/code/tan_llama
LLAMA_DIR=/home/awee/code/llama.cpp
g++ -O2 -std=c++17 \
  -I$LLAMA_DIR/include -I$LLAMA_DIR/ggml/include \
  tools/triatt_rolling_ppl.cpp \
  -L$LLAMA_DIR/build-360/bin -lllama -lggml-base -lggml \
  -Wl,-rpath,$LLAMA_DIR/build-360/bin \
  -o bin/triatt_rolling_ppl
```

### Reproduce the bug
```bash
cd /home/awee/code/tan_llama
MODEL9=/home/awee/models/turboquant9b/Qwen_Qwen3.5-9B-TQ3_4S.gguf
WIKI=/home/awee/code/llama.cpp/wikitext-2-raw/wiki.test.raw

# This should show eviction but doesn't:
LLAMA_TRIATT_SCORE_MODEL=1 LLAMA_TRIATT_UTILITY_SIGN=-1 \
LLAMA_TRIATT_SCORE_SIGNED_COEF=-1.0 \
./bin/triatt_rolling_ppl --model "$MODEL9" --file "$WIKI" --ngl 99 \
  --mode triatt --prefill 512 --eval-tokens 256 \
  --triatt-tokens 64 --budget 64 --prefix 16 --recent 32 \
  2>&1 | grep -E "evict|score summary|capture|rolling_ppl"

# Expected: score summary + eviction log
# Actual: only "capture complete" then rolling_ppl
```

### Confirm eviction works in llama-perplexity (control)
```bash
cd /home/awee/code/llama.cpp
LLAMA_TRIATT_EVICT_BUDGET=32 \
./build-360/bin/llama-perplexity -m "$MODEL9" -f "$WIKI" -ngl 99 \
  --triatt-rework-tokens 32 --chunks 1 -c 512 2>&1 \
  | grep -E "score summary|eviction"
# Expected: shows score summary + eviction applied
```

---

## Trusted Results

### Minimal proof witness

9B trusted probe witness showing live runtime eviction:

- artifact:
  - [triatt_rolling_ppl_qwopus9b_probe_recheck2_20260413.log](/home/awee/code/tan_llama/artifacts/triatt_rolling_ppl_qwopus9b_probe_recheck2_20260413.log)
- evidence:
  - `capture complete`
  - `score summary`
  - `eviction budget ... status=applied`

### 9B runtime witness

Config:

- `prefill=512`
- `eval_tokens=128`
- `triatt_tokens=64`
- `budget=64`
- `prefix=16`
- `recent=32`
- `adaptive_mode=12`

Artifacts:

- baseline:
  - [triatt_runtime_witness_qwopus9b_baseline_20260413.txt](/home/awee/code/tan_llama/artifacts/triatt_runtime_witness_qwopus9b_baseline_20260413.txt)
- triatt:
  - [triatt_runtime_witness_qwopus9b_triatt_20260413.txt](/home/awee/code/tan_llama/artifacts/triatt_runtime_witness_qwopus9b_triatt_20260413.txt)

Result:

- baseline rolling PPL: `9.277347`
- triatt rolling PPL: `9.249639`

### 27B runtime witness

Same config:

- `prefill=512`
- `eval_tokens=128`
- `triatt_tokens=64`
- `budget=64`
- `prefix=16`
- `recent=32`
- `adaptive_mode=12`

Artifacts:

- baseline:
  - [triatt_runtime_witness_qwopus27b_baseline_20260413.txt](/home/awee/code/tan_llama/artifacts/triatt_runtime_witness_qwopus27b_baseline_20260413.txt)
- triatt:
  - [triatt_runtime_witness_qwopus27b_triatt_20260413.txt](/home/awee/code/tan_llama/artifacts/triatt_runtime_witness_qwopus27b_triatt_20260413.txt)

Result:

- baseline rolling PPL: `12.702019`
- triatt rolling PPL: `11.053831`

### Interpretation

The runtime path is now trusted enough to continue score-model work:

- eviction is live in the probe
- 9B improves slightly
- 27B improves materially on this witness
- the next bottleneck is scorer quality / stability, not wiring

### 2026-04-13 scorer sweep result

On the trusted witness contract:

- `prefill=512`
- `eval_tokens=128`
- `triatt_tokens=64`
- `budget=64`
- `prefix=16`
- `recent=32`
- `adaptive_mode=12`

The current default scorer remains best among the tested cheap variants.

9B:

- baseline: `9.277347`
- triatt default (`utility_sign=+1`, `lambda=0.25`): `9.249639`
- triatt `utility_sign=-1`, `lambda=0`: `12.949860`
- triatt `utility_sign=+1`, `lambda=0`: `12.371451`
- triatt linear model (`signed=-1,pos=+1,age=0.5,absz=0.25`, `lambda=0.25`): `12.016446`
- triatt `lambda=0.5`: `12.401943`

27B:

- baseline: `12.702019`
- triatt default (`utility_sign=+1`, `lambda=0.25`): `11.053831`
- triatt `lambda=0`: `11.403726`

Conclusion:

- keep the current default scorer as the runtime baseline
- do not pursue sign-flip or simple linear-model variants as the next step
- the next real extension should be a better feature model, not more scalar-coefficient poking

### 2026-04-13 qwen35 4B witness

Available local 4B witness is BF16 rather than TQ3_4S:

- model:
  - `/home/awee/models/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-BF16.gguf`
- cache contract:
  - `K=f16`
  - `V=f16`

Artifacts:

- baseline:
  - [triatt_runtime_witness_qwen35_4b_bf16_baseline_20260413.txt](/home/awee/code/tan_llama/artifacts/triatt_runtime_witness_qwen35_4b_bf16_baseline_20260413.txt)
- triatt:
  - [triatt_runtime_witness_qwen35_4b_bf16_triatt_20260413.txt](/home/awee/code/tan_llama/artifacts/triatt_runtime_witness_qwen35_4b_bf16_triatt_20260413.txt)

Result:

- baseline rolling PPL: `8.203648`
- triatt rolling PPL: `11.183225`

Read:

- the current default runtime scorer is **not** uniformly beneficial across qwen35 sizes
- 9B and 27B improve on the trusted TQ3_4S witnesses
- 4B BF16 regresses badly on the same retention contract

Implication:

- we should not treat the current heuristic scorer as a family-wide production default yet
- qwen35 now clearly needs either:
  - size-conditioned policy, or
  - KLD-guided scorer fitting before broader rollout

### 2026-04-13 paper-style scorer witness

Implemented a separate runtime score mode that follows the original TriAttention structure more closely:

- pre-RoPE center phase alignment
- trigonometric distance preference
- additive norm term
- mean-over-offset aggregation
- same selector and same witness contract

Control knob:

- `LLAMA_TRIATT_SCORE_MODEL=2`

Witness config:

- `prefill=512`
- `eval_tokens=128`
- `triatt_tokens=64`
- `budget=64`
- `prefix=16`
- `recent=32`
- `adaptive_mode=12`

Artifacts:

- 4B BF16:
  - [triatt_runtime_witness_qwen35_4b_bf16_triatt_paper_20260413.txt](/home/awee/code/tan_llama/artifacts/triatt_runtime_witness_qwen35_4b_bf16_triatt_paper_20260413.txt)
- 9B TQ3_4S:
  - [triatt_runtime_witness_qwopus9b_triatt_paper_20260413.txt](/home/awee/code/tan_llama/artifacts/triatt_runtime_witness_qwopus9b_triatt_paper_20260413.txt)
- 27B TQ3_4S:
  - [triatt_runtime_witness_qwopus27b_triatt_paper_20260413.txt](/home/awee/code/tan_llama/artifacts/triatt_runtime_witness_qwopus27b_triatt_paper_20260413.txt)

Results:

- 4B BF16:
  - baseline: `8.203648`
  - old runtime default: `11.183225`
  - paper-mode: `8.200394`
- 9B TQ3_4S:
  - baseline: `9.277347`
  - old runtime default: `9.249639`
  - paper-mode: `9.113473`
- 27B TQ3_4S:
  - baseline: `12.702019`
  - old runtime default: `11.053831`
  - paper-mode: `10.568799`

Read:

- this is the first runtime scorer that behaves consistently across the tested qwen35 sizes
- it fixes the 4B regression
- it beats the previous heuristic on 9B and 27B
- the next step should now be KLD-guided refinement on top of paper-mode, not more work on the old heuristic

---

## Next Steps (Priority Order)

1. **Freeze the runtime baseline**:
   - weighted-standardized scorer
   - `utility_sign=+1`
   - `lambda=0.25`

2. **Target**: rolling_ppl ≤ baseline on 9B with budget ≤ 50% of context.

3. **Next extension**:
   - move to richer score features or KLD-calibrated fitting
   - do not keep exploring naive sign/lambda sweeps
