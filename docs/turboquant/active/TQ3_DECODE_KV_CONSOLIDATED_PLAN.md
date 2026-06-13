# TQ3 Decode + KV Consolidated Plan

**Date:** 2026-04-13

## Purpose

This is the current canonical plan for the TurboQuant runtime line.

It consolidates the overlapping threads from:

- `../proven/TQ3_4S_V2_RELEASE.md`
- `../proven/TQ3_4S_PERFORMANCE_DASHBOARD.md`
- `../proven/TQ3_RUNTIME_AND_RESULTS.md`
- `../procedures/TEST_SUITE.md`
- `FP4_KV_PLAN.md`
- `archive/exploration-legacy/TQ3_0_PROD_QJL_PROPOSAL.md`
- `llama.cpp/docs/triattention_rework_plan.md`

The main correction is simple:

- **decode speed**
- **KV quality**
- **runtime eviction**

are different projects and must not be mixed into one headline.

---

## Executive Summary

### 1. Weight path

`TQ3_4S` is already the public weight format that matters.

Current proven story:

- high quality for its size
- strong prompt-path speed after the MMQ rewrite
- decode speed is now much closer to the practical bar

### 2. Decode target

If the goal is to roughly **double decode speed**, the highest-ROI work is:

- token-generation CUDA kernel work
- MMVQ / `vec_dot` / decode-path simplification
- not runtime eviction
- not speculative new quant formats first

### 3. KV target

If the goal is to improve long-context quality without giving up memory efficiency, the best next direction is:

- keep `TQ3_4S` weights unchanged
- improve **KV representation**, especially `K`
- start with a **K-only FP4-style / block-floating** experiment

### 4. TriAttention status

TriAttention is now a **deferred research track**.

Reason:

- it is not the cleanest path to decode-speed gains
- its full-pass evaluation contract is still not production-clean
- it should not block the main decode + KV roadmap

---

## What Is Already Proven

### 2026-04-13 branch start state

Private execution branch:

- `/home/awee/code/llama.cpp` `experiment/tq3-decode-kv-plan`

Current decode anchors on that branch:

- Qwopus 27B `tg128`: `21.67 +/- 0.03 tok/s`
- Qwen 9B `tg128`: `64.14 +/- 0.03 tok/s`

Artifacts:

- `artifacts/llama_bench_qwopus27b_tg128_private_decode_branch_20260413.txt`
- `artifacts/llama_bench_qwopus9b_tg128_private_decode_branch_20260413.txt`

Important correction:

- the current private and public CUDA decode sources are effectively identical in:
  - `ggml/src/ggml-cuda/vecdotq.cuh`
  - `ggml/src/ggml-cuda/mmvq.cu`
  - `ggml/src/ggml-cuda/ggml-cuda.cu`
  - `ggml/src/ggml-cuda/mmq.cu`
  - `ggml/src/ggml-cuda/mmq.cuh`
- therefore, any remaining `tg128` gap is not explained by a private/public source delta in those files

Another correction:

- the widely repeated `27B tg128 = 23.2 tok/s` headline is stale
- the reproducible artifacted range for the current public/runtime line is about `21.5-21.7 tok/s`

### Weight / kernel line

Validated:

- `TQ3_4S` is the correct published weight format
- the MMQ tile-loader rewrite delivered the major prompt-path win
- the Blackwell decode improvements materially improved TG
- quality stayed intact on the validated weight-path checks

Practical headline:

- weight-side kernel work is real and worth shipping

### KV line

Validated:

- low-bit KV remains valuable
- `K` is more sensitive than `V`
- a mixed or asymmetric KV policy is the right direction

Strong prior:

- `K` should get the better representation
- `V` can stay cheaper longer

---

## What We Should Stop Mixing Together

### Do not use one headline for all of these

These are separate:

1. `TQ3_4S` weight-format quality
2. prompt-path CUDA speed
3. decode-path CUDA speed
4. KV-cache compression quality
5. runtime cache eviction

Each one needs its own benchmark and claim.

### Do not use full-pass `llama-perplexity` as a TriAttention proof

Current evidence says:

- standard full-pass `llama-perplexity` is not a trustworthy runtime-eviction harness
- do not claim runtime TriAttention quality wins from that path unless eviction is explicitly observed in the log

---

## Canonical Priorities

## Priority 1: Decode Speed

### Goal

Push real TG throughput materially higher without harming model quality.

### Why this is first

- this is where the user-visible gap still matters
- this is where the best chance of a `~2x` improvement exists
- this is independent of speculative KV research

### Focus areas

1. `vecdotq.cuh` / MMVQ decode hot path
2. Blackwell-specialized TG path
3. packed activation reuse where it helps decode
4. decode-side kernel fusion only when it is measured

### Immediate execution order

1. keep the current `TQ3_4S` weight decode path frozen as the branch baseline
2. treat `vecdotq.cuh` / `mmvq.cu` / `ggml-cuda.cu` parity between private and public as established
3. move the next decode investigation to the KV attention path:
   - `ggml/src/ggml-cuda/fattn-common.cuh`
   - `ggml/src/ggml-cuda/fattn.cu`
   - `src/llama-kv-cache.cpp`
4. only return to `vecdotq.cuh` if a measured branch shows a real delta there

### Success metrics

- `llama-bench -n 128` or equivalent TG benchmark
- strict chat smoke
- no PPL regression on the relevant weight path

### Non-goals for this phase

- no new weight format
- no full FP4 activation pipeline
- no runtime-eviction dependency

---

## Priority 2: KV Quality At Similar Memory

### Goal

Improve long-context quality without materially increasing KV memory.

### Best first bet

Start with:

- `weights = TQ3_4S`
- `K = FP4-style block-floating`
- `V = current known-good path`

This is the cleanest hybrid experiment.

### Why K first

- attention score quality is more sensitive to `K`
- `Q·K` errors get amplified by softmax
- `V` corruption is usually cheaper to postpone

### Recommended representation

Do **not** start with “true HiFloat4 everywhere”.

Start with:

- software block-floating 4-bit K format
- shared exponent / log-like scale per block
- optional outlier side buffer later

This is closer to what we can implement and benchmark inside `llama.cpp`.

### Suggested sequence

1. `K = block-float-4`, `V = f16`
2. `K = block-float-4`, `V = q8_0`
3. optional outlier side buffer for `K`
4. only later consider `V = block-float-4`

### Benchmark contract

Compare against:

- `f16 / f16`
- `q8_0 / f16`
- `q4_0 / f16`
- `tq3_0 / f16`
- new `bf4_k / f16`

On:

- Qwen3.5-9B
- Qwopus3.5-27B

Metrics:

- long-context quality
- decode speed
- prompt speed
- KV buffer size
- chat stability

---

## Priority 3: Advanced KV Research

These are worthwhile, but not primary until Priority 1 and 2 are cleaner.

### A. QJL residual on KV

Still plausible for:

- `K`-side dot-product correction
- quality improvement at similar nominal bitrate

But it is a second-stage KV experiment, not the first.

### B. Fused decode for KV attention

Very relevant for speed, especially:

- single-token decode
- no intermediate score-vector write

But best done after the KV representation is chosen.

### C. TriAttention runtime eviction

Keep as research only.

Return to it only if:

- decode path is no longer the main bottleneck
- KV format work has stabilized
- runtime evaluation contract is cleaner

---

## Deprioritized / Deferred

### TriAttention as a mainline product feature

Deferred.

Reason:

- unclear production evaluation path
- not the shortest route to speed
- too easy to confuse with decode and KV-format work

### Full FP4 activation pipeline

Deferred.

Reason:

- high implementation risk
- large kernel surface
- not the first efficient move

### New weight formats before decode recovery

Deferred.

Reason:

- `TQ3_4S` is already the correct published weight story
- weight innovation should not fragment the current runtime line

---

## Required Benchmark Discipline

Every future claim must declare which layer it belongs to.

### Weight quality claim

Needs:

- like-for-like PPL
- same model family
- no runtime-policy conflation

### Prompt-speed claim

Needs:

- `llama-bench` PP
- explicit context
- explicit build and GPU

### Decode-speed claim

Needs:

- `llama-bench` TG
- strict chat smoke
- no rolling-PPL probe speed substituted for TG

### KV-quality claim

Needs:

- same weights
- same model
- same KV baseline
- long-context quality test
- chat or retrieval sanity if relevant

### Runtime-eviction claim

Needs:

- explicit `score summary`
- explicit `eviction`
- not just “feature enabled”

---

## Immediate Next Steps

### Step 1

Freeze the current known-good decode baseline:

- 9B TG
- 27B TG
- coherent chat

### Step 2

Audit the current decode hot path:

- `ggml/src/ggml-cuda/vecdotq.cuh`
- related decode MMVQ path
- compare against the earlier faster Blackwell state

### Step 3

Write the concrete `K-only` block-floating KV experiment plan against the real llama.cpp KV codepaths.

### Step 4

Only after Step 2 and 3:

- prototype the new KV `K` format
- benchmark quality vs memory vs TG

---

## Canonical Headline Going Forward

Use this as the current project framing:

**TurboQuant mainline = TQ3_4S weights + faster decode + better KV quality.**

Not:

- TriAttention-first
- FP4-everywhere
- new weight-format churn

The next serious work items are:

1. recover and extend decode speed
2. improve KV quality with a better `K` representation
