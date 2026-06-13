# Speculative Decode Hybrid Plan

Date: 2026-04-14

## Goal

Make speculative decode scalable across models without depending on per-model self-decode caches as the default shipped artifact.

## Core Direction

Use a three-layer stack:

1. `runtime local cache`
2. `universal static cache`
3. optional `compact learned drafter`

The first two layers should already cover a large share of easy continuation wins. The learned drafter should only be introduced if those layers plateau.

## Why

Per-model self-decode caches are strong but operationally weak:

- they require per-model generation
- they are prompt-distribution dependent
- they are awkward to ship
- they do not scale cleanly across model families

A universal static cache built from mixed prompt corpora is weaker but much more scalable. Runtime local cache then recovers session-specific repetition.

## Architecture

### Layer 1: Runtime Local Cache

- built from prompt tokens and accepted generated tokens
- always enabled
- zero offline build requirement
- strongest on repetition and local structure

### Layer 2: Universal Static Cache

- built offline from mixed prompt corpora
- intended to be shared across models in the same tokenizer family
- should prioritize:
  - code skeletons
  - QA patterns
  - math formatting
  - list / table / schema structure
  - common prose continuations

### Layer 3: Compact Learned Drafter

- only if layers 1 and 2 saturate
- should reuse target LM head when possible
- should be much smaller than full DFlash sidecars
- should use n-gram coverage to reduce required capacity

## First Deliverable

Build and measure a universal static cache pipeline.

Success criteria:

- one static cache serves qwen35 `4B`, `9B`, and `27B`
- improves over runtime-local-only on at least two prompt families
- stays far smaller and simpler than a learned drafter

## Benchmark Matrix

### Models

- qwen35 4B
- qwopus / qwen35 9B TQ3_4S
- qwopus / qwen35 27B TQ3_4S

### Cache Variants

1. runtime-local only
2. universal static only
3. universal static + runtime-local
4. model-self static upper bound

### Prompt Families

- factual QA
- structured explanation
- code
- math
- open-ended prose

### Metrics

- decode tok/s
- accept / reject / no-guess counts
- accepted-token percentage
- correctness / output sanity

## Practical Dataset Sources

Start small and mix domains. Do not optimize for sheer size first.

- local prompt sets already in `tan_llama/prompts/*.jsonl`
- small curated public prompts from Hugging Face datasets
- hand-written structured prompts for formatting coverage

The first universal cache should be small and easy to rebuild.

## Decision Rule

If universal static + runtime local gets close to the model-self upper bound, stay on the cache path.

If it clearly stalls well below the upper bound, then invest in a compact learned drafter.

## Current Findings (2026-04-14)

### Small Curated Beats Large Mixed

Measured locally with `llama-lossy-ngram`:

- large mixed cache `universal_v2` (`150K`) is too noisy
- small curated caches transfer better
- `universal_v1` (`33K`) is currently the best scalable candidate
- `ngram_synthetic.bin` (`49K`) is also competitive

### 27B Qwen3.5 TQ3_4S (6-prompt slice, c=4096, n=64)

- local only: `23.60 t/s`
- synthetic: `23.59 t/s`
- universal_v1: `23.81 t/s`
- self27 upper bound: `23.53 t/s`
- mixed universal_v2: `23.48 t/s`

Interpretation:

- the static cache does not need to be large to be useful
- the mixed universal cache adds conflicting predictions
- the small curated universal cache already matches or slightly beats the model-self upper bound on this slice

### 9B Qwen3.5 TQ3_4S (6-prompt slice, c=4096, n=64)

- local only: `70.83 t/s`
- synthetic: `70.53 t/s`
- universal_v1: `70.16 t/s`
- mixed universal_v2 on a broader 8-prompt slice was clearly worse

Interpretation:

- `universal_v1` is not yet a speed win on 9B, but it remains competitive
- the scalable path is still viable if we prune toward high-value n-grams rather than broad mixed coverage

### Tactical Conclusion

Do not scale up corpus size blindly.

Next work should be:

1. curate and prune the universal static cache
2. keep it small
3. test transfer on 4B / 9B / 27B
4. only add more data if it improves mean `t/s`, not just acceptance
