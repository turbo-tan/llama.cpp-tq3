# TQ3 Build-360 Recovery And Regression Guard

**Date**: 2026-04-09  
**Purpose**: preserve the exact investigation trail around the old fast `build-360` binary, document what is proven, and prevent future regressions from being misdiagnosed or silently reintroduced.

## Why This Exists

We had a preserved old binary:

- `/home/awee/code/llama.cpp/build-360/bin/llama-bench`

It benchmarked far better than the later public/runtime state:

- `pp2048 = 335.69 ± 0.07`
- `tg128 = 23.90 ± 0.08`

But later source rebuilds from apparently related commits did **not** reproduce the same decode speed. This document records the difference between:

- what is actually proven by binaries
- what is actually proven by source
- what still needs recovery

## Ground Truth

### Preserved fast binary

Binary:

- `/home/awee/code/llama.cpp/build-360/bin/llama-bench`

Embedded build metadata:

- `5c10e1e68 (8664)`

Measured on RTX 5060 Ti:

- `pp2048 = 335.69 ± 0.07`
- `tg128 = 23.90 ± 0.08`

These numbers are real and reproducible from the preserved binary.

### Preserved fast binary is model-neutral

The preserved `build-360` decode jump is not a Qwopus-only artifact.

Measured on plain `Qwen_Qwen3.5-27B-TQ3_4S.gguf`:

- `pp2048 = 334.24 ± 0.37`
- `tg128 = 24.02 ± 0.04`

This matters because it proves the missing speed is in the runtime path itself,
not in one model's chat behavior or template.

### Recovered source sanity

A clean worktree from `5c10e1e68` did **not** build as-is. Minimal fixes were required:

- resolve merge markers in `ggml/src/ggml-cuda/mmq.cu`
- use shared turbo WHT declaration in `ggml/src/ggml-cuda/ggml-cuda.cu`

Recovered verification branch:

- `/tmp/llama.cpp-5c10e1e68`
- branch: `recovery/5c10e1e68-buildfix`
- commit: `2d0c2b8d5`

Recovered behavior:

- `llama-simple-chat`: coherent
- `llama-server`: coherent

Important:

- recovered `5c10e1e68` source sanity does **not** prove it reproduces the old fast decode path
- it only proves the old line was capable of sane chat/server behavior

## Public Main State

Public `llama.cpp-tq3` currently has these important fixes:

- `de8a2cb7c`
  - fix shared turbo WHT declaration in CUDA backend
- `9c61a1cdc`
  - revert bad `TQ3_4S` `VDR=16` regression
- `755de5813`
  - repair `TQ3_4S` Blackwell MMVQ vec-dot

Current public verified behavior:

- build clean
- `llama-simple-chat` sane
- `llama-server` sane

Current public measured performance:

- `pp2048 = 336.53 ± 0.80`
- `tg128 = 14.97 ± 0.05`

Conclusion:

- public `main` is healthy
- public `main` does **not** yet reproduce the old `build-360` decode speed

## Most Important Finding

The old fast binary cannot be explained by the clean public source state alone.

That means:

- the preserved `build-360` came from a dirty or otherwise unrecovered worktree state
- simply restoring one visible file is not enough
- prompt and decode must be treated separately

The updated evidence now narrows this further:

- prompt is already basically recovered on public `main`
- the missing gap is overwhelmingly decode-only

## What We Proved About Missing Pieces

### Not the explanation

These are **not** the missing miracle patch:

- `mmq.cuh` private vs public
  - currently identical in the compared trees
- `mmvq.cu`
  - no decisive difference in the recovered path
- linker/build issues
  - those were real, but separate from decode speed

### Still likely relevant

These commits in `/home/awee/code/llama.cpp` are the best surviving trail:

1. `8484314bf`
   - `perf: restore VDR=8 for TQ3_4S MMVQ on SM120 (Blackwell)`
2. `733424df9`
   - `Checkpoint: VDR=8 and scale table optimizations in place`
3. `d61a76563`
   - `perf: TQ3_4S MMA packed-X activation staging`
4. `d49be0649`
   - `Checkpoint: TQ3_4S CUDA optimisations (VDR=8, scale table, packed-X, SM120 row blocking)`

These are the strongest surviving candidates behind the old fast line.

## What The Qwen Correctness Gate Proved

We switched from Qwopus-only smoke checks to a stricter baseline:

- `/home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf`

This was important because Qwen is a cleaner correctness oracle for the decode path.

### Safe public path on Qwen

Current public `main`:

- `pp2048 = 334.66 ± 0.99`
- `tg128 = 14.76 ± 0.10`
- `llama-simple-chat`: coherent

### Historical simple `VDR=8` path on Qwen

Recovered simple historical `VDR=8` vec-dot body with normal scale decode:

- `tg128 = 21.58 ± 0.22`
- `llama-simple-chat`: broken
  - emitted empty or truncated `<think>` blocks instead of a proper answer

This is the most important new recovery result so far.

It means:

- the old simple `VDR=8` path really does reproduce most of the lost decode speed
- but it fails the correctness gate on plain Qwen
- therefore the obvious recovered fast path is not a safe shipping candidate

### Updated interpretation

The preserved `build-360` decode speed is real, but the nearest source-backed
fast path we can replay today is still broken.

So either:

- `build-360` included an additional dirty-worktree correctness fix that we still
  have not recovered, or
- the fast bench path hid generation problems that only show up clearly under
  stricter chat checks

Either way, we must not treat the recovered fast `VDR=8` path as valid.

## Critical Distinction

There are two different `VDR=8` stories:

### Historical simple `VDR=8`

From `8484314bf` / `733424df9`:

- macro-level `VDR=8` path
- paired with the old scale-table path
- documented in the old dashboard/session log as the major decode jump

### Later repaired subgroup-aware wide-dot path

This is the later investigation path:

- mathematically cleaner
- coherent
- but only small gains in current testing

Do **not** assume these two are equivalent.

They are different code paths and should be benchmarked separately.

## Recovery Ladder

This is the shortest trustworthy reconstruction order.

### Step 1: exact historical decode path

Apply only:

- `8484314bf` `vecdotq.cuh`
- `733424df9` `tq3_4s_scale.cuh`

Then measure:

- `tg128`
- `llama-simple-chat`
- `llama-server`

This answers:

- whether the old decode jump came from the historical `VDR=8 + scale table` pair

Status:

- partially answered
- the fast historical family does reproduce most of the decode jump
- but it is not correct enough under the Qwen chat gate

### Step 2: add historical prompt win

Apply:

- `d61a76563`

Then measure:

- `pp2048`
- `tg128`

This answers:

- whether the old prompt win is recreated without harming decode

### Step 3: only then test larger checkpoint state

If step `1 + 2` still misses `build-360`, test:

- `d49be0649`

But only after the narrower reconstruction, so we do not blur the source of the gain.

## Regression Guard Rules

These rules should be followed every time a speed path changes.

### Rule 1: never trust a dashboard first

Before quoting a speedup publicly, rerun:

- `llama-bench -p 2048 -n 0 -r 3`
- `llama-bench -p 0 -n 128 -r 3`
- one `llama-simple-chat` sanity run
- one `llama-server` `/v1/chat/completions` sanity run

### Rule 2: bench and chat are different gates

We must verify both:

- benchmark speed
- actual chat coherence

A fast bench binary is not enough.

Add one more explicit gate:

- one plain `Qwen_Qwen3.5-27B-TQ3_4S.gguf` chat smoke test

Qwopus alone is not enough for correctness screening.

### Rule 3: preserve source state, not just binary claims

Whenever a binary is unusually fast:

- create a branch immediately
- commit the exact source
- record the build path
- record the benchmark output in `artifacts/`

If this had been done for `build-360`, we would not be reconstructing it now.

### Rule 4: separate prompt and decode stories

Do not merge these into one vague “speedup” claim:

- prompt (`pp2048`)
- decode (`tg128`)

They can improve independently and can come from different code paths.

### Rule 5: never assume `VDR` changes are mathematically safe

For `TQ3_4S`, `VDR` changes affect the subgroup contract.

Every `VDR` change must be gated by:

- coherent text generation
- no gibberish
- short PPL check
- `tg128`

### Rule 6: use a canonical recovery branch for experiments

For old fast-state recovery, use one named branch only, for example:

- `recovery/build-360-shortpath`

Do not spread the recovery across unrelated branches.

## Current Best Interpretation

- `build-360` remains the ground-truth fast artifact
- public `main` remains the ground-truth safe source state
- the missing speed is decode-only
- the obvious recovered historical `VDR=8` path is fast-but-broken
- the next valid path is to optimize the safe subgroup-aware implementation,
  not to ship the recovered fast path unchanged

What we know:

- old `build-360` fast decode is real
- current public `main` is correct but slower on decode
- the strongest missing candidate is the historical `VDR=8 + scale table` path, not the newer repaired wide-dot body

What we do **not** yet know:

- whether `8484314bf + 733424df9` alone reproduces the old decode jump
- whether `d61a76563` is enough to finish the old prompt side on top

## Current Action Plan

1. reproduce historical `VDR=8 + scale table`
2. measure `tg128`
3. verify chat/server coherence
4. add `d61a76563` packed-X
5. measure `pp2048`
6. only then compare to `build-360`

## Confirmed Recovery Result (2026-04-09)

The first exact short-path replay was tested on:

- `llama.cpp-tq3` branch:
  - `recovery/build-360-shortpath`

Applied historical pieces:

- `8484314bf`:
  - historical `VDR=8` vec-dot path
- `733424df9`:
  - historical `tq3_4s_scale.cuh`

Measured result:

- `tg128 = 21.82 ± 0.05`

This is a major recovery relative to the current repaired public line:

- public repaired wide-dot path:
  - `tg128 = 14.97 ± 0.05`

But the same short-path replay failed the coherence gate:

- `llama-simple-chat` output became gibberish / punctuation-heavy corrupted reasoning text

So the current confirmed interpretation is:

- historical `VDR=8 + scale table` recovers most of the lost decode speed
- but by itself it is **not safe**
- the missing final piece is whatever made the old fast line both:
  - fast
  - coherent

This narrows the remaining search sharply:

- the decode speed win is real and reproducible from the old historical contract
- the remaining missing piece is a correctness/stability fix layered on top of that contract

## One-Sentence Summary

The old fast `build-360` binary is real, but the exact source state behind it was not preserved cleanly; the shortest safe reconstruction path is to replay the surviving historical `VDR=8 + scale table` decode path first, then add the historical packed-X prompt path, and gate every step with both benchmark and live chat checks.
