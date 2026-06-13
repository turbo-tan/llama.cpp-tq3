# TQ3_4S Canonical Speed Plan

**Date**: 2026-04-09  
**Status**: Canonical active plan  
**Scope**: `TQ3_4S` runtime speed only

## Current Verified State

### Public safe baseline

- Branch:
  - `llama.cpp-tq3 main`
- Safe public fix:
  - revert bad `VDR=16` MMVQ change
- Known-good behavior:
  - `llama-simple-chat` sane
  - `llama-server` sane

### Current measured performance

For `Qwopus3.5-27B-v3-TQ3_4S` on RTX 5060 Ti:

- Public safe `pp2048`:
  - about `325 tok/s` class
- Local repaired wide-dot experiment:
  - `pp2048 = 328.84 ± 3.33 tok/s`
- Public safe `tg128`:
  - `12.52 ± 0.02 tok/s`
- Local repaired wide-dot experiment:
  - `12.78 ± 0.15 tok/s`

For plain `Qwen_Qwen3.5-27B-TQ3_4S` on the same box:

- Public safe `pp2048`:
  - `334.66 ± 0.99 tok/s`
- Public safe `tg128`:
  - `14.76 ± 0.10 tok/s`
- Preserved `build-360`:
  - `pp2048 = 334.24 ± 0.37 tok/s`
  - `tg128 = 24.02 ± 0.04 tok/s`

This confirms the missing gap is decode-only and not specific to Qwopus.

### Current quality signal

- Local repaired wide-dot experiment:
  - `10`-chunk PPL:
  - `7.8485 ± 0.39251`

## What Has Actually Shipped

These are the only changes that proved useful enough to keep in the public story:

1. Packed-X `dp4a` staging
   - real prompt win
2. MMA packed-X follow-up
   - smaller additional prompt win
3. Public linker/runtime fix
   - `turbo_wht` link fix
   - revert broken `VDR=16`

## What We Have Disproved

These should not be treated as live plan items anymore:

- naive native scalar kernel
  - massively slower than MMQ
- widened repack variants
  - no useful real win
- row-widening / pair-hoist schedule tweaks
  - regressed
- shared-half scale staging
  - regressed
- register-scale broadcast via `__shfl_sync`
  - regressed badly
- blockslot-only CTA contract
  - correct but only parity
- macro-only `VDR=16`
  - mathematically wrong
  - caused gibberish
- simple historical `VDR=8` replay as a shipping path
  - recovers decode speed
  - fails the Qwen chat correctness gate

## Where The Real Bottlenecks Still Are

### Prompt path

Remaining prompt gap is mostly in:

- `load_tiles_tq3_4s`
  - still does runtime mini-float decode with `ldexpf`
- MMQ bridge overhead
  - still amortized well enough that naive native kernels do not win

### Decode path

Remaining decode gap is mostly in:

- WHT rotation overhead
- MMVQ contract efficiency
- kernel launch / staging overhead around the TQ3 path

The latest recovery work adds one important constraint:

- the old fast decode class can be replayed with the simple historical `VDR=8`
  family, but that path is not correct enough to keep
- so the decode target is not to restore old simple `VDR=8` unchanged
- it is to make the safe subgroup-aware path faster

## Important Diff Findings

From direct code comparison between private `llama.cpp` and public `llama.cpp-tq3`:

- `mmq.cuh` is currently identical between the two trees
  - there is no hidden private `mmq.cuh` miracle patch missing from public
- the active local decode experiment is in:
  - `vecdotq.cuh`
- `load_tiles_tq3_4s` in `mmq.cuh` still uses:
  - `ldexpf(1 + mantissa / 32, exp)`
  - in both trees

This means the remaining easy prompt-side item from the old plan is still valid:

- replace runtime scale decode in `load_tiles_tq3_4s` with a table/LUT path

## Ranked Next Steps

### Tier 1: Best evidence-backed next moves

1. Scale LUT in `load_tiles_tq3_4s`
   - reason:
     - both trees still pay `ldexpf` in the hot loader
   - expected impact:
     - prompt-side only
     - low risk

2. Clean `VDR` sweep with one implementation family
   - compare:
     - `VDR=4`
     - `VDR=8`
     - `VDR=16`
   - using the repaired subgroup-aware vec-dot body
   - gate on:
     - coherence
     - `tg128`
     - short PPL

   This is now the only valid `VDR` family to optimize.
   The historical simple `VDR=8` family is no longer a candidate.

3. Native WHT op for decode-side pre-rotation work
   - reason:
     - decode gap is not being solved by vec-dot width alone
   - expected impact:
     - decode-side only
     - medium risk

### Tier 2: Only after Tier 1

4. Fused rotate + quantize kernel
   - only if prompt-side profiling still shows launch/memory tax worth chasing

5. Hybrid attention policy
   - only if decode remains the main blocker after WHT work

## Explicit Non-Goals For The Next Pass

Do not spend another cycle on:

- new scalar native kernels
- another blockslot-only branch
- more shared/regscale experiments
- another blind row-blocking rewrite
- any `VDR` change that is only a macro change

## Success Gates

We only keep a change if it satisfies both:

1. correctness
   - no gibberish
   - coherent `llama-simple-chat` / `llama-server`
   - one plain `Qwen_Qwen3.5-27B-TQ3_4S.gguf` chat smoke test
   - no PPL regression outside noise
2. real speed
   - prompt or decode gain large enough to matter
   - not just low-single-digit noise

## Canonical Reading Order

1. `TQ3_MOONSHOT_MASTER_LOG.md`
2. `TQ3_SPEED_UPDATED_PLAN.md`
3. `TQ3_NATIVE_PROMPT_KERNEL_DESIGN.md`
4. `TQ3_Q3KS_BOTTLENECK_MAP.md`
