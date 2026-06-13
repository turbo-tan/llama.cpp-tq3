# TQ3_4S Next Steps — 2026-04-02

Canonical attempt history now lives in:

- [TQ3_MOONSHOT_MASTER_LOG.md](/home/awee/code/tan_llama/docs/turboquant/TQ3_MOONSHOT_MASTER_LOG.md)

Use this note for ordered fronts and decision rules.
Use the master log for per-attempt history and measured outcomes.

## 2026-04-03 Update

This note is now the working moonshot plan.

Two corrections:

1. Use the yesterday `KLD` search as the quality anchor.
- Do not let later partial screens or broken scorer runs overwrite the signal.

2. Treat speed as a separate hard target.
- The bar is no longer just "faster than today".
- The real bar is to approach the `Q3_K_S` class on prompt speed.

## Where We Are

### Clean base

- `TQ3_4S`
- size: `12.923 GiB`
- full-pass `c=2048` PPL: `6.8224 +/- 0.04534`

### Best proven mixed policy

- `TQ3_4S + last8 FFN -> Q5_K`
- size: `13.296 GiB`
- full-pass `c=2048` PPL: `6.6947 +/- 0.04370`
- full-pass chunk summary:
  - mean: `6.5318`
  - median: `6.5259`
  - p95: `6.8083`
  - max: `7.4996`

### Best KLD-only search result so far

- `TQ3_4S + last12 FFN -> Q5_K`
- size: `13.483 GiB`
- `Mean KLD = 0.13358`

### Yesterday's proven KLD ladder

Against the stable `Q8_0` base on `wiki.test.raw`, `100` chunks, `c=2048`:

- `last6_ffn_q5k`
  - `Mean KLD = 0.138196`
  - worse than anchor

- `last8_ffn_q5k`
  - `Mean KLD = 0.136296`
  - first stable mixed anchor

- `last10_ffn_q5k`
  - `Mean KLD = 0.134840`
  - first clear improvement

- `last12_ffn_q5k`
  - `Mean KLD = 0.13358`
  - best finished pure sparse override so far

Interpretation:

- the signal is real
- the win came from broader late-FFN protection
- later role-aware and banded quick screens did not yet beat this clean ladder

### Key comparison points

- `Q3_K_S`
  - size: `11.445 GiB`
  - full-pass `c=2048` PPL: `6.8630 +/- 0.04583`
  - full-pass chunk summary:
    - mean: `6.6415`
    - median: `6.6204`
    - p95: `6.8722`
    - max: `7.5342`

- `TQ3_1S`
  - size: `12.923 GiB`
  - full-pass `c=2048` PPL: `6.9807 +/- 0.04690`

## What The Search Already Taught Us

1. The base format is good enough.
- The real gains now come from policy, not from blindly inventing another format.

2. Late FFN protection is the strongest lever found so far.
- Small attention add-ons helped less than expected.

3. The current mixed-policy line improves quality without changing the format.
- That means the next work should be more targeted, not broader.

4. Runtime speed is still the main deployment weakness.
- Quality is now competitive.
- Prompt throughput is not.

5. Tom's lesson is correct: configuration and allocation beat elegant theory until proven otherwise.
- The best practical wins so far came from policy choices, not fancy correction stages.
- For our line, that means:
  - keep weight work focused on `TQ3_4S`
  - use policy to buy quality
  - use kernel work only where the hot path is clearly identified

6. The latent-reasoning / Mamba post is not a near-term `TQ3_4S` action item.
- It is useful as strategic inspiration:
  - hidden-state compute instead of visible token reasoning
  - fixed-memory reasoning loops
- It is not a direct speed fix for our current CUDA weight path.
- Near term, it does not compete with:
  - KLD-guided tensor policy
  - prompt-path speed work

## The Redundancies To Remove

### Policy redundancy

- Stop promoting full `last-N` windows forever.
- Split FFN into:
  - `ffn_down`
  - `ffn_gate`
  - `ffn_up`
- Measure each family separately.

### Rescue-level redundancy

- Stop assuming `Q5_K` everywhere.
- Test:
  - `Q4_K` first
  - then `Q5_K`
  - only `Q6_K` if the gain is worth the size

### Search redundancy

- Stop broad sweeps.
- Use:
  - short family screens
  - then greedy add/remove search under a size budget

### Runtime redundancy

- Stop expecting surrogate `q8` staging or generic `MMQ` to magically catch up.
- Keep the clean cuBLAS path as baseline.
- Treat faster kernels as a separate moonshot front.
- Do not use dirty staged-kernel numbers as release evidence.

## Ordered Fronts

## Front 1 — Best quality per extra MiB

Objective:
- beat `last8 FFN -> Q5_K` at equal or lower size
- if possible, recover most of `last12` quality without paying the full `13.483 GiB`

Next tests:
1. `last8 down_proj -> Q5_K`
2. `last8 gate_proj -> Q5_K`
3. `last8 up_proj -> Q5_K`
4. `last8 (gate + down) -> Q5_K`
5. same set with `Q4_K`

Decision rule:
- rank by `quality gain / size cost`

Current note:
- do not restart broad manual sweeps until there is a stable scorer path for these family splits

## Front 2 — KLD-guided greedy policy

Objective:
- stop guessing by layer windows

Method:
1. start from clean `TQ3_4S`
2. test one tensor-family upgrade at a time
3. keep only upgrades with the best `ΔKLD / ΔMiB`
4. build greedily until the budget is hit

This should replace ad hoc brute-force batches.

Immediate same-model eval rule:

1. source `PPL`
2. quant `PPL`
3. `quant/source` ratio
4. matching-source `KLD`

This is now the standard response to the "random raw PPL across unrelated models" criticism.

## Front 3 — Speed baseline discipline

Objective:
- separate quality wins from kernel moonshots

Rules:
1. keep one known-good cuBLAS baseline
2. do not mix dirty-branch speed numbers into release claims
3. only compare new kernels to that baseline

Current hard target:

- `Q3_K_S pp2048`: about `689 tok/s`
- known-good `TQ3_4S` cuBLAS path: about `315 tok/s`

This means the moonshot target is not a cosmetic gain.
It needs to close a very large gap.

What the investigation already showed:

- `Q3_K_S` avoids the expanded-weight writeback that hurts `TQ3_4S`
- current `TQ3_4S` losses are dominated by dequant / transform overhead
- the surrogate `MMQ` path is the wrong shape for this format

So the correct speed thesis is:

- make `TQ3_4S` behave more like a direct `K`-quant hot path
- reduce or eliminate intermediate expanded-weight traffic
- stop paying surrogate-pack overhead just to enter generic kernels

## Front 4 — Moonshot kernel

Objective:
- recover enough prompt throughput to matter

Status:
- not the main line
- keep as background work

Constraint:
- no public claims until a new kernel beats the known-good cuBLAS path

### Updated moonshot direction

Do not resume the old `TQ3_0` native-prefill line.

That was useful only as a reminder that:
- bypassing a bad generic bridge can help

For the real weight formats, the moonshot is now:

1. stay on `TQ3_1S/TQ3_4S`
2. target prompt processing first, not decode
3. optimize the actual hot path identified in traces
4. learn from `Q3_K_S`:
   - fewer intermediate writes
   - fewer surrogate conversions
   - simpler tensor-core-friendly dataflow

The next kernel question is not:
- "can we invent more math?"

It is:
- "can we cut enough hot-path traffic and staging overhead to move `315 -> 450+` first?"

If the answer is no, stop before another large kernel detour.

## Immediate Actions

1. finish the same-model `Qwopus f16 -> TQ3_4S` `KLD` compare
2. use that as the template for future public comparisons:
- source `PPL`
- quant `PPL`
- ratio
- matching-source `KLD`
3. keep `last8_ffn_q5k` as the practical tuned variant
4. keep `last12_ffn_q5k` as the quality-max variant
5. only resume family-isolated policy search after the scorer path is stable again
6. for speed, work from the `Q3_K_S` comparison docs before writing new kernels

## Publish Guidance

Public base release:
- `TQ3_4S`

Private / tuned variant:
- `TQ3_4S + last8 FFN -> Q5_K`

Research-only quality-max point:
- `TQ3_4S + last12 FFN -> Q5_K`

Do not present:
- chunk-summary wins and headline full-pass wins as the same thing
- dirty kernel experiments as final speed results
