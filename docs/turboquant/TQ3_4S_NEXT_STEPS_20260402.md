# TQ3_4S Next Steps — 2026-04-02

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

- Stop expecting the current TQ3_4S kernel path to solve itself.
- Keep the clean cuBLAS path as baseline.
- Treat faster kernels as a separate moonshot front.

## Ordered Fronts

## Front 1 — Best quality per extra MiB

Objective:
- beat `last8 FFN -> Q5_K` at equal or lower size

Next tests:
1. `last8 down_proj -> Q5_K`
2. `last8 gate_proj -> Q5_K`
3. `last8 up_proj -> Q5_K`
4. `last8 (gate + down) -> Q5_K`
5. same set with `Q4_K`

Decision rule:
- rank by `quality gain / size cost`

## Front 2 — KLD-guided greedy policy

Objective:
- stop guessing by layer windows

Method:
1. start from clean `TQ3_4S`
2. test one tensor-family upgrade at a time
3. keep only upgrades with the best `ΔKLD / ΔMiB`
4. build greedily until the budget is hit

This should replace ad hoc brute-force batches.

## Front 3 — Speed baseline discipline

Objective:
- separate quality wins from kernel moonshots

Rules:
1. keep one known-good cuBLAS baseline
2. do not mix dirty-branch speed numbers into release claims
3. only compare new kernels to that baseline

## Front 4 — Moonshot kernel

Objective:
- recover enough prompt throughput to matter

Status:
- not the main line
- keep as background work

Constraint:
- no public claims until a new kernel beats the known-good cuBLAS path

## Immediate Actions

1. finish `last12 FFN -> Q5_K` full-pass run
2. start family-isolated FFN screens
3. add a greedy `ΔKLD / ΔMiB` search mode to the search script
4. stop broad mixed sweeps unless a family screen justifies them

## Publish Guidance

Public base release:
- `TQ3_4S`

Private / tuned variant:
- `TQ3_4S + last8 FFN -> Q5_K`

Do not present:
- chunk-summary wins and headline full-pass wins as the same thing
- dirty kernel experiments as final speed results
