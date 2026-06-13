# TurboQuant K-style 3.5-bit exploration

Date: 2026-03-31

## Branch

- target branch: `feature/tq3_1k-explore`
- working repo: `/home/awee/code/llama.cpp`

## Why This Line Exists

The modern 27B comparison ladder changed the target:

- plain `Q4_0` is not the strongest 4-ish-bit baseline anymore
- `Q4_K_M` is materially stronger on the 27B witness
- `IQ4_XS` is also right at that stronger bar on the same 100-chunk gate
- so the next serious TurboQuant weight question is no longer "beat `Q4_0`"
- it is "can TurboQuant geometry help a lower-bpw `K`-style quantizer enough to matter against the modern bar?"

That is the purpose of this line.

## First Validation: `TQ4_K`

The first experiment was deliberately simple:

- take native `Q4_K`
- apply WHT/random-sign rotation before quantization
- dequantize in the rotated domain
- inverse WHT back to the original domain

This gives a tensor-level `TQ4_K` screen without yet inventing a new low-bpw block.

### 9B witness

Real tensor screen on:

- `/home/awee/models/bartowski/Qwen_Qwen3.5-9B-GGUF/Qwen_Qwen3.5-9B-Q8_0.gguf`

Results:

- `Q4_K RMSE`: `0.001245`
- `Q4_K dot`: `0.019799`
- `TQ4_K RMSE`: `0.001180`
- `TQ4_K dot`: `0.018985`
- `RMSE ratio`: `0.948`
- `dot ratio`: `0.959`

Interpretation:

- about `5.2%` better RMSE
- about `4.1%` better dot error

### 27B witness

Real tensor screen on:

- `/home/awee/models/bartowski/Qwen_Qwen3.5-27B-GGUF/Qwen_Qwen3.5-27B-Q8_0.gguf`

Results:

- `Q4_K RMSE`: `0.001019`
- `Q4_K dot`: `0.016223`
- `TQ4_K RMSE`: `0.000974`
- `TQ4_K dot`: `0.015716`
- `RMSE ratio`: `0.956`
- `dot ratio`: `0.969`

Interpretation:

- about `4.4%` better RMSE
- about `3.1%` better dot error

## Current Conclusion

This is enough to say:

- TurboQuant-style WHT geometry still helps even when the baseline quantizer is already stronger than `Q4_0`
- the improvement survives on both the `9B` and `27B` witnesses
- therefore a `K`-style TurboQuant line has real legs

This is not yet a product result.

What is proven:

- tensor-level `TQ4_K` > native `Q4_K` on both witnesses

What is not yet proven:

- model-level PPL vs `Q4_K_M`
- runtime cost
- whether a lower-bpw `TQ3_K` contract can retain enough of this gain

Important scope guard:

- `TQ4_K` is **not** the product path
- it is only the upper-bound proof that the geometry works
- do not spend more product-design time on a `Q4_K_M`-class density line
- all active design work from here is on `TQ3_K`

## `TQ3_K` War Plan

The goal is not to stack TQ on top of an already-quantized `Q4_K_M` file.

The goal is:

- keep TurboQuant geometry (`WHT + sign`)
- borrow the stronger local scale modeling ideas from `K`-quants
- stay in a `3.x`-bit budget

### Working design direction

Use a 256-value super-block like `K` quants, but allocate bits roughly like:

1. super-block transform
- apply deterministic sign flip
- apply WHT over the full 256-value super-block, or blockwise over 32/64-value tiles inside it

2. local low-bit payload
- keep a 3-bit centroid payload as the main weight carrier
- still use TurboQuant-style centroid coding, not uniform scalar q4

3. richer local metadata
- replace the plain `TQ3_1S` dual-half scale with a `K`-style hierarchy:
  - per-subblock scale groups
  - maybe shared mins or shifts
  - maybe one coarse super-scale plus several finer local scales

### Current target

`TQ3_K_v0` should aim for:

- better local scale modeling than `TQ3_1S`
- materially lower bpw than `Q4_K_M`
- simpler first prototype than a full runtime format

### What Failed

Two scale-only `TQ3_K_v0` contracts were screened in the private harness:

- coarse-per-64 with nibble relative scales at `3.375 bpw`
- coarse-per-64 with full-u8 relative scales at `3.500 bpw`

Results were negative on both witnesses.

### 9B witness

- `TQ3_1S RMSE`: `0.002398`
- `TQ3_1S dot`: `0.038327`
- latest `TQ3_K_v0 RMSE`: `0.002676`
- latest `TQ3_K_v0 dot`: `0.042866`
- loss versus `TQ3_1S`: about `+11.6%`

### 27B witness

- `TQ3_1S RMSE`: `0.001974`
- `TQ3_1S dot`: `0.031802`
- latest `TQ3_K_v0 RMSE`: `0.002199`
- latest `TQ3_K_v0 dot`: `0.035201`
- loss versus `TQ3_1S`: about `+11.4%`

Conclusion:

- scale-only `TQ3_K_v0` is not enough
- more scale resolution alone is not the missing ingredient
- keep the existing `TQ3K_PROTO` plumbing, but stop tuning pure scale-only variants

### What The Existing Notes Already Tell Us

The strongest reusable signal from the older `TQ3` notes is:

- the remaining gap is not mainly a centroid/codebook issue
- compact extra amplitude/shape terms matter
- shared-shift helped
- affine-half helped more

Relevant earlier 9B prototype signal from the main plan:

| Prototype | Bytes/block | RMSE | Dot RMSE |
|---|---:|---:|---:|
| `TQ3_1S` | `16` | `0.002398` | `0.013558` |
| shared-shift proto | `18` | `0.002364` | `0.013333` |
| affine-half proto | `20` | `0.002247` | `0.012702` |

Interpretation:

- the next `TQ3_K` attempt should add a compact shift/affine degree of freedom
- not just a better scale ladder

### Active Focus

Only one private contract line is active now:

1. keep `TQ4_K` only as the upper-bound geometry proof
2. keep the existing `TQ3K_PROTO` harness plumbing
3. find a winning `TQ3_K`
4. re-screen each serious `TQ3_K` candidate on `9B` and `27B`

The public benchmark target is now explicitly:

- `Q4_K_M`
- `IQ4_XS`

not `Q4_0`.

The product constraint is also explicit:

- the line only matters if it preserves the 16 GB deployment story
- therefore the active search space is `TQ3`-class only
- not `TQ4_K`

### External Input To Steal From

The current public comparator review says:

- `Q4_K_M` is winning mainly on hierarchical local metadata over 256-value super-blocks
- `IQ4_XS` / IQ-family formats are winning mainly on better objective / importance-aware quantization
- Unsloth Dynamic 2.0 is winning mainly on heterogeneous per-layer / per-tensor quant selection and better calibration

Practical implication for this line:

- steal `Q4_K_M` ideas first:
  - coarse + fine hierarchy
  - better local block modeling
- steal IQ / Unsloth ideas second:
  - better objective
  - mixed calibration
  - selective tensor/layer protection

Important caveat:

- Unsloth Dynamic 2.0 is only partly inspectable from public docs and repos
- the exact GGUF generation path does not appear fully reproducible/open for every model
- so treat Unsloth as a benchmark and idea source, not as an implementation dependency

## Current Failed `TQ3_K` Attempts

These are now recorded dead ends unless a new fact appears:

1. scale-only coarse-per-64 `TQ3_K_v0`
2. finer-scale-only coarse-per-64 `TQ3_K_v0`
3. shift-only coarse-per-64 `TQ3_K_v0`
4. packed `4-bit scale + 4-bit shift` affine-at-`3.5 bpw` `TQ3_K_v0`

All of them remained roughly `11-13%` worse than `TQ3_1S` on both `9B` and `27B`.

Interpretation:

- the answer is not a tiny tweak to the current `v0` contract
- the next candidate must change the hierarchy, not just the quantization of one metadata byte

## Next `TQ3_K` Candidates

The next search should stay narrow and concrete.

### Candidate A: `TQ3_K_v1` shared-shift hierarchy

Reuse the strongest compact signal from the older `TQ3` work:

- one coarse scale per 64-value rotated tile
- one shared shift term per 64-value tile
- then two 32-value local scale bytes under that shared shift

Why this is first:

- shared-shift was the best compact scale-model lead in the earlier notes
- it is more expressive than scale-only or shift-only
- it still keeps the metadata structure simple enough for the current harness

### Candidate B: `TQ3_K_v1` diagonal-conditioned shared-shift

Same as Candidate A, but add the cheap diagonal/WUSH-lite conditioning before local quantization.

Why:

- diagonal conditioning was a small but real win in the older `TQ3` line
- it is worth stacking on a stronger contract, not worth using alone

### Candidate C: hybrid `TQ3_K` plus sparse protection

If uniform `TQ3_K` still misses badly:

- keep the best compact `TQ3_K` base
- allow a tiny promoted subset of the most sensitive layers/blocks

Why:

- the old adaptive-weight line proved the first `0.05-0.10 bpw` is very high leverage
- this may be the only way to close the modern quality gap without giving up the size story

## Immediate Next Step

Implement Candidate A first:

- shared-shift hierarchy inside the current `TQ3K_PROTO` path
- then re-screen on `9B`
- then `27B`

Only if that moves materially should Candidate B be added on top.

### `TQ3_K_v1` Design Direction

The next contract should try:

- 64-value rotated tiles inside the 256 super-block
- 3-bit centroid payload
- one coarse scale per 64
- one compact shift / affine correction per 32 or per 64
- target budget around `3.5-3.75 bpw`

The success bar stays:

- clearly better than `TQ3_1S`
- preserve as much of the `TQ4_K` gain as possible
- stay meaningfully below the `Q4_K_M` density class
