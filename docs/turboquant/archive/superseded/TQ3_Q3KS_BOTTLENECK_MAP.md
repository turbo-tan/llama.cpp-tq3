# TQ3_4S vs Q3_K_S Bottleneck Map

Date: 2026-04-04

## Purpose

This note turns the scattered pipeline diagrams and benchmark notes into one
explicit comparison:

- what `Q3_K_S` appears to do well
- where `TQ3_4S` loses time
- which branches have already tested each suspected bottleneck

Use this to choose the next moonshot branch.

## Throughput Anchors

Historical benchmark anchors from [BENCHMARK_PROTOCOL.md](../../procedures/BENCHMARK_PROTOCOL.md):

| Format | PP tok/s | TG tok/s |
|---|---:|---:|
| `Q3_K_S` | `689` | `20.7` |
| `TQ3_4S` | `327` | `14.6` |

Current fixed prompt-probe anchors on `Qwopus3.5-27B-v3-TQ3_4S`:

| Path | Prompt tok/s |
|---|---:|
| public known-good | `307.97` |
| private rebased baseline | `246.67` |
| latest native `Q3_K`-shaped tile branch | `253.03` |

Artifacts:

- [qwopus_prompt_probe_public_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_public_20260403.txt)
- [qwopus_prompt_probe_private_base2_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_base2_20260403.txt)
- [qwopus_prompt_probe_private_native_q3k_tile_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_native_q3k_tile_20260404.txt)

## Pipeline Comparison

### `Q3_K_S`

High-level prompt path:

1. load native quant blocks
2. expand into a `Q3_K`-shaped tile
3. feed directly into the `q8_0_16 x q8_1` MMA path
4. write back accumulators

Why it looks strong:

- native quant tile layout already matches the MMA path
- no WHT rotation step
- no extra activation-side scratch buffer
- mature prompt and decode kernels

### `TQ3_4S`

Current prompt path still effectively does:

1. load native `TQ3_4S` blocks
2. unpack 3-bit codes + per-group scales
3. activation-side WHT handling
4. stage through surrogate `q8`
5. run MMQ through the generic q8 path or a partially adapted `Q3_K`-style path

Why it loses:

- extra work exists before the MMA core even starts
- tile shape and staging have not been truly native to the final MMA kernel
- the current path is still partly a compatibility bridge, not a purpose-built prompt kernel

## Bottleneck Table

| Stage | `Q3_K_S` status | `TQ3_4S` status | What we learned |
|---|---|---|---|
| Weight tile load | native | partly bridged | native `Q3_K`-shaped tile helped a bit, but not enough |
| Weight metadata decode | simple scales/mins | 3-bit codes + 4 group scales | helper/LUT cleanups did not move the needle much |
| Activation prep | ordinary q8 staging | extra TQ3 rotation logic | fused rotate+q8 staging still lost |
| MMA core | native `q8_0_16 x q8_1` path | partially adapted | matching the tile shape improved only slightly |
| Temporary buffers | modest | extra rotated float/q8 staging | removing one temp buffer was not enough |
| Decode path | mature MMVQ | still slower than `Q3_K_S` | decode remains a separate open front |

## What Existing Attempts Already Covered

### 1. Scalar scale decode

Tried:

- shared helper cleanup
- LUT cleanup

Result:

- no meaningful prompt win

Interpretation:

- scalar scale decode is not the dominant bottleneck

### 2. Storage layout only

Tried:

- widened repack
- selective widened repack
- FFN-only repack
- attention-only repack
- same-size `pack4`

Result:

- no branch closed the gap

Interpretation:

- storage locality alone is not enough

### 3. Activation-side staging

Tried:

- fused activation rotate + q8 staging

Result:

- `241.60 tok/s`

Interpretation:

- removing the rotated float temp buffer is not the main win
- isolated staging microbench now shows `TQ3` rotate + `q8` staging is about
  `1.240x` slower than plain `q8` staging:
  - [bench_tq3_q8_stage_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_q8_stage_20260404.txt)

### 4. Weight tile shape

Tried:

- native `Q3_K`-shaped `TQ3_4S` tile emission

Result:

- `253.03 tok/s`

Interpretation:

- there was a real mismatch here
- fixing it helps
- but it is still not enough to explain the full speed gap
- isolated native-dot microbench now shows the inner weight contract is about
  `1.467x` better than the bridge:
  - [bench_tq3_native_dot_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_native_dot_20260404.txt)

## Bottleneck Ranking Right Now

Most likely remaining bottlenecks:

1. the prompt MMA path is still not truly native for `TQ3_4S`
2. surrogate `q8` staging still leaks overhead into the prompt path
3. the decode/MMVQ side remains slower than `Q3_K_S`
4. CTA-level producer/consumer overlap is still weak compared with the working-set reduction a native path could exploit

Less likely now:

- scale decode
- simple code packing
- family-selective repack
- one extra temp buffer by itself

## Practical Next Branches

### Prompt-side branch

Best next shot:

- a truly native prompt kernel for `TQ3_4S`
- not another bridge to an existing q8 path

That means:

- native `TQ3_4S` tile contract
- native MMA contract
- no surrogate q8 abstraction in the hot prompt path

### Decode-side branch

Best next supporting front:

- compare `Q3_K_S` and `TQ3_4S` MMVQ/decode path directly
- focus on:
  - vecdot load pattern
  - scale placement
  - shared-memory traffic

## Related Docs

- [TQ3_MOONSHOT_MASTER_LOG.md](/home/awee/code/tan_llama/docs/turboquant/TQ3_MOONSHOT_MASTER_LOG.md)
- [BENCHMARK_PROTOCOL.md](../../procedures/BENCHMARK_PROTOCOL.md)
- [TQ3_4S_ROTATED_MMQ_KERNEL_DESIGN.md](/home/awee/code/tan_llama/docs/turboquant/TQ3_4S_ROTATED_MMQ_KERNEL_DESIGN.md)
- [PIPELINE.md](/home/awee/code/tan_llama/docs/archive/PIPELINE.md)
