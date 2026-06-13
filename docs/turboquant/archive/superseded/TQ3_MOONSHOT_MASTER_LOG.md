# TQ3 Moonshot Master Log

Date: 2026-04-04

## Purpose

This is the single source of truth for the `TQ3_4S` speed moonshot.

Use this file to answer:

- what is the current prompt-speed baseline
- what has already been tried
- what worked
- what failed
- what the next serious branch should be

This consolidates recent work that had been spread across:

- `TQ3_4S_ROTATED_MMQ_KERNEL_DESIGN.md`
- `TQ3_4S_NEXT_STEPS_20260402.md`
- `archive/moonshot-legacy/LEGACY_SPEED_HANDOVER.md`
- `archive/moonshot-legacy/LEGACY_PP_SPEED_DESIGN.md`
- `archive/moonshot-legacy/LEGACY_TQ3_VS_Q3KS_INVESTIGATION.md`

## Current North Star

Goal:

- keep `TQ3_4S` quality
- close the prompt-speed gap to `Q3_K_S`

Current practical reference:

- public `Qwopus3.5-27B-v3-TQ3_4S` prompt probe:
  - `307.97 tok/s`
  - artifact:
    - [qwopus_prompt_probe_public_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_public_20260403.txt)

Historical bar:

- `Q3_K_S pp2048`: about `689 tok/s`

This means the problem is not cosmetic.
It is a large structural gap.

## Stable Quality Story

Current quality line is already good enough to separate from the speed moonshot:

- `TQ3_4S` full-pass quality is strong
- mixed policy improves it further
- same-model `Qwopus f16 -> TQ3_4S` stayed close on PPL

So this document is about speed/dataflow, not whether `TQ3_4S` is viable at all.

## Baselines

### Public / known-good reference

- prompt probe:
  - `307.97 tok/s`
  - artifact:
    - [qwopus_prompt_probe_public_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_public_20260403.txt)

### Private local baseline before the latest moonshot branch

- prompt probe:
  - `246.67 tok/s`
  - artifact:
    - [qwopus_prompt_probe_private_base2_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_base2_20260403.txt)

This private baseline matters because some private experiments happened on top of a rebased tree and should be compared against the exact local baseline, not only the public bar.

## What We Now Know About The Bottleneck

The current `TQ3_*` prompt path still does:

1. source activation on device
2. device copy to a temporary float buffer
3. TQ3 rotation (`WHT`) into that temp buffer
4. quantize rotated activations into `q8_1`
5. generic `MMQ` on surrogate `q8`

That is the core reason the work shifted away from tiny decode tweaks.

The problem is deeper than:

- scale decode math
- simple storage shuffle
- selecting only FFN or only attention weights

## Attempt Log

### Attempt 1: scale-decode helper cleanup

Idea:

- unify repeated `TQ3_4S` scale decode logic in the CUDA hot path

Result:

- no win
- prompt path still below baseline

Recorded in:

- `TQ3_4S_ROTATED_MMQ_KERNEL_DESIGN.md`

### Attempt 2: constant LUT / decode cleanup

Idea:

- make scale/lookup decode cheaper

Result:

- no win
- decode regressed

Recorded in:

- `TQ3_4S_ROTATED_MMQ_KERNEL_DESIGN.md`

### Attempt 3: shallow `Q3_K`-style MMQ trait/tile mimic

Idea:

- imitate the visible `Q3_K` tile shape in the existing `TQ3_4S` path

Result:

- no win
- prompt still far below the public bar

Conclusion:

- `Q3_K_S` advantage is structural, not just one tile constant

### Attempt 4: full-model widened repack

Idea:

- repack `TQ3_4S` at upload time into a prompt-fast widened layout
- pre-expand per-group scale bytes into fp16 deltas

Result:

- architecture compiled
- full-model repack did not fit on the 16 GB card
- VRAM use jumped too high before first useful measurement

Artifact:

- [qwopus_prompt_probe_private_repack_20260303.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_repack_20260303.txt)

Conclusion:

- full widened repack is not deployable on this target

### Attempt 5: selective widened repack with `+1 GiB` cap

Idea:

- keep widened repack
- cap extra VRAM
- select a subset of tensors under budget

Important bug found and fixed:

- alloc-size selection and upload selection diverged
- upload repacked tensors that alloc had left plain
- fixed by driving alloc/init/upload from one shared selector

Measured result:

- selective repack:
  - `257.31 tok/s`
  - artifact:
    - [qwopus_prompt_probe_private_repack_cap1g_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_repack_cap1g_20260404.txt)

Conclusion:

- stable and benchmarkable
- still clearly below both the public baseline and the target class

### Attempt 6: family isolation on widened repack

Purpose:

- determine whether widened repack hurts mostly in FFN or attention

Measured results:

- FFN-only repack:
  - `257.27 tok/s`
  - artifact:
    - [qwopus_prompt_probe_private_repack_ffn_cap1g_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_repack_ffn_cap1g_20260404.txt)
- attention-only repack:
  - `256.15 tok/s`
  - artifact:
    - [qwopus_prompt_probe_private_repack_attn_cap1g_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_repack_attn_cap1g_20260404.txt)

Conclusion:

- not a family-specific problem
- widened repack itself is the issue

### Attempt 7: same-size `pack4` layout

Idea:

- keep the original 16-byte `TQ3_4S` block size
- pack each 8-value group into one contiguous 32-bit chunk:
  - `[d_byte, qs0, qs1, qs2]`
- goal:
  - better locality
  - no extra bytes moved
  - no extra VRAM

Measured result:

- one-shot prompt probe:
  - `252.32 tok/s`
  - artifact:
    - [qwopus_prompt_probe_private_pack4_nocnv_nowarm_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_pack4_nocnv_nowarm_20260404.txt)

Conclusion:

- even same-size locality improvement is not enough
- the missing speed is not explained just by scattered `d + qs` fetches

### Attempt 8: fused activation-side rotate + q8 staging

Idea:

- stop allocating and filling a temporary rotated float buffer
- quantize directly from the original activation tensor into `block_q8_1_mmq`
- perform the TQ3 rotation inside the quantization kernel

Measured result:

- one-shot prompt probe:
  - `241.60 tok/s`
  - artifact:
    - [qwopus_prompt_probe_private_fused_q8_tq3_nocnv_nowarm_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_fused_q8_tq3_nocnv_nowarm_20260404.txt)

Conclusion:

- removing the temporary rotated float buffer is still not enough
- the prompt bottleneck is deeper than activation-side copy/rotate staging
- the next serious branch has to target the native prompt kernel itself, not another staging cleanup

### Attempt 9: native `Q3_K`-shaped `TQ3_4S` tile loader

Idea:

- keep the existing `q8_0_16` MMA path
- change `TQ3_4S` tile emission so it finally matches the `Q3_K`-style tile shape that path expects
- duplicate each subgroup scale across the two 4-value halves instead of writing the old `Q8_0`-style tile layout

Measured result:

- one-shot prompt probe:
  - `253.03 tok/s`
  - artifact:
    - [qwopus_prompt_probe_private_native_q3k_tile_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_native_q3k_tile_20260404.txt)

Conclusion:

- this was better than the fused/staging branch
- but it still stayed below the private local baseline
- matching the visible `Q3_K` tile shape was not the breakthrough

### Attempt 10: CTA-level native vs compact-predecode microbench

Purpose:

- stop guessing about the prompt bottleneck
- compare exact CTA contracts directly before touching production kernels

Measured result:

- artifact:
  - [bench_tq3_native_cta_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_native_cta_20260404.txt)
- raw native CTA:
  - `4.412 us/launch`
  - `0.507x`
- cached raw-block CTA:
  - `4.429 us/launch`
  - `0.506x`
- compact predecode CTA with consumer-lane predecode:
  - `4.467 us/launch`
  - `0.501x`
- cooperative compact-predecode CTA:
  - `2.189 us/launch`
  - `1.023x`
- all variants now match numerically:
  - diff `0`

Conclusion:

- raw native unpack is the wrong inner-loop contract
- the compact exact contract is viable
- but only when predecode is staged cooperatively in the right CTA shape
- the next real kernel branch is:
  - cooperative compact predecode feeding a native prompt inner loop

### Attempt 11: activation-side prepack side branch

Purpose:

- test whether prepacking the `q8` activation halves once in shared memory
  removes another meaningful chunk of prompt cost

Result:

- the side branch is still mathematically wrong
- current packed-`y` prototype does not match the bridge result yet
- so it is not trustworthy enough to drive kernel design

What survives from this branch:

- the x-side conclusion did not change
- cooperative compact predecode on the weight side is now the only exact CTA
  contract that matches the bridge speed envelope

### Attempt 12: lighter producer mapping plus `bfe` unpack

Purpose:

- cut redundant producer work inside the cooperative compact-predecode path
- try a cheaper `bfe`-based unpack helper on the exact CTA contract

Measured result:

- artifact:
  - [bench_tq3_native_cta_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_native_cta_20260404.txt)
- bridge CTA:
  - `2.193 us/launch`
- cooperative compact-predecode CTA:
  - `2.225 us/launch`
  - `0.985x`
- lane-owned block decode + `bfe` unpack:
  - `2.204 us/launch`
  - `0.995x`
- all exact:
  - diff `0`

Conclusion:

- reducing redundant producer work helped a little
- `bfe` unpack did not create a breakthrough by itself
- this branch stays near bridge parity, not beyond it

### Attempt 13: register-scale broadcast

Purpose:

- test the expert suggestion to keep subgroup scales in producer registers and
  broadcast them with `__shfl_sync`

Measured result:

- direct `shfl` in the hot loop:
  - `4.419 us/launch`
  - `0.496x`
- hoisted `shfl` before the hot loop:
  - `4.520 us/launch`
  - `0.485x`
- both exact:
  - diff `0`

Conclusion:

- for this CTA contract, shared-memory scales are better than register-broadcast
- `__shfl_sync` scale delivery is not the next lever

### Attempt 14: lighter producer mapping stabilised

Purpose:

- rerun the lighter producer ownership path after the latest bench cleanup to
  see whether it can actually beat the bridge, not just match it

Measured result:

- artifact:
  - [bench_tq3_native_cta_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_native_cta_20260404.txt)
- bridge CTA:
  - `2.220 us/launch`
- compact cooperative CTA:
  - `2.181 us/launch`
  - `1.018x`
- lane-owned block decode + `bfe` unpack:
  - `2.146 us/launch`
  - `1.034x`
- all exact:
  - diff `0`

Conclusion:

- this is the first exact microbench win over the bridge
- the best current contract is:
  - cooperative compact predecode
  - lighter producer ownership
  - shared scales
  - `bfe` unpack

### Attempt 15: fake double-buffer branch

Purpose:

- check whether a two-slot shared layout changes anything before doing real
  producer/consumer overlap

Measured result:

- lane8 `db` variant:
  - `2.187 us/launch`
  - `1.015x`
- exact:
  - diff `0`

Conclusion:

- just adding a second slot without real overlap is not the next win
- real overlap is still untested

## Measured Low-Level Invariants

These are not full prompt probes, but they now constrain what the next native
kernel must achieve.

### Native weight contract microbench

Artifact:

- [bench_tq3_native_dot_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_native_dot_20260404.txt)

Measured:

- native direct block dot:
  - `23.994 us/launch`
- dequant-bridge baseline:
  - `35.188 us/launch`
- speedup:
  - `1.467x`

Meaning:

- replacing the inner bridge math is real and worth doing
- but it did not survive CTA-level reality
- treat this as a useful local invariant, not a prompt-speed estimate

### Activation staging microbench

Artifact:

- [bench_tq3_q8_stage_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_q8_stage_20260404.txt)

Measured:

- plain `q8` staging:
  - `44.486 us/launch`
- `TQ3` rotate + `q8` staging:
  - `55.144 us/launch`
- overhead:
  - `1.240x`

Meaning:

- activation-side rotation is a real tax
- but it is smaller than the weight-side bridge penalty
- the next native kernel still has to change CTA-level dataflow, not only the
  block contract

### Current exact CTA takeaway

Artifact:

- [bench_tq3_native_cta_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_native_cta_20260404.txt)

Authoritative result:

- bridge CTA:
  - `2.189 us/launch`
- cooperative compact-predecode CTA:
  - `2.189 us/launch`
  - `1.023x` in one exact run, effectively parity
- raw native / cached raw / naive compact-predecode:
  - all around `4.4 us/launch`
  - about `0.5x`

Interpretation:

- the old block-dot-only projection was too optimistic
- the current bottleneck is the x-side predecode and lane formation step
- cooperative compact predecode is the first exact CTA contract worth porting
- there is still no proven route to `600+ tok/s`

## External Lessons We Should Keep

### From Tom’s work

- configuration and safe deployment matter
- do not assume the elegant correction step is the production win
- separate the clean success story from speculative kernel branches

### From `polarengine-vllm`

- layout and load pattern matter more than scalar decode tweaks
- keep compressed weights in VRAM
- avoid surrogate staging when possible
- benchmark by actual kernel shape, not just abstract theory

## Current Best Interpretation

What has been disproved:

- scale decode is the main bottleneck
- widened repack is a viable win
- FFN-only or attention-only widened repack will rescue speed
- same-size `pack4` locality alone will rescue speed

What still looks live:

1. cooperative compact predecode on the x-side
2. cheaper unpack in that producer path:
   - `bfe` already helped
   - `prmt + bfe + minimal lop3` is still the direction
3. multi-row weight reuse
4. packed activation halves in shared memory
5. overlap and double-buffering only after the mult-row packed-activation path
   is stable

Important correction from the latest step-back:

- the cooperative blockslot path is now proven correct in isolation
- but the clean two-path `bridge` vs `blockslot` benchmark only reaches parity
- so blockslot staging is **not** a justified runtime port yet
- the mixed CTA kitchen-sink harness should no longer be treated as a reliable
  correctness oracle for this branch

## Rules For Future Entries

Any new moonshot attempt should record:

1. exact idea
2. files touched
3. one artifact path
4. one prompt-speed number
5. conclusion in one sentence

If that is not written here, the attempt is not considered real history.

### Attempt 16: exact multi-row tile with packed activations

Purpose:

- stop trying to win with single-row CTA tuning alone
- reuse one compact predecoded weight slab across multiple prompt rows
- test whether activation-side prepack is the missing ingredient

Measured result:

- artifact:
  - [bench_tq3_multrow_cta_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_multrow_cta_20260404.txt)
- exact `2` rows/warp without packed activations:
  - `4.449 us/launch`
  - `1.000x`
- exact `2` rows/warp with packed activations:
  - `3.737 us/launch`
  - `1.190x`
- exact `4` rows/warp with packed activations:
  - best measured `2.607 us/launch`
  - `1.693x`

Conclusion:

- weight reuse alone is not enough
- weight reuse plus packed activation halves is the first large CTA-level win
- this is the first moonshot branch with a credible path beyond bridge parity

### Attempt 17: precombined scale tensor on mult-row branch

Purpose:

- remove the remaining `sw * xsc` multiply from the mult-row consumer loop
- test whether more shared-memory scale staging helps the `4`-row packed-activation path

Measured result:

- artifact:
  - [bench_tq3_multrow_cta_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_multrow_cta_20260404.txt)
- `2`-row packedx + combined scales:
  - about `4.379 us/launch`
  - worse than plain packedx
- `4`-row packedx + combined scales:
  - about `4.368 us/launch`
  - far worse than plain `4`-row packedx

Conclusion:

- precombining scales into a larger shared tensor is the wrong direction
- keep plain `4`-row packed activation reuse
- do not trade compute for more shared-memory scale traffic here

### Attempt 18: push reuse to `8` rows per warp-group

Purpose:

- test whether the mult-row packed-activation win keeps scaling with even more
  row reuse

Measured result:

- artifact:
  - [bench_tq3_multrow_cta_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_multrow_cta_20260404.txt)
- exact `8`-row packed-activation variant:
  - about `4.401 us/launch`
  - only `1.003x`
- exact `4`-row packed-activation variant in the same run:
  - `2.607 us/launch`
  - `1.693x`

Conclusion:

- `8` rows is too much for this contract
- register pressure kills the reuse benefit
- current sweet spot is `4` rows plus packed activation halves

### Attempt 19: real runtime TQ3 consumer hook swap

Purpose:

- test whether the synthetic CTA win maps to the real runtime by itself
- route `GGML_TYPE_TQ3_4S` through dedicated consumer entry points inside
  `mmq.cuh` without changing the inner consumer schedule yet

Files touched:

- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`

Measured result:

- artifact:
  - [qwopus_prompt_probe_private_tq3_consumer_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_tq3_consumer_20260404.txt)
- old private prompt baseline:
  - `246.67 tok/s`
- TQ3-specific consumer hook swap:
  - `247.25 tok/s`
  - about `1.002x`

Conclusion:

- the trait-level hook swap alone is basically noise
- the real runtime gap is not “missing TQ3 entry points”
- the next runtime branch must make the TQ3 consumer diverge for real:
  - `tile_y` contract
  - consumer schedule
  - inner hot loop

### Attempt 20: widen TQ3 MMA row blocking in the real runtime

Purpose:

- test the closest runtime analogue of the synthetic `4-row + packed x` win
- widen only the active NVIDIA MMA consumer schedule for `TQ3_4S`

Files touched:

- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`

Measured result:

- artifact:
  - [qwopus_prompt_probe_private_tq3_rows4_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_tq3_rows4_20260404.txt)
- prior TQ3 consumer hook swap:
  - `247.25 tok/s`
- widened row blocking:
  - `245.87 tok/s`
  - about `0.994x`

Conclusion:

- the bench-style row reuse does not survive this exact MMQ contract as a
  simple `rows_per_warp` widening
- the real TQ3 runtime path needs a deeper consumer hot-loop change, not just
  a wider row schedule

### Attempt 21: collapse duplicated half-group scales in the real TQ3 MMA loop

Purpose:

- exploit a real TQ3-specific invariant in the active NVIDIA MMA path
- `load_tiles_tq3_4s` writes the same subgroup scale twice, once for each
  4-value half inside the 8-value group
- replace `C0*dA0 + C1*dA1` with `(C0 + C1) * dA`

Files touched:

- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`

Measured result:

- artifact:
  - [qwopus_prompt_probe_private_tq3_scalecollapse_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_tq3_scalecollapse_20260404.txt)
- prior hook-only branch:
  - `247.25 tok/s`
- scale-collapse branch:
  - `249.17 tok/s`
  - about `1.008x`

Conclusion:

- this is the first real TQ3-specific consumer hot-loop win in the runtime
- still small, but it is a true end-to-end gain rather than dispatch noise

### Attempt 22: pair-hoist the collapsed-scale loop

Purpose:

- reduce repeated `dA` access further by hoisting one scale load per output
  pair in the TQ3 collapsed-scale loop

Files touched:

- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`

Measured result:

- artifact:
  - [qwopus_prompt_probe_private_tq3_pairhoist_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_tq3_pairhoist_20260404.txt)
- pair-hoist branch:
  - `242.11 tok/s`
  - about `0.972x` vs the scale-collapse branch

Conclusion:

- the extra loop reshaping regressed
- keep the simpler collapsed-scale loop as the best current runtime branch

### Attempt 23: step-back correctness proof for cooperative blockslot staging

Purpose:

- stop using the mixed CTA timing harness as the correctness oracle
- validate the full cooperative blockslot flow in isolation:
  - staged `x` packs
  - staged `x` scales
  - staged `y` packs
  - staged `y` scales
  - final accumulated output

Files touched:

- `/home/awee/code/llama.cpp/tests/tq3-coop-blockslot-correctness.cu`

Measured result:

- binary:
  - `/home/awee/code/tan_llama/artifacts/bin/tq3-coop-blockslot-correctness`
- all staged components matched reference:
  - `fail_px=0x00`
  - `fail_xd=0x00`
  - `fail_py=0x00`
  - `fail_yd=0x00`
- final output matched exactly:
  - `out_ref=-24.655697`
  - `out_stage=-24.655697`
  - `diff=0`

Conclusion:

- cooperative blockslot staging is mathematically correct in isolation
- the earlier mixed-harness corruption signal was not sufficient evidence that
  the blockslot math itself was wrong

### Attempt 24: clean two-path `bridge` vs `blockslot` benchmark

Purpose:

- benchmark only the proven-correct blockslot flow against the bridge
- remove packed-y, regscale, and other noisy side branches from the timing path

Files touched:

- `/home/awee/code/llama.cpp/tests/bench-tq3-blockslot-clean.cu`

Measured result:

- binary:
  - `/home/awee/code/tan_llama/artifacts/bin/bench-tq3-blockslot-clean`
- bridge:
  - `2.216 us/launch`
- blockslot:
  - `2.214 us/launch`
- speedup:
  - `1.001x`
- sample output:
  - `bridge=-24.655697`
  - `blockslot=-24.655697`
  - `diff=0`

Conclusion:

- the cooperative blockslot branch is correct
- but it is effectively parity with the bridge, not a breakthrough
- this branch should not be ported into the real runtime until a cleaner
  microbench win exists

### Attempt 25: verified real runtime `packed-x` prompt-path win

Purpose:

- port the successful activation-pack reuse idea into the real CUDA MMQ path
- specialize the live `TQ3_4S` consumer so prompt-side `y` packs are staged
  before the inner loop instead of being re-read in the old pattern

Files touched:

- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`
- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/mmq.cu`
- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`
- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/quantize.cu`
- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/quantize.cuh`
- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh`
- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/getrows.cu`
- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/tq3_4s_scale.cuh`
- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/tq3_4s_repack.cuh`

Measured result:

- private build:
  - `343.06 ± 1.30 tok/s`
- apples-to-apples public fork rerun:
  - `325.19 ± 0.83 tok/s`
- speedup:
  - about `1.055x`
- artifact:
  - [qwopus_prompt_probe_packedx_2row_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_packedx_2row_20260404.txt)

Conclusion:

- this is the first verified real runtime prompt-speed win that clearly beats
  the public fork on the same machine
- the earlier `347.13 tok/s` note was too high and has been superseded by the
  verified rerun
- the gain is real, but still far from the long-range `Q3_K_S` target

### Decode Recovery Update: preserved `build-360` is fast, but the obvious replay is broken

Purpose:

- use the plain Qwen 27B `TQ3_4S` model as a stricter correctness gate
- determine whether the lost `build-360` speed is real decode improvement or a
  model-specific artifact

Measured preserved `build-360` results:

- Qwopus:
  - `tg128 = 23.90 ± 0.08`
- plain Qwen:
  - `tg128 = 24.02 ± 0.04`
  - `pp2048 = 334.24 ± 0.37`

Measured current safe public results:

- Qwopus:
  - `tg128 = 14.97 ± 0.05`
- plain Qwen:
  - `tg128 = 14.76 ± 0.10`
  - `pp2048 = 334.66 ± 0.99`

Recovered historical simple `VDR=8` vec-dot replay:

- plain Qwen:
  - `tg128 = 21.58 ± 0.22`
- but `llama-simple-chat` failed the correctness gate:
  - emitted empty or truncated `<think>` output instead of a proper answer

Conclusion:

- the missing `build-360` gap is decode-only and model-neutral
- the obvious recovered simple `VDR=8` fast path reproduces most of the decode
  speed but is not correct enough to ship
- the next valid path is to optimize the safe subgroup-aware decode family, not
  to restore the old simple fast path unchanged
