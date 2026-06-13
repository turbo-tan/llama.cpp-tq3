# TQ3_4S Rotated-Domain MMQ Kernel Design

Date: 2026-04-01

Canonical attempt history:

- [TQ3_MOONSHOT_MASTER_LOG.md](/home/awee/code/tan_llama/docs/turboquant/TQ3_MOONSHOT_MASTER_LOG.md)

This file stays useful for deep kernel-specific notes.
The master log is the one place that should record every measured moonshot attempt.

## Objective

Design the first `TQ3_4S` prompt-processing kernel that can plausibly close the speed gap to `Q3_K_S` without giving up the current quality advantage.

Current 27B `c=2048` quality anchor:

- `TQ3_4S`: `6.7727`
- `Q3_K_S`: `6.7970`
- `IQ4_XS`: `6.8334`

Current speed reality:

- `Q3_K_S pp2048`: about `689 tok/s`
- known-good `TQ3_4S` cuBLAS path: about `315 tok/s`
- rotated-domain cuBLAS prototype: `265 tok/s` on a dirty branch, with a broken decode baseline nearby

So the kernel target is not a small win.
It is:

- recover the real `TQ3_4S` baseline cleanly
- then build a new prompt kernel with enough headroom to approach the `Q3_K_S` class

## Current Priority

The moonshot kernel path is no longer the only active line.

As of the first `KLD` runs, the highest-ROI path is:

1. `KLD`-guided mixed-precision search on top of `TQ3_4S`
2. keep the staged prompt kernel as a background moonshot
3. only revisit a new format tag if mixed-policy + kernel work both plateau

Why:

- `TQ3_4S` already wins on full-pass `PPL`
- raw `KLD` is still weak, so distribution-tail reduction is the real quality problem
- sparse protection is already showing measurable `KLD` improvement
- the kernel moonshot is still far from `Q3_K_S` speed

So the best current release path is:

- `TQ3_4S`
- plus `KLD`-guided sparse protection
- plus later kernel improvements if they land

## What Q3_K_S Is Doing Better In Code

The practical speed lesson is now explicit in the CUDA code:

- `Q3_K_S` (`GGML_TYPE_Q3_K`) has a native `MMQ` path:
  - `load_tiles_q3_K`
  - `VDR_Q3_K_Q8_1_MMQ`
  - `vec_dot_q8_0_16_q8_1_mma`
- `TQ3_4S` still enters `MMQ` by first collapsing itself into a surrogate `Q8`-style tile:
  - `load_tiles_tq3_4s`
  - `VDR_Q8_0_Q8_1_MMQ`
  - `vec_dot_q8_0_q8_1_mma`

Relevant files:

- `llama.cpp-tq3/ggml/src/ggml-cuda/mmq.cuh`
- `llama.cpp-tq3/ggml/src/ggml-cuda/mmq.cu`
- `llama.cpp-tq3/ggml/src/ggml-cuda/ggml-cuda.cu`

So the current difference is not just "one kernel is faster".
It is structural:

- `Q3_K_S` gets a format-native prompt kernel
- `TQ3_4S` pays extra staging work to look like a generic quant type

This matches the observed speed gap:

- `Q3_K_S pp2048`: about `689 tok/s`
- known-good `TQ3_4S` cuBLAS path: about `315 tok/s`

The immediate moonshot implication is:

- stop treating surrogate `Q8` staging as an acceptable long-term answer
- move `TQ3_4S` toward a format-native prompt path, the way `Q3_K_S` already is

## KLD Status

Base comparison setup:

- base logits: `Q8_0`
- `wiki.test.raw`
- `c=2048`
- `100` chunks

Current anchor model:

- `TQ3_4S + last8 FFN -> Q5_K`
- artifact:
  - [kld_cmp_27b_tq3_4s_last8ffn_q5k_c2048_100ch_20260401.txt](/home/awee/code/tan_llama/artifacts/kld_cmp_27b_tq3_4s_last8ffn_q5k_c2048_100ch_20260401.txt)
- result:
  - `Mean KLD = 0.136296 ± 0.003009`
  - `Median KLD = 0.026029`
  - `RMS Δp = 8.674 ± 0.097 %`
  - `Same top p = 89.264 ± 0.097 %`

Interpretation:

- the format is not failing on average-token behavior
- it is failing on a relatively small tail of badly distorted tokens
- this is why `PPL` can look very strong while `Mean KLD` still looks weak

This gives the current optimization target:

- reduce tail distortion
- not just shave average `PPL`

## KLD Search Progress

First sparse-protection sweep results against the `0.136296` anchor:

- `last6_ffn_q5k`
  - [kld_cmp_27b_tq3_4s_last6_ffn_q5k_c2048_100ch_20260401_154843.txt](/home/awee/code/tan_llama/artifacts/kld_cmp_27b_tq3_4s_last6_ffn_q5k_c2048_100ch_20260401_154843.txt)
  - `Mean KLD = 0.138196`
  - worse than anchor

- `last10_ffn_q5k`
  - [kld_cmp_27b_tq3_4s_last10_ffn_q5k_c2048_100ch_20260401_154843.txt](/home/awee/code/tan_llama/artifacts/kld_cmp_27b_tq3_4s_last10_ffn_q5k_c2048_100ch_20260401_154843.txt)
  - `Mean KLD = 0.134840`
  - first clear improvement

- `last8_ffn_q6k`
  - still in progress at time of this update
  - running around `0.134-0.135` in the later chunks

Conclusion so far:

- protecting too little is not enough
- a slightly broader late-FFN protection band helps `KLD`
- stronger protection (`q6_k`) may help too, but it needs the final result

This is enough signal to justify continuing the `KLD`-guided mixed-policy line before spending more time on external baseline benchmarking.

## Trace Evidence

Short traced run artifact:

- [bench_27b_tq3_4s_trace_short_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_trace_short_20260401.txt)

Representative stage timings:

- `m=10240 n=512 k=5120`: `rotate ~0.10 ms`, `src1_f16 ~0.03 ms`, `kernel ~26.5 ms`
- `m=17408 n=512 k=5120`: `rotate ~0.09-0.13 ms`, `src1_f16 ~0.03 ms`, `kernel ~44-45 ms`
- `m=5120 n=512 k=17408`: `rotate ~0.48-0.49 ms`, `src1_f16 ~0.15 ms`, `kernel ~44-46 ms`

Conclusion:

- activation rotation is not the bottleneck
- fp16 conversion is not the bottleneck
- the native rotated MMA kernel itself is the bottleneck

This is the main reason the first native kernel is not yet viable.

## Failure History

Keep this history. It defines the design constraints.

## 2026-04-03 Moonshot Log

### Attempt 1: Shared device helper for `TQ3_4S` scale decode

Change:

- factor the repeated `E3M5 -> float` decode logic into one shared CUDA device helper
- touch points:
  - `vecdotq.cuh`
  - `mmq.cuh`
  - `getrows.cu`

Why:

- the same scale decode math was duplicated in multiple `TQ3_4S` hot paths
- centralizing it is the smallest possible cleanup before a bigger MMQ change

Measurement:

- model:
  - `Qwopus3.5-27B-v3-TQ3_4S.gguf`
- exact same prompt probe on private vs public binaries
- artifacts:
  - [qwopus_prompt_probe_private_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_20260403.txt)
  - [qwopus_prompt_probe_public_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_public_20260403.txt)
  - [qwopus_decode_probe_private_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_decode_probe_private_20260403.txt)
  - [qwopus_decode_probe_public_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_decode_probe_public_20260403.txt)

Result:

- prompt speed:
  - private moonshot build: `242.05 tok/s`
  - public baseline: `307.97 tok/s`
- decode speed:
  - private moonshot build: `14.41 tok/s`
  - public baseline: `14.36 tok/s`

Conclusion:

- not a win
- prompt path regressed
- decode is essentially unchanged

Interpretation:

- removing source duplication alone does not help the real bottleneck
- the next attempt should be a true constant-table/LUT path for `TQ3_4S` scale decode, or a more structural MMQ representation change

### Attempt 2: Constant-table `TQ3_4S` scale decode

Change:

- replace the dynamic `E3M5 -> float` decode math with a `256`-entry CUDA constant lookup table
- same three hot paths:
  - `vecdotq.cuh`
  - `mmq.cuh`
  - `getrows.cu`

Artifacts:

- [qwopus_prompt_probe_private_lut_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_lut_20260403.txt)
- [qwopus_decode_probe_private_lut_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_decode_probe_private_lut_20260403.txt)

Result:

- prompt speed:
  - `240.04 tok/s`
- decode:
  - prompt eval: `11.29 tok/s`
  - decode eval: `8.50 tok/s`

Conclusion:

- also not a win
- prompt path is still well below the clean public baseline
- decode regressed badly on the short probe

Interpretation:

- scale-byte decode is not the main problem
- the next idea should stop touching scalar decode details and instead attack the structural issue:
  - `TQ3_4S` still stages itself into a surrogate `q8` MMQ tile
  - `Q3_K_S` gets a format-native prompt tile path

## Lessons From `polarengine-vllm`

Reference repo:

- `/home/awee/code/polarengine-vllm`

Useful takeaway:

- the interesting part is not the brand or paper framing
- the useful part is the implementation discipline around **packed inference layout**

What appears transferable:

1. Packed layout is a first-class optimization target

- `polarengine-vllm` explicitly repacks low-bit codes so the kernel reads them contiguously
- that matches what our own failed micro-attempts already imply:
  - scalar scale decode is not the bottleneck
  - tile width alone is not the bottleneck
  - memory layout and traffic pattern matter more

2. Keep the compressed representation in VRAM

- their forward path keeps codes resident and runs the math around that representation
- the key lesson for us is not “copy PolarQuant”
- it is:
  - stop paying repeated surrogate staging costs if we want prompt speed

3. Share transformed activations across sibling projections

- they cache the transformed activation across Q/K/V
- even if FWHT is not our dominant cost, shared transformed-input reuse is still the correct architecture
- the point is to avoid redoing any per-activation prep when the same hidden state fans out into multiple projections

4. Tensor-role policy matters as much as the kernel

- their mixed-bit defaults align with what our own `KLD` search already found
- this reinforces that policy quality and runtime design should keep moving together

What does **not** transfer directly:

- their current recommended path is mostly not their custom Triton kernel
- it is a better-packed quant plus a fast existing backend
- that is important because it means:
  - a heroic custom kernel is not automatically the highest-ROI answer

Implication for `TQ3_4S`:

- the next serious prompt-speed step should be a **native packed prompt layout**
- not more decode-math tinkering
- not more `q8`-surrogate patchwork

So the current moonshot direction should be:

- native packed `TQ3_4S` prompt representation
- contiguous loads designed around the packed block
- pre-expanded/prepacked metadata where it reduces repeated hot-path work
- transformed-input reuse across sibling projections where practical

### 1. fp16 dequant + cuBLAS

This is the old working path:

1. read quantized `TQ3_4S`
2. dequantize
3. run inverse WHT inside the dequant path
4. write expanded fp16 weights to global memory
5. call cuBLAS GEMM

Why it failed:

- inverse WHT is in the hottest part of the weight path
- full fp16 weight expansion is written to global memory
- GEMM then reads the same expanded buffer back

This path is correct, but too expensive.

### 2. Existing surrogate-q8 MMQ

Current `mmq.cuh` routes `GGML_TYPE_TQ3_4S` through:

- `load_tiles_tq3_4s`
- fake `q8` surrogate packing
- generic `vec_dot_q8_0_q8_1_mma`

Why it failed:

- it fights the format instead of using it directly
- it repacks 3-bit centroid blocks into a surrogate q8 form
- it spends shuffle and packing work just to enter a generic path

Latest concrete finding:

- the path was not merely "slow"; it was also broken
- `GGML_TYPE_TQ3_4S` was missing from the MMQ tile-size switch tables in `mmq.cuh`
- that produced out-of-bounds shared-memory writes in `mul_mat_q<GGML_TYPE_TQ3_4S, 128>`
- after fixing the tile-size table, the crash is gone, but the path is still slow

Artifacts:

- crash smoke:
  - [bench_27b_tq3_4s_mmq_smoke_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_mmq_smoke_20260401.txt)
- memcheck:
  - [bench_27b_tq3_4s_mmq_memcheck_p512_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_mmq_memcheck_p512_20260401.txt)
- post-fix blocking smoke:
  - [bench_27b_tq3_4s_mmq_blocking_p512_afterfix_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_mmq_blocking_p512_afterfix_20260401.txt)
- post-fix non-blocking smoke:
  - [bench_27b_tq3_4s_mmq_smoke_afterfix_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_mmq_smoke_afterfix_20260401.txt)

Post-fix smoke numbers:

- `pp512 = 118.91 tok/s`
- `tg2 = 8.82 tok/s`

Conclusion:

- MMQ is now debuggable and no longer crashing on the prompt path
- but even after the layout fix it is still much slower than the known-good cuBLAS baseline
- so MMQ is not the active speed path

### 2b. Why this path still loses

From the current code:

- `TQ3_4S` `load_tiles_tq3_4s` decodes each block into surrogate `q8` levels
- then uses the generic `q8_0 x q8_1` MMA path
- this means extra:
  - scale decode
  - centroid decode
  - lane shuffles
  - packing into surrogate int8 form

`Q3_K_S` avoids that exact trap:

- it uses a native `Q3_K` tile loader
- it keeps its own per-format layout semantics alive through the tile path
- then enters a dedicated MMA routine instead of pretending to be generic `Q8`

That is the design lesson we should copy.

### 3. Marlin-style fused inverse-WHT kernel

This was the wrong architecture.

Why it failed:

- it still centered the design on reconstructing original-domain weights
- it was much slower than the cuBLAS baseline
- it increased complexity without removing the real traffic bottleneck

### 4. First native rotated MMA kernel

This was the first correct architecture, but the first implementation is still too naive.

Measured artifacts:

- [bench_27b_tq3_4s_c2048_rotmma_proto_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_c2048_rotmma_proto_20260401.txt)
- [bench_27b_tq3_4s_c2048_rotmma_proto_v2_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_c2048_rotmma_proto_v2_20260401.txt)

Observed:

- v1: `pp2048 48.89`, `tg10 14.12`
- v2: `pp2048 39.30`, `tg10 13.86`

Why it failed:

- simplistic CTA tile shape
- no pipelining / double buffering
- decode not overlapped with MMA

## Next Kernel Step

Do not start another broad prototype.

Implement one narrow objective:

### Objective

Build the first `TQ3_4S` prompt kernel that removes surrogate `Q8` staging from the hot path.

### Target properties

1. keep `TQ3_4S` block semantics alive inside the tile loader
2. avoid writing an expanded fp16 weight buffer to global memory
3. avoid packing into surrogate `Q8` just to enter generic MMA
4. preserve the current correctness path as fallback

### Minimum success bar

- beat the known-good `TQ3_4S` `MMQ` debug path by a wide margin
- then beat the `~315 tok/s` cuBLAS baseline
- only after that worry about approaching `Q3_K_S`

### Concrete implementation order

1. trace the current `Q3_K` MMA tile path end-to-end
2. write down the exact interfaces `TQ3_4S` must match
3. replace one stage at a time:
   - first tile load semantics
   - then MMA consumption path
4. keep every new path behind a flag until:
   - prompt smoke
   - PPL smoke
   - one real prompt-speed number

### Do not do next

- do not resume the old `TQ3_0` native-prefill detour
- do not add more broad rotated-kernel theory before removing the current surrogate staging cost
- do not publish any speed story from dirty branches
- shared-memory / store pattern still poor

Important:

- this does not disprove the rotated-domain design
- it only disproves the first naive kernel implementation

Reference:

- [marlin_kernel_postmortem.md](/home/awee/code/tan_llama/docs/turboquant/marlin_kernel_postmortem.md)

### 5. First staged prompt kernel

Measured artifacts:

- [bench_27b_tq3_4s_trace_stage_prompt512_smoke_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_trace_stage_prompt512_smoke_20260401.txt)
- [bench_27b_tq3_4s_trace_stage_prompt512_tile128_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_trace_stage_prompt512_tile128_20260401.txt)

Observed:

- first staged smoke with an accidental `TILE_N = 32` effective implementation:
  - `pp512 37.18`
  - `tg2 12.11`
- after fixing the selector/implementation mismatch and actually reusing decoded weights across `128` columns:
  - `pp512 83.43`
  - `tg2 11.68`

Why this matters:

- the original staged result was artificially bad because `prompt_512` was still only amortizing decode over `32` columns
- once widened to `128`, prompt speed improved by about `2.25x`

Conclusion:

- the staged path still trails the `~315 tok/s` cuBLAS baseline badly
- but it is no longer a trivial dead end
- widening `N` really does help, so reuse/amortization is a real lever here

## Core Insight

Do not inverse-transform the weights in the prompt hot path.

For an orthonormal transform `R`:

```text
W_orig^T x = (R W_orig)^T (R x)
```

`TQ3_4S` already stores quantized weights in the rotated domain.

So the right prompt architecture is:

1. keep weights in the rotated domain
2. rotate activations once
3. multiply in the rotated domain
4. never materialize original-domain fp16 weights globally

This removes the two real structural costs:

- inverse WHT in the weight dequant loop
- the global fp16 weight expansion buffer

## Active Kernel Design

Name:

- `tq3_4s_rot_mma`

Purpose:

- prompt / prefill only
- `TQ3_4S` weights
- rotated activations
- native tensor-core MMA

Non-goals:

- do not solve TG first
- do not revive Marlin
- do not keep the fake-q8 MMQ abstraction
- do not keep iterating the current WMMA-first prototype as the main line

## Dataflow

### Old path

```text
TQ3_4S weights
  -> inverse WHT + dequant
  -> fp16 global buffer
  -> cuBLAS GEMM
```

### New path

```text
TQ3_4S weights (already in rotated domain)
  -> decode directly into shared-memory fp16 tiles

activations
  -> rotate once with ggml_cuda_tq3_rotate_act
  -> convert to fp16 tile

native MMA in rotated domain
```

That is the whole design.

## Design Pivot

After tracing the rotated path and reading the local ExLlamaV2 and ExLlamaV3 CUDA implementations, the next main line is no longer:

- one WMMA-first kernel with larger tiles
- one tiny scalar packed-dot kernel

The new main line is:

- staged packed GEMM
- specialized by `M`
- specialized by kernel shape / `K/N` tile size
- rotated-domain by construction
- register fragment decode feeding tensor-core style inner loops

Why:

- current trace shows the kernel is the bottleneck, not rotation
- the first native WMMA prototypes are far too slow
- the first `rot_dot` prototype is also too slow
- ExLlamaV2 teaches the value of shape selection and packed decode
- ExLlamaV3 teaches the value of staged packed GEMM, cp.async, and cooperative shape-specific kernels

## New Active Plan

### Phase 1: Shape Table and Dispatch

Adopt an ExLlamaV3-style shape table instead of one generic launch.

Initial shape buckets:

- shape A: `M <= 4`
- shape B: `M <= 16`
- shape C: `M <= 64`
- shape D: large prompt `M > 64`

Each shape chooses:

- tile `M`
- tile `K`
- tile `N`
- thread/block size
- shared-memory stages

### Phase 2: Staged Packed GEMM

Stop thinking in terms of "dot kernel vs WMMA kernel".
The target is a staged packed GEMM:

- packed weight loads
- staged activation tile loads
- register fragment decode
- pipelined inner loop
- no global fp16 weight materialization

This is closer to ExLlamaV3 than either our WMMA prototype or the tiny `rot_dot` prototype.

### Phase 3: Register Decode Fragments

Decode `TQ3_4S` blocks into register fragments directly:

- packed byte loads
- `half2` fragment decode
- centroid/scale application in registers

Keep decode close to the compute loop.
Do not decode to a large scalar scratch array.

### Phase 4: Pipeline

Use staged loads and overlap:

- activation tile load
- packed weight tile load
- register decode
- compute

The important lesson from ExLlamaV3 is that the pipeline and shape table are first-class, not optional polish.

### Phase 5: Re-evaluate Small-Bucket GEMV

Only after the staged GEMM path exists do we revisit whether a tiny `M <= 4` packed-dot path is useful for decode.

## Current Failure Boundary

What is still true:

- `TQ3_4S` quality is already good enough to justify optimization
- the rotated-domain design is still the right math path

What is no longer the active bet:

- bigger WMMA tiles
- more shared-memory staging
- more minor launch-shape tuning on the current native kernel

Those ideas have already failed to move the result enough.

## Concrete Task List

### Task 0: Keep the current good baseline intact

- Preserve the known-good cuBLAS/dequant path as the fallback.
- Do not regress the `PP 315 / TG 14.6` baseline while experimenting.

Expected win:

- `0%` direct speed gain
- but prevents us from optimizing on a broken baseline again

### Task 1: Add shape-table dispatch

- Add a `tq3_4s_rot_gemm_shape` selector.
- Hardcode 3-4 initial shapes, similar in spirit to ExLlamaV3.
- Dispatch by `M`, `N`, `K`, and CC.

Expected win:

- small alone: `0-5%`
- but required for every bigger improvement after it

### Task 2: Build one staged packed GEMM kernel

- Start with the medium/large prompt bucket, not decode.
- Use staged shared-memory activation tiles.
- Keep packed weights in place.
- Decode directly into register fragments.

Expected win:

- medium: `15-30%` over the current `TQ3_4S` prompt baseline if done competently
- this is the first task with real headline potential

### Task 3: Add cp.async-style pipelining / stage overlap

- Overlap tile movement with compute.
- Double-buffer the hot path.
- Keep the decode close to the fragment load.

Expected win:

- medium-large: `10-25%` on top of Task 2
- likely necessary to get out of the low-300s

### Task 4: Specialize small-`M` decode path

- After the staged prompt kernel exists, revisit `M <= 4`.
- Make the tiny path share the same fragment decode logic.
- Do not maintain a completely separate scalar path.

Expected win:

- decode-side only: `5-20%`
- useful, but not the main reason to pursue this line

### Task 5: Retune shape table

- Evaluate `K/N` tile sizes after the first real staged kernel exists.
- Only then add more shapes or retune block dimensions.

Expected win:

- small-medium: `5-15%`
- mostly polish after the architecture is correct

## Best-Win Estimate

This is not a proof. It is the current best engineering estimate.

### Near-term realistic target

If Tasks 1-3 land well:

- prompt speed: from `315` to about `450-550 tok/s`
- decode speed: from `14.6` to about `16-18 tok/s`

That would still trail `Q3_K_S`, but it would stop being obviously non-competitive.

### Aggressive target

If the staged packed GEMM path really works and shape tuning is good:

- prompt speed: about `550-650 tok/s`
- decode speed: about `18-20 tok/s`

This is the first point where we could start arguing that the quality advantage justifies the remaining speed gap.

### Stretch target

Matching or beating `Q3_K_S` prompt speed:

- `~689 tok/s`

Possible in principle, but not what I would promise from the current evidence.
That should be treated as stretch, not baseline expectation.

- only subgroup broadcasts
- no 32-lane butterfly
- no q8 repack

## Activation Path

Reuse the existing rotation primitive:

- [tq3-native.cu](/home/awee/code/llama.cpp/ggml/src/ggml-cuda/tq3-native.cu)

For prompt processing:

1. copy activation chunk to scratch
2. rotate once
3. convert to fp16
4. stage into `B_sh`

This is the right asymmetry:

- rotating activations is cheaper than inverse-transforming weights inside every dequant block

## MMA

Use native fp16 tensor-core MMA:

- `m16n8k16` / `16x16x16` class fragments
- fp16 inputs
- fp32 accumulation

Each `K=32` step becomes two `K=16` MMA steps.

## Why This Can Actually Win

This design removes all three of the wrong costs at once:

1. no inverse WHT in the weight loop
2. no global fp16 weight write
3. no global fp16 weight re-read

The rotated-domain cuBLAS prototype only improved prompt speed modestly because it still expanded weights into a global fp16 buffer.

This kernel is the first design that removes the real bottleneck rather than moving it around.

## Implementation Plan

### Phase 0

Restore the real `TQ3_4S` baseline on branch before trusting any delta.

### Phase 1

Add:

- `ggml/src/ggml-cuda/tq3_4s_rot_mma.cu`
- `ggml/src/ggml-cuda/tq3_4s_rot_mma.cuh`

Implement:

- CTA `64x32x32`
- subgroup decode
- fp16 shared-memory tiles
- fp32 MMA accumulation

Dispatch behind:

- `GGML_CUDA_TQ3_4S_ROT_MMA=1`

### Phase 2

Use it only for prompt / prefill and only for:

- `src0->type == GGML_TYPE_TQ3_4S`
- large enough batch / prompt shapes

Keep rotated-domain cuBLAS fallback.

### Phase 3

Benchmark with artifacts only:

- `pp2048`
- `pp512`
- `tg10`
- `100`-chunk `c=2048` PPL

### Phase 4

Only after PP is fixed:

- optimize TG / `vec_dot_tq3_4s_q8_1`

## Decision Rule

If this kernel cannot move `TQ3_4S` materially beyond the old cuBLAS path, then the format may simply not have enough speed headroom to justify chasing `Q3_K_S`.

But this is the first kernel design that actually matches:

- the format math
- the memory profile
- the current failure history

## Supersedes

This is the active design now.

Older docs were useful, but no longer the main plan:

- [PP_SPEED_DESIGN.md](/home/awee/code/tan_llama/docs/turboquant/PP_SPEED_DESIGN.md)
- [MARLIN_KERNEL_SPEC.md](/home/awee/code/tan_llama/docs/turboquant/MARLIN_KERNEL_SPEC.md)

## One-Line Summary

The kernel to build is:

- **rotated-domain `TQ3_4S` decode**
- **rotated activations**
- **shared-memory fp16 tiles**
- **native MMA**
- **no inverse WHT and no global fp16 weight expansion**

## 2026-04-03 Repack Shot

### Attempt 4: CUDA load-time prompt-fast repack

Change:

- add an env-gated CUDA repack path:
  - `GGML_CUDA_TQ3_4S_REPACK=1`
- repack `TQ3_4S` weights at tensor upload time into prompt-fast blocks:
  - keep the 3-bit payload compact
  - widen the 4 per-8 scale codes into final fp16 prompt deltas once
- thread a real `tq3_4s_repacked` flag into CUDA `MMQ` so the prompt kernel can consume the repacked layout

Files in private `llama.cpp`:

- `ggml/src/ggml-cuda/tq3_4s_repack.cuh`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/mmq.cuh`
- `ggml/src/ggml-cuda/mmq.cu`

Artifact:

- [qwopus_prompt_probe_private_repack_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_repack_20260403.txt)

Measured outcome:

- code path compiles cleanly
- full-offload load on the 16 GB test card fails before first token
- CUDA allocation request jumps to:
  - `15518.79 MiB`

Conclusion:

- the prompt-fast repack architecture is now real in code
- but an all-layer repack is too large for the current 16 GB deployment target
- the next serious step is:
  - selective repack for only the hottest tensors/layers
  - or a same-size native prompt layout

### Attempt 5: 1 GiB selective repack

Change:

- keep the load-time repack path
- add a deterministic selector shared by:
  - CUDA alloc sizing
  - tensor init
  - upload/repack
- cap extra VRAM with:
  - `GGML_CUDA_TQ3_4S_REPACK_EXTRA_MIB=1024`

Files in private `llama.cpp`:

- `ggml/src/ggml-cuda/tq3_4s_repack.cuh`
- `ggml/src/ggml-cuda/ggml-cuda.cu`

Artifacts:

- [qwopus_prompt_probe_private_base2_20260403.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_base2_20260403.txt)
- [qwopus_prompt_probe_private_repack_cap1g_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_repack_cap1g_20260404.txt)

Measured outcome:

- private non-repack baseline:
  - `246.67 tok/s`
- public reference baseline on the same probe:
  - `307.97 tok/s`
- selective repack with `+1 GiB` cap:
  - `257.31 tok/s`

What changed technically:

- the first selective version exposed a real bug:
  - upload would repack tensors that alloc sizing had left plain
- fixing `set_tensor()` to respect the same selector made the path stable and benchmarkable

Conclusion:

- selective repack is now a real working prompt-path experiment
- it improves slightly over the current private baseline
- but it is still well below the public reference prompt speed
- widening the stored layout at upload time is not enough by itself
- the next step is to isolate whether repacking:
  - FFN-only
  - attention-only
  - or the mixed late-layer subset
  changes the result in a meaningful way

### Attempt 6: family isolation on selective repack

Artifacts:

- [qwopus_prompt_probe_private_repack_ffn_cap1g_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_repack_ffn_cap1g_20260404.txt)
- [qwopus_prompt_probe_private_repack_attn_cap1g_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_repack_attn_cap1g_20260404.txt)

Measured outcome:

- FFN-only repack:
  - `257.27 tok/s`
- attention-only repack:
  - `256.15 tok/s`
- mixed selective repack:
  - `257.31 tok/s`
- public reference baseline:
  - `307.97 tok/s`

Conclusion:

- the slowdown is not specific to FFN or attention families
- any path that widens `TQ3_4S` into the larger repacked storage loses roughly the same amount
- the core problem is the repacked layout itself:
  - more bytes moved
  - not better enough access locality to pay for that expansion
- the next serious moonshot is no longer “select a better subset”
- it is:
  - a same-size prompt-fast layout
  - or a truly native prompt kernel that consumes the original packed payload without surrogate widening

### Attempt 7: same-size `pack4` layout

Idea:

- keep `TQ3_4S` at the same 16-byte block size
- repack each 8-value group into one contiguous 32-bit word:
  - `[d_byte, qs0, qs1, qs2]`
- goal:
  - one leader load per group
  - no widened storage
  - no extra VRAM budget needed

Files in private `llama.cpp`:

- `ggml/src/ggml-cuda/tq3_4s_repack.cuh`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/mmq.cu`
- `ggml/src/ggml-cuda/mmq.cuh`

Artifact:

- [qwopus_prompt_probe_private_pack4_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_pack4_20260404.txt)

Measured outcome:

- same-size `pack4`:
  - `254.08 tok/s`
- public reference baseline:
  - `307.97 tok/s`

Conclusion:

- even a same-size contiguous-group layout does not recover the prompt-speed gap
- the bottleneck is deeper than:
  - repeated scale decode
  - widened repack storage
  - or simple `d + qs` fetch locality
- the remaining serious path is now a true native prompt kernel or a more fundamental dataflow change, not another storage shuffle
