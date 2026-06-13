# TQ3 Native Prompt Kernel Design

Date: 2026-04-04

## Goal

Design a **true native prompt kernel** for `TQ3_4S` that keeps the current
quality story but closes the prompt-speed gap to `Q3_K_S`.

This note is not another vague moonshot sketch. It answers:

- what the kernel must compute
- what the current bridge path is wasting
- what a native kernel would load, store, and accumulate
- which parts are mathematically exact
- which claims are already supported by a small experiment

## Mathematical Invariant

`TQ3` quantization stores weights in a rotated domain.

The forward transform used by the quantizer is:

```text
R(x) = H(Sx) / sqrt(32)
```

where:

- `S` is the fixed sign-flip diagonal from `TQ3_0_SIGNS`
- `H` is the 32-point Walsh-Hadamard transform

This transform is orthogonal, so:

```text
<w, x> = <R(w), R(x)>
```

That is the key fact that makes a native prompt kernel possible:

- weights can stay in the rotated domain
- activations only need to be rotated once per 32-value block
- the dot product is still exact apart from quantization error

## Exact `TQ3_4S` Block Contract

From [ggml-common.h](/home/awee/code/llama.cpp/ggml/src/ggml-common.h):

- block size: `QK_TQ3_0 = 32`
- `block_tq3_4s` layout:
  - `d[4]`: 4 scale bytes, one per group of 8
  - `qs[12]`: 32 packed 3-bit indices
- total: `16 B` per 32 weights

So for `256` logical weights:

- `8` blocks
- raw weight bytes = `8 * 16 = 128 B`

For `Q3_K`:

- `256` weights per block
- block size = `110 B`

So raw global weight traffic is:

- `TQ3_4S`: `128 B / 256 weights`
- `Q3_K`: `110 B / 256 weights`

That is only a `1.16x` raw-byte penalty, not a `2x` or `3x` penalty.
The full prompt gap must therefore come from **decode/dataflow**, not just raw
GGUF bytes.

## Current Bridge Path

The current prompt path still effectively behaves like this:

```text
weights (TQ3_4S)
  -> unpack into a q8/q3k-like MMQ tile
  -> feed generic q8-style MMA path

activations
  -> float source
  -> rotate
  -> quantize to q8_1
  -> feed generic MMQ path
```

Even after cleanup, this still means:

1. weights are expanded into a large MMQ x-tile
2. activations are staged into surrogate `q8`
3. the MMA core does not consume the original compressed weight contract

## Important Calculation: Tile Size Is Not The Main Difference

From [mmq.cuh](/home/awee/code/llama.cpp/ggml/src/ggml-cuda/mmq.cuh):

```text
MMQ_MMA_TILE_X_K_Q8_0 = 2*128 + 2*128/4 + 4 = 324 ints = 1296 B
MMQ_MMA_TILE_X_K_Q3_K = 2*128 +   128/2 + 4 = 324 ints = 1296 B
```

So the **shared-memory tile footprint is identical** for:

- `Q8_0`-style bridge tiles
- `Q3_K`-style tiles

This matters because it kills one bad hypothesis:

- the remaining gap is **not** because `TQ3_4S` uses a larger x-tile than
  `Q3_K`

The difference has to be in:

- how expensive it is to populate the tile
- how much extra staging happens before the tile
- or whether the tile itself should disappear entirely

## Activation-Side Cost

The current fused activation quantizer in
[quantize.cu](/home/awee/code/llama.cpp/ggml/src/ggml-cuda/quantize.cu)
does this per 32-value activation block:

1. read `32` floats: `32 * 4 = 128 B`
2. apply sign flips + 32-point WHT
3. quantize to `32` int8 values + one scale

For `128` activation values, it writes one `block_q8_1_mmq`:

- `128` int8 values
- `16 B` of scale/sum metadata
- total `144 B`

For `256` activation values:

- input read: `256 * 4 = 1024 B`
- q8 output write: `2 * 144 = 288 B`

Earlier fused-activation cleanup removed the extra temporary rotated float
buffer, but prompt speed still regressed to:

- [qwopus_prompt_probe_private_fused_q8_tq3_nocnv_nowarm_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_fused_q8_tq3_nocnv_nowarm_20260404.txt)
- `241.60 tok/s`

Conclusion:

- activation-side temp-buffer traffic is real
- but it is **not** the dominant explanation for the full prompt-speed gap

## What A Native Prompt Kernel Should Do

The native prompt kernel should stop expanding `TQ3_4S` into a surrogate MMQ
x-tile.

Instead, for each `256`-value K-slab:

1. rotate and quantize the activation tile once
2. keep that activation tile in shared memory as compact `q8`
3. load compressed `TQ3_4S` weights as `8` raw `16 B` blocks
4. decode each 3-bit group directly into `packed4` int8 registers
5. use `dp4a` on those native 4-lane packs
6. accumulate with the subgroup scale and activation scale

In other words:

```text
compressed TQ3_4S weight block
  -> register unpack (not giant x_tile)
  -> dp4a against rotated q8 activation packs
  -> float accumulation
```

## Exact Inner Product Contract

For one `TQ3_4S` block:

- 32 weights
- 4 subgroups of 8
- each subgroup stores:
  - 1 scale byte
  - 8 packed 3-bit indices

Let:

- `dA` be the activation q8 scale for the 32-value activation block
- `dW[g]` be the decoded scale for subgroup `g`
- `L(idx)` map a 3-bit index to one of the 8 centroid levels
- `qa[k]` be the rotated activation quantized to int8

Then the block dot is:

```text
dot_block = sum_{g=0..3} dA * dW[g] * sum_{r=0..7} L(idx[g,r]) * qa[g,r]
```

That inner sum can be implemented as `dp4a`:

- 8 values per subgroup
- 2 `dp4a` ops per subgroup
- 4 subgroups per block
- `8 dp4a` ops per 32-value block

For one `256`-value K-slab:

- `8` blocks
- `64 dp4a` ops
- `32` scale applications

## Why This Is Better Than The Bridge

### Current bridge

For `256` weights:

- raw compressed weights: `128 B`
- expanded x-tile contract: `1296 B`

Expansion factor:

```text
1296 / 128 = 10.125x
```

So the current prompt path is paying more than a `10x` expansion on the x-side
just to enter the generic MMA contract.

### Native prompt contract

For the same `256` weights:

- raw compressed weights in shared or registers: about `128 B`
- no giant surrogate x-tile
- unpack only the few 4-lane chunks needed for the current MMA step

This is the real reason a native kernel is still worth chasing even after the
smaller layout experiments failed.

## Small Experiment That Supports This Direction

Block-level microbenchmark:

- binary:
  - [bench-tq3-native-dot](/home/awee/code/tan_llama/artifacts/bin/bench-tq3-native-dot)
- artifact:
  - [bench_tq3_native_dot_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_native_dot_20260404.txt)

Measured result:

- native direct block dot:
  - `23.994 us/launch`
- dequant-then-dot baseline:
  - `35.188 us/launch`
- speedup:
  - `1.467x`
- numerical difference:
  - `0`

## Fourth Small Experiment: Cooperative Compact Predecode CTA

The first CTA-level compact-predecode attempt was exact but slow because the
same 8 consumer lanes were also doing the predecode work. That serialized the
predecode in the wrong shape and hid the value of the compact contract.

So the next microbench changed only the **producer shape**:

- keep the exact compact contract
  - `8` packed4 subgroup-halves
  - `4` subgroup scales
- but stage it cooperatively with the same 32-lane block-fill pattern that the
  fast bridge path already uses

Artifact:

- [bench_tq3_native_cta_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_native_cta_20260404.txt)

Measured result:

- bridge CTA:
  - `2.239 us/launch`
- native raw-block CTA:
  - `4.412 us/launch`
  - `0.507x`
- native cached-raw-block CTA:
  - `4.429 us/launch`
  - `0.506x`
- compact predecode CTA with consumer-lane predecode:
  - `4.467 us/launch`
  - `0.501x`
- cooperative compact-predecode CTA:
  - `2.189 us/launch`
  - `1.023x`
- numerical difference:
  - `0`

This is the first CTA-level result that says something actionable:

- the compact contract itself is not the problem
- raw native unpack in the hot loop is the problem
- compact predecode must be staged cooperatively before the dot loop

So the next kernel target is no longer a raw-block native slab. It is:

- cooperative compact predecode
- followed by a native inner loop that consumes only the compact exact packs

## Immediate Implementation Target

The next production branch should not start from raw `block_tq3_4s` unpack in
the hot loop.

It should start from the first exact CTA contract that hit the bridge envelope:

1. cooperatively predecode each `TQ3_4S` block into:
   - `8` packed4 subgroup-halves
   - `4` subgroup scales
2. keep that compact contract in shared memory
3. let only the consumer lanes run the `dp4a` loop
4. postpone activation-side prepack until it is exact in the microbench

This is the first branch that is worth porting into production CUDA code.

Update after the correctness-first step-back:

- that sentence is too optimistic for the current blockslot branch
- cooperative blockslot staging is now proven correct in isolation
- but the clean two-path benchmark only reaches parity with the bridge:
  - `bridge_clean: 2.216 us`
  - `blockslot_clean: 2.214 us`
  - `1.001x`
- so the current blockslot contract is a valid experimental baseline, not a
  production port candidate

## What The Last Expert-Guided Variants Changed

Two direct tests narrowed the next step further:

1. lighter producer mapping plus `bfe` unpack
   - exact
   - improved the cooperative path slightly
   - still only reached near-parity with the bridge
2. register-scale broadcast with `__shfl_sync`
   - exact
   - badly regressed, both inline and hoisted

So the next production branch should keep:

- cooperative compact predecode
- shared-memory scales
- lighter producer ownership where one lane owns one block decode

And it should avoid:

- `__shfl_sync` scale delivery inside or around the hot dot loop

Latest exact CTA microbench says the best current contract is now:

- bridge CTA:
  - `2.220 us/launch`
- lane-owned block decode + `bfe` unpack:
  - `2.146 us/launch`
  - `1.034x`
- exact:
  - diff `0`

So the first production kernel port should now use:

1. cooperative compact predecode
2. one-lane-per-block producer ownership
3. `bfe`-based unpack helper
4. shared-memory scales

This does **not** prove the full prompt kernel will be `1.455x` faster.
But it does prove an important lower-level fact:

- the native block contract is cheaper than a dequant-bridge for the same math

That is enough to justify the next branch.

What it does **not** justify:

- porting the current cooperative blockslot staging path into `mmq.cuh`
- using the mixed CTA harness as the primary correctness signal

The new discipline for this line is:

1. prove correctness in a standalone test
2. benchmark only `bridge` vs the new contract in a clean two-path bench
3. only then consider runtime integration

## Second Small Experiment: Activation-Side Tax

Activation staging is also now measured in isolation.

- binary:
  - [bench-tq3-q8-stage](/home/awee/code/tan_llama/artifacts/bin/bench-tq3-q8-stage)
- artifact:
  - [bench_tq3_q8_stage_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_q8_stage_20260404.txt)

Measured result:

- plain `q8` staging:
  - `44.486 us/launch`
- `TQ3` rotate + `q8` staging:
  - `55.144 us/launch`
- overhead:
  - `1.240x`

So the activation-side WHT tax is real, but it is still materially smaller than
the weight-side bridge penalty measured by the native-dot microbench.

## Mathematical Ceiling For The Next Branch

Artifact:

- [tq3_native_prompt_math_20260404.txt](/home/awee/code/tan_llama/artifacts/tq3_native_prompt_math_20260404.txt)

Using the public fixed prompt-probe anchor:

- current public prompt speed:
  - `307.97 tok/s`
- measured native weight-contract gain:
  - `1.467x`

If a new prompt kernel only inherits that local weight-side speedup, the simple
ceiling is:

```text
307.97 * 1.467 = 451.79 tok/s
```

That is important because it rules out one bad assumption:

- **swapping only the inner dequant-bridge for a native dot is not enough to
  reach `600 tok/s`**

To hit `600 tok/s` from the same public anchor, we still need:

```text
600 / 307.97 = 1.948x total
1.948 / 1.467 = 1.328x extra beyond the measured native weight-contract gain
```

To match the historical `Q3_K_S` prompt bar (`689 tok/s`), we still need:

```text
689 / 307.97 = 2.237x total
2.237 / 1.467 = 1.525x extra beyond the measured native weight-contract gain
```

So the next branch must do more than change one inner contract.
It must also improve **CTA-level dataflow**:

- better activation reuse
- less shared-memory tile traffic
- less bridge write/read churn
- better overlap between producer and consumer warps

## Working-Set Calculation For A Native CTA

Take one `256`-value K-slab and a CTA that amortizes one activation tile across
`8` output rows.

### Current bridge-like contract per row

- weight-side surrogate x-tile:
  - `1296 B`
- amortized activation tile:
  - `288 B / 8 = 36 B`
- total:
  - `1332 B / row`

### Native prompt contract per row

- raw compressed weights:
  - `128 B`
- amortized activation tile:
  - `36 B`
- total:
  - `164 B / row`

Ratio:

```text
1332 / 164 = 8.12x
```

This is why a native CTA is still worth building even after the smaller
micro-optimizations failed:

- at the CTA working-set level, the bridge contract is still grossly inflated

But the ceiling calculation above also tells us something equally important:

- **that working-set reduction will only matter if the kernel actually turns it
  into better occupancy / overlap / throughput**
- a naive native rewrite can still miss the bar

## Proposed Native CTA Shape

Keep the successful parts of the current prompt path:

- quantize/rotate activations once per K-slab
- share that activation tile across output rows

Change only the x-side contract.

Proposed CTA responsibilities:

1. **Producer warps**
   - read float activations
   - apply sign + WHT
   - quantize to compact q8 packs in shared memory

2. **Consumer warps**
   - load compressed `TQ3_4S` blocks for a row tile
   - unpack subgroup codes into `packed4` int8 registers
   - run `dp4a`
   - accumulate subgroup-scaled partial sums

3. **Writeback**
   - same float accumulation/writeback contract as current MMQ path

## Third Small Experiment: CTA-Level Reality Check

Artifact:

- [bench_tq3_native_cta_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_native_cta_20260404.txt)

Measured result on an exact 8-row CTA microbench:

- compact shared predecode CTA:
  - `2.215 us/launch`
- native on-demand raw-block CTA:
  - `4.564 us/launch`
  - `0.485x` of compact predecode
- native raw-block cached-in-shared CTA:
  - `4.680 us/launch`
  - `0.473x` of compact predecode
- numerical differences:
  - `0` for both native variants

This is the first CTA-level result that really changes the design.

It says:

- the slow part is not just redundant global loads of the raw `16 B` block
- caching raw `block_tq3_4s` in shared still does not help
- the expensive part is repeated bit unpack + level materialization inside the hot inner loop

So the best design is **not**:

- giant MMQ bridge tile
- or raw on-demand unpack per `dp4a`

The best design is now:

- a **compact predecode CTA**
- predecode each block into exactly `8` packed4 subgroup-halves + `4` subgroup scales
- keep that compact contract in shared memory
- run the inner loop only on those compact predecoded packs

That preserves the important insight from the native direction:

- no huge surrogate x-tile

but also preserves the key practical lesson from the CTA microbench:

- some staging is absolutely required
- it just has to be the **minimal exact staging contract**, not the old MMQ compatibility tile

## Implementation Ladder

The next code branch should be built in this order:

1. **One-row native slab kernel**
   - one warp handles one `256`-value K-slab for one row
   - activations come from one compact shared `q8` tile
   - weights stay as raw `block_tq3_4s`
   - goal:
     - validate exactness against the current bridge path

2. **Multi-row CTA**
   - one producer group stages the rotated `q8` activation slab once
   - multiple consumer warps reuse it across several rows
   - goal:
     - convert the `8.12x` working-set reduction into real prompt speed

3. **Double-buffered producer/consumer overlap**
   - while consumer warps finish slab `k`, producer warps stage slab `k+1`
   - goal:
     - hide more of the activation-side staging cost

4. **Only then evaluate wider MMA ideas**
   - if the native CTA still underperforms, the next step is a different MMA
     contract or a more aggressive pack, not another decode tweak

## What The Next Code Branch Should Implement

Not another repack.
Not another scalar decode cleanup.
Not another q8 bridge tweak.

The next real branch should:

1. add a dedicated native prompt kernel for `GGML_TYPE_TQ3_4S`
2. load raw `block_tq3_4s` blocks directly
3. keep activations in compact rotated-q8 form
4. unpack weight codes into register `packed4` lanes
5. accumulate with `dp4a`
6. never materialize the large surrogate x-tile

## What This Design Does Not Solve Yet

- decode / MMVQ path for token generation
- `TQ3_1S`
- `TQ3_4SE` / `TQ3_4SV`
- public branch integration

## Runtime Mapping Check

The first real runtime hook test was intentionally narrow:

- `GGML_TYPE_TQ3_4S` was routed to dedicated TQ3 consumer entry points in
  `mmq.cuh`
- but the consumer body still matched the old generic schedule

Measured real prompt result:

- [qwopus_prompt_probe_private_tq3_consumer_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_private_tq3_consumer_20260404.txt)
- old private baseline:
  - `246.67 tok/s`
- hook swap only:
  - `247.25 tok/s`

So the bench win does not map into runtime automatically.
The next runtime branch has to change the real TQ3 consumer contract, not just
the function pointers that dispatch into it.

## Verified Runtime Port

The first runtime port that produced a real prompt-speed gain was narrower than
the earlier moonshot sketches:

- keep the live MMQ structure
- keep the real TQ3 consumer path
- stage prompt-side activation packs in the consumer path before the inner loop

Verified real-model result on the private build:

- [qwopus_prompt_probe_packedx_2row_20260404.txt](/home/awee/code/tan_llama/artifacts/qwopus_prompt_probe_packedx_2row_20260404.txt)
- private build:
  - `343.06 ± 1.30 tok/s`
- apples-to-apples public fork rerun:
  - `325.19 ± 0.83 tok/s`
- gain:
  - about `1.055x`

This does not validate every earlier mult-row idea.
It does validate one specific lesson:

- activation-pack reuse can survive contact with the real runtime

So the design state is now:

- cooperative blockslot alone: correct but parity
- narrow packed-activation runtime port: real win
- full native prompt kernel: still an open longer-range branch

This note is only for the **prompt kernel**.

## Related Docs

- [TQ3_MOONSHOT_MASTER_LOG.md](/home/awee/code/tan_llama/docs/turboquant/TQ3_MOONSHOT_MASTER_LOG.md)
- [TQ3_Q3KS_BOTTLENECK_MAP.md](/home/awee/code/tan_llama/docs/turboquant/TQ3_Q3KS_BOTTLENECK_MAP.md)
- [TQ3_4S_ROTATED_MMQ_KERNEL_DESIGN.md](/home/awee/code/tan_llama/docs/turboquant/TQ3_4S_ROTATED_MMQ_KERNEL_DESIGN.md)
- [BENCHMARK_PROTOCOL.md](../../procedures/BENCHMARK_PROTOCOL.md)
