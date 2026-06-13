# TQ3 Multi-Row Tile Kernel Design

Date: 2026-04-04

## Goal

Move beyond CTA parity and change the prompt-path arithmetic intensity by
reusing the same compact predecoded `TQ3_4S` weight slab across multiple prompt
rows.

This design is the next branch after the exact CTA microbench reached only a
small win over the bridge:

- bridge CTA:
  - `2.220 us/launch`
- best exact compact-coop variant:
  - `2.146 us/launch`
  - `1.034x`

That is real progress, but not enough. The only credible next lever is to
raise reuse.

## Core Change

Old effective dataflow:

```text
for each row:
  load / stage weight tile
  dp4a
```

New target dataflow:

```text
load / predecode weight tile once
reuse it across multiple prompt rows
```

## Threadblock Mapping

Recommended sketch:

- CTA: `256` threads = `8` warps
- each CTA owns one `K=256` slab
- one slab contains `8` compressed `TQ3_4S` blocks

### Predecode phase

- warp `0..7` each owns one `32`-value compressed weight block
- each owner warp expands its block into the compact exact contract:
  - `8` packed4 subgroup-halves
  - `4` subgroup scales

Shared layout:

```text
w_p4[8][8]   // 8 blocks, 8 subgroup-halves per block
w_sc[8][4]   // 8 blocks, 4 subgroup scales per block
```

This keeps:

- no decode in the hot dot loop
- dp4a-ready packed weights
- exact numeric behavior

### Consume phase

After `__syncthreads()`:

- warp `w` processes `4` prompt rows
- rows:
  - `4*w + 0`
  - `4*w + 1`
  - `4*w + 2`
  - `4*w + 3`

So one CTA covers:

- `32` prompt rows
- against the same predecoded weight slab

## Register Blocking

Each active consumer lane keeps:

```text
float acc0, acc1, acc2, acc3;
```

for the four rows owned by the warp.

This changes the hot loop from:

```text
w = shared_weight
x = one row
acc += dp4a(x, w)
```

to:

```text
w = shared_weight

acc0 += dp4a(x0, w)
acc1 += dp4a(x1, w)
acc2 += dp4a(x2, w)
acc3 += dp4a(x3, w)
```

## Before vs After Reuse

For one subgroup-half pack:

### Before

- one shared weight load
- one `dp4a` stream
- one accumulator

Effective:

- `1` weight load -> `1` row use

### After

- one shared weight load
- `4` `dp4a` operations
- `4` row accumulators

Effective:

- `1` weight load -> `4` row uses

So the dp4a work per shared weight load increases by about `4x`.

That does not guarantee a full `4x` kernel speedup, because:

- activation loads still scale with rows
- reductions still cost something
- register pressure rises

But it is the first design that changes arithmetic intensity materially enough
to make a ~2x class win plausible.

## MMA Loop Sketch

For each `blk in 0..7` and subgroup-half `lane in 0..7`:

```text
w  = w_p4[blk][lane]
sw = w_sc[blk][subgroup]

x0 = packed q8 half for row0
x1 = packed q8 half for row1
x2 = packed q8 half for row2
x3 = packed q8 half for row3

acc0 += dp4a(w, x0) * sw * dx0
acc1 += dp4a(w, x1) * sw * dx1
acc2 += dp4a(w, x2) * sw * dx2
acc3 += dp4a(w, x3) * sw * dx3
```

Hot-loop rule:

- no bit ops
- no scale decode
- no lane packing

Only:

- shared weight reads
- activation pack reads
- `dp4a`
- multiply-accumulate

## Register Pressure

Main new cost:

- `4` accumulators per lane instead of `1`
- extra activation packs per iteration

Likely pressure points:

- `acc0..acc3`
- temporary activation packs `x0..x3`
- subgroup scale `sw`

This can lower occupancy, but that trade is intentional: we are spending
registers to buy reuse.

## Occupancy Tradeoff

Expected trade:

- lower occupancy than the single-row compact-coop kernel
- higher arithmetic intensity
- better amortization of weight staging

This is acceptable if:

- shared weight tile is reused enough times per CTA
- the kernel becomes compute-heavier and less staging-bound

## Measured Results

Current exact microbench artifact:

- [bench_tq3_multrow_cta_20260404.txt](/home/awee/code/tan_llama/artifacts/bench_tq3_multrow_cta_20260404.txt)

What actually happened:

- `2` rows/warp without activation prepack:
  - exact
  - about parity with single-row baseline
- `2` rows/warp with packed activation halves:
  - exact
  - modest win in the best run
- `4` rows/warp with packed activation halves:
  - exact
  - best measured `2.607 us/launch`
  - `1.693x` faster than the comparable single-row baseline
- `8` rows/warp with packed activation halves:
  - exact
  - fell back to about parity
  - register pressure erased the reuse gain

What failed:

- precombined scale tensors for mult-row:
  - both `2`-row and `4`-row variants regressed badly
  - more shared-memory scale traffic is not the answer

So the current best interpretation is:

- weight reuse alone is not enough
- weight reuse plus packed activation halves is the first real breakthrough
- current sweet spot is the plain `4`-row packed-activation contract
- the next branch should improve that contract, not replace it with either
  larger shared scale tensors or even wider row reuse

## What This Design Keeps

- cooperative compact predecode
- shared-memory weight tile
- no extra global traffic for weights
- exact numeric contract

## What This Design Changes

- shared weights are no longer consumed by only one row
- each warp becomes a multi-row compute warp
- register blocking becomes explicit

## Sketch Code

A compilable sketch exists in the private `llama.cpp` tree:

- [tq3-multrow-kernel-sketch.cu](/home/awee/code/llama.cpp/tests/tq3-multrow-kernel-sketch.cu)

That file is not yet wired into production `mmq.cu`. It exists to make the new
mapping concrete before porting it into the runtime path.
