# Q4_0_TQ Launch Pad

Date: 2026-03-27

This note upgrades the earlier sketch into an execution plan for the
weight-native `Q4_0_TQ` path.

Goal:

- keep `TQ3_0` as the KV-oriented `3.5 bpv` format
- define one concrete weight-first `3.5 bpv` candidate
- turn that candidate into an implementation task list

Short conclusion:

- `TQ3_0` is already the right `3.5 bpv` KV format
- a dedicated weight variant is worth exploring only if it gives a cleaner
  matmul contract than current `TQ3_0`
- the first candidate should be a runtime-native weight format, not another transform-heavy format
- user-visible naming should be `Q4_0_TQ`
- internal prototype naming can stay `TQ3_W_v0` until a real ggml type exists

## Current Ground Truth

`TQ3_0` is already `3.5 bpv`:

- 12 bytes packed indices
- 2 bytes block scale
- 14 bytes per 32 values

So the question is not "do we need a 3.5-bit format?"

The real question is:

- do we want one `3.5 bpv` format for both KV and weights
- or one for KV and one for weights

## Design Decision

We should separate the roles.

### Keep `TQ3_0` for KV

Reason:

- it already wins on memory footprint
- it now has a viable CUDA KV decode path
- long-context value comes mainly from compact KV storage, not from the cleanest possible matmul contract

### Explore `Q4_0_TQ` for weights

Reason:

- weight inference is dominated by matmul structure
- `TQ3_0` requires format-specific activation rotation and dispatch handling
- that is survivable for KV, but expensive as a long-term weight story

## Dynamic Shapes vs Fixed CUDA Tiles

The model matrices must stay dynamic.

What can stay fixed:

- warp width
- tile sizes
- shared-memory staging shape
- per-kernel iteration constants

What must not become a format contract:

- model hidden size
- FFN expansion size
- prompt length
- chat-template token count

This matters because the current `TQ3_0` weight bug appears only on some
multi-token prefill shapes. That is a kernel-contract problem, not a reason to
define the next format around fixed model dimensions.

The `Q4_0_TQ` target should behave like `q4_0` operationally:

- logical matrices are dynamic
- kernels may process them in fixed-size tiles
- out-of-range lanes may read zeros or null-filled padding
- correctness must not depend on a special prompt shape

## Concrete Candidate: `TQ3_W_v0`

The first candidate should be intentionally boring at runtime.

### Runtime priorities

The kernel should do only this:

1. load packed block
2. unpack 3-bit codes
3. apply simple local scale
4. feed dot or MMA staging immediately

The kernel should not do:

- inverse WHT
- activation pre-rotation
- centroid-domain transforms that need cross-lane reconstruction

### Block contract

Candidate block:

- group size: 32 weights
- payload: 32 x 3-bit codes = 96 bits = 12 bytes
- metadata: 2 bytes total
- total: 14 bytes = `3.5 bpv`

This keeps the same storage budget as `TQ3_0`, but changes the meaning of the bits.

### Proposed metadata layout

Use the 2 metadata bytes as:

- 1 byte shared exponent or scale code for the whole block
- 1 byte refinement field

The refinement byte is the main design lever.

Three plausible uses for it:

1. secondary scale for one half-block of 16 values
2. offset / zero-point style correction
3. tiny codebook selector for the block

For `TQ3_W_v0`, pick the simplest option:

- byte 0: block scale
- byte 1: half-block scale delta

That gives:

- 16 values with base scale
- 16 values with adjusted scale

Why this is the right first candidate:

- still exactly `3.5 bpv`
- runtime is simple
- better local fidelity than one flat block scale
- still tile-friendly

### Proposed block struct

Reference C-style sketch:

```c
#define QK_TQ3_W_V0 32

typedef struct {
    uint8_t qs[12];   // 32 x 3-bit codes
    uint8_t s0;       // base block scale code
    int8_t  ds1;      // signed half-block scale delta
} block_tq3_w_v0;
```

Interpretation:

- `qs` holds the 32 packed 3-bit values
- `s0` encodes the primary scale for values 0..15
- `ds1` adjusts the scale for values 16..31

Derived scales:

- `scale0 = decode_scale(s0)`
- `scale1 = scale0 * decode_delta(ds1)`

This keeps the decode contract simple:

- one scale for the first half
- one derived scale for the second half
- no transform stage

## Proposed Decode Contract

Interpret each 3-bit code as a signed level from a fixed 8-level table.

Example signed levels:

```text
[-7, -5, -3, -1, +1, +3, +5, +7]
```

Then reconstruct as:

- first 16 values: `w = s0 * lut[q]`
- second 16 values: `w = s1 * lut[q]`

Where:

- `s0` comes from the main scale byte
- `s1` is derived from the base scale plus the refinement byte

This gives a block-local two-scale structure without increasing storage.

### Reference decode pseudocode

```c
static const int8_t TQ3_W_LEVELS[8] = { -7, -5, -3, -1, 1, 3, 5, 7 };

float scale0 = decode_scale_u8(block.s0);
float scale1 = scale0 * decode_delta_i8(block.ds1);

for (int i = 0; i < 32; ++i) {
    const uint8_t q = unpack_3bit(block.qs, i);
    const float level = (float) TQ3_W_LEVELS[q];
    const float scale = i < 16 ? scale0 : scale1;
    out[i] = scale * level;
}
```

Where:

- `decode_scale_u8()` should map the scale byte to a positive real scale
- `decode_delta_i8()` should map the signed delta to a small multiplicative correction

Recommended first mapping:

- `decode_scale_u8(s) = exp2((int)s - 127)` or a simpler affine proxy during prototyping
- `decode_delta_i8(d) = exp2(d / 32.0f)`

This is not yet the final quantizer choice.
It is the first runtime contract to benchmark.

## Why `TQ3_W_v0` Is Better Aligned To Weight Kernels

Compared with `TQ3_0`, this candidate:

- has no runtime transform dependency
- has no activation-domain contract
- can be unpacked directly into q8/fp16 staging
- matches MMQ/MMVQ-style kernel thinking much more naturally

That means:

- simpler MMVQ path
- simpler MMQ bridge
- easier profiling
- easier reasoning about whether the format is worth keeping

## What We Give Up Relative To `TQ3_0`

Likely tradeoffs:

- lower quantization quality at the same `3.5 bpv`
- less elegant signal modeling
- weaker reconstruction on difficult blocks

That is acceptable for a first candidate.

The first candidate does not need to beat `TQ3_0` on quality.
It needs to answer a sharper question:

- can a weight-native `3.5 bpv` format materially reduce runtime cost?

## Go / No-Go Criteria

Continue only if `TQ3_W_v0` satisfies at least one:

1. noticeably simpler runtime than `TQ3_0`
2. noticeably faster PP than `TQ3_0`
3. noticeably faster TG than `TQ3_0`
4. similar quality with less kernel complexity

Stop if:

- quality is obviously unacceptable
- runtime is not simpler
- performance does not improve enough to justify a new type

## Launch Criteria

Call `Q4_0_TQ` launch-ready only if all of these are true:

1. `llama-quantize` can emit the format from a trusted higher-precision source
2. GGUF load path preserves the intended format identity
3. plain prompt and chat-template prompt both stay numerically sane
4. backend-op guard cases stay green on prompt-like and decode-like shapes
5. perplexity is within the agreed guardrail relative to the chosen baseline
6. runtime contract is simpler than rotated `TQ3_0` weights

If a candidate misses `3`, `4`, or `5`, do not optimize it yet.

## Implementation Plan

### Phase 0: Paper Contract

Deliverables:

- exact block struct for `TQ3_W_v0`
- exact decode formula
- exact q8 bridge target for MMQ-style staging

Tasks:

- [x] define `block_tq3_w_v0` bit layout
- [x] choose fixed 8-level decode table
- [x] write decode pseudocode for one 32-value block
- [ ] define final scale and scale-delta encoding

### Phase 1: Scalar Reference

Deliverables:

- one scalar quantize/dequant reference implementation
- one block-error report against trusted source tensors

Checkpoint on current prototype:

- implementation exists in `llama.cpp/tests/test-tq3w-prototype.cpp`
- exact size stays at `14` bytes per block
- synthetic corpus result:
  - `Q4_0_TQ_v0 RMSE = 0.173998`
  - `TQ3_0 RMSE = 0.211066`
  - `Q4_0 RMSE = 0.086597`
  - `Q4_0_TQ_v0 dot = 0.984397`
  - `TQ3_0 dot = 1.191813`
  - `Q4_0 dot = 0.488415`
- real tensor corpus result from `Qwen3.5-9B-Q8_0.gguf`
  - source tensors:
    - `blk.0.attn_qkv.weight`
    - `blk.0.ffn_down.weight`
    - `blk.31.attn_k.weight`
  - `Q4_0_TQ_v0 RMSE = 0.002913`
  - `TQ3_0 RMSE = 0.002722`
  - `Q4_0 RMSE = 0.001414`
  - `Q4_0_TQ_v0 dot = 0.016470`
  - `TQ3_0 dot = 0.015388`
  - `Q4_0 dot = 0.007934`

Interpretation:

- the candidate is already better than rotated `TQ3_0` on this scalar synthetic corpus
- on the real Qwen tensor corpus it is close to `TQ3_0`, but still slightly worse
- it is still about `2x` worse than `Q4_0`
- that is acceptable for Phase 1 because the runtime contract is much simpler

Tasks:

- [x] finalize first-pass `decode_scale_u8()` and `decode_delta_i8()`
- [ ] quantize one real tensor corpus block-by-block
- [x] quantize one real tensor corpus block-by-block
- [x] compare RMSE against `TQ3_0`
- [x] confirm the format still lands at exactly `3.5 bpv`
- [x] add dot-product proxy gate against `TQ3_0`
- [ ] add `Q4_0` baseline to a real tensor corpus run

Exit rule:

- quality is not obviously catastrophic
- implementation remains simpler than rotated `TQ3_0`

### Phase 2: Backend Contract

Deliverables:

- one exact backend-op repro for decode-like shapes
- one exact backend-op repro for prompt-like shapes
- one documented rule for padding and out-of-range lanes

Tasks:

- [ ] define the q8/fp16 activation contract for `Q4_0_TQ`
- [ ] define zero-fill behavior for partial tiles
- [ ] document row/column-major staging assumptions
- [ ] prove dynamic model shapes are handled by padding, not by special cases

Exit rule:

- same logical op works for `n=1` and prompt-like `n>1`
- no activation rotation is required

### Phase 3: CUDA Microkernel

Deliverables:

- one block-level CUDA kernel path
- one benchmark against current `TQ3_0` and `q4_0`

Tasks:

- [ ] implement one direct packed-block decode path
- [ ] feed MMQ-style staging without transform work
- [ ] benchmark prompt-like shapes first
- [ ] keep the smallest possible kernel surface until correctness is stable

Exit rule:

- prompt-like backend-op cases are green
- speed is at least directionally competitive with current rotated `TQ3_0`

### Phase 4: End-to-End Launch Pad

Deliverables:

- quantize CLI path
- GGUF metadata path
- model-level sanity and perplexity gates

Tasks:

- [ ] add a real `Q4_0_TQ` ggml type or equivalent runtime hook
- [ ] wire `llama-quantize` end to end
- [ ] add backend-op tests for both plain and chat-like prompt shapes
- [ ] add perplexity comparison to the required gate
- [ ] run one long-context benchmark with `TQ3_0` KV

Exit rule:

- `Q4_0_TQ` weights + `TQ3_0` KV is correct first
- only then start speed tuning

## Immediate Recommendation

Do not spend more time trying to turn rotated `TQ3_0` weights into the final
weight format.

Use the current `TQ3_0` findings as input:

- KV path stays valuable
- dynamic-shape fragility in prefill is a warning sign
- the next weight candidate should behave like `q4_0` at runtime
- zero/null padding is acceptable inside tiles

That is the practical launch pad for `Q4_0_TQ`.

Exit criteria:

- block layout is unambiguous
- runtime decode has no transform operations

### Phase 1: CPU Reference Path

Deliverables:

- scalar reference quantize
- scalar reference dequantize
- block-level correctness test

Tasks:

- [ ] add reference quantize function in a scratch note or prototype file
- [ ] add reference dequantize function
- [ ] compare block MSE against `TQ3_0` on sample tensors

Exit criteria:

- decode correctness is proven
- quality is at least plausible enough for a GPU prototype

### Phase 2: CUDA Microkernel

Deliverables:

- one block-level CUDA microbenchmark
- one block dot kernel against q8 activations

Tasks:

- [ ] implement one CUDA unpack-and-dot microkernel for `TQ3_W_v0`
- [ ] benchmark against current `TQ3_0` MMVQ block math cost
- [ ] record instruction and memory behavior

Exit criteria:

- block runtime is clearly simpler than `TQ3_0`
- no transform logic appears in the hot path

### Phase 3: MMVQ-Style Integration

Deliverables:

- one vector-matmul path using the new block

Tasks:

- [ ] integrate `TQ3_W_v0` into an MMVQ-like path first
- [ ] use the same tiny regression shapes we already use for `TQ3_0`
- [ ] benchmark TG-like decode shapes

Exit criteria:

- TG-like performance is meaningfully better than `TQ3_0`
- no regression in correctness harness

### Phase 4: MMQ / Prefill Evaluation

Deliverables:

- one MMQ-side bridge or direct tile path

Tasks:

- [ ] decide whether `TQ3_W_v0` should bridge through q8 staging or direct MMA packing
- [ ] benchmark PP against current `TQ3_0`
- [ ] compare end-to-end PP and TG against q4 baseline

Exit criteria:

- either clear value is shown
- or the variant is discarded early

## Immediate Next Tasks

These are the next concrete tasks to execute now.

1. Lock the format contract for `TQ3_W_v0`.
2. Write the reference decode formula for one block.
3. Define the exact q8 bridge target for the first CUDA microkernel.
4. Reuse the existing `TQ3_0` microbenchmark style to build a `TQ3_W_v0` block benchmark.

## Recommended First Build Slice

Do not add a new GGML type yet.

First build:

- one standalone design note
- one standalone block reference implementation
- one standalone CUDA microbenchmark

Only after that:

- decide whether the type deserves full integration

## Practical Recommendation

The best architecture today is:

- `TQ3_0` for KV cache
- current `TQ3_0` kernels as the transitional weight path
- `TQ3_W_v0` as the experiment for a true weight-native `3.5 bpv` format

That is the cleanest way to avoid mixing:

- KV-first design goals
- weight-first kernel goals
