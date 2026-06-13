# FP4 KV Plan — HiFloat4 and Fused Decode

**Date:** 2026-04-13

## Verdict

HiFloat4-style KV is worth planning, but not as a blind replacement for current 4-bit KV.

The right framing is:

- `HiFloat4` is a **same-class bitrate replacement** for `q4`-style KV, aimed at **better accuracy / dynamic range**
- `Fused-KV-Cache` is a **decode-kernel/dataflow idea**, aimed at **lower decode overhead**
- `TriAttention` is a **retention / eviction policy**, aimed at **smaller active KV**

These are complementary, not substitutes.

## What The References Actually Say

### TriAttention handover

Current blocker is still runtime eviction correctness / observability:

- probe path does not yet give a trusted eviction signal
- score path recently hit non-finite values and needed guarding
- coefficient tuning is premature until runtime eviction is proven live

Source:

- [TRIATTENTION_HANDOVER.md](TRIATTENTION_HANDOVER.md)

### HiFloat4

HiFloat4 inference paper describes:

- 64 x 4-bit values per unit
- 32 bits shared scaling metadata
- average `4.5 bits/value`
- three-level scaling hierarchy
- better average accuracy than NVFP4 in their inference study

Important implication for us:

- this is **not smaller than a typical 4-bit block format in spirit**
- the value is **range handling / accuracy**, not immediate size reduction versus `q4`-class KV

### Fused-KV-Cache

The repo demonstrates:

- single-token decode
- one fused kernel for `QK^T`, softmax, and `AV`
- score vector kept in shared memory instead of written to global

That is a valid decode direction for llama.cpp, but it is a **kernel fusion optimization**, not a quant format.

## What This Means For Our KV Roadmap

### 1. HiFloat4 should target `q4`-class KV, not `tq3_0`

If we compare against:

- `fp16` or `fp8` KV: HiFloat4 is a strong memory win
- `q4_0` / `q4`-class KV: HiFloat4 is mainly an **accuracy-format trade**
- `tq3_0` KV: HiFloat4 is **larger**, so it cannot replace the turbo line on memory

So the clean question is not:

- "should HiFloat4 replace TQ3 KV?"

It is:

- "can HiFloat4 replace `q4` KV as a better 4-bit baseline?"

### 2. On NVIDIA / llama.cpp, HiFloat4 is software first

The HiFloat4 papers are not a drop-in CUDA/NVIDIA implementation path for us.

For our stack:

- Blackwell has native NVIDIA low-precision directions such as NVFP4
- HiFloat4 itself is not something llama.cpp can natively issue as hardware math today
- so our first implementation would be **software-stored FP4 KV with software dequant**

That means:

- likely **quality gain**
- uncertain decode speed
- no guaranteed throughput win over current `q4` kernels

### 3. Fused decode is orthogonal and likely more important for speed

If we adopt an FP4 KV format, the main decode-speed win probably does **not** come from the format alone.

It comes from:

- smaller KV bandwidth
- fused single-token decode kernel
- no intermediate score-vector global write
- direct dequant-consume path inside attention

That makes `Fused-KV-Cache` relevant as a decode-template reference, especially for:

- `B=1`
- `T_q=1`
- long-cache decode

## Recommended Architecture

### Track A: Fix TriAttention first

Do not stack FP4-KV work on top of untrusted TriAttention runtime eviction.

Exit criteria:

- runtime eviction confirmed in probe
- finite scores only
- 9B and 27B rolling-PPL witness trusted

### Track B: Add a software FP4 KV type

Add a new KV-only type, separate from weights.

Recommended first format:

- `GGML_TYPE_HF4_0` or similar KV-only enum
- block size `64`
- 64 x 4-bit payload
- 32-bit metadata block
- software dequant to `f16` or `f32` in attention path

Why block-64:

- matches the HiFloat4 grouping idea
- keeps metadata amortization simple
- aligns better with head-dim multiples we already see in qwen35

### Track C: Use asymmetric KV policy

Do not force both K and V to the same new format on day 1.

Start with:

- `K = HF4`
- `V = f16` or `q8_0`

Reason:

- `K` drives score computation and is often more forgiving to compact representation if the scale model is good
- `V` corruption is directly reflected in context output and can damage decode quality faster

Then test:

1. `K=HF4, V=f16`
2. `K=HF4, V=q8_0`
3. `K=HF4, V=HF4`

### Track D: After correctness, fuse decode

Once software FP4 KV is numerically stable:

- add a decode-only fused attention kernel for `T_q=1`
- keep prefill on the existing path
- target the KV read + score + softmax + V accumulate chain

That is the point where `Fused-KV-Cache` becomes directly actionable.

## What Not To Assume

- HiFloat4 will not automatically beat `tq3_0` on memory
- HiFloat4 will not automatically beat `q4_0` on speed in software
- Fused decode does not fix bad score models
- TriAttention does not fix bad quantization

## Concrete Near-Term Plan

### Phase 0

Finish TriAttention runtime proof:

1. restore the missing runtime enable/export path cleanly
2. re-run the 9B `prefill=12 budget=8 triatt_tokens=8` witness
3. verify no non-finite score summaries
4. verify eviction fires in probe logs

### Phase 1

Prototype `HF4` KV storage only:

1. add new KV-only enum
2. add CPU reference quant/dequant
3. add CUDA dequant path
4. wire `type_k=hf4_0`
5. keep `type_v=f16`

Success criteria:

- no gibberish in chat
- rolling PPL within acceptable delta vs `q4_0` KV baseline
- KV memory same class as `q4_0`, but quality better if the format is doing real work

### Phase 2

Benchmark asymmetric KV:

1. `q8_0/f16`
2. `q4_0/f16`
3. `hf4_0/f16`
4. `tq3_0/f16`
5. `hf4_0/q8_0`

On:

- Qwen3.5-9B
- Qwopus3.5-27B

Metrics:

- rolling PPL
- prompt tok/s
- decode tok/s
- KV buffer size

### Phase 3

Decode fusion:

1. isolate single-token decode path
2. fuse score staging + softmax + V accumulation
3. first with `f16` KV
4. then with `hf4_0` K or `hf4_0` KV

## Bottom Line

HiFloat4-style KV is a plausible **better 4-bit KV baseline**.

It is **not** the next thing to ship before TriAttention runtime correctness is fixed.

The strongest combined strategy is:

1. fix TriAttention runtime eviction
2. add software `HF4` KV as a quality-oriented 4-bit KV line
3. only then pursue fused single-token decode for real speed gains
