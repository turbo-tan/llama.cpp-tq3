# TurboQuant Test Suite

Date: 2026-03-27

This file is the minimum guardrail suite for TurboQuant work in `llama.cpp`.

Current naming split:

- user-facing weight target: `Q4_0_TQ`
- current low-level ggml/CUDA type under test: `TQ3_0`

Use it for:

- CUDA dispatch changes
- MMQ/MMVQ changes
- native TQ3 kernel changes
- KV-cache type changes
- CPU SIMD changes

If a change touches one of those areas, run the matching section below before calling it done.

## Why This Exists

We already had a real regression where `TQ3_0` was blocked in one CUDA path but still entered MMVQ through another dispatch path during decode with `tq3_0` KV cache.

The lesson is:

- correctness tests alone are not enough
- benchmark-only checking is too late
- dispatch policy needs explicit regression coverage

So the suite must check three layers:

1. unit and block-level correctness
2. focused backend regression cases
3. end-to-end model behavior and throughput
4. perplexity quality guard

## Required By Change Type

### 1. CUDA dispatch or MMVQ/MMQ policy changes

Must run:

- CUDA unit tests for TQ3 helpers and tile loaders
- focused `test-backend-ops` `MUL_MAT` cases
- TinyLlama `llama-bench` with `q4_0` baseline
- TinyLlama `llama-bench` with `tq3_0` weight path
- TinyLlama `llama-bench` with `tq3_0` KV cache if KV dispatch is touched

Must add a new test when:

- a type is excluded from one dispatch path
- a new env-gated policy is introduced
- a non-contiguous view/permuted path is involved

### 2. Native TQ3 kernel work

Must run:

- block-level CUDA tests
- the smallest backend-op case that reaches the new kernel
- fixed-prompt `llama-bench` A/B against current stable path
- one quality sanity generation with deterministic prompt

Must add a new test when:

- the kernel changes decode or WHT logic
- the kernel adds a new tile contract
- the kernel introduces a shape restriction

### 3. KV-cache changes

Must run:

- prompt-only benchmark
- decode-only or `tg1` benchmark
- `type_k=tq3_0` and `type_v=f16`
- a bounded timeout repro to catch stalls

If the change adds sparse or gated KV work:

- compare ON/OFF on the same KV baseline
- use at least one long-context run where the optimization is actually active
- record the threshold used
- record measured or estimated skip ratio

Must add a new test when:

- decode dispatch differs from prefill dispatch
- cache tensors become views or permuted tensors
- a path is forced to cuBLAS as a safety fallback

### 4. CPU SIMD work

Must run:

- block-level vec-dot correctness
- backend-op `MUL_MAT` correctness
- TinyLlama CPU-only decode benchmark
- q4 vs tq3 comparison on the same build

Must add a new test when:

- a new AVX2 or NEON kernel is added
- scalar fallback behavior changes
- a new fused decode plus dot path is introduced

## Baseline Commands

Run these from `/home/awee/code/llama.cpp`.

### Build

```bash
cmake -S . -B build
cmake --build build -j 16 --target llama-bench llama-cli
```

If tests are enabled in that build, also build the relevant test targets.

## CUDA Unit Tests

These are the lowest-level correctness gates for TQ3 CUDA work.

Expected rule:

- keep them green before and after any dispatch or kernel change

Commands:

```bash
cmake --build build -j 16 --target test-tq3-cuda test-tq3-load-tiles
ctest --test-dir build --output-on-failure -R 'test-tq3-cuda|test-tq3-load-tiles'
```

What they protect:

- TQ3 block decode correctness
- inverse WHT behavior
- tile-loader and native block contracts

## Focused Backend Regression

Use `test-backend-ops` to exercise small `MUL_MAT` cases against CPU reference.

Minimum filters to run after CUDA dispatch work:

```bash
cmake --build build -j 16 --target test-backend-ops
./build/bin/test-backend-ops test -o MUL_MAT -p 'type_a=tq3_0'
```

Focused non-contiguous regression for the KV/MMVQ class of bug:

```bash
./build/bin/test-backend-ops test -o MUL_MAT -p 'type_a=tq3_0,type_b=f32,m=128,n=1,k=256,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=512,o=1'
```

Focused Qwen FFN prefill regressions for the `Q4_0_TQ` weight path
(implemented today as internal type `TQ3_0`):

```bash
./build/bin/test-backend-ops test -o MUL_MAT -b CUDA0 -p 'type_a=tq3_0,type_b=f32,m=4096,n=1,k=12288'
./build/bin/test-backend-ops test -o MUL_MAT -b CUDA0 -p 'type_a=tq3_0,type_b=f32,m=4096,n=13,k=12288'
./build/bin/test-backend-ops test -o MUL_MAT -b CUDA0 -p 'type_a=tq3_0,type_b=f32,m=4096,n=32,k=12288'
./build/bin/test-backend-ops test -o MUL_MAT -b CUDA0 -p 'type_a=tq3_0,type_b=f32,m=4096,n=64,k=12288'
```

Short end-to-end Qwen sanity probe for the same bug class:

```bash
timeout 20s ./build/bin/llama-completion -m /home/awee/models/Qwen3.5-9B-tq3_0.gguf -ngl 24 -c 512 -n 24 --temp 0 -p 'The capital of France is'
```

Expected rule after the fix:

- all backend-op Qwen FFN cases above stay green
- the short Qwen prompt must stop producing `????????`

Current caveat:

- `MUL_MAT_VEC_FUSION` discovery rows exist in the broad CSV output, but exact filtered `tq3_0` invocations still return `0/0 tests passed`
- until that harness/filtering issue is fixed, do not treat fusion as a required guardrail for `TQ3_0`
- use the four plain `MUL_MAT` Qwen FFN cases above as the authoritative red/green set

If the change is specifically about view-backed or non-contiguous tensors, add a focused case and run it directly.

Required principle:

- every dispatch exclusion or fallback rule must have at least one backend-op regression case

## End-To-End Benchmarks

Use fixed TinyLlama prompts so results stay comparable.

Model examples used in this project:

- `/home/awee/models/tinyllama-1.1b-q4_0.gguf`
- `/home/awee/models/tinyllama-1.1b-tq3_0.gguf`

### q4_0 Baseline

```bash
./build/bin/llama-bench -m /home/awee/models/tinyllama-1.1b-q4_0.gguf -ngl 999 -fa 0 -ctk f16 -ctv f16 -p 512 -n 1 -r 1 --no-warmup --progress
```

Record:

- `pp512`
- `tg1`
- VRAM if relevant

### tq3_0 Weight Path

```bash
./build/bin/llama-bench -m /home/awee/models/tinyllama-1.1b-tq3_0.gguf -ngl 999 -fa 0 -ctk f16 -ctv f16 -p 512 -n 1 -r 1 --no-warmup --progress
```

Record:

- `pp512`
- `tg1`
- compare against the current stable TQ3 baseline, not only q4

### Perplexity Guard

If a change touches:

- `llama-quantize`
- TQ3 weight kernels
- MMQ/MMVQ math for model weights
- any optimization that could trade quality for speed

run a perplexity comparison on the same corpus.

Standard corpus:

```bash
cd /home/awee/code/llama.cpp
./scripts/get-wikitext-2.sh
```

Standard guard script:

```bash
/home/awee/code/tan_llama/scripts/test_tq3_perplexity.sh
```

Default comparison:

- baseline: `/home/awee/models/tinyllama-1.1b-q4_0.gguf`
- target: `/home/awee/models/tinyllama-1.1b-tq3_0.gguf`
- corpus: `/home/awee/code/llama.cpp/wikitext-2-raw/wiki.test.raw`
- context: `512`
- chunks: `8`

Default fail limits:

- `MAX_PPL_DELTA=5.0`
- `MAX_PPL_RATIO=1.25`

You can override them when testing a different pair:

```bash
BASE_MODEL=/path/to/q4.gguf \
TARGET_MODEL=/path/to/tq3.gguf \
CHUNKS=16 \
MAX_PPL_DELTA=2.0 \
MAX_PPL_RATIO=1.10 \
/home/awee/code/tan_llama/scripts/test_tq3_perplexity.sh
```

Current lesson from this guard:

- speed claims are not enough for TQ3 weights
- every serious weight-path optimization must carry a perplexity comparison

### tq3_0 KV Decode Repro

This is the regression case that caught the dispatch bug.

```bash
timeout 90s ./build/bin/llama-bench -m /home/awee/models/tinyllama-1.1b-q4_0.gguf -ngl 999 -fa 0 -ctk tq3_0 -ctv f16 -p 512 -n 1 -r 1 --no-warmup --progress
```

Pass condition:

- command exits normally
- prompt pass completes
- generation pass completes
- no timeout `124`

If a future change touches KV decode dispatch, this command is mandatory.

### Sparse-V Follow-On Rule

If we add attention-gated `V` dequant skipping:

- do not validate only at short context
- run one short sanity check and one long-context benchmark
- compare against the same KV type with sparse-`V` disabled

Minimum expected matrix:

- `f16 KV` baseline
- `q8_0 KV` with sparse-`V` OFF vs ON
- `tq3_0 KV` with sparse-`V` OFF vs ON

Minimum questions to answer:

- did decode speed improve
- at what context length did it improve
- did quality change relative to the same KV baseline
- what threshold was used

### Value Proof: Smaller KV And Still Fast

Use the local proof script when the question is not just "is it correct?" but:

- does `q4 weights + tq3_0 KV` still buy real KV reduction
- while staying close enough to the `q4_0 KV` speed baseline on this machine

Script:

```bash
/home/awee/code/tan_llama/scripts/test_tq3_value.sh
```

Default assertions:

- `tq3_0` K cache size is smaller than `q4_0`
- PP ratio is at least `0.50`
- TG ratio is at least `0.25`

Thresholds can be overridden with:

```bash
MIN_PP_RATIO=0.75 MIN_TG_RATIO=0.75 /home/awee/code/tan_llama/scripts/test_tq3_value.sh
```

## Quality Sanity Check

Use one deterministic generation after any performance-oriented CUDA change.

Example:

```bash
./build/bin/llama-cli -m /home/awee/models/tinyllama-1.1b-tq3_0.gguf -ngl 999 -fa 0 -n 32 -p 'The capital of France is'
```

Check:

- output is coherent
- no obvious repetition collapse
- no NaN or garbage output

If speed changes but quality worsens, treat it as a failed change.

## CPU Suite

Run this for CPU SIMD work or any change to CPU TQ3 vec-dot.

### CPU-only benchmark

```bash
./build/bin/llama-bench -m /home/awee/models/tinyllama-1.1b-q4_0.gguf -ngl 0 -fa 0 -p 128 -n 32 -r 1 --no-warmup --progress
./build/bin/llama-bench -m /home/awee/models/tinyllama-1.1b-tq3_0.gguf -ngl 0 -fa 0 -p 128 -n 32 -r 1 --no-warmup --progress
```

Always compare:

- same machine
- same build
- same prompt shape

## Rules For Future Appends

Append to this file when:

- a new fallback or dispatch exclusion is added
- a new kernel path becomes selectable
- a new benchmark repro catches a real failure
- a new architecture-specific CPU path lands

For each append, include:

- exact command
- what it protects
- pass or fail condition
- when it is mandatory

Do not replace old regressions just because the current bug is fixed.
Old bug repros stay in the suite unless they become obsolete for a documented reason.

## Non-Contiguous TQ3_0 MMVQ Regression (added 2026-03-27)

This test covers the non-contiguous/view-backed TQ3_0 MUL_MAT shape that caused TG hangs.
Mandatory after any MMVQ dispatch or TQ3_0 vec-dot change.

```bash
cmake --build build -j 16 --target test-backend-ops
./build/bin/test-backend-ops test -o MUL_MAT \
    -p 'type_a=tq3_0,type_b=f32,m=128,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=512,o=1'
```

Pass condition: `2/2 backends passed OK`

What it protects: non-contiguous TQ3_0 KV cache MMVQ path (permuted K tensor in attention).
