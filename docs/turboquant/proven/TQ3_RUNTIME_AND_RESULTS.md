# TurboQuant Weights in llama.cpp

## Overview

`TQ3_1S` and `TQ3_4S` are custom low-bit weight formats built for the TurboQuant runtime fork:

- `TQ3_1S`
  - smaller, more aggressive
  - optimized for maximum size reduction
- `TQ3_4S`
  - slightly larger
  - optimized to stay much closer to source quality

These formats are designed for:

- `llama.cpp`-style GGUF deployment
- fast local inference on commodity GPUs
- practical quality retention at much lower memory cost

Runtime:

- public runtime fork:
  - `https://github.com/turbo-tan/llama.cpp-tq3`

## What The Method Does

At a high level, the method uses:

1. Blockwise transform-domain quantization

- weights are processed in fixed-size blocks
- each block is represented in a compact low-bit code form instead of stored directly as fp16/bf16 values

2. Rotation-assisted compression

- the runtime applies a Walsh-Hadamard-style rotation idea so quantization happens in a more compression-friendly domain
- in practice, this improves low-bit behavior compared with naive direct-domain quantization

3. Small metadata plus packed codes

- the model stores compact codes
- plus lightweight per-block scale information
- this is what lets the format stay much smaller than `Q8_0` or `BF16`

4. Runtime-side reconstruction for inference

- inference does not treat the model as plain dense fp16 weights
- the custom runtime knows how to interpret the packed `TQ3_*` blocks and run them correctly

## Format Roles

### `TQ3_1S`

Use `TQ3_1S` when:

- memory is extremely tight
- you want the highest compression from this family
- you accept a larger quality drop than `TQ3_4S`

Practical summary:

- around the `3`-bit class
- very strong size reduction from `Q8_0`
- best for “fit first” deployment

### `TQ3_4S`

Use `TQ3_4S` when:

- you want the better quality member of the pair
- you still need a major memory reduction
- you want a more balanced release format

Practical summary:

- around the `4`-bit class
- still much smaller than `Q8_0` and `BF16`
- materially closer to source quality than `TQ3_1S`

## Typical Compression

Approximate size reduction:

- `BF16 -> TQ3_1S`
  - about `80%+` smaller
- `BF16 -> TQ3_4S`
  - about `74%` smaller
- `Q8_0 -> TQ3_1S`
  - about `60%` smaller
- `Q8_0 -> TQ3_4S`
  - about `48%` smaller

These are practical ballpark figures, not guaranteed exact values for every model family.

## Quality Positioning

General behavior:

- `TQ3_1S`
  - stronger compression
  - larger quality loss
- `TQ3_4S`
  - less aggressive
  - better quality retention

One completed same-model example:

- `Qwopus3.5-27B-v3`
  - source `f16`: `6.2137` PPL
  - `TQ3_4S`: `6.3433` PPL
  - about `+2.09%` vs source on that run

That is the kind of tradeoff `TQ3_4S` is meant to deliver:

- much smaller model size
- while staying close to source quality

## Why A Custom Runtime Is Required

These GGUFs are **not** standard stock `llama.cpp` types.

They require the public TurboQuant runtime fork:

- `https://github.com/turbo-tan/llama.cpp-tq3`

If a runtime does not understand the custom tensor type ids, the model will fail to load.

## Multimodal Note

For multimodal model families such as Qwen/Qwopus vision-capable releases:

- the main GGUF may also require a matching `mmproj` GGUF
- that projector should be loaded alongside the text model in the runtime

## Simple Summary

If you want the smallest possible release in this family:

- choose `TQ3_1S`

If you want the better quality release in this family:

- choose `TQ3_4S`

If you want to run either:

- use `llama.cpp-tq3`

## Verified Prompt Result

One verified real runtime result on `Qwopus3.5-27B-v3-TQ3_4S`:

- private TurboQuant build:
  - `343.06 ± 1.30 tok/s`
- public fork rerun on the same machine:
  - `325.19 ± 0.83 tok/s`
- test:
  - `pp2048`
- GPU:
  - `RTX 5060 Ti`

That result came from a CUDA `TQ3_4S` prompt-path update that stages
activation packs more efficiently in the live MMQ path.
