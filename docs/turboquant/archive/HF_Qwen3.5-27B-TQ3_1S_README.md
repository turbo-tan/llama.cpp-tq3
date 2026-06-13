---
license: mit
language:
- en
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- qwen
- qwen3.5
- quantization
- turboquant
- wht
base_model:
- Qwen/Qwen3.5-27B
---

# Qwen3.5-27B-TQ3_1S

`Qwen3.5-27B-TQ3_1S` is a GGUF quantization of `Qwen/Qwen3.5-27B` using **TQ3_1S**, a 3.5-bit weight format based on:

- Walsh-Hadamard rotation
- 8 centroid quantization
- dual half-block scales

This release is aimed at one practical outcome:

- **near-Q4_0 quality**
- **about 10% smaller than Q4_0**
- **small enough to fit fully on a single 16 GB RTX 5060 Ti** in the tested `llama.cpp` setup

## Headline Result

Gold-standard `wiki.test.raw` pass, `c=512`, full `580` chunks:

| Format | PPL | Size |
|---|---:|---:|
| Q4_0 | `7.2431 +/- 0.0482` | 14.4 GB |
| TQ3_1S | `7.2570 +/- 0.0480` | 12.9 GB |

Measured gap:

- `+0.0139` PPL
- about `0.19%`

Safe interpretation:

- `TQ3_1S` is **near-Q4_0 quality**
- `TQ3_1S` is **about 1.5 GB smaller**
- on this 27B model, that size reduction is enough to change deployment on a 16 GB GPU

## Important Caveat

This model card does **not** claim that TQ3_1S is universally faster than native `Q4_0` under the same conditions.

The practical speed win in the tested setup comes mainly from:

- `TQ3_1S` fitting fully on GPU
- while `Q4_0` does not fit fully on GPU on the same 16 GB card

So this is primarily a **deployment / fit advantage** story, not a blanket kernel-speed claim.

## Files

- `Qwen3.5-27B-TQ3_1S.gguf`

## Base Model

- Base model: [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B)

## Recommended Runtime

This model is intended for the public TQ3 runtime fork:

- GitHub: `https://github.com/turbo-tan/llama.cpp-tq3`

It requires TQ3_1S runtime support and will not run on a stock `llama.cpp` build unless that support is present.

## Example

```bash
./build/bin/llama-server \
  -m Qwen3.5-27B-TQ3_1S.gguf \
  -ngl 99 \
  -fa on \
  -c 4096
```

## Quantization Notes

TQ3_1S uses a 32-element block layout:

```text
[d0: fp16][d1: fp16][qs: 12 bytes]
```

That is:

- 16 bytes per 32 weights
- 4.0 bits per weight at the block level

## Credit

This work is inspired by the broader line of transform-based quantization methods, especially RaBitQ-style Walsh-Hadamard rotation ideas, adapted here for LLM weight quantization in GGUF / llama.cpp.

## Limitations

- This 27B result does **not** imply that plain TQ3_1S is equally strong on smaller dense models.
- In internal testing, 9B models were much less forgiving at this bitrate.
- This release is a practical 27B deployment artifact, not a universal claim about all model scales.

## License

Same model license terms as the base model apply.
