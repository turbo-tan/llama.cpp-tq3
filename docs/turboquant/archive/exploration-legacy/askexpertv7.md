# TQ3_4S Expert Consultation — 2026-03-31

## What We Found

fp16 scales in TQ3_1S waste 9.12 bits each. Real entropy is only 6.88 bits per scale. Only 2364 unique fp16 values used out of 65536 (3.6%). Dynamic range is 3.07 bits.

**8-bit scales match fp16 quality exactly** — verified on both 9B and 27B witnesses with zero RMSE difference.

This frees 2 bytes per 16-byte block. We tested what to do with them:

## Screening Results (tensor-level RMSE, 9B + 27B witnesses)

### Scale granularity (same 3-bit centroids + WHT, u8 scales)

| Design | Scales | Group size | bpw | vs TQ3_1S (9B) | vs TQ3_1S (27B) |
|--------|--------|-----------|-----|----------------|-----------------|
| TQ3_1S (current) | 2×fp16 | per-16 | 4.000 | 1.000 | 1.000 |
| **TQ3_4S** | **4×u8** | **per-8** | **4.000** | **0.955** | **0.956** |
| TQ3_8S | 8×u8 | per-4 | 5.000 | 0.868 | 0.870 |
| TQ3_16S | 16×u8 | per-2 | 7.000 | 0.692 | 0.698 |

### Other 2-byte bonus options (all at 4.0 bpw)

| Option | Layout | vs TQ3_1S (9B) |
|--------|--------|----------------|
| 4×u8 per-8 scales (D) | best | 0.955 |
| u8 d+d + u8 shift+shift (A2) | | 0.969 |
| u8 d+d + fp16 shift (A) | | 0.984 |

### Extended designs

| Design | bpw | vs TQ3_1S (9B) |
|--------|-----|----------------|
| 4×scale + 2×shift (E) | 4.500 | 0.926 |
| 4×scale + 4×shift (F) | 5.000 | 0.894 |
| 8×u8 pure scales (D2) | 5.000 | 0.868 |

### Failed approaches (all worse than TQ3_1S)

| Attempt | bpw | vs TQ3_1S | Why it failed |
|---------|-----|-----------|---------------|
| TQ3_K_v0 scale-only | 3.500 | 0.986 | Only 1.4% better, not enough |
| TQ3_K_v1 shared-shift | 4.000-4.250 | 1.152 | Shift doesn't help 3-bit centroids |
| TQ3_1S_K amortized 4-bit scales | 3.312 | 1.185 | Scale quantization too lossy |
| TQ4 symmetric (naive) | ~4.28 | 1.25× Q4_K | Needs better quantizer |

## TQ3_4S Implementation (done, needs review)

### Block structure

```c
typedef struct {
    ggml_half d;          // max group scale (fp16)
    uint8_t   ratios[2];  // 4 × 4-bit scale ratios packed
    uint8_t   qs[12];     // 32 × 3-bit packed indices
} block_tq3_4s;           // 16 bytes = 4.0 bpw (same as TQ3_1S)
```

### Scale encoding

```
d = max of 4 group scales (fp16, full precision for magnitude)
ratios = 4 × 4-bit indices into ratio table:
  {0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625,
   0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 1.75}

group_scale[g] = d * ratio_table[4-bit index for group g]
```

### Decode

```
For each group g (0..3), 8 elements:
  scale = fp16_to_float(d) * ratio_table[nibble(ratios, g)]
  for j in 0..7:
    rotated[g*8+j] = centroids[3-bit index] * scale
Inverse WHT over 32 elements → output
```

### What's done

- Type ID 46 (`GGML_TYPE_TQ3_4S`) registered
- Block struct in `ggml-common.h`
- Quantizer + dequantizer in `ggml-quants.c`
- CPU type traits in `ggml-cpu.c`
- Validation in `ggml-quants.c`
- Model loader + quantize tool support
- 27B model quantized (13.2 GB)
- Roundtrip test passes (sane values)

### What's NOT done

- CUDA MMQ kernel is still not trustworthy
- CPU vec_dot (falls back to dequant → GEMM)
- Fresh re-quantized model-level validation is still needed

### Runtime bring-up result

The previously missing CUDA runtime pieces have now been wired up in the private `llama.cpp` branch:

- CUDA dequant kernel path
- CUDA `get_rows`
- CUDA MMVQ vec_dot path
- CUDA dispatch

Two real issues showed up during bring-up:

1. `TQ3_4S` initially crashed in CUDA MMQ because of a missing `GGML_TYPE_TQ3_4S` q8_1 layout case in `mmq.cuh`. This is fixed.
2. The dedicated `TQ3_4S` MMQ path still triggers an illegal memory access, so MMQ is temporarily disabled for `TQ3_4S` to allow real model-level testing through the working CUDA dequant + cuBLAS path.

### Model-level reality check

Short 27B PPL runs on the existing quantized model are bad on both GPU and CPU:

| Format | Chunks | Device | PPL |
|--------|--------|--------|-----|
| TQ3_4S | 10 | GPU (`-ngl 99`) | `8.1119 +/- 0.41463` |
| TQ3_4S | 10 | CPU (`-ngl 0`) | `8.1138 +/- 0.41475` |

This means:

- the CUDA runtime is now working well enough to execute real model-level tests
- CPU and GPU agree closely
- the current problem is **not** the CUDA runtime anymore
- it is either the existing `Qwen_Qwen3.5-27B-TQ3_4S.gguf` being stale/bad, or the `TQ3_4S` quantizer/dequantizer contract itself

So the next correct step is:

1. regenerate `TQ3_4S` from a known-good source with the current quantizer
2. rerun the same short PPL gate
3. only then come back to MMQ optimization

## Questions for Expert

1. **Ratio table design**: I used 16 non-uniform levels biased toward smaller ratios (0.125-1.75). Is there a better distribution? Should it be learned from data?

2. **Quantizer refinement**: Currently I do 4 iterations of scale refinement per group, then find the best ratio table entry. Should the indices be re-quantized after snapping to the ratio table? (I do this but want to confirm it's correct.)

3. **CUDA kernel strategy**: For the MMVQ vec_dot, each thread handles one group of 8 elements. The scale lookup is `d * ratio_table[nibble]` — one extra multiply vs TQ3_1S. Is this worth a dedicated kernel or should we reuse TQ3_1S's kernel with a scale adapter?

4. **Block size**: 16 bytes = same as TQ3_1S. Should we consider 18 bytes (4.5 bpw) with 2 extra shift bytes? The screening showed 7.4% RMSE improvement at 4.5 bpw (option E: 4 scales + 2 shifts). Still smaller than Q4_K_M.

5. **Superblock variant**: Would a 256-element superblock with 1 fp16 super-scale + 32 × 4-bit sub-scales be better? Same 4.0 bpw but different amortization. The screening used 32-element blocks — haven't tested superblock layout.

## Target Ladder

| Format | bpw | PPL (27B, 100ch) | Size 27B |
|--------|-----|-------------------|----------|
| Q4_K_M | 4.83 | 6.81 | 15.5 GB |
| IQ4_XS | 4.25 | 6.81 | 14.2 GB |
| Q3_K_S | 3.44 | 6.96 | 11.4 GB |
| **TQ3_4S** | **4.00** | **TBD (~6.8-7.0?)** | **12.9 GB** |
| TQ3_1S | 4.00 | 7.11 | 12.9 GB |

The 4.5% tensor RMSE improvement needs to translate to ≥0.15 PPL improvement to beat Q3_K_S.

## Files Modified

```
ggml/include/ggml.h              — GGML_TYPE_TQ3_4S = 46
ggml/src/ggml-common.h           — block_tq3_4s struct
ggml/src/ggml-quants.c           — quantize/dequantize + validation
ggml/src/ggml-quants.h           — declarations
ggml/src/ggml.c                  — type traits + quantize dispatch
ggml/src/ggml-cpu/ggml-cpu.c     — CPU type traits
ggml/src/ggml-cpu/ops.cpp        — get_rows + other op dispatch
ggml/src/ggml-cuda/ggml-cuda.cu  — type support lists (partial)
include/llama.h                  — LLAMA_FTYPE_MOSTLY_TQ3_4S
src/llama-quant.cpp              — ftype→type mapping
src/llama-model-loader.cpp       — ftype name + detection
tools/quantize/quantize.cpp      — CLI entry
```

## Branch

`feature/tq3_1k-explore` on `/home/awee/code/llama.cpp`
