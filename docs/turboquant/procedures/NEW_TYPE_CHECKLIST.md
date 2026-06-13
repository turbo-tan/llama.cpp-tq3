# New Quant Type Checklist

Every file that must be modified when adding a new TQ3 quantization type.
Based on TQ3_4S and TQ3_4SE implementation experience.

## Core Type Registration (must do first)

| # | File | What to add |
|---|------|-------------|
| 1 | `ggml/include/ggml.h` | `GGML_TYPE_XXX` enum value, bump `GGML_TYPE_COUNT` |
| 2 | `ggml/src/ggml-common.h` | `block_xxx` struct + `static_assert` on size |
| 3 | `ggml/src/ggml-quants.h` | Function declarations: `quantize_row_xxx_ref`, `dequantize_row_xxx`, `quantize_xxx` |
| 4 | `ggml/src/ggml-quants.c` | Quantizer + dequantizer implementations |
| 5 | `ggml/src/ggml.c` | Type traits: `.type_name`, `.blck_size`, `.type_size`, `.is_quantized`, `.to_float`, `.from_float_ref` |
| 6 | `ggml/src/ggml.c` | Quantize dispatch: `case GGML_TYPE_XXX: result = quantize_xxx(...)` |

## llama.cpp Integration

| # | File | What to add |
|---|------|-------------|
| 7 | `include/llama.h` | `LLAMA_FTYPE_MOSTLY_XXX` enum value |
| 8 | `src/llama-quant.cpp` | `case LLAMA_FTYPE_MOSTLY_XXX: return GGML_TYPE_XXX;` |
| 9 | `src/llama-model-loader.cpp` | ftype→string mapping + ggml_type→ftype mapping |
| 10 | `tools/quantize/quantize.cpp` | `{ "XXX", LLAMA_FTYPE_MOSTLY_XXX, "description" }` |

## CPU Backend

| # | File | What to add |
|---|------|-------------|
| 11 | `ggml/src/ggml-cpu/ggml-cpu.c` | CPU type traits: `.from_float`, `.vec_dot`, `.vec_dot_type`, `.nrows` |
| 12 | `ggml/src/ggml-cpu/ops.cpp` | Add `case GGML_TYPE_XXX:` alongside existing TQ3 types in mul_mat etc. |

## CUDA Backend

| # | File | What to add |
|---|------|-------------|
| 13 | `ggml/src/ggml-cuda/common.cuh` | `ggml_cuda_type_traits<GGML_TYPE_XXX>` with qk, qr, qi |
| 14 | `ggml/src/ggml-cuda/convert.cu` | Dequant kernel + `dequantize_row_xxx_cuda` + register in `get_to_fp16/fp32_cuda` |
| 15 | `ggml/src/ggml-cuda/getrows.cu` | Get-rows kernel or dispatch case |
| 16 | `ggml/src/ggml-cuda/ggml-cuda.cu` | Add `case GGML_TYPE_XXX:` in supported type lists |
| 17 | `ggml/src/ggml-cuda/vecdotq.cuh` | MMVQ vec_dot kernel (for TG speed) |
| 18 | `ggml/src/ggml-cuda/mmvq.cu` | MMVQ dispatch case |
| 19 | `ggml/src/ggml-cuda/mmq.cuh` | MMQ load_tiles kernel (for PP speed) |
| 20 | `ggml/src/ggml-cuda/mmq.cu` | MMQ dispatch + type traits |
| 21 | `ggml/src/ggml-cuda/template-instances/mmq-instance-xxx.cu` | MMQ template instance |

## Minimum Viable (cuBLAS fallback, no optimized kernels)

Steps 1-16 are enough for a working format using cuBLAS dequant→GEMM.
Steps 17-21 are for optimized MMVQ/MMQ kernels (3-5x faster PP, ~30% faster TG).

## Notes

- Block size (`blck_size`) must be a power of 2 for CUDA alignment
- `qr` in CUDA type traits = number of elements per dequantized output (usually 2 for 3-bit)
- `qi` = number of 32-bit integers per block for MMVQ indexing
- The cuBLAS path (steps 1-16) gives correct results but slow PP (~220 tok/s vs ~700 tok/s)
- Always test with 1-chunk PPL trial before running 100 chunks
