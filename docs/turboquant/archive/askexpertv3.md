# TQ3_0 CUDA — Expert Help v3: MMQ Works for KV Cache but Not Weights

## What Changed Since v2

Expert's step 1 solved: **fp16 inputs + fp32 compute** (`CUBLAS_COMPUTE_32F`) fixes cublas correctness. The bug was fp16 accumulation (`CUBLAS_COMPUTE_16F`), not fp16 storage.

Expert's step 3 solved: **exact q8_0 requantization** in `load_tiles_tq3_0` — warp reduce amax, exact scale, exact q8_0 layout. Unit test passes (4/4 blocks match CPU bit-for-bit).

## The Current Bug

**MMQ produces correct output for KV cache attention but garbage for weight matmul**, using the exact same `load_tiles_tq3_0` kernel.

### Test results

```
# KV cache (q4_0 weights, tq3_0 KV cache) — MMQ enabled for TQ3_0
PP 530 tok/s, TG 82 tok/s, output: "The capital of France is" ✅

# Weight matmul (tq3_0 weights, tq3_0 KV cache) — MMQ enabled for TQ3_0  
PP 44 tok/s, TG 8.8 tok/s, output: "The capital of France isnt^[^[^[^[" ❌

# Weight matmul (tq3_0 weights) — MMQ disabled, cublas fp16+fp32compute
PP 16 tok/s, TG 3.7 tok/s, output: "The capital of France is Paris" ✅
```

### What's different between the two paths

Both go through `ggml_cuda_op_mul_mat` → `ggml_cuda_op_mul_mat_q` → MMQ kernel → `load_tiles_tq3_0`.

**KV cache attention (works):**
- `src0` = K tensor after `ggml_permute(0,2,1,3)` — **non-contiguous**
- Shape: `[head_dim=128, n_kv, n_heads=32, 1]` after permute → `[128, 32, n_kv, 1]`
- `src0_is_contiguous = false` → data copied via `ggml_cuda_cpy_tensor_2d` per `(i02, i03)` slice
- Each slice: `row_diff` rows of `ne00=128` elements = 4 TQ3_0 blocks per row
- `ne02 = n_kv` iterations in the outer loop

**Weight matmul (broken):**
- `src0` = weight tensor — **contiguous**
- Shape: `[2048, 2048, 1, 1]` (e.g., attn_q weight)
- `src0_is_contiguous = true` → `src0_dd_i = src0->data` directly, no copy
- Single iteration: `ne02=1, ne03=1`
- `row_diff = ne01 = 2048`, `ne00 = 2048` → 64 TQ3_0 blocks per row, 2048 rows

### What I've verified

1. **Standalone unit test** (`tests/test-tq3-load-tiles.cu`): GPU kernel produces exact q8_0 match vs CPU for 4 different blocks. All d values and all 32 qs values match.

2. **Dequant kernel** (`convert.cu`): Standalone test passes at 4M elements. GPU dequant output matches CPU bit-for-bit for real model data (verified by dumping first block from actual model load).

3. **The same `load_tiles_tq3_0` function** handles both KV cache and weight paths — it's a function pointer in the MMQ type traits. No code difference.

### My hypothesis

The difference is in how `ggml_cuda_op_mul_mat` sets up `src0_dd_i`:

- **KV cache (non-contiguous):** `ggml_cuda_cpy_tensor_2d` copies one 2D slice at a time into a temp buffer. The temp buffer is contiguous and correctly laid out as sequential `block_tq3_0` structs.

- **Weights (contiguous):** `src0_dd_i = src0->data` — points directly to the GGUF mmap'd data. The MMQ kernel reads this as `block_tq3_0 *` with stride `stride` between rows.

The `stride` parameter in `load_tiles_tq3_0` is `ne00 / QK_TQ3_0` = number of blocks per row. For weights: `2048/32 = 64`. The kernel accesses `block_tq3_0 * at kbx0 + i*stride + blk`. If the weight tensor's row stride in bytes doesn't match `stride * sizeof(block_tq3_0)`, the kernel reads wrong data.

For TQ3_0: `sizeof(block_tq3_0) = 14 bytes`. Row stride = `64 * 14 = 896 bytes`. But `ggml_row_size(TQ3_0, 2048) = (2048/32) * 14 = 896`. So the stride should be correct.

Unless the GGUF file has padding between rows, or the tensor's `nb[1]` doesn't match `ne00/QK * sizeof(block)`.

## Questions

1. Is there a padding/alignment issue in how GGUF stores TQ3_0 weight tensors that causes `src0->data` to have a different layout than sequential `block_tq3_0` structs?

2. Should I add a debug check that verifies `src0->nb[1] == ggml_row_size(src0->type, src0->ne[0])` before entering the MMQ path?

3. Could the issue be in how `ggml_cuda_op_mul_mat` computes `kbx0` for the MMQ kernel? The `kbx0` calculation might differ between the contiguous and non-contiguous paths.

## How to Reproduce

```bash
cd /home/awee/code/llama.cpp
git checkout experiment/tq3-weight-quant

# Build
cd build && cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) llama-completion llama-quantize

# Quantize TinyLlama to TQ3_0
./bin/llama-quantize --allow-requantize /home/awee/models/tinyllama-1.1b-q4_0.gguf /home/awee/models/tinyllama-1.1b-tq3_0.gguf TQ3_0

# Test KV cache MMQ (WORKS) — re-enable MMQ by removing TQ3_0 exclusion in ggml-cuda.cu
# Edit: remove "&& src0->type != GGML_TYPE_TQ3_0" from use_mul_mat_q
echo "Hi" | timeout 20 ./bin/llama-completion \
    -m /home/awee/models/llama2-7b-q4_0.gguf \
    -ngl 99 --fit off -c 2048 -n 50 \
    --cache-type-k tq3_0 --cache-type-v f16 -fa off --no-warmup \
    -p "The capital of France is" -cnv

# Test weight MMQ (BROKEN) — same binary with MMQ enabled
timeout 20 ./bin/llama-completion \
    -m /home/awee/models/tinyllama-1.1b-tq3_0.gguf \
    -ngl 99 --fit off -c 64 -n 5 --no-warmup -no-cnv \
    -p "The capital of France is" --seed 42

# Run unit tests
nvcc -O2 -o /tmp/test_tq3 tests/test-tq3-cuda.cu && /tmp/test_tq3           # 9/9 pass
nvcc -O2 -o /tmp/test_lt tests/test-tq3-load-tiles.cu && /tmp/test_lt        # 4/4 pass
```

## Key Files

- `ggml/src/ggml-cuda/mmq.cuh:3218` — `load_tiles_tq3_0` (exact q8_0 requant, unit tested)
- `ggml/src/ggml-cuda/ggml-cuda.cu:1545` — `src0_is_contiguous` (differs between KV and weights)
- `ggml/src/ggml-cuda/ggml-cuda.cu:1627` — contiguous path: `src0_dd_i = src0->data`
- `ggml/src/ggml-cuda/ggml-cuda.cu:1765` — non-contiguous path: `ggml_cuda_cpy_tensor_2d`
- `ggml/src/ggml-cuda/ggml-cuda.cu:2255` — `use_mul_mat_q` exclusion line
- `tests/test-tq3-load-tiles.cu` — standalone unit test (4/4 pass)
- `tests/test-tq3-cuda.cu` — dequant unit tests (9/9 pass)

## Branch

`experiment/tq3-weight-quant` on `charpdev/llama.cpp`
