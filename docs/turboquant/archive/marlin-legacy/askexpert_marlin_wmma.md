# Ask Expert: Marlin-style Fused Dequant+GEMM with WHT for TQ3_4S

## Context

Implementing fused TQ3_4S dequantization + GEMM kernel using CUDA wmma (Tensor Cores) to achieve 2x speedup over current cuBLAS path (315 → 600+ tok/s prompt processing).

**Current status:** Kernel compiles and runs, WHT (Hadamard transform) produces correct weight values, but final GEMM output is wrong.

## Problem

TQ3_4S format stores WHT-transformed 3-bit quantized weights in 32-element blocks. After dequantizing, must apply inverse WHT to recover original weights before GEMM.

**Verified correct:**
- Dequantization: converts packed 3-bit indices + E3M5 scales to fp16
- WHT: W[0:4] after WHT = [-0.0180, 0.0229, 0.0050, -0.0203] matches cuBLAS [-0.0178, 0.0217, -0.0046, 0.0193] (small rounding diff)
- X input: X[0:4] = [0.8569, -0.8003, -0.0435, -1.0693] matches cuBLAS

**Wrong:**
- Final output: Y[0:4] = [1.8174, 0.2260, 0.9136, -0.5886] vs cuBLAS [0.3660, 0.4091, -1.4446, 0.7209]

## Current Implementation

```cuda
// Process K in multiples of 32 (WHT block size)
for (int k = 0; k < K; k += 32) {
    // 1. Dequant full 32-elem block per row (warp collaborative)
    // 2. Apply WHT via warp shuffles
    // 3. Split into two 16x16 tiles for wmma
    
    for (int k_sub = 0; k_sub < 32; k_sub += 16) {
        // Load W_tile[16][16] from WHT-transformed data
        // Load X_tile[16][16] from activations
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        
        wmma::load_matrix_sync(a_frag, &W_tile[0][0], 16);  // row-major
        wmma::load_matrix_sync(b_frag, &X_tile[0][0], 16);  // col-major
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
}

// Store output: Y[m + b*M] (column-major)
wmma::store_matrix_sync(&Y_tile[0][0], c_frag, 16, wmma::mem_row_major);
```

## Data Layouts

**cuBLAS reference:**
```cpp
cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    M, batch, K,
    W,  // [K, M] col-major → transposed to [M, K]
    X,  // [K, batch] col-major
    Y,  // [M, batch] col-major, ld=M
```

**Kernel storage:**
- W_blocks: `[m * (K/32) + k_blk]` — M-major, each block is 32 elements along K
- X: `[b * K + k]` — row-major [batch, K]
- Y: `[m + b * M]` — column-major [M, batch]

**Tile loading:**
- W_tile: [16][16] row-major, represents W[m:m+16][k:k+16] after WHT
- X_tile: [16][16] col-major, represents X[k:k+16][b:b+16]

## Questions

1. **WHT block alignment:** WHT operates on 32-element blocks, but wmma uses 16x16 tiles. Is splitting a WHT-transformed 32-elem block into two 16-elem halves valid? Or does this break the WHT properties?

2. **wmma fragment layout:** With `row_major` A and `col_major` B fragments, does wmma expect:
   - A stored as `A[row][col]` with stride=16?
   - B stored as `B[row][col]` with stride=16 (leading dim = #rows)?
   
   Current: `load_matrix_sync(b_frag, &X_tile[0][0], 16)` where X_tile is `[16][16]` C array.

3. **Accumulation across K:** Processing K in two 16-tile chunks per 32-elem block. Does `mma_sync` correctly accumulate when called multiple times with same `c_frag`?

4. **Output layout:** Storing with `mem_row_major` then writing `Y[m + b*M]`. Is this correct for column-major output with ld=M?

## Attempted Fixes

- ✅ Fixed output write: `Y[m + b*M]` for column-major
- ✅ Added WHT with correct signs and 1/sqrt(32) scaling
- ✅ Process K in multiples of 32 (WHT block size)
- ❌ Tried different fragment layouts (row/col major) — no change
- ❌ Verified W and X values going into wmma — both correct

## Reference Code

- Working cuBLAS dequant with WHT: `ggml-cuda/convert.cu:912` (`dequantize_block_tq3_4s`)
- Broken Marlin kernel: `ggml-cuda/tq3_4s_marlin.cu:40` (`tq3_4s_marlin_gemm_kernel`)
- Test harness: `ggml-cuda/ggml-cuda.cu:1441` (compares first 16 outputs)

## Expected Outcome

Kernel should produce same output as cuBLAS reference (within fp16 rounding). Currently off by 5-10x in magnitude.

## Environment

- CUDA 12.x, Compute Capability 8.9 (RTX 4090)
- llama.cpp fork with TQ3_4S support
- Model: Qwen3.5-27B quantized to TQ3_4S (4.0 bpw)
