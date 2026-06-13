# TQ3_4S Marlin-Style Fused GEMM Kernel

## Problem

Current path: `dequantize_block_tq3_4s<__half>` (78.8% of time) → CUTLASS h16816 GEMM (9.4%)

The dequant kernel reads 12.9 GB quantized weights, writes 12.9 GB fp16, then GEMM reads 12.9 GB fp16.
Total memory traffic: 25.8 GB per forward pass. Q3_K_S MMQ: 11.4 GB (no intermediate).

## Solution: Fused Dequant+GEMM

Each GEMM thread block loads its tile of TQ3_4S blocks, dequants in registers, feeds fp16 directly to tensor core. No intermediate fp16 buffer written to global memory.

## TQ3_4S Block Structure

```c
typedef struct {
    uint8_t d[4];    // 4 × E3M5 scales (one per 8-element group)
    uint8_t qs[12];  // 32 × 3-bit indices packed
} block_tq3_4s;      // 16 bytes = 4.0 bpw
```

E3M5 decode (fast, no ldexpf):
```c
float decode_e3m5(uint8_t b) {
    uint32_t e = ((b >> 5) - 9 + 127) & 0xFF;
    uint32_t m = (b & 31) << 18;
    return __uint_as_float((e << 23) | m);
}
```

3-bit unpack for group g, element r (0..7):
```c
const uint8_t * qp = qs + g * 3;
uint32_t packed = qp[0] | (qp[1] << 8) | (qp[2] << 16);
uint8_t idx = (packed >> (3 * r)) & 7;
```

Centroids (fixed, 8 values):
```c
const float C[8] = {-2.1519f,-1.3439f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3439f,2.1519f};
```

## Kernel Design

### Thread Block Layout
- Block: (32, 8) = 256 threads (8 warps)
- Each warp handles one row of the output tile
- Tile size: 128 rows × 64 cols (tunable)

### Per-Warp Work
For each K-tile (32 elements = 1 TQ3_4S block):
1. Lane 0 reads 16-byte block from global memory
2. Broadcast scale bytes (4 × E3M5) to all lanes
3. Broadcast packed qs bytes to all lanes
4. Each lane decodes its element: `val = C[idx] * scale` → fp16
5. Feed fp16 values to tensor core fragment

### MMA Instruction
RTX 5060 Ti (sm_120) supports:
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (PTX)
- Or CUDA C++: `wmma::mma_sync` with `wmma::fragment<matrix_a, 16, 16, 16, half, row_major>`

### Memory Access Pattern
- Weight tile: 128 rows × 32 elements = 128 × 16 bytes = 2 KB per K-step
- Activation tile: 64 cols × 32 elements = 64 × 32 × 2 bytes = 4 KB per K-step
- Output tile: 128 × 64 × 4 bytes = 32 KB (in registers)

### Pseudocode

```cuda
__global__ void tq3_4s_gemm_fp16(
    const block_tq3_4s * __restrict__ A,  // [M/32][K] blocks
    const half * __restrict__ B,           // [K][N] fp16
    float * __restrict__ C,                // [M][N] fp32
    int M, int K, int N) {

    // Tile indices
    const int m0 = blockIdx.x * 128;
    const int n0 = blockIdx.y * 64;
    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;

    // Output accumulator (in registers)
    float acc[128/8][64/32] = {0};  // 16 × 2 = 32 floats per warp

    // K-loop
    for (int k = 0; k < K; k += 32) {
        // Load TQ3_4S tile (128 rows × 32 elements)
        // Each warp loads 16 rows
        const int row = m0 + warp_id * 16 + lane / 2;
        const int blk_k = k / 32;
        const block_tq3_4s * blk = A + row * (K/32) + blk_k;

        // Decode to fp16 (in registers)
        half vals[32];  // 32 fp16 values for this warp's row
        // ... decode E3M5 + unpack 3-bit + multiply centroids

        // Load activation tile (64 cols × 32 elements)
        // ... load B[k:k+32][n0:n0+64] into shared memory

        // MMA
        // ... wmma::mma_sync(acc, vals_frag, B_frag, acc)
    }

    // Store output
    // ... store acc to C[m0:m0+128][n0:n0+64]
}
```

## Expected Performance

- Eliminates 78.8% dequant overhead
- Memory traffic: 12.9 GB (quantized) + 12.9 GB (activations) = 25.8 GB
  vs current: 12.9 GB (quantized) + 12.9 GB (fp16 write) + 12.9 GB (fp16 read) + activations
- Expected PP: 600-900 tok/s (vs 315 current, 689 Q3_K_S)

## Files to Create/Modify

- `ggml/src/ggml-cuda/tq3_4s_gemm.cu` — new fused kernel
- `ggml/src/ggml-cuda/ggml-cuda.cu` — dispatch to new kernel for TQ3_4S prefill
- Keep fp16 cuBLAS as fallback for small batch sizes

## Reference

- Marlin paper: https://arxiv.org/abs/2408.11743
- GPTQ-Marlin implementation in vllm
- llama.cpp NVFP4 kernel (similar fused approach for 4-bit)
