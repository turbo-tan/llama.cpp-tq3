# ExLlamaV2 Quantization Analysis

## Repository Cloned
Location: `~/exllama_env/exllamav2`
Source: https://github.com/turboderp/exllamav2

## Key Findings

### 3-bit Quantization Implementation

**File**: `exllamav2/exllamav2_ext/cuda/quant/qdq_3.cuh`

#### Packing Strategy
```cuda
// Permutation for 3-bit values (32 elements = 96 bits = 3 uint32)
// v9997775 55333111  u8886664 44222000  (u, v lsb)
// vjjjhhhf ffdddbbb  uiiiggge eecccaaa
// vtttrrrp ppnnnlll  usssqqqo oommmkkk
```

**Key differences from TQ3_4S**:
1. **No WHT**: Direct 3-bit packing without Walsh-Hadamard Transform
2. **Bit-level permutation**: Shuffles bits for better memory access patterns
3. **Vectorized dequant**: Uses half2 operations for 2 elements at once

#### Dequantization
```cuda
__forceinline__ __device__ void dequant_3bit_32(
    const uint32_t q_0, q_1, q_2,  // 3 uint32 = 96 bits = 32 × 3-bit
    half2 (&dq)[16],                // Output: 16 half2 = 32 half
    int stride
)
```

**Process**:
1. Unpack 3-bit indices from uint32
2. Add bias (1024) for fp16 conversion trick
3. Multiply by scale factors (1/8, 1/64 for different positions)
4. Subtract bias to get final value

**No lookup table** - uses arithmetic instead of centroid lookup!

### GEMM Kernel Structure

**File**: `exllamav2/exllamav2_ext/cuda/q_gemm.cu`

#### Key Features
1. **Fused dequant + GEMM**: Dequantizes on-the-fly during matrix multiply
2. **Multiple kernel variants**: Different kernels for different M sizes (1, 2, 3, 4...)
3. **Permutation-aware**: Handles bit permutations efficiently
4. **Tensor core usage**: Uses WMMA for actual GEMM after dequant

#### Kernel Selection
```cuda
// comp_units/unit_exl2_3a.cu
fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_3a(
    const int max_m,      // Max M dimension
    const int perm,       // Permutation type
    bool r_weights,       // Row-major weights
    bool mul_r_weights,   // Multiply by row weights
    int kn_size           // K dimension size
)
```

**Adaptive kernel selection** based on problem size!

### Comparison with TQ3_4S

| Feature | TQ3_4S | ExLlamaV2 EXL2 |
|---------|--------|----------------|
| **Transform** | WHT (32-element butterfly) | None (direct packing) |
| **Centroids** | 8 fixed values | Arithmetic (no lookup) |
| **Packing** | 4 groups × 8 elements | Bit-level permutation |
| **Dequant** | Lookup + WHT + signs | Arithmetic + bias trick |
| **GEMM** | Separate or fused | Always fused |
| **Kernel variants** | Single kernel | Multiple (M-adaptive) |
| **Tensor cores** | Attempted (failed) | Successfully used |

## Why ExLlamaV2 is Faster

### 1. No WHT Overhead
- TQ3_4S: 5 shuffle steps per 32 elements
- EXL2: Direct unpack, no transform

### 2. Arithmetic Dequant
- TQ3_4S: Centroid lookup (memory access)
- EXL2: Arithmetic operations (compute-bound, faster)

### 3. Adaptive Kernels
- TQ3_4S: One-size-fits-all
- EXL2: Optimized kernel per M size

### 4. Better Tiling
- TQ3_4S: Failed to tile efficiently (WHT constraint)
- EXL2: No transform constraint, tiles freely

## Correction From Trace Evidence

That earlier diagnosis is too simplistic for the current `TQ3_4S` path.

Short traced run:

- [bench_27b_tq3_4s_trace_short_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_trace_short_20260401.txt)

What the trace shows:

- activation rotation is small
- fp16 conversion is small
- the native `TQ3_4S` kernel itself is the bottleneck

Representative timings:

- `rotate`: about `0.09` to `0.49 ms`
- `src1_f16`: about `0.02` to `0.15 ms`
- `kernel`: about `26` to `46 ms`

So the main problem is not "TurboQuant math is inherently too expensive".
The main problem is:

- poor packed-kernel design
- poor tiling / pipelining
- no `M`-adaptive specialization
- decode not feeding MMA efficiently

This changes the takeaway:

- do not abandon the rotated-domain design yet
- steal ExLlamaV2's kernel structure ideas first

## Lessons for TQ3_4S

### 1. Add `M`-adaptive kernels

This is the biggest direct lesson to steal.

We should not use one prompt kernel for every shape.
At minimum:

- short `M`
- medium prompt `M`
- large prompt `M`

need different tile choices and scheduling.

### 2. Redesign packed decode for MMA feeding

Current `TQ3_4S` kernel is too naive.

What to steal from EXL2:

- decode directly into MMA-friendly tiles
- avoid extra staging work
- use more arithmetic / register-side unpack where possible

### 3. Keep TurboQuant geometry, but optimize around it

`TQ3_4S` already stores weights in the rotated domain.

So we are still doing TurboQuant if we:

- keep rotated-domain weights
- rotate activations once
- multiply in the rotated domain

That is still the core idea.
The runtime problem is the kernel implementation, not the existence of the transform.

### 4. Only consider format changes after kernel lessons are exhausted

Removing WHT or inventing a new no-WHT format is a much larger change.

It may still be worth exploring later for a new format family, but it should not be the first conclusion from the current trace data.

## Recommendation

For `TQ3_4S`, the right near-term move is:

1. keep the TurboQuant rotated-domain design
2. copy the good kernel ideas from EXL2:
   - `M`-adaptive specialization
   - better packed decode
   - less staging
3. improve the native packed kernel until it beats the current cuBLAS path

For a future format family, EXL2-style arithmetic dequant may be useful in a new `TQ3_x` design, but that is a separate exploration from fixing `TQ3_4S`.

## Files to Study Further

1. `exllamav2_ext/cuda/quant/qdq_3.cuh` - 3-bit dequant implementation
2. `exllamav2_ext/cuda/q_gemm.cu` - Fused GEMM kernel
3. `exllamav2_ext/cuda/comp_units/unit_exl2_3a.cu` - Kernel selection
4. `convert.py` - Quantization process

## Next Steps

1. Study EXL2 `M`-adaptive kernel selection more closely
2. Map those kernel-shape ideas onto `TQ3_4S`
3. Simplify the current native packed kernel around decode + MMA feeding
4. Benchmark against the current `TQ3_4S` cuBLAS path after each kernel pass

## Concrete Lessons To Copy

After reading the local ExLlamaV2 CUDA code more closely, the most important lesson is this:

- ExLlamaV2 is not winning with one giant WMMA tile
- it is winning with many small, specialized packed dot-product kernels

Relevant local files:

- `/home/awee/exllama_env/exllamav2/exllamav2/exllamav2_ext/cuda/q_gemm.cu`
- `/home/awee/exllama_env/exllamav2/exllamav2/exllamav2_ext/cuda/q_gemm_kernel.cuh`
- `/home/awee/exllama_env/exllamav2/exllamav2/exllamav2_ext/cuda/comp_units/kernel_select.cuh`
- `/home/awee/exllama_env/exllamav2/exllamav2/exllamav2_ext/cuda/quant/qdq_3.cuh`

### 1. Kernel family, not one kernel

`q_gemm.cu` picks kernels by:

- `max_m`
- permutation kind
- `kn_size` (`32` or `64`)

So the first practical fix for `TQ3_4S` is:

- stop relying on one native kernel
- add explicit kernel families for:
  - very small `M`
  - medium `M`
  - large prompt `M`

### 2. Packed dot-product, not WMMA-first design

`q_gemm_kernel.cuh` accumulates packed dequantized slices with:

- `half2 dq[...]`
- `dot22_8_h`
- `dot22_16_h`
- `dot22_32_h`

The important thing is:

- decode stays local
- compute stays in registers
- accumulation is tailored to the quant block size

This is very different from the current `TQ3_4S` prototype, which:

- decodes into shared tiles
- then feeds WMMA

That is probably the wrong first architecture for `TQ3_4S`.

### 3. Arithmetic/vector dequant matters

`qdq_3.cuh` uses:

- bit permutation
- `half2`
- arithmetic dequant tricks

This suggests a better `TQ3_4S` direction:

- decode 8-lane subgroups into `half2` or register vectors
- avoid scalar centroid lookup in the hot path if possible
- use small in-register centroid tables or arithmetic-friendly remaps

### 4. Group/scales scheduling is precomputed

ExLlamaV2 does not rediscover structure inside the hot loop.
It carries:

- `group_map`
- `perm`
- preloaded scales

So for `TQ3_4S`, we should do the same kind of prep work:

- precompute whatever rotation/scale metadata the kernel needs
- keep the hot loop focused on:
  - packed load
  - decode
  - fused dot

## Practical TQ3_4S Kernel Direction

If we translate ExLlamaV2's lessons literally, the next `TQ3_4S` kernel should be:

1. packed dot kernel first
   - not WMMA first
2. `M`-specialized
3. `K/N` block-size specialized
4. register / `half2` driven
5. rotated-domain by construction

That keeps us inside TurboQuant while copying the right runtime lessons.
