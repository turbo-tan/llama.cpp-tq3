# Marlin Kernel Morning Brief - 2026-04-01

## 🎯 Major Progress Overnight!

We've narrowed down the Marlin kernel issue significantly. The problem is now isolated to wmma fragment layout/accumulation.

## Current Status

### ✅ What's Working
1. **WHT (Walsh-Hadamard Transform)** - Fully correct
   - W values after WHT match cuBLAS exactly
   - Verified: W[0][0:4] = [-0.0180, 0.0229, 0.0050, -0.0203] ✓

2. **X values** - Correct
   - X[0:4][0] = [0.8569, -0.8003, -0.0435, -1.0693] ✓

3. **First output element** - Nearly correct!
   - test[0] = 0.3794 vs ref[0] = 0.3660
   - Only 3.7% error (was 5-10x magnitude before)

### ❌ The Problem
**Elements beyond the first have wrong values, including wrong signs:**
```
test = [0.3794, -0.3424,  0.1636,  0.5256]
ref  = [0.3660,  0.4091, -1.4446,  0.7209]
         ✓       WRONG    WRONG    WRONG
              (sign!)   (sign!)
```

## Root Cause Hypothesis

The wmma computation is producing correct results for element [0][0], but incorrect results for other elements. This suggests:

1. **Fragment layout issue**: The way we're loading/storing fragments may be transposing or scrambling elements beyond [0][0]

2. **Accumulation issue**: The c_frag accumulation across K iterations may not be distributing correctly across all 16x16 elements

## Current Configuration
```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> a_frag;  // W
wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;  // X
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;  // Y

// Storage
wmma::store_matrix_sync(&Y_tile[0][0], c_frag, 16, wmma::mem_row_major);
```

This should compute: W^T * X (which is what cuBLAS does)

## Next Steps

### Option 1: Check wmma Element Distribution
The wmma fragments distribute elements across threads in a specific pattern. Element [0][0] might be handled differently than other elements. Need to:
- Check NVIDIA docs for wmma fragment layout
- Verify how elements map to threads
- Ensure our indexing matches the expected layout

### Option 2: Try Different Fragment Layouts
Systematically try:
- `mem_col_major` for C fragment storage (already tried, didn't help)
- Different combinations of A/B fragment layouts
- Check if stride parameter needs adjustment

### Option 3: Simplify Test Case
Create a minimal 16x16 * 16x16 test with known values to verify wmma behavior in isolation.

## Files Modified
- `/home/awee/code/llama.cpp/ggml/src/ggml-cuda/tq3_4s_marlin.cu`
  - Added extensive logging (W, X, Y at each step)
  - Current commit: `9b82ce734`

## Performance Target
Once fixed, expect **600+ tok/s** (2x current 315 tok/s) for prompt processing.

## Key Insight
Getting the first element nearly correct (3.7% error) is HUGE progress! This means:
- The overall computation flow is correct
- WHT is working
- wmma is being called correctly
- The issue is purely in element layout/distribution

We're very close! 🚀
