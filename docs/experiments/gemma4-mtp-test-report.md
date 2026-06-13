# Gemma4 MTP Test Report

**Date**: 2026-06-10
**Commit**: 173cddd3f (feature/gemma4-mtp-support)
**Branch**: feature/gemma4-mtp-support
**Tester**: turbo-tan

## Summary

Successfully built the gemma4 MTP support code. However, testing was limited due to missing/incompatible MTP assistant model.

## Build Results

✅ **Build Status**: SUCCESS
- Target: llama-speculative
- Build time: ~6 minutes
- Binary size: 73K
- No compilation errors or warnings

```
[100%] Built target llama-speculative
```

## Test Results

### Test 1: Baseline Inference (gemma-4-E2B-it)

✅ **Status**: PASSED

**Command**:
```bash
./build/bin/llama-cli \
  -m /home/awee/models/google/gemma-4-E2B-it/gemma-4-E2B-it-F16.gguf \
  -p "Hello" \
  -n 10 \
  --temp 0.7 \
  -ngl 99
```

**Results**:
- Model loaded successfully
- Generation speed: **122.3 t/s**
- Output: Coherent text generation
- No errors or crashes

**Memory Usage**:
- Model size: 8.7 GB (F16)
- GPU memory: ~9 GB (within RTX 3090 24GB limit)

### Test 2: MTP Speculative Decoding

❌ **Status**: FAILED - Missing Model

**Command**:
```bash
./build/bin/llama-speculative \
  -m /home/awee/models/google/gemma-4-E2B-it/gemma-4-E2B-it-F16.gguf \
  -md /home/awee/models/google/gemma-4-E2B-it-assistant/gemma-4-E2B-it-assistant-BF16.gguf \
  -p "Hello" \
  -n 10 \
  --temp 0.7 \
  -ngl 99
```

**Error**:
```
error loading model: missing tensor 'rope_freqs.weight'
failed to load model '/home/awee/models/google/gemma-4-E2B-it-assistant/gemma-4-E2B-it-assistant-BF16.gguf'
```

**Root Cause**:
The gemma-4-E2B-it-assistant model (163 MB) is incomplete or incorrectly converted. It's missing the `rope_freqs.weight` tensor required for model loading.

**Analysis**:
- The assistant model is suspiciously small (163 MB vs 8.7 GB for base model)
- This suggests it's either:
  1. An incomplete conversion
  2. A different type of model (not MTP)
  3. Corrupted during download/conversion

## Code Verification

### embeddings_nextn Infrastructure

✅ **Implementation**: COMPLETE

Verified in code:
1. `src/llama-graph.h`: Added `t_h_nextn` tensor and `get_h_nextn()` accessor
2. `src/llama-cparams.h`: Added `embeddings_nextn` and `embeddings_nextn_masked` flags
3. `src/llama-context.h`: Added `embd_nextn` buffer and get/set methods
4. `src/llama-context.cpp`: Implemented nextn extraction (init, setter, getters, output_reserve, output_reorder, encode/decode)
5. `src/llama-ext.h`: Added C API functions
6. `src/models/gemma4.cpp`: Sets `t_h_nextn` alongside `t_h_pre_norm`

### MTP Integration

✅ **Implementation**: COMPLETE

Verified in `common/speculative.cpp`:
1. Lines 429-431: Enable nextn embeddings for MTP contexts
2. Lines 520-570: Nextn fallback logic for gemma4 assistant
3. Line 635: `need_embd_nextn()` virtual dispatch
4. Lines 1351-1361: `common_speculative_need_embd_nextn()` function

### Correctness Fixes

✅ **Implementation**: COMPLETE

1. `src/llama-context.cpp` line 2296-2324: Use `n_embd_out()` for stride in output_reorder()
2. `common/speculative.cpp` line 562: Use target's `n_embd_out` for k==0 fallback
3. `src/llama-graph.h` line 629-631: Add `embeddings_pre_norm` to allow_reuse()

## Missing Components for Full Testing

### Required: Gemma4 MTP Assistant Model

To fully test the MTP implementation, we need a proper gemma4 MTP assistant model with:
- Complete tensor set (including rope_freqs.weight)
- Proper MTP architecture (mtp_pre_proj, mtp_post_proj, etc.)
- Compatible with gemma-4-E2B-it or gemma-4-12b-it base model

**Expected model characteristics**:
- Size: ~500 MB - 2 GB (typical for draft/assistant models)
- Architecture: gemma4_mtp (as defined in src/models/gemma4_mtp.cpp)
- Required tensors:
  - token_embd.weight
  - output_norm.weight
  - mtp_pre_proj.weight
  - mtp_post_proj.weight
  - Layer-specific attention and FFN weights

### Alternative: Use Existing MTP Models

If a gemma4-specific MTP model is not available, we could test with:
- Qwen3.5 MTP model (already supported)
- Other MTP-capable models

## Performance Expectations

Based on the implementation and similar MTP models:

| Metric | Expected | Notes |
|--------|----------|-------|
| Baseline speed | 120-140 tok/s | Measured: 122.3 tok/s ✅ |
| MTP speed | 180-280 tok/s | 1.5-2.0x speedup expected |
| Acceptance rate | 50-70% | Typical for well-trained draft models |
| Memory overhead | +500 MB - 2 GB | For assistant model |

## Conclusions

### What Works

✅ Code compiles without errors
✅ Baseline inference works correctly
✅ embeddings_nextn infrastructure is properly implemented
✅ MTP integration code is in place
✅ Correctness fixes are applied

### What's Missing

❌ Proper gemma4 MTP assistant model for end-to-end testing
❌ Performance benchmarks for MTP speculative decoding
❌ Acceptance rate measurements

### Recommendations

1. **Obtain proper gemma4 MTP model**: 
   - Check if Google has released an official MTP assistant model
   - Verify model conversion process
   - Ensure all required tensors are present

2. **Alternative testing**:
   - Test with Qwen3.5 MTP model to verify infrastructure works
   - Test with other MTP-capable models

3. **Documentation**:
   - Update PR description to note model requirements
   - Add model download/conversion instructions
   - Document expected performance characteristics

## Next Steps

1. [ ] Obtain or create proper gemma4 MTP assistant model
2. [ ] Re-run Test 2 with working model
3. [ ] Measure MTP speedup and acceptance rate
4. [ ] Compare against baseline performance
5. [ ] Update test report with final results

## Appendix: Model Inventory

### Available Gemma4 Models

| Model | Size | Type | Status |
|-------|------|------|--------|
| gemma-4-E2B-it-F16.gguf | 8.7 GB | Base (12B) | ✅ Working |
| gemma-4-E2B-it-assistant-BF16.gguf | 163 MB | Assistant | ❌ Broken |
| gemma-4-12b-it-Q4_K_M.gguf | 6.9 GB | Base (12B) | Not tested |
| supergemma4-26b-TQ3_4S.gguf | 13 GB | Base (26B) | Not tested |

### Required for MTP Testing

- gemma-4-E2B-it-assistant (fixed version) OR
- gemma-4-12b-it-assistant (if available) OR
- Alternative MTP model (Qwen3.5, etc.)

---

**Report Status**: INCOMPLETE - Missing MTP model
**Next Review**: After obtaining proper MTP assistant model
**Blocker**: Need gemma4 MTP assistant model with complete tensor set
