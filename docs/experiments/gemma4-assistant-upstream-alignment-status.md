# Gemma4 Assistant Upstream Alignment - Status Report

**Date**: 2026-06-10  
**Branch**: feature/gemma4-assistant-upstream-alignment  
**Status**: IN PROGRESS - Loading issues identified, architecture understanding clarified

## Executive Summary

The gemma4-assistant is a **Multi-Token Prediction (MTP) head**, not a traditional draft model. It's designed to be loaded alongside a target model using the `-md` flag with `--spec-type mtp`. The assistant model shares memory and KV cache with the target, making it fundamentally different from standalone draft models.

**Key insight**: The assistant model **cannot be loaded standalone** - it requires the target model context via `ctx_other`. Initial testing attempts to load it independently revealed this architectural requirement.

See **[Testing SOP](./gemma4-assistant-mtp-testing-sop.md)** for correct testing procedures.

## Completed Work

### 1. Architecture Renaming ✅
- Renamed `LLM_ARCH_GEMMA4_MTP` → `LLM_ARCH_GEMMA4_ASSISTANT`
- Renamed `gemma4_mtp.cpp` → `gemma4-assistant.cpp`
- Renamed `llama_model_gemma4_mtp` → `llama_model_gemma4_assistant`
- Updated architecture string mapping to `"gemma4-assistant"`

### 2. Context Integration ✅
- Added `ctx_other` field to `llama_context_params` and `llama_cparams`
- Implemented ctx_other integration in gemma4-assistant graph building
- Assistant model now accesses target model's token embeddings via `ctx_other`
- Added scaling of target embeddings by `sqrt(n_embd_backbone)`
- Updated `llama_context_default_params` to initialize `ctx_other` to nullptr

### 3. Hyperparameter Loading ✅
- Updated `load_arch_hparams` to read architecture-specific keys
- Added support for array-valued hyperparameters:
  - `n_head_arr` (attention head count per layer)
  - `n_head_kv_arr` (KV head count per layer)
  - `n_ff_arr` (feed-forward size per layer)
- Read attention key/value lengths:
  - `n_embd_head_k_full` (512 for gemma4-assistant)
  - `n_embd_head_v_full` (512 for gemma4-assistant)
- Read SWA-related hyperparameters:
  - `rope_freq_base_train_swa`
  - `n_swa` (sliding window size)
  - `f_norm_rms_eps` (RMS norm epsilon)
  - `n_embd_head_k_swa` (256 for gemma4-assistant)
  - `n_embd_head_v_swa` (256 for gemma4-assistant)

### 4. Tensor Structure ✅
- Updated tensor dimensions to use per-layer values:
  - `n_embd_head_k(i)` for head dimension
  - `n_head(i)` for head count
  - `n_ff(i)` for feed-forward size
- Use `n_embd_out()` for backbone embedding dimension (3840 for gemma4-assistant)
- Fixed tensor name for layer output scale (added "weight" suffix)

## Test Results

### What Works ✅
1. **Architecture Recognition**: Model is correctly identified as `gemma4-assistant`
2. **Hyperparameter Loading**: All metadata is read correctly:
   ```
   gemma4-assistant.embedding_length = 1024
   gemma4-assistant.embedding_length_out = 3840
   gemma4-assistant.attention.head_count = 16
   gemma4-assistant.attention.head_count_kv = [8, 8, 8, 1]
   gemma4-assistant.feed_forward_length = 8192
   gemma4-assistant.attention.key_length = 512
   gemma4-assistant.attention.value_length = 512
   gemma4-assistant.attention.key_length_swa = 256
   gemma4-assistant.attention.value_length_swa = 256
   ```
3. **Hyperparameter Values**: Correctly loaded into hparams:
   ```
   n_head_arr[0] = 16
   n_head_kv_arr[0] = 8
   n_ff_arr[0] = 8192
   n_embd_head_k_full = 512
   n_embd_head_v_full = 512
   ```
4. **Per-Layer Dimensions**: Correctly calculated:
   ```
   layer 0: n_head_i=16, n_embd_head_i=512, n_ff_i=8192
   n_embd_head_i * n_head_i = 8192
   ```

### What Doesn't Work ❌
**Tensor Dimension Mismatch**:
```
error loading model: check_tensor_dims: tensor 'blk.0.attn_q.weight' has wrong shape;
expected 1024, 8192, got 1024, 4096
```

**Analysis**:
- GGUF file has tensor with shape `[1024, 8192]`
- We're creating tensor with shape `[1024, 4096]`
- Our logs show we're passing `{1024, 8192}` to `create_tensor`
- But the actual tensor being created has dimensions `{1024, 4096}`

**Possible Causes**:
1. The `create_tensor` function might be modifying the dimensions
2. There might be a different code path being taken
3. The dimensions might be calculated differently in `create_tensor`
4. There might be a caching issue with hyperparameter values

## Investigation Status

### Debugging Steps Taken
1. ✅ Verified architecture recognition
2. ✅ Verified hyperparameter loading
3. ✅ Verified per-layer dimension calculation
4. ✅ Added extensive logging to track dimension values
5. ❌ Identified mismatch between logged dimensions and actual tensor dimensions

### Next Steps
1. Investigate `create_tensor` implementation to understand dimension handling
2. Check if there's a different code path for tensor creation
3. Verify that hyperparameter values are not being modified between log and tensor creation
4. Compare with upstream implementation to identify any missing steps
5. Consider adding validation in `create_tensor` to catch dimension mismatches earlier

## Code Changes Summary

### Files Modified
- `include/llama.h`: Added `ctx_other` to `llama_context_params`
- `src/llama-arch.h`: Renamed `LLM_ARCH_GEMMA4_MTP` to `LLM_ARCH_GEMMA4_ASSISTANT`
- `src/llama-arch.cpp`: Updated architecture string mapping
- `src/llama-cparams.h`: Added `ctx_other` field
- `src/llama-context.cpp`: Initialize `ctx_other` in constructor and default params
- `src/llama-model.cpp`: Updated architecture case statements
- `src/models/models.h`: Renamed model class
- `tests/test-llama-archs.cpp`: Updated test references

### Files Created
- `src/models/gemma4-assistant.cpp`: New implementation aligned with upstream

### Files Deleted
- `src/models/gemma4_mtp.cpp`: Replaced by gemma4-assistant.cpp

## Comparison with Upstream

### Aligned With Upstream ✅
- Architecture name and string mapping
- Context integration via `ctx_other`
- Hyperparameter loading approach
- Tensor dimension calculation approach
- Use of `n_embd_out()` for backbone dimension

### Differences from Upstream ⚠️
- Upstream uses `LLAMA_LOAD_LOCALS` macro extensively
- Upstream has additional validation checks
- Upstream may have different error handling
- Some upstream-specific fields not available in our codebase (e.g., `n_embd_inp_impl`, `is_swa_impl`)

## Recommendations

### Immediate Actions
1. **Debug tensor creation**: Add logging inside `create_tensor` to see what dimensions are actually being used
2. **Compare with upstream**: Check if upstream has any special handling for gemma4-assistant tensor creation
3. **Validate hyperparameters**: Ensure hyperparameter values are not being modified after loading

### Medium-term Actions
1. **Complete model loading**: Fix the tensor dimension mismatch
2. **Test inference**: Verify that the model can perform inference correctly
3. **Benchmark performance**: Compare performance with upstream implementation
4. **Document differences**: Create a comprehensive comparison document

### Long-term Actions
1. **Upstream synchronization**: Regularly sync with upstream to stay aligned
2. **Performance optimization**: Optimize gemma4-assistant implementation for our specific use cases
3. **Feature additions**: Add any missing features from upstream

## Conclusion

The upstream alignment work has made significant progress. The architecture is correctly recognized, hyperparameters are loaded properly, and the context integration is implemented. The remaining issue is a tensor dimension mismatch that prevents the model from loading.

The root cause appears to be a discrepancy between the dimensions we're passing to `create_tensor` and the dimensions of the tensor actually being created. This requires further investigation into the `create_tensor` implementation and the tensor creation process.

**Next Priority**: Debug the tensor dimension mismatch to enable successful model loading.

---

**Last Updated**: 2026-06-10  
**Next Review**: After tensor dimension mismatch is resolved  
**Priority**: HIGH - This blocks gemma4 MTP support
