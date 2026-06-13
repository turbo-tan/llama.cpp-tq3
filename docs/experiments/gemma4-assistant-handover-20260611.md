# Gemma4 Assistant MTP Support - Handover Document

**Date**: 2026-06-11  
**Branch**: `feature/gemma4-assistant-upstream-alignment` (in llama.cpp-tq3 worktree)  
**Status**: IN PROGRESS - Compilation issues with llama-kv-cache changes

## Executive Summary

We are implementing gemma4-assistant MTP (Multi-Token Prediction) support for the llama.cpp-tq3 fork. The goal is to enable speculative decoding using gemma4-assistant models as draft models for gemma4 target models.

**Current Status**: Compilation issues with llama-kv-cache changes. The implementation is 80% complete but has compilation errors that need to be resolved.

## What Has Been Done

### 1. Layer-wise Adaptive KV Cache Precision (Phase A5)

Implemented layer-wise adaptive KV cache precision where different layers can use different quantization types:
- Top N layers use high precision (e.g., Q8_0)
- Bottom layers use low precision (e.g., Q4_0)

**Files Modified**:
- `include/llama.h`: Added `type_k_low`, `type_v_low`, `n_layers_high_precision` to `llama_context_params`
- `src/llama-cparams.h`: Added corresponding fields to internal `llama_cparams`
- `src/llama-kv-cache.h`: Added member variables and constructor parameters
- `src/llama-kv-cache.cpp`: Implemented layer-wise precision logic
- `common/common.cpp`: Added CLI arguments (`--cache-type-k-low`, `--cache-type-v-low`, `--n-layers-high-precision`)
- `common/common.h`: Added fields to `common_params`
- `common/arg.cpp`: Added argument parsing

### 2. Gemma4 Assistant Architecture Support

- Added `LLM_ARCH_GEMMA4_ASSISTANT` architecture
- Implemented `llama_model_gemma4_assistant` class
- Added `ctx_other` parameter for shared memory between target and assistant models
- Implemented MTP speculative decoding logic

### 3. Compilation Fixes

Fixed multiple compilation errors:
- Fixed `layer_share_cb` type errors in llama-kv-cache.h
- Fixed `llama_kv_cells_vec` vs `llama_kv_cells` type mismatches
- Fixed `override_arch` parameter in llama_model_params
- Added TQ3 quantization types (TQ3_0, TQ3_1S, TQ3_4S)
- Fixed `llama_cast` template issues
- Fixed flash attention cache type validation

## Current Issues

### 1. llama-kv-cache Constructor Mismatches

**Error**: Multiple calls to `llama_kv_cache` constructor don't match the updated signature.

**Affected Files**:
- `src/llama-memory-hybrid.cpp`: Line 34
- `src/llama-model.cpp`: Lines 2035, 2052

**Issue**: The constructor signature was updated to include adaptive KV cache parameters, but some call sites weren't updated.

**Fix Required**: Update all `llama_kv_cache` constructor calls to match the new signature:

```cpp
llama_kv_cache(
    const llama_model & model,
    const llama_hparams & hparams,
    ggml_type type_k,
    ggml_type type_v,
    bool v_trans,
    bool offload,
    bool unified,
    uint32_t kv_size,
    uint32_t n_seq_max,
    uint32_t n_pad,
    uint32_t n_swa,
    llama_swa_type swa_type,
    llama_memory_t mem_other,
    const layer_filter_cb & filter,
    const layer_reuse_cb & reuse,
    int32_t n_layers_high_precision = 0,
    ggml_type type_k_low = GGML_TYPE_COUNT,
    ggml_type type_v_low = GGML_TYPE_COUNT
)
```

### 2. Compilation Errors Summary

Current compilation errors (as of last build):

```
src/llama-memory-hybrid.cpp:34: error: no matching function for call to 'llama_kv_cache::llama_kv_cache(...)'
src/llama-model.cpp:2035: error: no matching function for call to 'llama_kv_cache::llama_kv_cache(...)'
src/llama-model.cpp:2052: error: no matching function for call to 'llama_kv_cache::llama_kv_cache(...)'
```

## What Needs to Be Done

### Immediate Tasks

1. **Fix llama-kv-cache Constructor Calls** (CRITICAL)
   - Update all call sites in `llama-memory-hybrid.cpp` and `llama-model.cpp`
   - Pass the new adaptive KV cache parameters (can use default values)
   - Test compilation

2. **Test Compilation**
   ```bash
   cd /home/awee/code/llama.cpp-tq3/build
   cmake --build . --target llama-cli -j8
   ```

3. **Test Basic Functionality**
   ```bash
   # Test basic model loading
   ./build/bin/llama-cli -m /path/to/model.gguf -p "Hello" -n 10
   
   # Test with adaptive KV cache
   ./build/bin/llama-cli -m /path/to/model.gguf -p "Hello" -n 10 \
     --cache-type-k q8_0 --cache-type-v q8_0 \
     --cache-type-k-low q4_0 --cache-type-v-low q4_0 \
     --n-layers-high-precision 10
   ```

### Medium-term Tasks

1. **Test MTP Speculative Decoding**
   ```bash
   # Test with gemma4-assistant as draft model
   ./build/bin/llama-speculative \
     -m /path/to/gemma4-target.gguf \
     --model-draft /path/to/gemma4-assistant.gguf \
     --spec-type mtp \
     --draft-max 4
   ```

2. **Performance Testing**
   - Compare performance with and without adaptive KV cache
   - Measure acceptance rates with MTP
   - Benchmark speed improvements

3. **Documentation**
   - Update README with usage examples
   - Document adaptive KV cache precision feature
   - Add examples to docs/turboquant/

## Known Issues and Workarounds

### Issue 1: Model Loading Fails with Tensor Mismatch

**Error**: `check_tensor_dims: tensor 'blk.0.attn_q.weight' has wrong shape`

**Cause**: The gemma4-assistant GGUF file has different tensor shapes than expected.

**Workaround**: The implementation expects specific tensor shapes. The GGUF file needs to match the expected architecture.

### Issue 2: GPU Memory Constraints

**Issue**: 24GB GPU memory is not enough for large models with MTP.

**Workaround**: Use smaller context sizes or quantization.

### Issue 3: Compilation Errors

**Issue**: Multiple compilation errors with llama-kv-cache changes.

**Workaround**: Fix all constructor call sites to match the updated signature.

## Testing Instructions

### 1. Build the Project

```bash
cd /home/awee/code/llama.cpp-tq3
cd build
cmake --build . --target llama-cli -j8
```

### 2. Test Basic Model Loading

```bash
./build/bin/llama-cli \
  -m /home/awee/models/google/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf \
  -p "Hello" \
  -n 10
```

### 3. Test Adaptive KV Cache

```bash
./build/bin/llama-cli \
  -m /home/awee/models/google/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf \
  -p "Hello" \
  -n 10 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-type-k-low q4_0 \
  --cache-type-v-low q4_0 \
  --n-layers-high-precision 10
```

### 4. Test MTP Speculative Decoding

```bash
./build/bin/llama-speculative \
  -m /home/awee/models/google/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf \
  --model-draft /home/awee/models/google/gemma-4-E2B-it-assistant/gemma-4-E2B-it-assistant-BF16.gguf \
  --spec-type mtp \
  --draft-max 4 \
  -p "Hello" \
  -n 50
```

## Files Changed

### Core Implementation
- `include/llama.h`: Added adaptive KV cache parameters
- `src/llama-cparams.h`: Added internal parameters
- `src/llama-kv-cache.h`: Added member variables
- `src/llama-kv-cache.cpp`: Implemented adaptive precision
- `src/llama-model.cpp`: Added model loading support
- `src/llama-context.cpp`: Added context support

### Common Utilities
- `common/common.h`: Added CLI parameters
- `common/common.cpp`: Implemented parameter parsing
- `common/arg.cpp`: Added argument parsing

### Documentation
- `docs/turboquant/active/PHASE_A5_LAYERWISE_ADAPTIVE_KV_PRECISION_PROGRESS.md`: Implementation progress
- `docs/turboquant/active/SPEED_PLAN_27B_MTP_OUT6K_20260610.md`: Speed optimization plan
- `docs/experiments/gemma4-assistant-handover-20260611.md`: This handover document

## Next Steps

1. **Fix compilation errors** (CRITICAL)
   - Fix all llama-kv-cache constructor calls
   - Test compilation
   - Test basic functionality

2. **Test MTP speculative decoding**
   - Test with gemma4-assistant models
   - Measure performance improvements
   - Document results

3. **Performance optimization**
   - Test adaptive KV cache precision
   - Measure speed improvements
   - Document best practices

## Contact and Support

For questions or issues:
- Check the documentation in `docs/turboquant/active/`
- Review the implementation in `src/llama-kv-cache.cpp`
- Review the PR at https://github.com/turbo-tan/llama.cpp-tq3/pull/27

## References

- PR: https://github.com/turbo-tan/llama.cpp-tq3/pull/27
- Upstream PR: https://github.com/ggml-org/llama.cpp/pull/23398
- Documentation: `docs/turboquant/active/PHASE_A5_LAYERWISE_ADAPTIVE_KV_PRECISION_PROGRESS.md`

---

**Last Updated**: 2026-06-11  
**Status**: IN PROGRESS - Compilation issues need to be resolved  
**Next Action**: Fix llama-kv-cache constructor calls and test compilation
