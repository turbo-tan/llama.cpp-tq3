# Gemma4 MTP Implementation - Status Handover

**Date**: 2026-06-10  
**Status**: IN PROGRESS - Segfault during warmup phase  
**Branch**: `feature/gemma4-assistant-upstream-alignment`

---

## Executive Summary

We have successfully implemented the layer counting alignment with upstream llama.cpp to support gemma4-assistant MTP models. The core refactoring is complete and the model loads successfully, but there is a segmentation fault during the warmup phase that prevents full MTP inference.

---

## Current Status

### ✅ Completed Work

1. **Layer Counting Refactoring** (136 files updated)
   - Added `n_layer_all` field to store total layer count from GGUF
   - Added `n_layer_nextn` field to store MTP/nextn layer count
   - Added `n_layer()` method that returns `n_layer_all - n_layer_nextn`
   - Removed `n_layer` member variable (now a method)
   - Updated all code to use `n_layer()` for regular layers or `n_layer_all` for total

2. **KV Cache Updates**
   - KV cache creation now iterates over `n_layer_all` (all layers including MTP)
   - Fixed `layers.resize()` to use `n_layer_all`
   - Fixed layer loops in `load_tensors()` to use `n_layer_all`
   - Added `n_layer_all` to `LLAMA_LOAD_LOCALS` macro

3. **Gemma4-Assistant Model Support**
   - Updated `gemma4-assistant.cpp` to set `n_layer_nextn = n_layer_all`
   - For gemma4-assistant: `n_layer() = 0` (no regular layers, all are MTP)
   - Model loads successfully with correct layer structure

4. **Context Creation Support**
   - Added `ctx_other` field to `llama_context_params`
   - Added validation to ensure `ctx_other` is set for gemma4-assistant models
   - Added support in `common_init_result` to detect gemma4-assistant and set `ctx_other`
   - Updated speculative example to detect gemma4-assistant and pass `ctx_tgt`

5. **KV Cache Sharing**
   - KV cache sharing is working (logs show layers being shared correctly)
   - Example output:
     ```
     layer   3: sharing with layer 47. k = 0x7f94ebb00000, v = 0x7f94ebb80000
     layer   0: sharing with layer 46. k = 0x7f96a1c00000, v = 0x7f96a1e00000
     layer   1: sharing with layer 46. k = 0x7f96a1c00000, v = 0x7f96a1e00000
     layer   2: sharing with layer 46. k = 0x7f96a1c00000, v = 0x7f96a1e00000
     ```

### ❌ Current Issue

**Segmentation fault during warmup phase:**
```
/home/awee/code/worktrees/tan_llama-main-ref/ggml/src/ggml-backend.cpp:194: GGML_ASSERT(buffer) failed
```

**GDB Backtrace:**
```
#0  __GI___wait4 (pid=4071457, stat_loc=0x0, options=0, usage=0x0)
#1  ggml_print_backtrace ()
#2  ggml_abort ()
#3  ggml_backend_buffer_get_type ()
#4  ggml_backend_buffer_is_host ()
#5  llama_kv_cache::set_input_k_idxs(...)
#6  llm_graph_input_attn_kv_iswa::set_input(...)
#7  llm_graph_result::set_inputs(...)
#8  llama_context::process_ubatch(...)
#9  llama_context::decode(...)
#10 llama_decode ()
#11 common_init_from_params(...)
#12 main ()
```

**Root Cause Analysis:**
- The crash occurs in `ggml_backend_buffer_is_host()` when checking `dst->buffer`
- The buffer pointer is null, causing the assertion to fail
- This happens during `set_input_k_idxs()` which is called during warmup
- The issue is likely that shared KV cache layers don't have their own buffers allocated

---

## Git Branches

### Main Branches

1. **`master`** (tan_llama)
   - Status: Clean, up to date
   - Contains: Base documentation and analysis documents
   - No code changes

2. **`feature/gemma4-assistant-upstream-alignment`** (tan_llama-main-ref worktree)
   - Status: **ACTIVE** - Contains all the layer counting refactoring
   - Contains: 136+ files modified with layer counting changes
   - Last commit: Layer counting refactoring complete
   - **This is the branch we're working on**

### Worktrees

1. **`/home/awee/code/worktrees/tan_llama-main-ref`**
   - Branch: `feature/gemma4-assistant-upstream-alignment`
   - Status: Active development
   - Build: Successful (compiles without errors)
   - Runtime: Segfault during warmup

2. **`/home/awee/code/worktrees/upstream-master-test`**
   - Branch: `master` (upstream)
   - Status: Reference only
   - Purpose: Compare with upstream implementation

---

## Files Modified

### Core Changes (136+ files)

**Header files:**
- `src/llama-hparams.h` - Added `n_layer_all`, `n_layer_nextn`, `n_layer()` method
- `src/llama-model.h` - Added `n_layer_all` to `LLAMA_LOAD_LOCALS` macro
- `src/llama-kv-cache.h` - Updated to use `n_layer_all`
- `src/llama-context.h` - Added `ctx_other` field
- `include/llama.h` - Added `ctx_other` to `llama_context_params`

**Implementation files:**
- `src/llama-hparams.cpp` - Implemented `n_layer()` method, updated all layer checks
- `src/llama-model.cpp` - Updated model loading to use `n_layer_all`
- `src/llama-kv-cache.cpp` - Updated KV cache creation to iterate over `n_layer_all`
- `src/llama-context.cpp` - Added `ctx_other` validation and handling
- `src/llama-model-saver.cpp` - Updated to save `n_layer_all`

**Model files:**
- `src/models/gemma4-assistant.cpp` - Set `n_layer_nextn = n_layer_all`
- `src/models/gemma4.cpp` - Updated to use `n_layer_all`
- All other model files - Updated to use `n_layer()` method

**Common files:**
- `common/common.cpp` - Added gemma4-assistant detection and `ctx_other` setting
- `common/common.h` - Added `ctx_tgt` field to draft params

**Example files:**
- `examples/speculative/speculative.cpp` - Added gemma4-assistant detection

---

## Key Code Changes

### 1. Layer Counting (src/llama-hparams.h)

```cpp
struct llama_hparams {
    uint32_t n_layer_all;        // Total layers (from GGUF block_count)
    uint32_t n_layer_nextn = 0;  // Number of MTP/nextn layers
    
    uint32_t n_layer() const {
        return n_layer_all - n_layer_nextn;
    }
};
```

### 2. Gemma4-Assistant (src/models/gemma4-assistant.cpp)

```cpp
void llama_model_gemma4_assistant::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    if (hparams.n_layer_nextn == 0) {
        hparams.n_layer_nextn = hparams.n_layer_all;
    }
    // For gemma4-assistant: n_layer() = 0 (no regular layers)
}
```

### 3. Context Creation (src/llama-context.cpp)

```cpp
if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
    if (params.ctx_other == nullptr) {
        throw std::runtime_error("Gemma4Assistant requires ctx_other to be set");
    }
    cparams.ctx_other = params.ctx_other;
}
```

### 4. KV Cache Sharing (src/llama-kv-cache.cpp)

```cpp
for (uint32_t il = 0; il < hparams.n_layer_all; il++) {
    if (share && other) {
        const int32_t il_share = share(il);
        if (il_share >= 0) {
            const auto & layer_share = other->layers[other->map_layer_ids[il_share]];
            layers.push_back(layer_share);  // Share the layer
            layers.back().il = il;
            continue;
        }
    }
    // ... create new layer ...
}
```

---

## Testing

### Test Command

```bash
cd /home/awee/code/worktrees/tan_llama-main-ref
./build/bin/llama-speculative \
    --model /home/awee/models/google/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf \
    --model-draft /home/awee/models/gemma-4-12B-it-assistant-Q8_0.gguf \
    --prompt "Hello" \
    --n-predict 10 \
    --ctx-size 512
```

### Expected Behavior

1. Target model loads successfully ✅
2. Draft model (gemma4-assistant) loads successfully ✅
3. KV cache layers are shared correctly ✅
4. Warmup phase completes successfully ❌ (segfault)
5. Speculative decoding runs ❌ (blocked by segfault)

### Actual Behavior

```
I main: detected gemma4-assistant draft model, setting ctx_other
I common_init_result: setting ctx_other for gemma4-assistant model
W llama_kv_cache: layer   3: sharing with layer 47
W llama_kv_cache: layer   0: sharing with layer 46
W llama_kv_cache: layer   1: sharing with layer 46
W llama_kv_cache: layer   2: sharing with layer 46
W common_init_from_params: warming up the model with an empty run
/home/awee/code/worktrees/tan_llama-main-ref/ggml/src/ggml-backend.cpp:194: GGML_ASSERT(buffer) failed
Segmentation fault (core dumped)
```

---

## Root Cause Analysis

### The Problem

The segfault occurs because:
1. Gemma4-assistant layers are shared from the target model's KV cache
2. Shared layers don't have their own buffers allocated
3. During warmup, `set_input_k_idxs()` tries to access the buffer
4. The buffer pointer is null, causing the assertion to fail

### Why This Happens

When we share a layer:
```cpp
layers.push_back(layer_share);  // Copy the layer struct
layers.back().il = il;           // Update the layer index
```

The layer struct contains pointers to K and V tensors:
```cpp
struct layer {
    ggml_tensor * k;
    ggml_tensor * v;
    // ...
};
```

These tensors have buffers, but the buffers belong to the target model's KV cache, not the draft model's KV cache. When we try to access the buffer in the draft model's context, it fails.

### Potential Solutions

1. **Don't share buffers for shared layers**
   - Allocate separate buffers for shared layers in the draft model
   - Pro: Clean separation of concerns
   - Con: Defeats the purpose of sharing (uses more memory)

2. **Fix buffer access to handle shared layers**
   - Check if buffer is null before accessing
   - Use the target model's buffer if available
   - Pro: Maintains memory efficiency
   - Con: More complex, potential for other issues

3. **Don't share layers at all for gemma4-assistant**
   - Create separate KV cache for draft model
   - Pro: Simplest solution
   - Con: Uses more memory, may not be the intended design

4. **Fix the warmup to skip shared layers**
   - Skip warmup for layers that are shared
   - Pro: Quick fix
   - Con: May hide other issues

---

## Next Steps

### Immediate (Priority: HIGH)

1. **Fix the segfault**
   - Investigate why shared layers don't have valid buffers
   - Implement one of the potential solutions above
   - Test that warmup completes successfully

2. **Test full MTP inference**
   - Once segfault is fixed, test speculative decoding
   - Verify acceptance rate and speedup
   - Compare with baseline (no MTP)

### Medium-term (Priority: MEDIUM)

3. **Optimize KV cache sharing**
   - Ensure memory efficiency is maintained
   - Profile memory usage
   - Verify that sharing actually saves memory

4. **Add comprehensive tests**
   - Test with different model sizes
   - Test with different context sizes
   - Test edge cases (empty prompts, long prompts, etc.)

### Long-term (Priority: LOW)

5. **Document the implementation**
   - Update documentation with final design
   - Add examples and usage instructions
   - Document performance characteristics

6. **Compare with upstream**
   - Run upstream version with same models
   - Compare behavior and performance
   - Identify any remaining differences

---

## Documentation

### Analysis Documents (in tan_llama/docs/experiments/)

1. `gemma4-mtp-test-report.md` - Initial test results
2. `gemma4-mtp-model-compatibility.md` - Model compatibility analysis
3. `gemma4-mtp-upstream-diff-analysis.md` - Upstream difference analysis
4. `gemma4-mtp-upstream-alignment-status.md` - Alignment status (this document)

### Code Documentation

- All changes are documented in commit messages
- Key functions have inline comments
- Architecture decisions are documented in analysis documents

---

## References

### Upstream Implementation

- Upstream branch: `master` (ggml-org/llama.cpp)
- Key files to reference:
  - `src/llama-hparams.h` - Layer counting
  - `src/llama-model.cpp` - Model loading
  - `src/llama-kv-cache.cpp` - KV cache creation
  - `src/models/gemma4-assistant.cpp` - Gemma4-assistant model

### Related Issues

- Original issue: Layer counting mismatch between our code and upstream
- Current issue: Segfault during warmup with shared KV cache layers

### Related PRs

- PR #27: gemma4 MTP support (turbo-tan/llama.cpp-tq3)
  - Status: Open
  - Contains: Initial embeddings_nextn infrastructure
  - Note: This PR is from before the layer counting refactoring

---

## Contact

For questions or clarifications:
- Review the analysis documents in `docs/experiments/`
- Check the commit history on `feature/gemma4-assistant-upstream-alignment` branch
- Run the test command to reproduce the issue
- Use GDB to debug the segfault

---

**Last Updated**: 2026-06-10  
**Next Review**: After segfault is fixed  
**Priority**: HIGH - This blocks gemma4 MTP support
