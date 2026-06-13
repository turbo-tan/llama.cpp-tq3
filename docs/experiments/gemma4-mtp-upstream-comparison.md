# Gemma4 MTP Porting Analysis: Upstream vs Our Implementation

**Date**: 2026-06-10  
**Status**: CRITICAL FINDINGS - Our porting has fundamental architectural issues

## Executive Summary

After analyzing upstream master's implementation, we discovered that our gemma4 MTP porting has **fundamental architectural differences** that make it incompatible with the actual gemma4-assistant models. The upstream implementation uses a completely different approach that we need to adopt.

## Key Differences

### 1. Architecture Name

| Aspect | Upstream | Our Implementation |
|--------|----------|-------------------|
| Architecture enum | `LLM_ARCH_GEMMA4_ASSISTANT` | `LLM_ARCH_GEMMA4_MTP` |
| Architecture string | `"gemma4-assistant"` | `"gemma4_mtp"` |
| Model class | `llama_model_gemma4_assistant` | `llama_model_gemma4_mtp` |
| Source file | `src/models/gemma4-assistant.cpp` | `src/models/gemma4_mtp.cpp` |

**Impact**: Our architecture name doesn't match the actual model metadata, causing loading failures.

### 2. Tensor Names

| Tensor | Upstream | Our Implementation |
|--------|----------|-------------------|
| Pre-projection | `LLM_TENSOR_NEXTN_PROJ_PRE` → `"nextn.pre_projection"` | `LLM_TENSOR_MTP_PRE_PROJ` → `"nextn.pre_projection"` |
| Post-projection | `LLM_TENSOR_NEXTN_PROJ_POST` → `"nextn.post_projection"` | `LLM_TENSOR_MTP_POST_PROJ` → `"nextn.post_projection"` |

**Status**: ✅ We fixed the string mappings to match, but the enum names are different.

### 3. Context Integration (CRITICAL)

**Upstream Approach**:
```cpp
// In gemma4-assistant.cpp
GGML_ASSERT(cparams.ctx_other != nullptr);
const auto * model_other = llama_get_model(cparams.ctx_other);

// Access target model's token embeddings
ggml_tensor * x = ggml_get_rows(ctx0, model_other->tok_embd, inp_tokens);
```

**Our Approach**:
```cpp
// In gemma4_mtp.cpp
// No ctx_other integration
// Uses model.tok_embd directly (wrong!)
```

**Impact**: This is the **most critical difference**. Upstream's gemma4-assistant model:
- Requires `ctx_other` to be set to the target model context
- Accesses the target model's token embeddings through `ctx_other`
- Uses the target model's vocabulary and embeddings

Our implementation:
- Doesn't use `ctx_other` at all
- Tries to use its own token embeddings (which don't exist in the assistant model)
- Is fundamentally incompatible with the actual model structure

### 4. Metadata Keys

| Metadata | Upstream | Our Implementation |
|----------|----------|-------------------|
| Predict layers | `LLM_KV_NEXTN_PREDICT_LAYERS` | Custom key lookup |
| Backbone size | `hparams.n_embd_inp()` (from `embedding_length_out`) | Custom key `"gemma4-assistant.embedding_length_out"` |

**Status**: We partially fixed this, but upstream uses a more systematic approach.

### 5. Speculative Decoding Integration

**Upstream**:
```cpp
// In common/speculative.cpp
llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
```

**Our Implementation**:
```cpp
// Similar approach but without ctx_other check
llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
```

**Status**: ✅ Our embeddings_nextn infrastructure is correct and matches upstream.

## Root Cause Analysis

### Why Our Implementation Fails

1. **Wrong Architecture Name**: We used `gemma4_mtp` instead of `gemma4-assistant`
   - The actual model has `general.architecture = gemma4-assistant`
   - Our code expects `general.architecture = gemma4_mtp`

2. **Missing ctx_other Integration**: The gemma4-assistant model is designed to work with a target model through `ctx_other`
   - It doesn't have its own token embeddings
   - It relies on the target model's embeddings
   - Our implementation doesn't provide this integration

3. **Incomplete Model Structure**: Our `gemma4_mtp.cpp` doesn't match the actual model's tensor structure
   - Missing proper handling of `nextn_proj_pre` and `nextn_proj_post`
   - Incorrect layer structure assumptions

### Why Upstream's Approach Works

1. **Correct Architecture Name**: Uses `gemma4-assistant` which matches the model
2. **Proper ctx_other Integration**: The model can access the target model's embeddings
3. **Complete Model Structure**: The implementation matches the actual model's tensor layout
4. **Server Integration**: The server properly sets `ctx_other` when creating the draft context

## What Works in Our Implementation

✅ **embeddings_nextn Infrastructure**: 
- The core infrastructure for extracting hidden states is correct
- Matches upstream's approach
- Can be reused

✅ **Speculative Decoding Integration**:
- Setting `embeddings_nextn` on target and draft contexts
- The fallback logic in the draft loop

✅ **Build System**:
- Code compiles successfully
- Dependencies are correct

## What Needs to Be Fixed

### Option 1: Align with Upstream (RECOMMENDED)

**Effort**: 3-5 days

**Changes Required**:
1. Rename `LLM_ARCH_GEMMA4_MTP` → `LLM_ARCH_GEMMA4_ASSISTANT`
2. Rename `gemma4_mtp.cpp` → `gemma4-assistant.cpp`
3. Rename `llama_model_gemma4_mtp` → `llama_model_gemma4_assistant`
4. Add `ctx_other` integration to the model
5. Update tensor enum names to match upstream
6. Update metadata key handling to match upstream
7. Update server code to set `ctx_other` for gemma4-assistant

**Benefits**:
- Fully compatible with actual gemma4-assistant models
- Aligned with upstream for easier maintenance
- Proper integration with target model

### Option 2: Keep Current Approach (NOT RECOMMENDED)

**Effort**: Unknown, likely 5-10 days

**Changes Required**:
1. Create a custom model format that works with our architecture
2. Convert gemma4-assistant models to our format
3. Maintain a fork divergent from upstream

**Drawbacks**:
- Incompatible with standard gemma4-assistant models
- Requires custom model conversion
- Diverges from upstream
- High maintenance burden

## Recommendation

**Adopt Option 1**: Align our implementation with upstream master.

**Rationale**:
1. Upstream's implementation is correct and tested
2. It matches the actual model structure
3. It's easier to maintain alignment with upstream
4. Our embeddings_nextn infrastructure can be reused
5. The effort is reasonable (3-5 days)

## Next Steps

1. **Immediate**: Document the findings in this analysis
2. **Short-term**: Create a new branch to align with upstream
3. **Medium-term**: Implement the changes and test with actual models
4. **Long-term**: Merge the aligned implementation into main

## Code References

### Upstream Files to Reference

- `src/llama-arch.cpp`: Architecture registration
- `src/llama-arch.h`: Architecture enum definitions
- `src/models/gemma4-assistant.cpp`: Model implementation
- `src/models/models.h`: Model class definitions
- `src/llama-context.cpp`: Context initialization with ctx_other
- `tools/server/server-context.cpp`: Server integration
- `common/speculative.cpp`: Speculative decoding integration

### Our Files to Modify

- `src/llama-arch.cpp`: Rename architecture
- `src/llama-arch.h`: Rename enum
- `src/models/gemma4_mtp.cpp` → `gemma4-assistant.cpp`: Complete rewrite
- `src/models/models.h`: Rename class
- `src/llama-context.cpp`: Add ctx_other handling
- `common/speculative.cpp`: Already correct, minor updates needed

## Conclusion

Our gemma4 MTP porting has **fundamental architectural issues** that make it incompatible with actual gemma4-assistant models. The upstream implementation uses a different approach with `ctx_other` integration that we need to adopt.

**The good news**: Our embeddings_nextn infrastructure is correct and can be reused. We just need to align the model implementation with upstream.

**The bad news**: This requires significant refactoring of the model code, but it's necessary for compatibility.

**The path forward**: Align with upstream master's implementation, reusing our embeddings_nextn infrastructure.

---

**Last Updated**: 2026-06-10  
**Next Review**: Before starting the refactoring work  
**Priority**: HIGH - This blocks gemma4 MTP support
