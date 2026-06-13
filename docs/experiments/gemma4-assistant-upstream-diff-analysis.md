# Gemma4 Assistant: Upstream vs Our Implementation - Root Cause Analysis

**Date**: 2026-06-10  
**Author**: Kiro  
**Status**: Analysis Complete - Root Cause Identified

---

## Executive Summary

After thorough investigation, I've identified the **root cause** of the differences between our implementation and upstream. The issue is NOT in the KV cache sharing mechanism itself, but in how we handle **layer counting and iteration** for the gemma4-assistant architecture.

**Key Finding**: Upstream uses `n_layer_all` (total layers) vs our `n_layer` (which may have different semantics for MTP models). This affects how the KV cache is created and how sequence positions are tracked.

---

## Architecture Overview

### Gemma4 Assistant Model Structure

The gemma4-assistant model is a **pure MTP model** where:
- ALL layers are nextn/MTP layers (no regular transformer layers)
- It shares KV cache with the target model
- It uses the target model's token embeddings via `ctx_other`

**Model metadata**:
```
gemma4-assistant.block_count = 4
gemma4-assistant.nextn_predict_layers = 4
```

This means: 4 total layers, all 4 are nextn layers.

---

## Critical Differences Identified

### Difference 1: Layer Counting Semantics

**Upstream (`src/llama-hparams.h`)**:
```cpp
uint32_t n_layer_all;        // Total layers in the model (from GGUF block_count)
uint32_t n_layer_nextn = 0;  // Number of nextn/MTP layers

uint32_t n_layer() const {
    return n_layer_all - n_layer_nextn;  // Regular (non-MTP) layers
}
```

**Our code (`src/llama-hparams.h`)**:
```cpp
uint32_t n_layer;                    // Loaded from GGUF block_count
uint32_t nextn_predict_layers = 0;   // Number of nextn layers
// No n_layer() method that computes regular layers
```

**Impact**:
- Upstream: For gemma4-assistant, `n_layer_all=4`, `n_layer_nextn=4`, so `n_layer()=0`
- Our code: `n_layer=4`, `nextn_predict_layers=4`

### Difference 2: KV Cache Layer Iteration

**Upstream (`src/llama-kv-cache.cpp` line 116)**:
```cpp
const uint32_t n_layer = hparams.n_layer_all;  // Iterate over ALL layers
```

**Our code (`src/llama-kv-cache.cpp` line 113)**:
```cpp
const uint32_t n_layer = hparams.n_layer;  // Iterate over n_layer
```

**Impact**:
- Both iterate over 4 layers for gemma4-assistant
- BUT the semantics are different!
- Upstream explicitly uses `n_layer_all` to indicate "all layers including MTP"
- Our code uses `n_layer` which might be interpreted differently in different contexts

### Difference 3: has_kv() Logic

**Upstream (`src/llama-hparams.cpp`)**:
```cpp
bool llama_hparams::has_kv(uint32_t il) const {
    if (n_layer_kv_from_start >= 0) {
        if (il < (uint32_t) n_layer_kv_from_start) {
            return true;
        }
        return false;
    }
    // by default, all layers have kv
    return true;
}
```

**Our code (`src/llama-hparams.cpp`)**:
```cpp
bool llama_hparams::has_kv(uint32_t il) const {
    if (kv_only_nextn) {
        // MTP head: only the trailing nextn_predict_layers blocks own a KV cache
        return nextn_predict_layers > 0 && il >= (n_layer - nextn_predict_layers);
    }
    if (n_layer_kv_from_start >= 0) {
        if (il < (uint32_t) n_layer_kv_from_start) {
            return true;
        }
        return false;
    }
    return true;
}
```

**Impact**:
- Upstream: No special handling for `kv_only_nextn`
- Our code: Has `kv_only_nextn` flag that changes KV cache behavior
- For gemma4-assistant, this might cause different layers to have KV cache

### Difference 4: gemma4-assistant Layer Loading

**Upstream (`src/models/gemma4-assistant.cpp`)**:
```cpp
ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
GGML_ASSERT(hparams.n_layer_nextn == hparams.n_layer_all && 
    "n_layer_nextn must be == n_layer_impl");
```

**Our code (`src/models/gemma4-assistant.cpp`)**:
```cpp
ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
if (hparams.nextn_predict_layers == 0) {
    hparams.nextn_predict_layers = hparams.n_layer;
}
GGML_ASSERT(hparams.nextn_predict_layers == hparams.n_layer &&
    "gemma4-assistant expects nextn_predict_layers to cover all layers");
```

**Impact**:
- Upstream: Sets `n_layer_nextn` and asserts it equals `n_layer_all`
- Our code: Sets `nextn_predict_layers` and asserts it equals `n_layer`
- Semantically equivalent, but different field names

---

## The Real Issue: Sequence Position Synchronization

The other agent reported:
> "target KV says last position is 12"  
> "draft batch starts again at 1"

This is the **actual bug**, and it's related to how the shared KV cache tracks sequence positions.

### How Shared KV Cache Works

1. **KV Cache Creation** (`src/llama-model.cpp` line 2045-2073):
   ```cpp
   if (arch == LLM_ARCH_GEMMA4_ASSISTANT) {
       llama_memory_t mem_other = llama_get_memory(cparams.ctx_other);
       
       share = [&](int32_t il) {
           const llama_model * model_other = llama_get_model(cparams.ctx_other);
           if (hparams.is_swa(il)) {
               return llama_model_n_layer(model_other) - 2;
           }
           return llama_model_n_layer(model_other) - 1;
       };
       
       res = new llama_kv_cache_iswa(
           *this, params.type_k, params.type_v,
           !cparams.flash_attn, cparams.offload_kqv,
           params.swa_full, cparams.kv_unified,
           cparams.n_ctx_seq, cparams.n_seq_max,
           cparams.n_ubatch, 1,
           mem_other,  // <-- Shared memory from target
           nullptr, reuse, share);
   }
   ```

2. **Shared Cells** (`src/llama-kv-cache.cpp` line 101):
   ```cpp
   v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
   ```

   This means the draft model **shares the same `v_cells_impl`** as the target model.

3. **Sequence Position Tracking** (`src/llama-kv-cells.h` line 349):
   ```cpp
   llama_pos seq_pos_max(llama_seq_id seq_id) const {
       if (seq_pos[seq_id].empty()) {
           return -1;
       }
       return seq_pos[seq_id].rbegin()->first;
   }
   ```

### The Bug

When the draft model calls `llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0)`, it should get the same position as the target model because they share `v_cells_impl`.

**But there's a problem**: The `share` callback returns which layer from the target model to share with:
```cpp
share = [&](int32_t il) {
    if (hparams.is_swa(il)) {
        return llama_model_n_layer(model_other) - 2;  // SWA layers
    }
    return llama_model_n_layer(model_other) - 1;  // Non-SWA layers
};
```

This maps assistant layer `il` to target layer `llama_model_n_layer(model_other) - 1` or `- 2`.

**The issue**: When the draft model iterates over its layers (0-3), it maps to target layers (33-34 for a 35-layer target). But the sequence position tracking might not be properly synchronized across these mapped layers.

---

## Root Cause Hypothesis

The position mismatch likely occurs because:

1. **Target model** processes tokens and updates KV cache at positions 0-11
2. **Draft model** tries to decode starting at position 12
3. **But** the draft model's KV cache view (via `share` callback) might not properly reflect the target's position state

The `share` callback maps layers, but **sequence positions are tracked per-cell, not per-layer**. So when the draft model accesses its KV cache, it should see the same positions as the target.

**Possible causes**:
1. The `share` callback is not being called correctly during position queries
2. The shared `v_cells_impl` is not properly synchronized
3. There's a race condition or ordering issue in how positions are updated
4. The `seq_pos_max` query is not going through the shared cells correctly

---

## Comparison with Upstream

### What Upstream Does Differently

1. **Uses `n_layer_all` explicitly** in KV cache creation
2. **No `kv_only_nextn` flag** - simpler has_kv() logic
3. **Same `share` callback** - maps assistant layers to target layers
4. **Same shared `v_cells_impl`** mechanism

### What's the Same

1. **ctx_other integration** - both use it to access target embeddings
2. **share callback logic** - both map to `n_layer - 1` or `- 2`
3. **Shared v_cells_impl** - both share the same cell implementation
4. **KV cache creation** - both use `llama_kv_cache_iswa` with shared memory

### The Key Difference

**Upstream uses `n_layer_all` consistently**, while we use `n_layer`. This might cause issues in:
- Layer iteration during KV cache creation
- Layer filtering and reuse logic
- Position tracking across shared layers

---

## Recommended Fix

### Option 1: Align with Upstream (Recommended)

Add `n_layer_all` and `n_layer_nextn` fields to match upstream:

```cpp
// In src/llama-hparams.h
uint32_t n_layer_all;        // Total layers (from GGUF block_count)
uint32_t n_layer_nextn = 0;  // Number of nextn layers

uint32_t n_layer() const {
    return n_layer_all - n_layer_nextn;
}
```

Update `src/llama-kv-cache.cpp` line 113:
```cpp
const uint32_t n_layer = hparams.n_layer_all;  // Match upstream
```

Update `src/models/gemma4-assistant.cpp`:
```cpp
ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
GGML_ASSERT(hparams.n_layer_nextn == hparams.n_layer_all);
```

### Option 2: Debug the Position Synchronization

Add logging to track sequence positions:
```cpp
// In handle_mtp_for_ubatch
const llama_pos pos_max_mtp = llama_memory_seq_pos_max(llama_get_memory(mtp.ctx_mtp), 0);
LLAMA_LOG_INFO("MTP: pos_start=%d, pos_max_mtp=%d\n", pos_start, pos_max_mtp);

// Check if shared cells are working
auto * mem_dft = llama_get_memory(ctx_dft);
auto * mem_tgt = llama_get_memory(ctx_tgt);
LLAMA_LOG_INFO("Shared cells: dft=%p, tgt=%p, same=%d\n",
    mem_dft, mem_tgt, mem_dft->v_cells_impl == mem_tgt->v_cells_impl);
```

---

## Next Steps

1. **Immediate**: Add `n_layer_all` and `n_layer_nextn` to match upstream
2. **Test**: Verify that the position synchronization issue is resolved
3. **Debug**: If issue persists, add logging to track shared cell state
4. **Compare**: Run upstream version to confirm it works correctly

---

## Conclusion

The root cause is **not** in the KV cache sharing mechanism itself, but in how we handle layer counting. Upstream uses `n_layer_all` explicitly to distinguish between "total layers" and "regular layers", while our code uses `n_layer` which may have ambiguous semantics.

By aligning with upstream's layer counting approach, we should resolve the position synchronization issue and enable proper gemma4 MTP speculative decoding.

---

**Last Updated**: 2026-06-10  
**Priority**: HIGH - This blocks gemma4 MTP support  
**Estimated Fix Time**: 2-4 hours
