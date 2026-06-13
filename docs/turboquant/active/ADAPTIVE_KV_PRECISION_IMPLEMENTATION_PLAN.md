# Phase A5: Layer-wise Adaptive KV Cache Precision — Implementation Plan

**Date:** 2026-06-11  
**Branch:** `perf/a5-layerwise-adaptive-kv` (in `turbo-tan/llama.cpp-tq3`)  
**Status:** Implementation started, compilation errors need fixing

## What This Feature Does

Layer-wise adaptive KV cache precision allows different quantization levels for different transformer layers. Top N layers (closer to output, more critical for quality) use high precision (e.g., Q8_0), while bottom layers (closer to input, more redundant) use low precision (e.g., Q4_0). This reduces KV cache memory while preserving quality.

**Expected impact:**
- Memory savings: 30-40% KV cache reduction when half layers use Q4_0 instead of Q8_0
- Quality impact: Minimal (<1% perplexity increase)
- Speed impact: Minimal (<5% slowdown)

## What Has Been Done

### 1. API Changes (committed as `bf11b80de`)

**Files modified:**
- `include/llama.h` — Added `type_k_low`, `type_v_low`, `n_layers_high_precision` to `llama_context_params`
- `src/llama-cparams.h` — Added matching fields to internal `llama_cparams`
- `common/common.h` — Added fields to `common_params` struct
- `common/arg.cpp` — Added CLI args: `--cache-type-k-low`, `--cache-type-v-low`, `--n-layers-high-precision`
- `common/common.cpp` — Added parameter conversion from `common_params` to `llama_context_params`
- `src/llama-kv-cache.h` — Added member variables for layer-wise precision settings
- `src/llama-kv-cache.cpp` — Implemented layer-wise precision logic in KV cache allocation
- `src/llama-context.cpp` — Context parameter handling (includes gemma4-assistant code that needs removal)

### 2. The Adaptive KV Logic (already in `llama-kv-cache.cpp`)

```cpp
// Layer-wise adaptive KV cache precision (Phase A5)
ggml_type layer_type_k = type_k;  // default: high precision
ggml_type layer_type_v = type_v;

if (n_layers_high_precision > 0 && type_k_low != GGML_TYPE_COUNT) {
    // Bottom layers use low precision
    if (il < (n_layer - n_layers_high_precision)) {
        layer_type_k = type_k_low;
    }
}
if (n_layers_high_precision > 0 && type_v_low != GGML_TYPE_COUNT) {
    if (il < (n_layer - n_layers_high_precision)) {
        layer_type_v = type_v_low;
    }
}

ggml_tensor * k = ggml_new_tensor_3d(ctx, layer_type_k, ...);
ggml_tensor * v = ggml_new_tensor_3d(ctx, layer_type_v, ...);
```

## What Needs To Be Done

### Step 1: Fix Compilation Errors

The code from `tan_llama` was copied wholesale and includes gemma4-assistant specific code that doesn't compile on this branch. The following files have errors that need fixing:

**`src/llama-context.cpp`:**
1. `llm_arch_supports_rs_rollback` — function not declared. Find it or remove the check.
2. `n_layer_all` — not a member of `llama_hparams` on this branch. Use `n_layer()` or `n_layer_kv`.
3. `llama_memory_params` — too many initializers. Removed `mem_other` field but initializer list still has it.
4. `set_embeddings` — redefined. There's a duplicate definition at line 891 and 1065.
5. `embd_nextn` — referenced but not declared. All `embeddings_nextn` code needs removal.

**`src/llama-kv-cache.h`:**
1. `llama_kv_cells_vec` — type not found. Check if it should be `llama_kv_cells` or add proper include.

**`src/llama-cparams.h`:**
- Already cleaned up (ctx_other and embeddings_nextn removed)

**`include/llama.h`:**
- Already cleaned up (ctx_other removed from llama_context_params)

### Step 2: Build

```bash
cd /home/awee/code/llama.cpp-tq3/build
cmake .. -DGGML_CUDA=OFF  # CPU-only for faster iteration
cmake --build . --target llama-cli -j8
```

### Step 3: Test with llama-server (not llama-cli)

**Important:** llama-cli doesn't support TQ3_4S quantization. Use llama-server instead:

```bash
cd /home/awee/code/llama.cpp-tq3/build
cmake --build . --target llama-server -j8
```

### Step 4: Run Benchloop Test

```bash
# Start server with adaptive KV cache
./build/bin/llama-server \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  -a qwen36-27b-mtp-tq3_4s-outq6 \
  --host 0.0.0.0 --port 8085 \
  --threads 8 --threads-batch 8 \
  --gpu-layers 99 -np 1 -c 65536 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --cache-type-k-low q4_0 \
  --cache-type-v-low q4_0 \
  --n-layers-high-precision 30 \
  --spec-type draft-mtp \
  --spec-draft-n-min 1 --spec-draft-n-max 2 --spec-draft-p-min 1.0 \
  --gpu-layers-draft 99 \
  --spec-draft-type-k q4_0 --spec-draft-type-v tq3_0 \
  --cache-ram 0 --jinja

# Run benchloop (speed suite only for quick test)
cd /home/awee/code/tan_llama
bench-loop run \
  --endpoint http://127.0.0.1:8085 \
  --provider openai_compat \
  --model qwen36-27b-mtp-tq3_4s-outq6 \
  --suites speed \
  --harness raw \
  --hardware "NVIDIA RTX 3090 24GB"
```

### Step 5: Compare Baseline vs Adaptive

Run two benchloop tests:
1. **Baseline:** `-ctk q8_0 -ctv q8_0` (uniform Q8_0)
2. **Adaptive:** `-ctk q8_0 -ctv q8_0 --cache-type-k-low q4_0 --cache-type-v-low q4_0 --n-layers-high-precision 30`

Compare:
- Speed score (t/s)
- Memory usage (nvidia-smi)
- Quality (benchloop quality score)

### Step 6: Tune Parameters

Try different values of `n_layers_high_precision`:
- 10 layers high precision (bottom 50 use low precision)
- 20 layers high precision (bottom 40 use low precision)
- 30 layers high precision (bottom 30 use low precision)

Find the optimal balance of memory savings vs quality.

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/llama-kv-cache.cpp` | KV cache allocation with adaptive precision |
| `src/llama-kv-cache.h` | KV cache header with member variables |
| `include/llama.h` | Public API (`llama_context_params`) |
| `src/llama-cparams.h` | Internal context parameters |
| `common/arg.cpp` | CLI argument parsing |
| `common/common.cpp` | Parameter conversion |

## Models Available for Testing

- `/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf` (winner, 14GB)
- `/home/awee/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q4_K_M.gguf` (smaller, ~10GB)
- `/home/awee/models/unsloth/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q3_K_M.gguf` (smallest, ~9GB)

## Previous Benchloop Results (for comparison)

From 2026-06-11 with baseline (no adaptive KV):
- Speed score: 72.2
- Generation: 51.42 t/s
- Prompt eval: 133.26 t/s
- Memory: ~16GB GPU

## Related Documentation

- `docs/turboquant/active/SPEED_PLAN_27B_MTP_OUT6K_20260610.md` — Overall speed plan
- `docs/turboquant/active/PHASE_A5_LAYERWISE_ADAPTIVE_KV_PRECISION_PROGRESS.md` — Design document
- `docs/turboquant/active/SPEED_PLAN_PROGRESS_20260611.md` — Overall progress report
