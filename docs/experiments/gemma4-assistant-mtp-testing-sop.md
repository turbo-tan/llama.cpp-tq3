# Gemma4 Assistant MTP Testing SOP

**Purpose**: Standard operating procedure for testing gemma4-assistant models  
**Scope**: Multi-Token Prediction (MTP) assistant models used with llama.cpp  
**Last Updated**: 2026-06-10

## Architecture Overview

### What is a Gemma4 Assistant?

The gemma4-assistant is a **Multi-Token Prediction (MTP) head**, not a traditional draft model. It's a small neural network (typically 78M params) designed to predict multiple future tokens from the target model's hidden states.

**Key characteristics:**
- Small model: 4 layers, 256 hidden size (vs target's 35 layers, 1536 hidden)
- Uses target model's embeddings via shared memory (`ctx_other`)
- Shares KV cache with target model (no separate cache)
- Predicts N tokens at once (block_size typically 2-3)
- Zero quality loss - output identical to non-speculative decoding

### How MTP Works

```
Target Model (gemma4-E2B-it)
    ↓
Hidden state from last layer
    ↓
Assistant (MTP head) predicts next N tokens
    ↓
Target model verifies predictions
    ↓
Accept correct predictions, reject wrong ones
    ↓
Repeat
```

This is **different from traditional speculative decoding**:
- Traditional: Draft model runs independently, target verifies
- MTP: Assistant uses target's hidden states, predicts in parallel

## Test Environment Setup

### Required Models

1. **Target model**: Full gemma4 model (e.g., `gemma-4-E2B-it-Q4_K_M.gguf`)
2. **Assistant model**: MTP head (e.g., `gemma-4-E2B-it-assistant-BF16.gguf`)

**Important**: The assistant model alone is **not valid** - it requires the target model.

### Build Requirements

The worktree must support:
- `--spec-type mtp` flag for MTP speculative decoding
- `gemma4_assistant` architecture recognition
- Shared memory integration (`ctx_other` parameter)

## Test Procedures

### Test 1: Server with MTP (Recommended)

This is the primary test for production use.

```bash
# Start server with MTP speculative decoding
./build/bin/llama-server \
    -m ./models/gemma-4-E2B-it-Q4_K_M.gguf \
    -md ./models/gemma-4-E2B-it-assistant-BF16.gguf \
    --spec-type mtp \
    --draft-block-size 3 \
    --draft-max 8 \
    --draft-min 0 \
    -ngl 99 -ngld 99 \
    -ctk q8_0 -ctv q8_0 \
    -fa on \
    -c 16384 \
    --host 127.0.0.1 \
    --port 8080
```

**Parameters explained:**
- `-m`: Target model (full gemma4)
- `-md`: Assistant/MTP head model
- `--spec-type mtp`: Use MTP speculative decoder
- `--draft-block-size 3`: Predict 3 tokens per step
- `--draft-max 8`: Max tokens to draft per generation
- `--draft-min 0`: Min tokens (0 = accept any)
- `-ngl 99 -ngld 99`: All layers to GPU (target and draft)
- `-ctk q8_0 -ctv q8_0`: KV cache quantization (q8_0 for accuracy)
- `-fa on`: Flash attention enabled
- `-c 16384`: Context length

**Expected output:**
```
llm_load_tensors: ggml ctx size =    0.33 MiB
llm_load_tensors: offloading 35 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 36/36 layers to #GPU0
llm_load_tensors:        CPU buffer size =   256.00 MiB
llm_load_tensors:      CUDA0 buffer size =  4864.00 MiB
...................................................................................................
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   102.40 MiB
llama_new_context_with_model: KV self size  =  102.40 MiB, K (q8_0):   51.20 MiB, V (q8_0):   51.20 MiB
llama_new_context_with_model: graph nodes  = 1098
llama_new_context_with_model: graph splits = 1
common_init_from_params: added  logit bias = -inf
common_init_from_params: added <|tool_response> logit bias = -inf
common_init_from_params: setting dry_penalty_last_n to ctx_size = 16384
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
srv          init: initializing slots, n_slots = 1
slot         init: id  0 | task -1 | new slot n_ctx_slot = 16384
main: model alias: gemma-4-E2B-it
main: server is listening on http://127.0.0.1:8080 - starting the main loop
```

### Test 2: CLI with MTP

For quick command-line testing:

```bash
# Basic test with MTP
./build/bin/llama-cli \
    -m ./models/gemma-4-E2B-it-Q4_K_M.gguf \
    -md ./models/gemma-4-E2B-it-assistant-BF16.gguf \
    --spec-type mtp \
    --draft-block-size 3 \
    --draft-max 8 \
    -ngl 99 \
    -p "What is the capital of France?" \
    -n 64
```

**Expected behavior:**
- Model loads both target and assistant
- Generates text with MTP speculative decoding
- Shows timing stats including draft acceptance rate

### Test 3: Speculative Example

Using the speculative example binary:

```bash
./build/bin/llama-speculative \
    -m ./models/gemma-4-E2B-it-Q4_K_M.gguf \
    --model-draft ./models/gemma-4-E2B-it-assistant-BF16.gguf \
    --spec-type mtp \
    -p "The capital of France is" \
    -n 32 \
    -ngl 99
```

**Note**: This requires the speculative example to support `--spec-type mtp`.

### Test 4: API Testing

After starting the server (Test 1), test the API:

```bash
# Test completion endpoint
curl http://127.0.0.1:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "What is the capital of France?",
        "max_tokens": 64,
        "temperature": 0.7
    }'
```

Expected response:
```json
{
    "id": "cmpl-...",
    "object": "text_completion",
    "created": ...,
    "model": "gemma-4-E2B-it",
    "choices": [
        {
            "text": " Paris",
            "index": 0,
            "logprobs": null,
            "finish_reason": null
        }
    ],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 1,
        "total_tokens": 9
    }
}
```

## Validation Criteria

### Loading Success

The model loads successfully when you see:
1. Both target and assistant models loaded
2. No tensor dimension errors
3. KV cache initialized for both models
4. Shared memory established (`ctx_other` configured)

### Runtime Success

The model runs successfully when:
1. Text generation completes without errors
2. Draft acceptance rate > 0% (check logs)
3. Response quality matches non-speculative output
4. No crashes or hangs during generation

### Performance Metrics

Check these metrics in server logs:
- `draft_n`: Number of drafts attempted
- `draft_n_accepted`: Number of drafts accepted
- Acceptance rate = `draft_n_accepted / draft_n * 100%`

**Target metrics:**
- Acceptance rate: 50-80% (varies by prompt)
- Speedup: 1.5-3x compared to non-speculative
- Memory: Target + Assistant (minimal overhead)

## Common Issues and Solutions

### Issue 1: Architecture Not Recognized

**Symptom:**
```
unknown model architecture: 'gemma4'
```

**Solution:**
- Ensure worktree has `gemma4_assistant` architecture support
- Check that assistant GGUF has `general.architecture = gemma4_assistant` metadata
- If missing, the GGUF was converted incorrectly

### Issue 2: Tensor Count Mismatch

**Symptom:**
```
wrong number of tensors; expected 48, got 46
```

**Cause:**
This happens when trying to load the assistant as a standalone model or with wrong architecture.

**Solution:**
- Always load assistant with `-md` flag alongside target model
- Use `--spec-type mtp` to select MTP decoder
- Don't try to load assistant alone with `-m`

### Issue 3: Dimension Mismatch

**Symptom:**
```
tensor 'blk.0.attn_q.weight' has wrong shape; expected X, got Y
```

**Cause:**
Hyperparameter loading issue, often related to SWA/full attention layer confusion.

**Solution:**
- Check that `n_layer_nextn` is set correctly
- Verify SWA pattern is loaded from GGUF metadata
- Ensure `n_embd_head_k_full` is read before being used

### Issue 4: Missing Rope Frequencies

**Symptom:**
```
missing tensor 'rope_freqs.weight'
```

**Cause:**
Gemma4 assistant models don't have rope_freqs tensors in GGUF.

**Solution:**
- Make `rope_freqs` tensor creation optional (`TENSOR_NOT_REQUIRED`)
- Use dynamic rope frequency generation if not provided

### Issue 5: No Draft Acceptance

**Symptom:**
- Model loads and runs
- `draft_n_accepted = 0` in logs

**Cause:**
- MTP head not properly connected to target
- Shared memory (`ctx_other`) not established

**Solution:**
- Verify `ctx_other` is set when creating assistant context
- Check that target model is loaded first
- Ensure both models use same vocabulary

## Debugging Procedures

### Enable Verbose Logging

Add `-v` or `--verbose` flag to see detailed loading and inference logs:

```bash
./build/bin/llama-cli \
    -m target.gguf \
    -md assistant.gguf \
    --spec-type mtp \
    -v \
    -p "test"
```

### Check Model Metadata

Verify GGUF metadata is correct:

```bash
./build/bin/llama-gguf-split --info assistant.gguf
```

Expected keys:
```
general.architecture = gemma4_assistant
gemma4_assistant.block_count = 4
gemma4_assistant.embedding_length = 256
gemma4_assistant.attention.head_count = 4
gemma4_assistant.attention.head_count_kv = [1, 1, 1, 1]
gemma4_assistant.attention.sliding_window = [1, 1, 1, 0]
```

### Monitor Memory Usage

Check GPU memory allocation:

```bash
watch -n 1 nvidia-smi
```

Expected:
- Target model: ~5GB (E2B)
- Assistant model: ~100MB
- KV cache: ~500MB (varies by context length)

## Test Checklist

Before marking MTP support as working:

- [ ] Target model loads without errors
- [ ] Assistant model loads with `-md` flag
- [ ] `--spec-type mtp` is recognized
- [ ] Shared memory (`ctx_other`) is established
- [ ] Text generation completes
- [ ] Draft acceptance rate > 0%
- [ ] Output quality matches non-speculative
- [ ] No crashes during extended generation
- [ ] API endpoint works correctly
- [ ] Performance shows measurable speedup

## References

- [AtomicChat GGUF Repository](https://huggingface.co/AtomicChat/gemma-4-E2B-it-assistant-GGUF)
- [Google Gemma4 Model Card](https://huggingface.co/google/gemma-4-E2B-it-assistant)
- [Upstream llama.cpp](https://github.com/ggerganov/llama.cpp)
- [atomic-llama-cpp-turboquant fork](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant)

## Version History

- **2026-06-10**: Initial version documenting MTP testing procedures
