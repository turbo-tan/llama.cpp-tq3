# Session Handover — 2026-04-14

Update 2026-04-20: active speculative-decode work has moved to `experiment/specdecode-next-v2-20260419` in `/home/awee/code/worktrees/llama-specdecode-next`. The recovered exact contract is `21.2203 -> 24.7688 tok/s` (`1.1672x`) with 10/10 token-exact output at `MIN_MARGIN=1.0`, `LOSSY_NGRAM_BATCH=384`. The previous hardcoded batch `512` OOMed before the 299 MB GPU shadow buffer allocation on RTX 5060 Ti 16GB. See `docs/turboquant/active/SPEC_DECODE_STATUS.md`.

## What Was Achieved Today

### GPU Shadow Buffer (SHIPPED)
- `ggml/src/ggml-cuda/ssm-shadow.cu`: persistent `cudaMalloc` buffer, D2D async copy
- Shadow save: ~0.5ms (was 26ms with CPU path)
- 8/10 exact on original 10 prompts, 10/10 correct answers
- +28.7% avg speed on cached prompts

### Root Cause of Non-Exactness (DIAGNOSED)
- pp2 ≠ tg1+tg1 for Qwen3.5 hybrid SSM — 0.1 logit diff after 2 tokens
- Confirmed across ALL quants (TQ3_4S, Q3_K_S, Q4_0) — NOT our bug
- Upstream hybrid recurrent-state path consistency issue
- Drift only affects continuation phrasing, never the primary answer

### EAGLE3 Pipeline (Historical, Not Active)
- Hidden state capture: 51K pairs from 4B
- Hidden state predictor: cosim=0.76 (21MB GGUF)
- Architecture correct: predict hidden → LM head matmul → token
- **Blocker**: CPU LM head matmul too slow (25ms). Need GPU matmul via ggml graph.
- `llama_model_get_output_tensor()` API added to expose LM head weights

### Gemma 4 Opportunity (DISCOVERED)
- 26B-A4B already runs at 100 tok/s on our GPU
- Pure attention → speculation would be EXACT (no SSM drift)
- **RedHat released EAGLE3 speculator**: `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`
- Also exists for 31B dense: `RedHatAI/gemma-4-31B-it-speculator.eagle3`
- Works with vLLM, can be ported to our framework

## Blocked Items

### TQ3_4S Quantization for Gemma4
- Private repo main has TQ3_4S quantizer (type 45) + Gemma4 model support
- But experiment branches have merge conflicts with main
- Need: clean build from main, BF16 source model for Gemma4
- The Q3_K_S we have can be requantized with `--allow-requantize` once build works

### EAGLE3 GPU Matmul (Historical)
- Need `ggml_mul_mat` on GPU with model's output tensor
- The tensor is on GPU, API to access it is added
- Need to build a mini ggml compute graph for the matmul
- This is the last piece for the proper EAGLE3 architecture

### Gemma 4 31B Dense
- Not downloaded locally
- Would be ~12GB in Q3_K_S, fits on 16GB
- EAGLE3 speculator available from RedHat

## Next Session Priority

This section is historical. For the current checklist-driven plan, use `docs/turboquant/active/ACTIVE_PROGRESS_PLAN.md`.

1. Revalidate speculative decode on current branch tip and save artifacts.

2. GPU-benchmark sparse V dequant on `experiment/decode-speed-plan-x`.

3. Keep TQ3_4S Gemma4 quantization as a lower-priority parked track unless it directly supports the two active items above.

## Key Files Changed Today

| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/ssm-shadow.cu` | GPU D2D shadow buffer |
| `src/llama-memory-recurrent.cpp` | GPU shadow save/restore |
| `src/llama-context.cpp` | `llama_memory_attn_seq_rm`, `llama_memory_recurrent_split` |
| `src/llama-model.cpp` | `llama_model_get_output_tensor()` |
| `include/llama.h` | New API declarations |
| `examples/jacobi/eagle-gpu.cpp` | EAGLE3 with LM head (WIP) |
| `examples/jacobi/pp2-vs-tg1.cpp` | Per-layer divergence diagnostic |
| `examples/jacobi/lossy-ngram.cpp` | Resync, gating improvements |

## Branches
- `experiment/specdecode-next-v2-20260419`: active recovered GPU-shadow speculation branch
- `experiment/persistent-decode`: earlier work (CPU shadow, n-gram cache)
- Main: TQ3_4S quantizer + Gemma4 model support

## Gemma4 TQ3_4S Quantization (BLOCKED)

Attempted quantizing SuperGemma4-26B Q4_K_M → TQ3_4S.
- Added TQ3_4S ftype mapping to quantizer ✅
- Quantizer starts but crashes on first tensor: `blk.0.attn_k.weight [2816, 2048]`
- `GGML_ASSERT(result == nrows * row_size)` — TQ3_4S block size doesn't align with Gemma4 dimensions
- Gemma4 uses n_embd=2816, which is not a multiple of TQ3_4S's expected block alignment
- This is the same blocker from the earlier `experiment/quantize-supergemma4` branch

Fix needed: adjust TQ3_4S quantizer to handle non-standard dimensions (padding or variable block size).

## 31B Download
- Q8_0 download from unsloth started in background
- Check: `tail /tmp/gemma31b_download.log`
- Once downloaded: same quantizer fix needed for TQ3_4S
