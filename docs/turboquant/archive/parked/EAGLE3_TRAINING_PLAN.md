# EAGLE3 Draft Head Training Plan

## Goal
Train a tiny EAGLE3 draft head for Qwen3.5-27B that predicts next tokens
from hidden states. Deploy on RTX 5060 Ti alongside the 27B model.
Use llama.cpp's existing EAGLE3 speculative decode + our seq_cp fork.

## Why This Beats N-gram Cache
- N-gram: memorizes specific sequences, ~90% on seen, ~50% on new
- EAGLE3: learns generation patterns, ~80-90% on EVERYTHING
- N-gram: 10-20MB external file, doesn't generalize
- EAGLE3: ~100-300MB weights, generalizes to any prompt

## VRAM Budget (RTX 5060 Ti 16GB)
- 27B TQ3_4S: 12.9GB
- KV cache (2K ctx): 128MB
- SSM state: 150MB
- Compute buffers: ~500MB
- Available for draft head: ~2.0GB
- EAGLE3 head (1-2 layers): ~100-300MB ✅

## H100 Training Plan (~2-4 hours)

### Step 1: Generate Training Data (1 hour)
Run 27B on H100 with hidden state capture:
- 5000 diverse prompts (alpaca + ShareGPT + code)
- 512 tokens per prompt
- Capture: hidden states at each layer + output tokens
- Output: ~50GB training data

### Step 2: Train EAGLE3 Head (1-2 hours)
- Architecture: 1-2 layer MLP on top of last hidden state
- Input: hidden state (5120 dims for 27B)
- Output: next token logits (248320 vocab)
- With feature projection: 5120 → 1024 → 248320
- Training: standard cross-entropy on next-token prediction
- Framework: PyTorch, standard training loop

### Step 3: Export to GGUF (10 minutes)
- Quantize head weights to Q8_0 or Q4_0
- Package as GGUF with EAGLE3 metadata
- llama.cpp loads it automatically with `--model-draft`

### Step 4: Test on 5060 Ti
```bash
./llama-server \
  -m Qwen3.5-27B-TQ3_4S.gguf \
  --model-draft eagle3-qwen35-27b.gguf \
  -ngl 99 --draft-max 4 \
  --speculative-type eagle3
```

## Alternative: Use Existing 4B as Draft
The Qwen3.5-4B (1.19GB TQ1_0) fits alongside the 27B.
But it's a full model, not a head — slower and lower acceptance.
Still worth testing as a baseline before training EAGLE3.

## Alternative: Block Diffusion (DDTree-style)
Train a small diffusion model that predicts K tokens in one pass.
More complex to train but potentially higher acceptance.
DDTree shows 8.2x on Qwen3 — but requires tree attention
which doesn't work with SSM. Our seq_cp fork handles this.

## Key Insight
llama.cpp already has EAGLE3 support (`COMMON_SPECULATIVE_TYPE_EAGLE3`).
We don't need to build the inference path — just train the head and
package it as GGUF. The seq_cp fork handles SSM rollback.

## Files to Check
- `common/speculative.cpp` — EAGLE3 speculative decode implementation
- `common/common.h:171` — speculative type enum
- `examples/speculative/` — speculative decode example
- `tools/server/server-task.cpp` — server integration
