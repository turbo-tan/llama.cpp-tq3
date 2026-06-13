# Jacobi Decode — Current Plan

**Branch:** `experiment/tq3-decode-next`
**Current target:** broaden speculative acceptance first, then scale to larger qwen35 models

## Why This Works

PP reads weights once → 2048 tokens → 700 tok/s
TG reads weights once → 1 token → 23 tok/s

Jacobi / lookahead reads weights once for multiple guessed tokens, then verifies a prefix.
When the draft is right, it amortizes expensive weight reads across multiple accepted tokens.

## Algorithm

```
# Initialize: guess next 8 tokens (e.g. repeat last token, or n-gram)
guesses = [last_token] * 8

while not converged:
    # ONE forward pass on all 8 tokens (batch, like PP)
    # Reads 12.9GB weights ONCE, processes 8 tokens
    logits[0..7] = forward(guesses[0..7])  # 28.8ms + 8*1.3ms = 39.2ms
    
    # Check: does logits[i] predict guesses[i+1]?
    new_guesses = [argmax(logits[i]) for i in range(8)]
    
    # Count how many are correct from the start
    n_correct = 0
    for i in range(7):
        if new_guesses[i] == guesses[i+1]:
            n_correct += 1
        else:
            break
    
    if n_correct == 7:  # all converged
        emit all 8 tokens
        break
    
    # Update guesses and iterate
    guesses = new_guesses
```

The key practical variant is now one-pass lookahead with a strong draft source:
- if the draft is weak, fall back to normal decode immediately
- if the draft is strong, verify a whole window in one pass and commit the accepted prefix

## What We Implemented

- New prototype binary: `llama-jacobi`
- Source:
  - `/home/awee/code/llama.cpp/examples/jacobi/jacobi.cpp`
  - `/home/awee/code/llama.cpp/examples/jacobi/CMakeLists.txt`
- The implementation uses:
  - n-gram draft from committed history
  - per-sequence state restore, not full-context restore
  - strict live-context gating before speculation
  - adaptive fallback to plain single-token decode when no real continuation exists

## Important Correction

The original plan assumed qwen35 hybrid could use simple rollback-free speculative flow.
That turned out to be false in this tree:

- full-context snapshot/restore worked for correctness but was too expensive
- stock `lookahead` / `lookup` style rollback hits M-RoPE / hybrid sequence-position failures on qwen35
- the working bridge is:
  - speculative verify in batch
  - commit only accepted prefix
  - use per-sequence state restore for correctness

So qwen35 hybrid is not “free rollback”, but it is still workable if the speculative entry is heavily gated.

## Current Measured Results

### Strong-repeat prompt, Qwen3.5-9B TQ3_4S

- baseline `llama-simple`: `36.24 t/s`
- `llama-jacobi`, `W=8`, `iters=1`: `80.96 t/s`
- speedup: `2.23x`
- acceptance: `100%`

Artifacts:
- `/home/awee/code/tan_llama/artifacts/llama_simple_9b_repeat_baseline_20260413_v2.txt`
- `/home/awee/code/tan_llama/artifacts/llama_jacobi_9b_repeat_w8_20260413.txt`

### Mixed structured prompt, Qwen3.5-9B TQ3_4S

- baseline: `37.85 t/s`
- current adaptive `llama-jacobi`: `34.00 t/s`
- still not a clean win
- acceptance remains prompt-sensitive

This tells us:
- the speculative machinery works
- the current limiter is draft quality, not verification math

## What Actually Moved the Needle

1. Fix the off-by-one draft bug
   - draft must use `history + sampled seed`
   - otherwise the n-gram cache is always one token behind

2. Enter speculation only on real live matches
   - generic prompts should not speculate
   - repetition / templated prompts should speculate aggressively

3. Keep the window modest
   - `W=8` is currently the best strong-repeat setting
   - larger windows collapse once acceptance weakens

## Next Plan

### Phase 1: Faster iteration on smaller qwen35
- run the same speculative path on 4B first
- determine which gates and window rules are model-agnostic

### Phase 2: Better draft quality
- improve context-only continuation scoring
- do not speculate on weak unigram/static matches
- prefer short windows on weak matches, longer windows on strong matches

### Phase 3: Soft speculative suffix
- keep a hard committed prefix
- keep a small editable speculative tail
- when new tokens are verified, allow the last `k` speculative tokens to be rewritten
- commit only the stable prefix before that tail

Why this matters:
- current prefix-only acceptance is brittle
- one early mismatch wastes the entire later draft
- many real prompts have near-miss drafts that become correct with one or two more tokens of context

Expected effect:
- better acceptance on natural text
- better punctuation / phrase correction
- less collapse on generic prompts

### Phase 4: Broaden beyond repetition
- target structured prompts with recurring phrases / schemas
- then test code and QA prompts

### Phase 5: Only then port upward
- once 4B behavior is stable, retest 9B
- then move to 27B / 35B

## Working Principle

This path is model-agnostic at the mechanism level:
- batch verify
- accepted-prefix commit
- draft from token history

But it is not model-agnostic in acceptance quality by default.
That comes from:
- tokenizer
- style entropy
- recurrence of local phrases
- prompt structure

So the practical strategy is:
- prove the gating and window policy on small qwen35 first
- then carry the best policy upward
