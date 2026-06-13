# Shadow State Decode — Virtual DOM for LLM Inference

**Branch:** `experiment/persistent-decode`
**Concept:** Like React's virtual DOM — speculate on a shadow copy, diff, commit matches.

## The Idea

```
Main State (committed)     Shadow State (speculative)
┌──────────────────┐       ┌──────────────────┐
│ KV cache pos 0-N │       │ KV cache pos 0-N+1 │
│ SSM state @ N    │       │ SSM state @ N+1    │
└──────────────────┘       └──────────────────┘
         │                          │
         │  token N+1 correct?      │
         │◄─── YES: swap shadow→main (pointer swap, ~0 cost)
         │                          │
         │  token N+1 wrong?        │
         │◄─── NO: discard shadow, keep main (free)
```

## Why This Works

Current Jacobi overhead: save state (0.5ms) + restore state (0.5ms) per window.
With shadow state: pointer swap (0.001ms) — 1000x faster.

## Memory Cost

SSM state for Qwen3.5-27B: ~50MB (32 SSM layers × ~1.5MB each)
KV cache: already handles rollback via seq_rm
Total extra memory: ~50MB (0.3% of 16GB VRAM)

## Implementation

### Data Structure
```cpp
struct shadow_state {
    // Two SSM state buffers — swap between them
    std::vector<uint8_t> ssm_state[2];
    int active = 0;  // which buffer is "committed"
    
    void save()    { /* copy active → shadow */ }
    void commit()  { active ^= 1; /* pointer swap */ }
    void discard() { /* nothing — shadow is just abandoned */ }
};
```

### Decode Loop (N=2)
```
1. Save SSM state to shadow buffer (one-time copy, 50MB)
2. Decode token N+1 (writes to KV cache + updates SSM state)
3. Sample token N+1 → get token N+2 prediction
4. Decode token N+2 speculatively (writes to KV + SSM shadow)
5. Verify: does the model agree with token N+2?
   YES → commit shadow (pointer swap), emit both tokens
   NO  → discard shadow, seq_rm KV for pos N+2, emit only token N+1
6. Repeat from step 1
```

### Key Insight
Step 2 and 4 are ONE batch decode (2 tokens, 1 weight read).
The verify happens by checking logits[1] == token N+2.
No separate verify pass needed — it's built into the batch decode.

### Expected Performance
- Current: 41.4ms for 1 token = 24.2 tok/s
- Shadow N=2: 46.7ms for 2 tokens × 80% acceptance = 1.6 tokens
  = 46.7ms / 1.6 = 29.2ms effective = 34.2 tok/s
- Shadow N=4: 57.3ms for 4 tokens × 70% acceptance = 2.8 tokens
  = 57.3ms / 2.8 = 20.5ms effective = 48.8 tok/s

## Files to Modify

1. `src/llama-memory-recurrent.cpp` — add shadow buffer support
2. `examples/jacobi/jacobi.cpp` — replace state save/restore with shadow swap
3. New: `src/llama-shadow-state.h` — shadow state manager

## TDD

Test 1: Shadow save/commit/discard cycle
Test 2: N=2 decode produces correct output (matches sequential)
Test 3: Speed improvement over baseline
Test 4: Memory usage stays within 50MB extra

## Why This Beats Draft Models

- Draft model: 1-2GB extra memory, separate forward pass, low acceptance on Qwen3.5
- Shadow state: 50MB extra memory, same model, same forward pass, built-in verify
- No SSM rollback problem — we just swap pointers

## CORRECTED Implementation (after analysis)

The overhead is NOT state copy (3ms). It's the EXTRA DECODE CALL (41ms).

### Current Jacobi flow (SLOW):
```
1. Save SSM state (3ms)
2. Decode [token + 7 guesses] as batch (41ms) — 1 weight read
3. Check which guesses match
4. Restore SSM state (3ms)
5. Decode [accepted tokens only] (41ms) — SECOND weight read!
Total: 88ms for ~4 accepted tokens = 45 tok/s... but actually slower
because of sync overhead between the two decodes.
```

### Shadow state flow (FAST):
```
1. Copy SSM state to shadow (0.33ms GPU-to-GPU)
2. Decode [current_token, guess_next] as 2-token batch (46.7ms)
   - Reads weights ONCE for both tokens
   - logits[0] = verified prediction for next token
   - logits[1] = prediction for token after that (bonus)
3. Check: does argmax(logits[0]) == guess_next?
   YES: accept both tokens, keep SSM state as-is (0ms)
   NO:  restore SSM from shadow (0.33ms), seq_rm KV pos+1 (0ms)
Total YES: 47ms for 2 tokens = 42.6 tok/s
Total NO:  47.3ms for 1 token = 21.1 tok/s
At 80% acceptance: 0.8*2 + 0.2*1 = 1.8 tokens per 47ms = 38.3 tok/s
```

### Key difference from current Jacobi:
- ONE decode call, not two
- Shadow restore only on mismatch (20% of the time)
- No serialization — GPU-to-GPU tensor copy

### Memory cost:
- SSM shadow: 150MB (fixed, context-independent)
- KV rollback: free (seq_rm is instant)
