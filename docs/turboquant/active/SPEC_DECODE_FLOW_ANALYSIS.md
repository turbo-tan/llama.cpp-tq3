# Speculative Decode Flow Analysis

## Current Flow (shadow copy)

```mermaid
sequenceDiagram
    participant Cache as N-gram Cache
    participant SSM as SSM State (150MB)
    participant Shadow as Shadow Copy (150MB)
    participant KV as KV Cache
    participant GPU as GPU Compute

    Note over SSM: State at position N

    Cache->>Cache: lookup guess for N+2
    
    rect rgb(255, 200, 200)
        Note over SSM,Shadow: SAVE: 26ms (64 × cudaMemcpy+sync)
        SSM->>Shadow: copy 150MB GPU→CPU
    end

    rect rgb(200, 255, 200)
        Note over GPU: DECODE: ~63ms (pp2 batch)
        GPU->>SSM: process [token_N+1, guess_N+2]
        GPU->>KV: write KV entries at N+1, N+2
    end

    alt guess correct (ACCEPT)
        Note over Shadow: discard shadow (free)
        Note over SSM: State at N+2 ✓
    else guess wrong (REJECT)
        rect rgb(255, 200, 200)
            Note over Shadow,SSM: RESTORE: 26ms (64 × cudaMemcpy+sync)
            Shadow->>SSM: copy 150MB CPU→GPU
        end
        KV->>KV: seq_rm entries at N+1, N+2
        rect rgb(200, 200, 255)
            Note over GPU: RE-DECODE: ~44ms (tg1)
            GPU->>SSM: process token_N+1 cleanly
            GPU->>KV: write KV entry at N+1
        end
    end
```

## Cost Analysis

```
ACCEPT path: 26ms (save) + 63ms (pp2) = 89ms for 2 tokens = 22.5 tok/s
REJECT path: 26ms (save) + 63ms (pp2) + 26ms (restore) + 44ms (redecode) = 159ms for 1 token = 6.3 tok/s
BASELINE:    44ms (tg1) for 1 token = 22.7 tok/s

At 90% accept: 0.9 × 89 + 0.1 × 159 = 96ms for 1.9 tokens = 19.8 tok/s — SLOWER than baseline!
```

**The shadow save/restore (52ms round trip) eats ALL the speculation gain.**

## The Git Insight

Git doesn't copy the whole repo for each commit. It stores:
1. The **current state** (working tree)
2. A **ref** to the parent commit
3. Only the **diff** (changed files)

For SSM state:
- The "working tree" = current R/S tensors (cell 0)
- The "parent commit" = state before speculation
- The "diff" = what the speculation changed

**We don't need to copy 150MB. We need to UNDO the diff.**

## The Undo Approach

```mermaid
sequenceDiagram
    participant SSM as SSM State (cell 0)
    participant GPU as GPU Compute
    participant KV as KV Cache

    Note over SSM: State at position N

    rect rgb(200, 255, 200)
        Note over GPU: DECODE token_N+1 only (tg1, 44ms)
        GPU->>SSM: process token_N+1
        GPU->>KV: write KV at N+1
    end

    Note over SSM: State at N+1 (committed, correct)
    Note over GPU: logits → verified_N+2, also get guess from cache

    alt have guess for N+2
        rect rgb(200, 255, 200)
            Note over GPU: DECODE guess_N+2 (tg1, 44ms)  
            GPU->>SSM: process guess_N+2
            GPU->>KV: write KV at N+2
        end
        
        Note over GPU: logits[N+1] already told us verified_N+2
        
        alt guess == verified (ACCEPT)
            Note over SSM: State at N+2 ✓
            Note over GPU: 2 tokens in 88ms = 22.7 tok/s (same as baseline)
        else guess != verified (REJECT)
            Note over SSM: State WRONG (processed wrong token)
            Note over GPU: Need to undo... back to copying
        end
    end
```

**Wait — this is just 2 sequential decodes. No speedup.**

## The REAL Insight: Batch Without Rollback

The speedup comes from pp2 (2 tokens in 63ms vs 2×44ms=88ms).
But pp2 requires knowing BOTH tokens upfront.

We know token_N+1 (from logits after decoding token_N).
We need to GUESS token_N+2.

```mermaid
sequenceDiagram
    participant SSM as SSM State
    participant GPU as GPU Compute

    Note over SSM: State at N-1

    rect rgb(200, 255, 200)
        Note over GPU: DECODE token_N (tg1, 44ms)
        GPU->>SSM: process token_N → state at N
    end
    
    Note over GPU: logits → token_N+1 (verified)
    Note over GPU: cache lookup → guess for N+2

    rect rgb(255, 255, 200)
        Note over GPU: KEY QUESTION
        Note over GPU: Can we batch [token_N+1, guess_N+2]
        Note over GPU: WITHOUT needing rollback?
    end
```

## The Double-Buffer Approach (no copy needed)

```mermaid
graph TD
    A[SSM has 2 cells on GPU already - 299MB allocated] --> B{Speculation?}
    B -->|Yes| C[Cell 0: decode token_N+1 → state at N+1]
    C --> D[Cell 0 is now committed at N+1]
    D --> E[seq_cp 0→1: cell 1 shares cell 0]
    E --> F[find_slot for seq 1: copies cell 0 data → cell 1]
    F --> G[Decode guess on seq 0 at N+2]
    G --> H{Guess correct?}
    H -->|Yes| I[Keep seq 0 at N+2, drop seq 1]
    H -->|No| J[Drop seq 0, restore from seq 1 at N+1]
    J --> K[Decode verified_N+2 on seq 0]
```

**The key: commit token_N+1 FIRST (single decode), THEN speculate on N+2.**

After committing N+1, the state is correct. The speculation on N+2 can be rolled back to N+1 (which is in cell 1).

But this is 2 separate decodes (44ms + 44ms = 88ms) not a pp2 batch (63ms). The pp2 gain is lost.

## The REAL Solution: Accept That pp2 Requires Upfront Knowledge

```
pp2 batch [A, B] at positions [N, N+1]:
- Attention: A sees pos 0..N-1, B sees pos 0..N
- SSM: processes A then B sequentially
- Cost: 63ms (one weight read)

To use pp2, we need BOTH A and B before the batch.
A = token_N+1 (known from previous logits)
B = guess_N+2 (from cache)

If B is wrong, SSM processed wrong token → need rollback.
Rollback cost = shadow save + restore = 52ms.
Net: 63ms + 52ms = 115ms for 1 token on reject.

The ONLY way to avoid rollback: never be wrong.
Or: make rollback cost → 0.
```

## Making Rollback Cost → 0

The RS buffer has 2 cells (299MB). Cell 0 = active, Cell 1 = empty.

**Before pp2 batch:** copy cell 0 → cell 1 ON GPU.
- 150MB at 448 GB/s = 0.33ms
- But needs CUDA API (not available in .cpp files)

**After pp2 batch (reject):** copy cell 1 → cell 0 ON GPU.
- 0.33ms

**Total rollback cost: 0.66ms** (vs 52ms currently)

```
ACCEPT: 0.33ms (save) + 63ms (pp2) = 63.3ms for 2 tokens = 31.6 tok/s (+39%)
REJECT: 0.33ms (save) + 63ms (pp2) + 0.33ms (restore) + 44ms (redecode) = 107.7ms for 1 token
At 90% accept: 0.9 × 63.3 + 0.1 × 107.7 = 67.7ms for 1.9 tokens = 28.1 tok/s (+24%)
```

## Implementation: Move Shadow to .cu File

```
ggml/src/ggml-cuda/ssm-shadow.cu:
  void ssm_shadow_save(tensors, n_tensors)  — cudaMemcpyAsync D2D × N, 1 sync
  void ssm_shadow_restore(tensors, n_tensors) — cudaMemcpyAsync D2D × N, 1 sync
```
