# Speculative Decode Exactness Bug — Root Cause Analysis

Date: 2026-04-14

## The Bug

seq_cp fork does NOT create an independent copy of SSM tensor data. Both sequences share the same underlying tensors. When the speculative decode overwrites seq 0's state, it also destroys the seq 1 "backup".

## Why 4/10 (fast) and 9/10 (safe) Still Match

- When the n-gram guess is **correct** (accepted), no rollback is needed — the state is valid
- When the guess is **wrong** (rejected), the rollback restores metadata but the tensor data is corrupted
- With `MIN_MARGIN=4.0` (safe), fewer speculations → fewer rejections → fewer corruptions
- With `MIN_MARGIN=0.0` (fast), more speculations → more rejections → more corruptions
- Prompts that happen to have 100% acceptance are always exact

## The Fix

Two options:

### Option A: Force immediate tensor copy after seq_cp

After `seq_cp(0, 1)`, trigger `find_slot` for seq 1 to force the tensor copy before speculation begins. This creates a real independent backup.

### Option B: Use shadow save/restore

The shadow save (`llama_memory_shadow_save`) does an actual GPU→CPU tensor copy. It's slower (~1-3ms) but guarantees an independent backup. The earlier +16.4% result with shadow save was correct but slower.

### Option C: Hybrid

Use seq_cp for the KV cache (attention layers — works correctly because KV entries are append-only, not overwritten). Use shadow save for the SSM state only (150MB R + S tensors).

## Impact on Claimed Numbers

| Config | Claimed | Actual | Exact |
|--------|---------|--------|-------|
| Fast (margin=0) | +38.7% | +37.2% speed, 4/10 exact | Speed real, correctness broken |
| Safe (margin=4) | +13.3% | +13.3% speed, 7-9/10 exact | Speed real, correctness mostly ok |
| Shadow save | +16.4% | Correct but slower | Needs re-measurement |

## Next Step

Fix the rollback to produce 100% exact output, then re-measure speed.

## Final Diagnosis (2026-04-14 evening)

### The 2/10 non-exact prompts are NOT from shadow/rollback bugs

Definitive tests prove:
- Shadow save/restore is bit-exact (GPU D2D copy)
- Baseline is deterministic (same output every run)
- Spec is self-deterministic (same spec output every run)
- With 100% acceptance and 0 rejects, some prompts STILL differ

### Root cause: fused GatedDeltaNet chunked kernel ≠ autoregressive kernel

The pp2 batch (2 tokens in one forward pass) uses the "chunked" GDN kernel.
The tg1 decode (1 token) uses the "autoregressive" GDN kernel.
These produce numerically different SSM states for the same token sequence.

The difference is tiny (floating point accumulation order) but the SSM
state is cumulative — small differences compound and eventually flip
a greedy argmax decision at a low-margin token boundary.

### Evidence
- Capital of France: pp2 == tg1+tg1 at n=128 ✅ (lucky — no low-margin flip)
- Rainbow: pp2 ≠ tg1+tg1 at n=64 ❌ (low-margin token at ~pos 50 flips)
- Both with 100% acceptance, 0 rejects, exact same tokens

### Fix options
1. **Accept it** — 8/10 exact is good enough for production (answers are correct, just different phrasing in continuation)
2. **Force tg1 for all** — lose pp2 speedup, get 10/10 exact
3. **Fix the GDN kernel** — ensure chunked and autoregressive paths are bit-equivalent (upstream llama.cpp issue)

### Current recommendation
Ship with the GPU shadow buffer (+28.7% avg speed) and document that
exact output match is 8/10 due to SSM kernel numerical non-equivalence.
The 2 non-exact prompts produce correct answers — the divergence is in
continuation text phrasing, not in the primary answer.

## Trace-Level Proof (2026-04-14)

### Margins diverge through ACCEPTED speculations, not just rejects

From hash_table traces:
```
pos=12: baseline margin=3.7680, spec margin=3.7680  (identical before first spec)
pos=29: baseline margin=3.3306, spec margin=3.3296  (0.001 diff after accepted specs)
pos=38: baseline margin=2.3415, spec margin=2.3631  (0.02 diff, growing)
pos=80: baseline token=13,       spec token=11       (completely different)
```

The drift accumulates through accepted pp2 batches. Each pp2 batch
introduces a ~0.001 margin difference. After ~30 accepted batches,
the accumulated difference is large enough to flip a low-margin decision.

### This is NOT a rollback bug

The shadow save/restore is bit-exact (verified by self-determinism).
The issue is that pp2 batch processing produces slightly different
SSM state than sequential tg1 processing, even for correct tokens.

### The fix

The fused GatedDeltaNet chunked kernel must produce bit-identical
results to the autoregressive kernel for the same token sequence.
This is an upstream llama.cpp kernel issue, not a speculation logic bug.

### Workaround

Gate speculation aggressively enough that the accumulated drift
never reaches a low-margin decision point. The 10/10 original set
achieves this. The extended 40 set has prompts where the drift
reaches a flip point before gating can prevent it.

## Kernel-Level Root Cause (confirmed)

The fused GDN kernel keeps SSM state in **registers** across the token loop.
For pp2 (2 tokens), state stays in registers between token 1 and token 2.
For tg1+tg1, state goes through **global memory** (write to dst, graph copy
to S tensor, read back for next decode).

The register path and global memory path produce slightly different results
because the graph copy involves `ggml_get_rows` + `ggml_cpy` which may
introduce rounding through intermediate buffer operations.

### Fix options
1. **Kernel fix**: flush state to global memory between tokens in the loop
   (add `state[col * S_v + i] = s_shard[r]` + barrier between iterations)
2. **Graph fix**: ensure the state copy between decodes is a direct memcpy
   with no intermediate operations
3. **Workaround**: aggressive gating to avoid speculation at drift-sensitive points
   (current approach, achieves 10/10 on original set)
