# TQ3_4S Decode Speed — Focus Document

**Date**: 2026-04-09  
**Goal**: TG 25+ tok/s on 27B RTX 5060 Ti  
**Current**: TG 16 tok/s (PP 360, chat works, PPL correct)  
**Gap**: +56% needed

---

## What has been tried and failed

| Attempt | Result | Why it failed |
|---|---|---|
| VDR=16 macro-only | 25 tok/s but garbage | Only computed 2 of 4 subgroups |
| VDR=8 original inline vec_dot | 24 tok/s but garbage | g=iqs/8 gives g=0,1 only (misses 2,3) |
| VDR=16 with subgroup loop | 13 tok/s, correct | Worse parallelism (qi/vdr=1) |
| Fused rotation+quantize kernel | +1.4% TG | Marginal, not the bottleneck |
| Native scalar prefill kernel | 32x slower | Wrong approach entirely |
| Naive native kernels | Parity at best | CTA microbench ceiling |
| Blockslot/row-blocking/regscale | Regressed | Various reasons |

## What works (current safe state: 755de5813)

- VDR=8 with subgroup-aware helper (`tq3_4s_dot_subgroup_q8_1`)
- `subgroup0 = iqs/4`, loop `s=0..1` → subgroups {0,1} and {2,3}
- Correct for `iqs = 0, 8` (qi=16, vdr=8 → 2 calls per block)
- TG: 16 tok/s, chat coherent, PPL bit-identical

## Why the gap exists

Q3_K_S gets 25.46 tok/s because:
- No WHT rotation overhead
- Mature optimized MMVQ kernel
- Smaller model (11.44 vs 12.91 GiB)

TQ3_4S decode pipeline per matmul:
1. Pool alloc temp buffer (float)
2. cudaMemcpyAsync activations → temp
3. tq3_rotate_act kernel (WHT rotation)
4. quantize_row_q8_1 (float → q8_1)
5. MMVQ kernel (vec_dot_tq3_4s_q8_1)

Steps 1-4 are overhead Q3_K_S doesn't have.

## Ranked next steps

### 1. Clean VDR sweep (per canonical plan)

Test VDR=4, 8, 16 with the SAFE subgroup-aware vec_dot.
Gate on: chat coherence + tg128 + 10ch PPL.

Current VDR=8 gives 16 tok/s. VDR=4 may be different.
VDR=16 gave 13 tok/s (worse). Need to confirm VDR=4.

### 2. Native WHT ggml op for decode

Instead of the 4-step overhead (alloc → memcpy → rotate → quantize),
add a `GGML_OP_TQ3_WHT` that does butterfly WHT in O(d log d) using
warp shuffles. This is the same kernel as `tq3_rotate_act_kernel`
but as a proper ggml op that can be fused into the graph.

Expected: eliminates steps 1-3, saves ~20% of decode time.

### 3. dp4a in vec_dot

Current vec_dot does 8 scalar float multiplies per subgroup.
Q3_K_S uses dp4a (int8 dot product) which is 4x fewer instructions.
Convert TQ3_4S centroids to int8 levels and use dp4a.

### 4. Hybrid Q8 attention (model change)

Store attn Q/K/V as Q8_0 (no rotation needed), keep FFN as TQ3_4S.
Eliminates rotation for ~35% of decode matmuls.
Requires re-quantization (new GGUF file).

---

## Rules (from steering)

- Test chat BEFORE any push
- 10+ chunk PPL minimum
- One change at a time
- Private branch only until verified
- No macro-only VDR changes
