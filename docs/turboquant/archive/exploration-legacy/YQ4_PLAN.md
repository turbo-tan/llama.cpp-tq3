# YQ4 — Next Generation Quantization Ideas

Date: 2026-03-29

## Ranking: Value vs Effort

Status note:

- `TQ4_0` is now **investigated and parked**, not yet adopted.
- It stays interesting because the first real 9B tensor screen is positive.
- It is parked because we still lack:
  - a real model-level PPL win over `Q4_0`
  - a user-facing chat-path quality check
  - runtime-cost evidence
- So it should not displace the active `TQ3` launch pad until those are proven.

| Rank | Idea | Expected Gain | Effort | Risk |
|------|------|--------------|--------|------|
| 1 | **TQ4_0** (WHT + Q4_0) | modest but real same-size quality gain | Low | Medium |
| 2 | **Importance-aware quant** | likely best next low-bpw path | Medium | Medium |
| 3 | **SVD skeleton + residual** | possible strong compression, heavy runtime cost | Medium | High |
| 4 | **Layer delta compression** | interesting only as a later secondary codec | Medium | High |
| 5 | **Spectral (DCT) compression** | mostly theoretical today | High | High |
| 6 | **Predictive (FLAC-style)** | storage-only unless integrated deeply | High | High |

---

## Rank 1: TQ4_0 — WHT + Q4_0 (INVESTIGATED, PARKED)

**Concept:** Apply WHT rotation before Q4_0 quantization. Quantize in the rotated domain and undo the rotation at dequant / runtime.

**Proof:** Tested on 100K synthetic blocks with outliers:
- Q4_0: RMSE 0.081 (baseline)
- TQ4_0: RMSE 0.057 (30% better, same 4.5 bpv)

**Why it works:** WHT spreads outliers across all 32 elements. After rotation, the distribution is more uniform → Q4_0 quantization error is lower.

**Block format:** same density class as `Q4_0` (18 bytes/32 values = 4.5 bpv)
```c
// Quantize: WHT(sign*x)/sqrt(32) -> Q4_0
// Dequant:  Q4_0_dequant -> inverse_WHT -> unsign
```

Important correction:

- the inference path is **not** literally identical to native `Q4_0`
- if the weights stay quantized in the rotated domain, runtime must undo that transform somewhere
- so this line is still attractive, but it is not “free”

**Files to touch:**
- `ggml/src/ggml-quants.c` — add `quantize_row_tq4_0` and `dequantize_row_tq4_0`
- `ggml/include/ggml.h` — add `GGML_TYPE_TQ4_0`
- `ggml/src/ggml.c` — register type traits
- `tools/quantize/quantize.cpp` — add `TQ4_0` option
- `ggml/src/ggml-cpu/ggml-cpu.c` — register vec_dot (reuse Q4_0's)

**Prototype test:** `tests/test-tq4-prototype.c` (see below)

**Expected PPL:** Q4_0 PPL - 0.3 to 0.5 (based on 30% RMSE improvement)

### Reality Check (2026-03-29, local 9B tensor screen)

The first local real-tensor prototype on the 9B witness is directionally positive, but much weaker than
the synthetic claim above.

Using five real tensors from:

- `/home/awee/models/bartowski/Qwen_Qwen3.5-9B-GGUF/Qwen_Qwen3.5-9B-Q8_0.gguf`

Measured with a tensor-level `Q4_0` vs `TQ4_0` prototype (`WHT -> Q4_0 -> inverse WHT`):

- `Q4_0 RMSE`: `0.001108`
- `TQ4_0 RMSE`: `0.001062`
- `Q4_0 dot`: `0.006253`
- `TQ4_0 dot`: `0.005991`
- ratio vs `Q4_0`: `0.958 / 0.958`

Interpretation:

- the idea has real legs
- but the observed gain on real 9B tensors is about `4.2%`, not `26-30%`
- therefore the next correct gate is a short `2`-chunk 9B PPL comparison, not an immediate big claim

---

## Rank 2: Importance-Aware Quantization

**Why it moved up:** This is the closest idea to what is already working for `TQ3`: spend bits only where the model seems to care.

**Concept:** Use activation/sensitivity statistics to identify important weight blocks. Quantize important blocks at a safer format, unimportant blocks more aggressively.

**Current skeptical read:**

- this is much more plausible than uniform global compression claims
- it matches what already worked for adaptive `TQ3` promotion
- unlike `TQ4_0`, it targets the low-bpw regime directly
- unlike SVD, it does not force a second matmul into inference

**Best near-term shape:**

1. start from a simple dense baseline
2. compute block sensitivity from a short calibration pass
3. allocate a small promoted budget or a mixed low/high format budget
4. validate on:
   - short PPL
   - 9B chat-path gate

This is the strongest remaining non-parked idea.

### Local 9B tensor screen (2026-03-29)

Using the same real 9B tensor subset as the `TQ4_0` reality check:

- base `TQ3_0`:
  - `RMSE 0.002230`
  - `dot 0.012585`
- importance-aware mix at `4.05 bpw`, promote hot blocks to `Q4_0`:
  - `RMSE 0.001477`
  - `dot 0.008214`
- importance-aware mix at `4.05 bpw`, promote hot blocks to `TQ4_0`:
  - `RMSE 0.001466`
  - `dot 0.008153`

Comparison versus uniform `Q4_0` on the same tensor screen:

- uniform `Q4_0`:
  - `RMSE 0.001108`
  - `dot 0.006253`

Gap from the better importance-aware mix (`TQ4_0` promoted) to uniform `Q4_0`:

- RMSE gap:
  - absolute: `0.001466 - 0.001108 = 0.000358`
  - relative: about `32.3%` worse than uniform `Q4_0`
- dot gap:
  - absolute: `0.008153 - 0.006253 = 0.001900`
  - relative: about `30.4%` worse than uniform `Q4_0`

Interpretation:

- importance-aware mixing is much better than plain `TQ3_0`
- `TQ4_0` is the better promoted target than plain `Q4_0`
- but at this `4.05 bpw` budget it is still clearly behind uniform `Q4_0`
- so this line has legs as a low-bpw research path, not yet as a replacement for `Q4_0`

### Budget ladder vs uniform `Q4_0`

Same tensor screen, base `TQ3_0`, promote the most dot-sensitive blocks to `TQ4_0`:

| Avg bpw | RMSE | Dot |
|---|---:|---:|
| `4.05` | `0.001466` | `0.008153` |
| `4.10` | `0.001414` | `0.007861` |
| `4.20` | `0.001312` | `0.007303` |
| uniform `Q4_0` | `0.001108` | `0.006253` |

Relative gap to uniform `Q4_0`:

- `4.05 bpw`
  - RMSE: about `32.3%` worse
  - dot: about `30.4%` worse
- `4.10 bpw`
  - RMSE: about `27.6%` worse
  - dot: about `25.7%` worse
- `4.20 bpw`
  - RMSE: about `18.4%` worse
  - dot: about `16.8%` worse

Interpretation:

- the line improves smoothly as budget rises
- but it does not close the `Q4_0` gap fast enough to become the near-term launch pad
- this remains a credible lower-bpw research direction, not the best current product bet

### WHT-aware ranking follow-up (2026-03-29)

We also tested whether the promotion ranking itself should be aware of the WHT geometry used by
`TQ4_0`, instead of ranking blocks only by original-domain dot error.

New ranking objectives screened on the same 9B tensor subset:

- `rot_sse`: rank by improvement in rotated-domain SSE (`WHT(ref)` vs `WHT(recon)`)
- `dot_rot`: rank by a normalized hybrid of original-domain dot-error gain plus rotated-domain SSE gain

Results for `TQ3_0 -> TQ4_0` importance-aware promotion:

| Avg bpw | Objective | RMSE | Dot |
|---|---|---:|---:|
| `4.05` | `dot` | `0.001466` | `0.008153` |
| `4.05` | `rot_sse` | `0.001455` | `0.008214` |
| `4.05` | `dot_rot` | `0.001610` | `0.008983` |
| `4.10` | `dot` | `0.001414` | `0.007861` |
| `4.10` | `rot_sse` | `0.001403` | `0.007920` |
| uniform `Q4_0` | n/a | `0.001108` | `0.006253` |

Interpretation:

- `rot_sse` slightly improves RMSE versus the plain `dot` ranking
- but it slightly worsens dot error
- the normalized hybrid `dot_rot` is clearly worse and should be discarded
- so the current best ranking objective remains the plain original-domain `dot` objective

Practical conclusion:

- `WHT + importance-aware` still has legs
- but the gain is coming mostly from the promoted `TQ4_0` block geometry itself
- not from switching the ranking objective into the rotated domain
- therefore this line stays a research path, not the next launch pad

---

## Rank 3: SVD Skeleton + Residual

**Concept:** Decompose W ≈ U×V^T (rank-r, fp16) + residual (TQ3_0). The skeleton captures the important structure; the residual handles the rest.

**Math:**
```
W (2048×2048) = U (2048×64 fp16) × V^T (64×2048 fp16) + R (TQ3_0)
Storage: 2×2048×64×2 + 2048×2048×3.5/8 = 524KB + 1.84MB = 2.36MB
vs Q4_0: 2048×2048×4.5/8 = 2.36MB
```
Same size as Q4_0, but quality = fp16 skeleton + TQ3_0 residual.

**Why better:** The rank-64 SVD captures ~90% of the weight variance. The residual has low variance → TQ3_0 is sufficient.

**Effort:** Requires SVD during quantization (offline, slow but one-time). Inference needs two matmuls (U×(V^T×x) + residual×x).

**Skeptical read:**

- the storage math is interesting
- the runtime cost is being understated
- this is not a same-kernel or same-throughput path
- quality could be good, but this is a systems project by itself, not a quick launch-pad experiment

So this stays interesting, but it is below importance-aware quantization in real ROI.

**Prototype:** Compute SVD of a weight matrix, measure residual quality.

---

## Rank 4: Layer Delta Compression

**Concept:** Adjacent transformer layers have similar weights. Store layer 0 fully, store deltas for layers 1-N.

**Math:**
```
delta[i] = layer[i] - layer[i-1]
|delta| << |layer|  (empirically ~10x smaller)
Quantize delta at 1-2 bits instead of 4 bits
```

**Storage:** Layer 0 at 4.5 bpv + layers 1-N at ~0.5 bpv = avg ~1 bpv for N=32 layers.

**Inference:** Reconstruct layer i = layer 0 + sum(delta[1..i]). Cache reconstructed layers.

**Risk:** Reconstruction error accumulates across layers. Need to verify quality doesn't degrade.

**Skeptical read:**

- this is better thought of as a secondary codec or archival/storage idea
- it is not a near-term inference-quality launch pad
- dense transformer layers are related, but not similar enough to assume cheap deltas without a lot of empirical work

---

## Rank 5: Spectral (DCT) Compression

**Concept:** Apply 2D DCT to weight matrices. Keep top-k% of coefficients. Like JPEG for weights.

**Math:**
```
W (2048×2048) → DCT2D → keep top 10% → quantize at 8-bit
Storage: 10% × 2048² × 8-bit + indices = ~0.8 bpv effective
```

**Risk:** High. DCT assumes smooth structure. LLM weights may not be smooth enough. Needs empirical validation.

**Skeptical read:**

- this is far from current evidence
- there is no sign yet that LLM weight matrices behave like image blocks strongly enough to justify this path
- keep as moonshot only

---

## Rank 6: Predictive (FLAC-style)

**Skeptical read:**

- mostly a storage idea unless the codec is integrated into runtime
- if it is integrated into runtime, it becomes another complex dequant path
- so this is not a next launch-pad candidate for model quality

---

## Exploration Order (after parking `TQ4_0`)

If we come back to the rest of this file, the practical order should be:

1. importance-aware quantization
2. real `TQ4_0` model-level PPL gate
3. SVD skeleton + residual
4. everything else

Short version:

- `TQ4_0`: parked until a real model-level win is proven
- importance-aware quantization: best remaining new exploration line
- SVD: later, if we are willing to pay a real runtime-complexity cost

---

## Prototype Plan

### Step 1: TQ4_0 Prototype (today)

```bash
# Build and run
gcc -O2 tests/test-tq4-prototype.c -lm -o /tmp/test_tq4
/tmp/test_tq4

# Expected output:
# Q4_0  RMSE: 0.081
# TQ4_0 RMSE: 0.057 (30% better)
# Same 4.5 bpv
```

### Step 2: TQ4_0 on Real Model

```bash
# Quantize Qwen3.5-9B to TQ4_0
llama-quantize Qwen3.5-9B-Q8_0.gguf Qwen3.5-9B-TQ4_0.gguf TQ4_0

# Compare PPL
llama-perplexity -m Qwen3.5-9B-Q4_0.gguf -f wiki.test.raw -c 512
llama-perplexity -m Qwen3.5-9B-TQ4_0.gguf -f wiki.test.raw -c 512
```

### Step 3: SVD Prototype

```python
import numpy as np
W = load_weight_matrix()  # 2048×2048
U, s, Vt = np.linalg.svd(W, full_matrices=False)
rank = 64
W_approx = U[:,:rank] @ np.diag(s[:rank]) @ Vt[:rank,:]
residual = W - W_approx
print(f"Residual variance: {residual.std():.4f} vs original: {W.std():.4f}")
```

---

## Success Criteria

| Format | Must beat | At bpv |
|--------|-----------|--------|
| TQ4_0 | Q4_0 PPL | 4.5 |
| SVD+residual | Q4_0 PPL | 4.5 |
| Importance-aware | Q4_0 PPL | 3.1 |

## The Moonshot

Combine all: SVD skeleton (fp16) + importance-aware TQ4_0 residual + layer deltas.

Expected: **Q4_0 quality at 1.5-2 bpv** — 3x smaller than Q4_0 with same quality.

This is the DNA/brain analogy realized: a compact structural skeleton (SVD) plus sparse high-precision corrections (importance-aware) plus temporal compression (layer deltas).
