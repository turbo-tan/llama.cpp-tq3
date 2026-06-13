# TQ3_0 — Outside The Box: Beating Q4_0

Date: 2026-03-28

## Status

Moonshot backlog only.

Do not use this as the active implementation plan.

Current canonical plan:

- `FIX_PLAN_2026-03-28.md`

Current active line:

- `TQ3_1S` quality work
- `TQ3_0` adaptive-KV work

## Status (2026-03-28)

Expert confirmed: valid as a moonshot backlog, not the current execution track.

**Ranking:**
- Idea 2 (codebook matmul): most interesting for TQ3_0, but new architecture — speculative
- Idea 3 (sparse-V): valid for long-context TG, not current PP bottleneck
- Idea 1 (bit-serial TC): high-risk, packing/accumulation overhead can erase paper win

**Current execution track:** Q4_1_TQ / new contract (see FIX_PLAN_2026-03-28.md).

## Three Ideas That Could Beat Native Q4_0

### Idea 1: Bit-Serial Tensor Cores (TC-FPx approach)

**Source:** FP6-LLM (USENIX ATC 2024) — achieved 1.69-2.65x faster than FP16.

**Concept:** Decompose 3-bit weights into 2-bit + 1-bit components. Run two tensor core
operations and combine. Tensor cores run at full speed, but on 22% less data than q4_0.

```
3-bit weight = (bit2 << 1) | bit1_0
= 2-bit component × 2 + 1-bit component

GEMM(W_3bit, X) = 2 × GEMM(W_2bit, X) + GEMM(W_1bit, X)
```

**Why it beats q4_0:**
- Same tensor core throughput as q4_0
- 22% less weight data to read (3.5 vs 4.5 bpv)
- Net: 1.28x faster at memory bandwidth limit

**Implementation:** Pack 3-bit weights into two separate bit-plane matrices. Run two
cublasGemmEx calls. Combine results. No custom kernel needed for the GEMM itself.

---

### Idea 2: Codebook MatMul (KLLM approach) — The Real Moonshot

**Source:** KLLM (2025) — avoids most dequantization and full-precision computation.

**Concept:** TQ3_0 has only 8 possible centroid values. For each activation block,
precompute all 8 possible products once, then the matmul is just lookups + additions.

```
For activation block x_rot[32]:
  precompute: partial[i] = centroid[i] × dot(x_rot, e_j)  for i=0..7, j=0..31
  
For each weight row:
  result = sum(partial[idx[j]] for j in 0..31)
         = 32 table lookups + 32 additions
         = 0 multiplies in the hot path
```

**Why it beats q4_0:**
- q4_0: 32 multiplies + 32 additions per weight row
- TQ3_0 codebook: 0 multiplies + 32 additions (after precompute)
- Precompute cost: 8 × 32 = 256 multiplies, amortized over 2048 weight rows
- Amortization: 256 / 2048 = 0.125 multiplies per row (vs 32 for q4_0)
- **Net: ~256x fewer multiplies in the hot path**

**The math:**
```
Y[row] = sum_j(centroid[idx[j]] × x_rot[j]) × rms[row]
       = rms[row] × sum_j(partial[idx[j]])
```

Where `partial[i] = centroid[i] × sum_j(x_rot[j] × delta(idx[j], i))` — precomputed once.

**Implementation sketch:**
```cuda
// Phase 1: precompute 8 partial sums per activation block (once per block)
__shared__ float partial[8];  // 8 centroid × activation dot products
for (int i = 0; i < 8; i++)
    partial[i] = centroid[i] * dot(x_rot_block, mask[i]);

// Phase 2: accumulate for each weight row (hot path — no multiplies)
float acc = 0;
for (int j = 0; j < 32; j++)
    acc += partial[idx[j]];  // table lookup + addition only
result = acc * rms;
```

**Expected speedup:** 3-5x over current TQ3_0, potentially 2x over q4_0.

---

### Idea 3: Sparse Attention + V Compression (TheTom's sparse-V)

**Concept:** At long context, most attention weights are near zero. Skip V dequant
for positions where attention weight < threshold.

```
for each query position:
    attn_weights = softmax(Q × K^T)
    # Only dequant V where attn_weight > threshold
    output = sum(attn_weight[i] × dequant(V[i]) for i where attn_weight[i] > 1e-6)
```

**Why it beats q4_0 at long context:**
- q4_0 V: dequant all N positions every token
- TQ3_0 V + sparse: dequant only ~5-10% of positions (attention is sparse)
- Net: 10-20x less V dequant work at 32K+ context

**Combined with TQ3_0 KV:** Already 4.6x less data to read. With sparse-V: another 10x
reduction in V dequant. Total: **46x less V work than q4_0 at long context**.

---

## The Winning Combination

For maximum speed at long context:

1. **Codebook MatMul** for weight inference (Idea 2) — beats q4_0 on PP
2. **TQ3_0 KV cache** — 4.6x less KV data
3. **Sparse-V dequant** (Idea 3) — 10x less V work at long context

This combination could achieve:
- PP: 2-3x faster than q4_0 (codebook matmul + less weight data)
- TG at 32K context: 5-10x faster than q4_0 (sparse V + compressed KV)

## Implementation Priority

1. **Codebook MatMul prototype** — test on isolated matmul benchmark first
2. **Sparse-V** — simple threshold gate, low risk
3. **Bit-serial tensor cores** — more complex, but proven by FP6-LLM

## References

- [FP6-LLM / TC-FPx](https://arxiv.org/abs/2401.14112) — bit-serial tensor cores, 1.69-2.65x vs FP16
- [KLLM](https://arxiv.org/abs/2507.23035) — index-based matmul, avoids dequantization
- [TheTom sparse-V](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md)
