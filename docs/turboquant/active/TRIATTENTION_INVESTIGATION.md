# TriAttention Investigation

**Paper:** TriAttention: Efficient Long Reasoning with Trigonometric KV Compression  
**arXiv:** https://arxiv.org/abs/2604.04921  
**Branch:** `experiment/triattention-kv-compression` (private repo)  
**Date:** 2026-04-11

## What It Does

TriAttention is a KV cache compression method that **evicts unimportant K/V tokens** during
generation, keeping only the top-scoring ones. Unlike prior methods (SnapKV, StreamingLLM)
that use post-RoPE attention scores to estimate importance, TriAttention uses **pre-RoPE Q/K
centers** — which are stable across positions — to predict which keys will be attended to.

Key results on AIME25 (Qwen3-8B, 32K generation):
- **10.7× KV memory reduction** while matching full attention accuracy
- **2.5× throughput** at equivalent accuracy
- Prior methods achieve ~half accuracy at same efficiency

## Core Insight

**Q/K concentration:** Pre-RoPE Q and K vectors cluster tightly around a fixed non-zero center
across all positions and contexts. This is stable because pre-RoPE vectors aren't affected by
positional rotation.

**Trigonometric series:** When Q/K are concentrated, the attention logit between query at
position i and key at position j reduces to a function of only the *distance* (i-j):

```
attn(i,j) ≈ f(i-j) = Σ_k A_k * cos(k * θ * (i-j) + φ_k)
```

This means: **the Q center determines which distances get high attention**, independent of
content. Keys at "preferred distances" from the query will always score high.

## How It Scores Keys

1. **Distance score:** Use the Q center + trigonometric series to score each key by its
   position relative to the current query. Keys at preferred distances get high scores.
2. **Norm score:** For heads where Q is less concentrated, use Q/K norms as a fallback.
3. **Adaptive weighting:** Balance distance vs norm score using Mean Resultant Length (MRL).
4. **Pruning:** Keep only top-K scored keys per head. Evict the rest permanently.

## Relevance to TQ3

### Why This Matters

Our TQ3_0 KV cache **quantizes all tokens** to 1.75 bpw. TriAttention suggests a
complementary approach: **evict unimportant tokens entirely**. Combined:

- TQ3_0 quantization: reduces memory per token (1.75 bpw vs 16 bpw)
- TriAttention eviction: reduces number of tokens kept

Together: potentially 10-20× KV memory reduction while maintaining quality.

### Implementation Complexity

TriAttention requires:
1. Pre-RoPE Q/K centers per head (calibration, done once per model)
2. Scoring keys at each generation step using the trigonometric series
3. Sparse KV cache — only top-K tokens per head retained

The sparse KV cache is the hard part — llama.cpp's KV cache is dense. Needs:
- New KV cache layout supporting per-head token eviction
- Modified flash attention for sparse K/V
- Calibration step to compute Q/K centers

### Simpler First Step

The trigonometric series insight suggests a simpler optimization: **bias attention toward
preferred distances** by adding a learned distance penalty per head. This is ALiBi but with
head-specific, learned distance preferences instead of a fixed linear penalty. Could be
implemented without changing the KV cache layout.

## Comparison

| Aspect | TQ3_0 KV | TriAttention |
|--------|----------|--------------|
| Approach | Quantize all tokens | Evict unimportant tokens |
| Memory reduction | ~9× (16→1.75 bpw) | ~10.7× |
| Quality impact | +1.3% PPL | Matches full attention |
| Implementation | Done ✅ | Complex (sparse KV) |
| Combinable | Yes | Yes |

## Next Steps

1. Read the code: https://github.com/WeianMao/triattention
2. Check if Q/K concentration holds for TQ3_4S weight-quantized models
   (WHT-rotated weights may affect pre-RoPE Q/K statistics)
3. Prototype the scoring function — compute Q/K centers from existing models
4. Assess llama.cpp integration complexity

## Open Questions

- Does Q/K concentration hold after TQ3_4S weight quantization?
- Can TriAttention eviction stack with TQ3_0 quantization of kept tokens?
- Is the 10.7× on top of full-precision KV, or does it stack with quantization?
