# TurboQuant for MoE: 4.6× KV Cache Compression Meets Hot Expert Dispatch

## The Problem

Running Qwen3.5-122B (256 experts, 48 MoE layers) on a single 16GB GPU requires careful VRAM management. At 32K context, the KV cache alone consumes ~5GB — a third of available memory. Every GB spent on KV cache is a GB that can't hold hot expert weights in VRAM.

## What We're Building

We're combining two complementary techniques to push MoE inference on consumer GPUs:

**1. Hot Expert Residency (our work in tan_llama)**
Pre-load frequently-used expert slices to GPU and skip redundant CPU→GPU copies. Using per-token resident masks, we only transfer cold experts — during decode, if 6/8 active experts are already resident, we copy near-zero bytes per layer.

**2. TurboQuant KV Cache Compression (Google Research, ICLR 2026)**
Compress KV cache from 8 bits to 3.25 bits per value using PolarQuant (random rotation → optimal scalar quantization) + QJL (1-bit residual correction). The key insight: random orthogonal rotation makes any distribution near-Gaussian, enabling simple uniform quantization with provably optimal distortion.

Building on TheTom's turboquant_plus implementation (which proved this works on Apple Silicon with llama.cpp), we're porting the CUDA backend for NVIDIA GPUs and integrating it with our MoE expert dispatch system.

## Why This Combination Matters

For a 122B MoE model at 32K context on 16GB VRAM:

| Component | Before | After |
|-----------|--------|-------|
| KV Cache | 5.0 GB (q8_0) | **1.1 GB** (turbo3) |
| Hot Experts | 2.1 GB | 2.1 GB |
| Freed VRAM | — | **+3.9 GB** |

That 3.9 GB can hold ~2× more hot experts in VRAM, further reducing CPU→GPU transfer and pushing decode throughput toward 30 tok/s on a single RTX 5060 Ti.

## The Math Behind TurboQuant

PolarQuant converts Cartesian coordinates to polar form — replacing per-block normalization constants (the hidden overhead in traditional quantization) with a single radius + angles. QJL then uses a 1-bit Johnson-Lindenstrauss projection on the residual to eliminate quantization bias.

The result: 3.25 bits/value with cosine similarity 0.95 to the original — and the rotation makes the kurtosis drop from 900 to 2.9 (essentially perfect Gaussian), which is why simple uniform quantization works so well after rotation.

## Status

- Hot expert dispatch: built, tested, awaiting GPU benchmark
- TurboQuant CUDA port: in progress, targeting t_llama_cpp (our llama.cpp fork)
- Target: Qwen3.5-122B-A10B at 30 tok/s decode on RTX 5060 Ti 16GB

References:
- TurboQuant paper: https://arxiv.org/abs/2504.19874
- PolarQuant: https://arxiv.org/abs/2502.02617
- turboquant_plus: https://github.com/TheTom/turboquant_plus
- Google Research blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

#LLM #MoE #Quantization #CUDA #LocalAI #TurboQuant
