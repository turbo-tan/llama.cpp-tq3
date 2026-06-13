# TQ3_4S v2: MMQ Tile Loader Rewrite — PP Nearly Doubled

## Benchmark Results — Qwen3.5-27B, RTX 5060 Ti 16GB

### Prompt Processing (PP512)

| Quant | Size | bpw | PP 512 (tok/s) | TG 128 (tok/s) |
|---|---|---|---|---|
| IQ4_XS | 13.9 GB | 4.25 | 1045 | 24.8 |
| **Q3_K_S** | **11.4 GB** | **3.44** | **826** | **26.1** |
| **TQ3_4S v2** | **12.9 GB** | **4.00** | **713** | **23.2** |
| TQ3_4S v1 | 12.9 GB | 4.00 | 360 | 23.3 |

### Prompt Processing (PP2048)

| Quant | PP 2048 (tok/s) | vs TQ3_4S v1 |
|---|---|---|
| **TQ3_4S v2** | **708** | **+97%** |
| TQ3_4S v1 | 360 | baseline |

### Gemma4 26B-A4B (MoE)

| Quant | PP 2048 (tok/s) | TG 128 (tok/s) | vs v1 |
|---|---|---|---|
| **TQ3_4S v2** | **2144** | 91.5 | **+358% PP** |
| TQ3_4S v1 | 468 | 93.9 | baseline |

### Quality (PPL — lower is better)

| Quant | PPL (27B, c=2048) |
|---|---|
| Q3_K_S | 6.80 |
| **TQ3_4S** | **6.16** |
| IQ4_XS | 6.06 |

TQ3_4S beats Q3_K_S on quality while being 13% larger.
PP speed gap closed from 2.6x to 1.16x.

## What Changed

Two-part rewrite of the MMQ tile loader for TQ3_4S:

**1. 16-thread-per-row structure** — Each thread independently decodes 16 weight
elements (2 subgroups) with zero warp shuffles. Old loader used 8 warps per tile
row with 7 shuffles per element. New loader matches Q3_K_S warp utilization.

**2. Scale baking** — Per-subgroup RMS scales are folded into the int8 tile values
using exact float centroids. This fixes a correctness bug where the old loader
used approximate int8 centroid levels (3-8% error) and only stored 1 of 4
per-subgroup scales.

PPL is bit-identical to the cuBLAS baseline (6.1629 on 27B 10-chunk).
