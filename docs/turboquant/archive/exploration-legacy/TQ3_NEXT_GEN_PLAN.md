# TQ3 Next Generation Plan — 2026-03-31

## Discovery: fp16 Scales Waste 9 Bits Each

Analysis of 100k real TQ3_1S blocks from the 27B model:
- Each fp16 scale has only **6.88 bits of entropy** in 16 bits
- Only 2364 unique values used out of 65536 (3.6%)
- Dynamic range: 3.07 bits
- **8-bit scales match fp16 quality exactly** (verified on 9B + 27B)

## Proven Results (tensor-level RMSE, both witnesses)

### Scale Compression (same indices, different scale precision)
| Precision | bpw | vs TQ3_1S |
|-----------|-----|-----------|
| fp16 | 4.000 | 1.000 |
| **8-bit** | **3.500** | **1.000** |
| 6-bit | 3.375 | 1.002 |
| 4-bit | 3.250 | 1.031 |

### Scale Granularity (u8 scales, same indices)
| Scales | Group | bpw | vs TQ3_1S |
|--------|-------|-----|-----------|
| 2×fp16 (current) | per-16 | 4.000 | 1.000 |
| **4×u8** | **per-8** | **4.000** | **0.955** |
| 8×u8 | per-4 | 5.000 | 0.868 |
| 16×u8 | per-2 | 7.000 | 0.692 |

## Recommended Formats

### TQ3_4S — Best Quality at Same Size
```
[d0:u8][d1:u8][d2:u8][d3:u8][qs:12B] = 16 bytes / 32 = 4.0 bpw
```
- 4 scales for 4 groups of 8 elements
- **4.5% better RMSE than TQ3_1S at same block size**
- Same GGUF block size — minimal code changes
- CUDA kernel: read 4×u8 instead of 2×fp16

### TQ3_2S — Best Size at Same Quality  
```
[d0:u8][d1:u8][qs:12B] = 14 bytes / 32 = 3.5 bpw
```
- Same quality as TQ3_1S (zero loss from 8-bit scales)
- 27B model: 12.9 GB → **11.3 GB** (1.6 GB saved for context)
- Same as TQ3_0 block size but with dual-scale quality

## Target Ladder

| Format | bpw | PPL (est) | Size 27B | Fits 16GB |
|--------|-----|-----------|----------|-----------|
| Q4_K_M | 4.83 | 6.81 | 15.5 GB | barely |
| IQ4_XS | 4.25 | 6.81 | 14.2 GB | tight |
| Q3_K_S | 3.44 | 6.96 | 11.4 GB | ✅ |
| **TQ3_4S** | **4.00** | **~6.9?** | **12.9 GB** | **✅** |
| TQ3_1S | 4.00 | 7.11 | 12.9 GB | ✅ |
| **TQ3_2S** | **3.50** | **~7.1** | **11.3 GB** | **✅** |

## Next Steps

1. **Implement TQ3_4S** — new block struct, quantizer, dequant, CUDA kernel
2. **Run 100-chunk PPL** on 27B to verify the 4.5% RMSE gain translates to PPL
3. If TQ3_4S PPL < 6.96 (Q3_K_S), we beat the modern bar
4. Implement TQ3_2S as the compact variant
5. Publish both to HF + GitHub

## Failed Approaches (for reference)
- TQ3_K_v0 scale-only: 11-13% worse than TQ3_1S
- TQ3_K_v1 shared-shift: 15% worse
- TQ3_1S_K amortized 4-bit scales: 19% worse
- TQ4 symmetric (naive): 25% worse than Q4_K
