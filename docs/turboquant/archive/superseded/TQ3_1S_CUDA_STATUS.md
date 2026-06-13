# TQ3_1S CUDA Kernel Status — 2026-03-30

## Verified Results (27B Qwen3.5, RTX 5060 Ti 16GB)

### Quality: 100 chunks, c=512, wiki.test.raw

| Model | PPL | ± | Size |
|-------|-----|---|------|
| Q4_0 27B | 7.0350 | 0.110 | 14.4 GB |
| TQ3_1S 27B | 7.1078 | 0.111 | 12.9 GB |

Gap: +0.073 PPL (1.0%) — error bars overlap. Parity confirmed.

### Speed

| Metric | TQ3_1S (ngl=99) | Q4_0 (ngl=50) |
|--------|-----------------|----------------|
| PP 512 | 221 tok/s | 74 tok/s |
| TG 10 | 15.3 tok/s | 8.1 tok/s |
| Fits 16GB | ✅ | ❌ |

TQ3_1S is 3x faster PP, 1.9x faster TG because it fits fully on GPU.

### Kernel Status

| Kernel | Status | Notes |
|--------|--------|-------|
| MMVQ `vec_dot_tq3_1s_q8_1` | ✅ Correct | Dual scale (d0/d1) per half-block |
| MMQ `load_tiles_tq3_1s` | ❌ Disabled | NaN after ~4 chunks — q8_0 single-scale tile can't represent dual scales |
| cuBLAS dequant | ✅ Correct | Used for prefill (PP), slower but correct |

### Known Issues

1. **MMQ NaN**: The q8_0 requant format stores one `d` per 32 elements.
   TQ3_1S needs two (d0 for elements 0-15, d1 for 16-31). Lane 0 stores
   d0, so groups 2-3 get wrong scale. Error accumulates → NaN at chunk 5+.

2. **9B TQ3_1S quality**: TQ3_1S at 3.5 bpw is too aggressive for 9B models.
   PPL ~19 vs Q4_0 ~8. Only viable at 27B+ scale.

### Validation Lessons

- 3-chunk PPL tests are insufficient — NaN bugs hide until chunk 5+
- Short chat tests (50 tokens) pass even with broken MMQ
- Minimum validation: 10+ chunk PPL comparison (GPU vs CPU)
