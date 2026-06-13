# Quick Comparison: 1 Chunk @ c=2048

## Results (wikitext-2-raw, c=2048, 1 chunk)

| Format | Size | PPL (1ch) | PPL (100ch) | Std Error |
|--------|------|-----------|-------------|-----------|
| UD-Q2_K_XL | 9.76 GB | ? | 7.53 | ±0.062 |
| TQ3_4S | 12.9 GB | 6.82 | 6.77 | ±0.054 |
| IQ4_XS | 13.27 GB | ? | 6.83 | ±0.055 |

## Observations

**TQ3_4S 1-chunk**: PPL 6.82 ±0.57
- Very close to 100-chunk result (6.77)
- High std error due to single chunk
- Shows format is stable/consistent

## For EXL3 Comparison

If we download EXL3 3.10 bpw (9.68 GB), we can:
1. Run 1-chunk test (quick, ~30 seconds)
2. Compare directly with TQ3_4S 1-chunk result

**Expected EXL3 3.10 bpw @ 1 chunk**: PPL ~6.9-7.1 (estimate)

This gives us a quick quality comparison without waiting for 100 chunks.

## Command for EXL3 Test

```bash
# If EXL3 downloaded and exllamav2 installed
python test_perplexity.py \
    -m /path/to/Qwen3.5-27B-EXL3-3.10bpw \
    -d wikitext \
    -c 2048 \
    --chunks 1
```

Or if converted to GGUF:
```bash
./llama-perplexity \
    -m /path/to/Qwen3.5-27B-EXL3-3.10.gguf \
    -ngl 99 -fa 1 -c 2048 \
    -f wikitext-2-raw/wiki.test.raw \
    --chunks 1
```

## Conclusion

**1-chunk test is sufficient** for quick quality comparison:
- Takes ~30 seconds vs 10 minutes for 100 chunks
- PPL difference from 100-chunk is minimal (6.82 vs 6.77 = 0.05)
- Good enough to determine if EXL3 is competitive with TQ3_4S
