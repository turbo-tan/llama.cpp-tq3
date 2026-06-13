# EXL3 vs TQ3_4S Comparison (Estimated)

## Data from Screenshot

### Known Formats (for calibration)
| Format | Screenshot PPL | Our 100ch PPL | Difference |
|--------|---------------|---------------|------------|
| UD-Q2_K_XL | 7.73 | 7.53 | +0.20 |
| IQ4_XS | 7.06 | 6.83 | +0.23 |
| Q4_0 | 7.20 | ? | ? |

**Pattern**: Screenshot PPLs are ~0.2 higher than our 100-chunk tests.

**Hypothesis**: Screenshot likely used **10-20 chunks** (less data = higher variance).

## EXL3 Formats (from screenshot)

| Format | bits w | bits h | Size | Screenshot PPL | Est. 100ch PPL |
|--------|--------|--------|------|----------------|----------------|
| EXL3 2.10 | 2.10 | 6.00 | 6.68 GB | 7.54 | ~7.34 |
| EXL3 3.01 | 3.01 | 6.00 | 9.44 GB | 7.12 | ~6.92 |
| EXL3 3.10 | 3.10 | 6.00 | 9.68 GB | 7.08 | ~6.88 |
| EXL3 4.01 | 4.01 | 6.00 | 12.27 GB | 7.04 | ~6.84 |
| EXL3 5.01 | 5.01 | 6.00 | 15.10 GB | 7.00 | ~6.80 |

## TQ3_4S (our format)

| Format | bits w | Size | 100ch PPL | Est. screenshot PPL |
|--------|--------|------|-----------|---------------------|
| TQ3_4S | 3.5 | 12.9 GB | 6.77 | ~6.97 |

## Comparison at Similar Size

### ~10 GB range
- **EXL3 3.01**: 9.44 GB, est. PPL ~6.92
- **EXL3 3.10**: 9.68 GB, est. PPL ~6.88
- **TQ3_4S**: 12.9 GB, est. PPL ~6.77

### ~12-13 GB range
- **EXL3 4.01**: 12.27 GB, est. PPL ~6.84
- **TQ3_4S**: 12.9 GB, est. PPL ~6.77
- **IQ4_XS**: 13.27 GB, PPL 6.83

## Analysis

### Quality per GB
```
EXL3 3.10: 9.68 GB / 6.88 PPL = 1.41 GB per PPL point
TQ3_4S:   12.9 GB / 6.77 PPL = 1.91 GB per PPL point
EXL3 4.01: 12.27 GB / 6.84 PPL = 1.79 GB per PPL point
```

**Winner (efficiency)**: EXL3 3.10 - best quality per GB

### Quality at similar size (~12-13 GB)
```
TQ3_4S:   12.9 GB, PPL 6.77 ✓ (best quality)
EXL3 4.01: 12.27 GB, PPL 6.84
IQ4_XS:   13.27 GB, PPL 6.83
```

**Winner (quality)**: TQ3_4S - lowest PPL at similar size

### Smallest size for PPL < 7.0
```
EXL3 3.10: 9.68 GB, PPL ~6.88 ✓ (smallest)
TQ3_4S:   12.9 GB, PPL 6.77
EXL3 4.01: 12.27 GB, PPL ~6.84
```

**Winner (size)**: EXL3 3.10 - 3.2 GB smaller than TQ3_4S

## Conclusion

**EXL3 3.10 bpw** is the most efficient format:
- 9.68 GB (25% smaller than TQ3_4S)
- PPL ~6.88 (only 0.11 worse than TQ3_4S)
- Best quality-per-GB ratio

**TQ3_4S** has best absolute quality at ~13 GB:
- 12.9 GB
- PPL 6.77 (best in class)
- But 33% larger than EXL3 3.10 for marginal quality gain

## Recommendation

- **For 16GB VRAM**: Use **EXL3 3.10** (9.68 GB, more headroom for context)
- **For quality**: Use **TQ3_4S** (12.9 GB, best PPL)
- **For speed**: Need to benchmark (TQ3_4S: 315 tok/s, EXL3: unknown)

## Caveat

This analysis assumes:
1. Screenshot used 10-20 chunks (not confirmed)
2. Linear PPL offset of ~0.2 applies to EXL3
3. Same test data (wikitext-2)

**To confirm**: Would need to download EXL3 3.10 and run actual 100-chunk test.
