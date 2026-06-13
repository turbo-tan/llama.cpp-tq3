# TQ3_4S Decode Bottleneck Analysis

**Date:** 2026-04-13
**Branch:** `experiment/tq3-decode-next`

## Profile Summary (27B, TG 64 tokens)

| Kernel | TQ3_4S (ms) | Q3_K_S (ms) | Ratio |
|---|---|---|---|
| MMVQ (need_check=true) | 50.6 | 47.8 | 1.06x |
| MMVQ (need_check=false) | 20.2 | 14.0 | 1.44x |
| **Total MMVQ** | **70.8** | **61.8** | **1.15x** |
| Total decode | ~87 ms | ~82 ms | 1.06x |

TQ3_4S decode: 24.2 tok/s vs Q3_K_S: 26.3 tok/s (8% gap)

## Why TQ3_4S MMVQ Is Slower

The TQ3_4S vec_dot per subgroup (8 elements):
1. Load 3 bytes of packed 3-bit indices
2. Unpack 8 indices via bit shifts (7 operations)
3. Look up 8 int8 levels from a static table
4. Pack into 2 int32 for dp4a
5. 2x dp4a calls
6. Scale: sumi * d_weight * (2.1519/127) * d_activation

Q3_K_S vec_dot per 32 elements:
1. Load 12 bytes of packed 3-bit indices + 2 bytes scales
2. Bit-shift decode (simpler packing)
3. 8x dp4a calls
4. Scale: sumi * d * ds

The TQ3_4S overhead: per-subgroup E3M5 scale decode (`ldexpf`) + the 3-bit
unpack is slightly more expensive than Q3_K_S's simpler packing.

## dmax Analysis

Only 0.8% of TQ3_4S subgroups have zero scale — `dmax` skip would save <1%.
Not worth implementing.

## Potential Improvements

1. **Fuse subgroup scale decode**: precompute all 4 scales once per block
   instead of per-subgroup. Currently `tq3_4s_ratio4s()` is called per subgroup.

2. **Use the MMQ tile loader approach for decode**: the PP path uses exact float
   centroids with scale baking. Could adapt for single-token decode.

3. **Reduce register pressure**: the 8 int8 level lookups use 8 registers.
   Could pack the lookup table into 2 int32 constants.

4. **VDR=16**: process 4 subgroups per thread instead of 2. Needs testing —
   may increase register pressure.

5. **Warp-cooperative decode**: multiple threads decode one block cooperatively,
   similar to the MMQ tile loader. Would reduce per-thread work but add shuffle overhead.

## Conclusion

The 8% decode gap vs Q3_K_S is inherent to the TQ3_4S format:
- 4 independent E3M5 scales per block (vs 1 scale in Q3_K_S)
- Non-uniform centroid levels (vs uniform in Q3_K_S)
- Per-subgroup scale decode overhead

A drastic improvement would require changing the decode algorithm, not just
tuning launch parameters. The most promising direction is fusing the scale
decode and using a precomputed LUT for the 3-bit → int8 mapping.
