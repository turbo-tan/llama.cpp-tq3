# TQ3_4S Speed Optimization — Expert Handover

## Status: PIVOT TO SPEED

Quality ceiling reached. TQ3_4S PPL 7.05 is our best — can't beat Q3_K_S (6.96)
due to WHT's inability to use importance weighting (fundamental limit).

**New goal: match or beat Q3_K_S speed at 12.9 GB.**

## Current Speed

| | Q3_K_S | TQ3_4S | Gap |
|---|---|---|---|
| PP 512 (tok/s) | 689 | 327 | 2.1x slower |
| TG 10 (tok/s) | 20.7 | 14.6 | 1.4x slower |
| Size | 11.4 GB | 12.9 GB | 13% larger |

## nsys Profile (27B, pp64 + tg5)

```
73.6%  dequantize_block_tq3_4s<__half>  — memory-bandwidth bound
 7.7%  cutlass h16816 gemm              — tensor core, fast
```

**The dequant kernel is 73.6% of total GPU time.**
Switched from fp32 to fp16 dequant (+49% PP). MMQ exists but is slower (105 tok/s)
because WHT inverse in tile loader is too expensive.

## Two Tasks

### Task 1: Custom fused dequant+GEMM kernel (PP target: 600+ tok/s)

Current path: dequant 12.9 GB → write 12.9 GB fp16 → cuBLAS reads 12.9 GB fp16.
Total bandwidth: 25.8 GB. Q3_K_S MMQ: 11.4 GB (no intermediate). Ratio = 2.3x ≈ gap.

The existing MMQ (load_tiles_tq3_4s) converts to q8_0 in shared memory but the
WHT inverse + centroid lookup + int8 requantization per tile load is slower than
just dequanting to fp16 and letting tensor cores do the work.

Need a NEW fused kernel that:
1. Loads TQ3_4S blocks into registers (not shared memory)
2. Decodes E3M5 + unpacks 3-bit + WHT inverse via warp shuffles
3. Feeds fp16 directly to tensor core MMA (no intermediate buffer)
4. Eliminates the 12.9 GB fp16 write entirely

This is a custom tensor core kernel, not standard MMQ.

### Task 2: Optimize TG decode speed (TG target: 18+ tok/s)

Current MMVQ vec_dot_tq3_4s_q8_1 works but has ~27% compute overhead.
TG is bandwidth-bound: 12.9 GB / 20.7 tok/s × (20.7/14.6) ≈ 18.3 GB effective.
Theoretical: 12.9 GB at Q3_K_S efficiency = 18.3 tok/s.

Optimizations for vec_dot_tq3_4s_q8_1 in vecdotq.cuh:
- __ldg() for read-only qs[] access
- Preload E3M5 scales into registers
- Shared memory centroid table (8 floats = 32 bytes)
- Fuse centroid lookup with WHT butterfly where possible

## Block Structure (TQ3_4S)

```c
typedef struct {
    uint8_t d[4];    // 4 × E3M5 scales for groups of 8 elements
    uint8_t qs[12];  // 32 × 3-bit packed indices
} block_tq3_4s;      // 16 bytes = 4.0 bpw
```

E3M5 decode: `scale = ldexpf(1.0f + (byte & 31) / 32.0f, (byte >> 5) - 9)`

Each group of 8 elements: 3 bytes packed as:
```
qs[0] = idx[0] | (idx[1] << 3) | (idx[2] << 6)
qs[1] = (idx[2] >> 2) | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7)
qs[2] = (idx[5] >> 1) | (idx[6] << 2) | (idx[7] << 5)
```

Dequant: `val = TQ3_0_CENTROIDS[idx] * scale`, then inverse WHT butterfly.

## Files to Modify

See `docs/turboquant/procedures/NEW_TYPE_CHECKLIST.md` for the full list.
For speed work, the key files are:
- `ggml/src/ggml-cuda/mmq.cuh` — load_tiles kernel
- `ggml/src/ggml-cuda/mmq.cu` — dispatch + type traits
- `ggml/src/ggml-cuda/vecdotq.cuh` — vec_dot optimization
- `ggml/src/ggml-cuda/mmvq.cu` — MMVQ dispatch
- `ggml/src/ggml-cuda/convert.cu` — dequant kernel (current bottleneck)

## Branch

Working branch: `/home/awee/code/llama.cpp` branch `feature/tq3_1k-explore`
