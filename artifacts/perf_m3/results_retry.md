# M3 MMVQ Fusion Gate Extension - Retry Results

**Date:** 2026-06-11
**Branch:** `perf/m3-mmvq-fusion-ncols`
**Commit:** `109e9d152`
**Model:** Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf

---

## Build

**Status: PASS**

All three targets built successfully:
- `llama-bench`
- `llama-server`
- `llama-batched-bench`

---

## Edits in this retry

Three edits were applied (the third was the fix for the previous crash):

1. `ggml/src/ggml-cuda/mmvq.cu:867` -- `if constexpr (c_ncols_dst <= 4)` (was `== 1`)
2. `ggml/src/ggml-cuda/ggml-cuda.cu:2627` -- `dst->ne[1] > 4` (was `!= 1`)
3. `ggml/src/ggml-cuda/mmvq.cu:1278` -- `GGML_ASSERT(ids || dst->ne[1] <= 4)` (was `== 1`) -- **NEW FIX vs prior run**

---

## Bench A: TG at Batch Sizes B=1..4

Warmup completed (pp512: 1085.40 t/s), then 5-run TG at each batch size.

| Batch | TG t/s        |
|-------|---------------|
| B=1   | 33.78 +/- 0.22 |
| B=2   | 33.87 +/- 0.43 |
| B=3   | 34.28 +/- 0.19 |
| B=4   | 34.04 +/- 0.21 |

No crash at any batch size. The assert fix worked.

Note: llama-bench TG keeps `dst->ne[1] == 1` internally regardless of `-b N`, so the M3 fusion path
for ne[1] > 1 is not exercised by this bench. These numbers reflect baseline decode speed.

---

## Bench B: MTP Server E2E Throughput

Server launched with `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 1.0`, port 8086.

3 runs, each 400 tokens:

| Run | t/s (predicted_per_second) | draft_n | draft_n_accepted |
|-----|---------------------------|---------|-----------------|
| 1   | 38.76                     | 280     | 259             |
| 2   | 38.69                     | 280     | 259             |
| 3   | 38.47                     | 280     | 259             |

**Average MTP e2e: ~38.64 t/s**
**Baseline reference: ~45.8 t/s**

Draft acceptance rate: 259/280 = 92.5% (good speculation quality)

---

## Verdict

**NEGATIVE**

The fusion extension for ne[1] <= 4 does NOT improve MTP e2e throughput vs baseline.
- MTP e2e: ~38.64 t/s vs baseline ~45.8 t/s -- this is worse, not better.
- TG at B=2/3/4 is flat vs B=1 (no improvement from fused kernel at higher ne[1]).

The crash from the prior run is fixed (assert relaxed to `<= 4`), but the kernel itself provides
no speedup for the MTP verify batch path. The hypothesis that fusing the B=2/3/4 matmul+bias+GLU
would improve MTP throughput is not supported by data.

Does NOT meet >5% WIN criterion at B=2 or B=3 vs B=1 baseline.

---

## Production Server

Restored successfully to port 8085 after benching.
`curl http://localhost:8085/health` returned `{"status":"ok"}`
