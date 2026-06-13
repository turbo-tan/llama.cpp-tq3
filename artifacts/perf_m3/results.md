# M3 MMVQ Fusion Gate Extension - Results

**Date:** 2026-06-11
**Branch:** `perf/m3-mmvq-fusion-ncols`
**Commit:** `f6c70e6e2`
**Model:** Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf

---

## Build

**Status: PASS**

All three targets built successfully:
- `llama-bench`
- `llama-server`
- `llama-batched-bench`

---

## Step 0: Edit Verification

Both edits confirmed correct:

- `ggml/src/ggml-cuda/mmvq.cu:867` - `if constexpr (c_ncols_dst <= 4)` (was `== 1`)
- `ggml/src/ggml-cuda/ggml-cuda.cu:2627` - `dst->ne[1] > 4` (was `!= 1`)

---

## Bench A: TG at Various Batch Sizes

Note: `llama-bench -t 0` caused a CPU assert (`cplan->n_threads > 0`); benches were run without `-t 0`.

| Batch | TG t/s |
|-------|--------|
| B=1   | 33.45 +/- 0.08 |
| B=2   | 33.38 +/- 0.36 |
| B=3   | 33.29 +/- 0.10 |
| B=4   | 33.12 +/- 0.08 |

**Note:** As expected, llama-bench TG keeps `dst->ne[1] == 1` internally regardless of `-b N`, so the M3 fusion path was NOT exercised by Bench A. Numbers are baseline decode speed for this model.

---

## Bench B: MTP Server E2E Throughput

**Status: CRASH**

Server crashed during MTP warmup decode with:

```
/home/awee/code/worktrees/tan_llama-main-ref/ggml/src/ggml-cuda/mmvq.cu:1278:
GGML_ASSERT(ids || dst->ne[1] == 1) failed
```

**Root cause:** The M3 edit relaxed the fusion dispatch gate (`dst->ne[1] > 4`) and the template instantiation guard (`c_ncols_dst <= 4`), but did NOT relax the assertion inside `ggml_cuda_mul_mat_vec_q` at mmvq.cu:1278:

```cpp
if (fusion) {
    GGML_ASSERT( !ids || dst->ne[2] == 1);  // MoE path: ne[2] must be 1
    GGML_ASSERT(  ids || dst->ne[1] == 1);  // non-MoE path: ne[1] must be 1  <-- fires
```

When the MTP verify pass calls `ggml_cuda_mul_mat_vec_q` with `ids == nullptr` and `dst->ne[1] == 2` (or 3/4), the assert fires because the original code only instantiated fused templates for ne[1]==1 in the non-MoE case. The gate change is incomplete: the inner assert guards the launch function body and was not updated to match.

**MTP e2e t/s: N/A (crash before measurement)**
**Baseline reference: ~45.8 t/s**

---

## Verdict

**BUG / NEGATIVE**

The edit is logically incomplete. The fusion gate was opened for ne[1] in 2..4 but the kernel launch function asserts ne[1]==1 for the non-MoE path (`ids == nullptr`). The server crashes on first MTP verify decode.

To fix this (NOT done here per SOP - no code fixes in bench runs):
- The assertion at mmvq.cu:1278 needs to be relaxed to allow `dst->ne[1] <= 4` when fusion is active for non-MoE paths, OR the fusion setup code needs to only pass `fusion` args when ids is non-null (MoE) for ne[1] > 1.

The change does NOT meet the >5% WIN criterion at B=2 or B=3 because Bench B crashed.

---

## Production Server

Restored successfully to port 8085 after benching.
`curl http://localhost:8085/health` returned `{"status":"ok"}`

Secondary process (35B model, port 18123, PID 1189945) was also killed for GPU clearance; it was not systemd-managed and its command line was captured:
```
/home/awee/code/llm-launch/release/bin/llama-server -m /home/awee/models/turboquant/tq3_4s/unsloth_35b_mtp/Qwen3.6-35B-A3B-MTP-TQ3_4S.gguf -a Qwen3.6-35B-A3B-MTP-TQ3_4S --host 0.0.0.0 --port 18123 -ngl 99 -c 200000 -ctk q4_0 -ctv tq3_0 -b 64 -ub 64 -fa on --threads 8 --threads-batch 8 --metrics --chat-template-file /home/awee/code/llm-launch/templates/qwen3_generic_no_empty_think.jinja --reasoning off --reasoning-format none --reasoning-budget 0 -np 1 --no-warmup --no-cache-prompt --cache-ram 0
```
