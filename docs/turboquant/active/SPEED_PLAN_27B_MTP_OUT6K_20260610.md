# Speed Plan: Qwen3.6-27B-MTP TQ3_4S outQ6K (PP + TG)

Date: 2026-06-10
Status: ACTIVE. Workflow rules: [docs/steering/perf-campaign-sop.md](../../steering/perf-campaign-sop.md)

## Campaign Ledger (always current - what we tried, where, verdict)

| Phase | Branch | Hypothesis | Result | Verdict |
|---|---|---|---|---|
| 0 | `perf/p0-baseline-3090` | Numbers stale (new GPU + rebase) | New build TG 33.7 vs old 8.6 (3.9x); PP 1103 ~ Q3KM parity; PPL witness 5.9720 | WIN (no code; winner validated) |
| A1 | `perf/a1-mtp-sweep` | Deeper MTP drafts win (acceptance 92 pct) | n-max 2 optimal: 45.8 t/s; depth collapse 92/80/71/52; p-min no-op | NEGATIVE for depth; config axis exhausted |
| A2 | `perf/a2-mmvq-verify-ncols` | TQ3 verify-batch kernels undertuned | Q3KM control shows identical scaling; nwarps variant within noise | NEGATIVE; falloff platform-generic |
| A3 | (not opened) | ncols=1 retune for SM86 | Killed by A2 control logic (exact B1 parity) | DEAD |
| A4 | (external) | Fewer bytes per token | Quality traded on UNMEASURED estimates; ceiling was <=5 pct | CLOSED; caused quality-first reset |
| L1 | `perf/l1-mtp-loop-overhead` | MTP loop has trimmable overhead | Ledger: verify 39-42ms, drafts 5.8, server 1.8, residual 8-12; hook/ckpt/cuda-graph theories falsified; p-min dead code found | PARTIAL WIN (instrumentation kept; trims pending) |
| M0 | `perf/m0-pass-trace` | 52 pct MBU is runtime fat, not physics | GPU 93 pct busy (no bubbles); matvecs at ~58 pct BW across ~416 small kernels; B=3 LOSES fusion + the tuned ncols1 kernel (dispatch gates fusion to ncols_dst==1) | WIN (two kernel targets found) |
| M1 | `perf/m1-pipelined-draft` | Draft latency hideable behind verify | pending | OPEN |
| M2 | `perf/m2-tree-verify` | Verify slots nearly free; widen pos-2 | pending | OPEN |
| M3 | `perf/m3-mmvq-fusion-ncols` | Extend MMVQ fusion + TQ3 special kernel to ncols 2-4; verify 37.8 -> ~33ms | Assert relaxed (109e9d152); no crash. Bench A: 33.78/33.87/34.28/34.04 t/s B=1..4 (flat; llama-bench ne[1] stays 1, new path not exercised). MTP e2e: 38.64 t/s avg vs baseline 45.8 t/s -- worse, not better. Draft accept 92.5 pct (259/280). | NEGATIVE (fusion extension at ne[1]<=4 provides no speedup; MTP e2e regresses) |

Do-not-repeat (carried from proven docs + this campaign): fused rotate+quantize
MMVQ (-3 pct), rotation caching, tile-loader micro-opts, MMQ without scale
baking (PPL 132+), QK 32->128 redefinition, deep sequential drafts (n-max > 2),
p-min tuning on draft-mtp (dead code), nwarps for ncols 2-4 (noise), quality
decisions from estimated KLD (forbidden).

## REVISION 2026-06-11 (2): Breaking the verify-pass box

The L1 ledger called 85 pct of the cycle "irreducible". Wrong framing - user
called it out. Three boxes to break, all lossless, all zero extra memory:

### Box 1 / M0: the MBU frontier

A single B=1 pass reads 13.38 GiB; at ~850 GB/s effective that is a ~16ms
floor. We measure 27.9ms = 52 pct MBU. Q3_K_M sits at the same plateau, which
proves the fat is in the RUNTIME (attention/non-matmul ops, per-matmul
rotate->quantize->mmvq chain, scheduling bubbles), not the format. Plan:
nsys-trace ONE decode pass and ONE B=3 verify pass, account every ms,
kernel by kernel. Every recovered MBU point speeds up every path including
non-spec decode. Phase M0, branch `perf/m0-pass-trace`, metric: a table that
sums to 27.9ms.

### Box 2 / M2: verify slots are nearly free

B=1 -> 27.9ms, B=3 -> 37.8ms: each extra verify token costs ~5ms because the
weights are read once per pass regardless. The A1 acceptance collapse punished
sequential DEPTH; a tree WIDENS instead: draft 2 candidates for position 2
from the same hidden state (one extra cheap MTP pass), verify all branches in
a batch-5 pass (~+10ms). Either candidate matching counts, so position-2
effective acceptance rises from 80 pct toward 95+. Greedy verification keeps
output bit-identical. Phase M2, branch `perf/m2-tree-verify`.

### Box 3 / M1: the cycle does not have to be sequential

At 92 pct acceptance the verify outcome is predictable. While the GPU runs
cycle N's verify, optimistically draft cycle N+1 on a side CUDA stream
assuming full acceptance (the MTP layer reads ~0.6 GiB - it interleaves into
the verify tail for almost nothing). 92 pct of the time drafts are ready when
verify lands; 8 pct of the time discarded work we would have lost anyway.
Deletes the 5.8ms draft segment and the sync gaps. Phase M1, branch
`perf/m1-pipelined-draft`.

Composite arithmetic: tree verify B~5 (~45ms) carrying ~3.4 tokens with
drafting hidden = ~75 t/s, before any M0 MBU gains. Current: 46-49 t/s.

Order: M0 (one GPU session, informs everything) -> M1 -> M2 -> M0-derived
kernel fixes. Ops track in parallel: PR #27 remaining CI failures
(build-cmake-pkg, python type-check x2, cpu-x64-high-perf, windows static),
then L0 production binary swap.

## REVISION 2026-06-11: Quality-First Reset

### What went wrong

Phase A4 (bytes-per-token audit) was executed as quality-for-bytes trading:
output-tensor downgrades and an MTP-head downgrade were created, then judged and
deleted based on ESTIMATED KLD numbers that were never measured (benchmarking was
disk-blocked the whole time). Two gate violations in one: quality decisions
without measurement, and quality spent on a lever whose ceiling was always <=5%
of decode bytes. User verdict: quality suffered. Verdict accepted.

### New hard rules

1. A4 is CLOSED, not deferred. No further byte-shaving of the winner recipe.
   The outQ5K file may be benchmarked later out of curiosity, but no recipe
   change ships on a speed argument.
2. Only two classes of speed work are allowed from here on:
   - LOSSLESS BY CONSTRUCTION: draft-then-verify schemes where the target model
     verifies greedily, so output is identical to baseline decoding. No KLD
     gate needed because the math cannot change the output.
   - MEASURED-NEUTRAL: runtime/kernel changes (KV layout, dispatch, batching)
     that must pass the witness PPL/KLD gate with MEASURED numbers before any
     claim. Estimated quality numbers are forbidden in decision-making.
3. Ops prerequisite: the disk is at 98% (42G free). Before any phase that
   produces model-sized artifacts, free or archive >=100G (move parked variants
   to HDD/HF per HF_UPLOAD_SOP). Disk pressure caused the A4 mess.

### What Gemma Diffusion actually is, and what we take from it

DiffusionGemma (Google, 2026) drafts a 256-token block of placeholder tokens and
iteratively refines it with bidirectional attention, converting decode from a
memory-bandwidth problem into an arithmetic problem: 1000+ t/s on H100, ~4x over
autoregressive. Two facts matter for us:

1. Google states its output quality is BELOW standard Gemma 4. Adopting a
   diffusion model directly is the same quality-for-speed trade we just banned,
   and our target (Qwen3.6-27B-MTP) is autoregressive - it cannot become a
   diffusion model without retraining.
2. The literature already shows the quality-preserving way to harvest the idea:
   use a small BLOCK-DIFFUSION model as the DRAFTER inside speculative decoding
   (DFlash, arXiv 2602.06036; SpecDiff-2; DiffuSpec). The diffusion drafter
   proposes a whole block in ONE forward pass; the AR target verifies; output is
   bit-identical to baseline. DFlash reports >6x lossless acceleration and 2.5x
   over EAGLE-3. This attacks exactly the bottleneck we measured in A1: our MTP
   head drafts sequentially and its acceptance collapses with depth
   (92/80/71/52% at 2/3/4/6) - parallel block drafting does not pay that
   compounding penalty.

### Revised track list (replaces A3/A4/A6 sequencing)

| Track | What | Quality risk | Effort | Expected |
|---|---|---|---|---|
| L0 | Swap production binary to rebased build | none (same model) | hours | non-spec floor 8.6 -> 32 t/s; spec path equal |
| L1a | Evaluate fork ngram spec types (`ngram-simple/map-k/map-k4v/mod/cache`) vs `mtp` per workload; route copy-heavy (dataextract) to ngram | none (lossless verify) | days, eval-only | recovers the weakest benchloop category |
| L1b | MTP+ngram hybrid drafting (fill draft slots from context ngrams when the MTP head is the bottleneck) | none (lossless verify) | small dev | raises drafted fraction above 68% of steps |
| L1c | Tree/multi-candidate MTP drafts: widen depth-2 instead of deepening (verify top-2 candidates for position 2 in one batch-4 pass) | none (lossless verify) | medium dev | attacks the 92->80% depth-2 acceptance drop |
| L2 | A5 unchanged: KV bandwidth at long context (sparse-V validation, asymmetric KV witness) | measured-neutral gate | days | main decode frontier at long ctx |
| L3 | DFlash-style block-diffusion drafter for Qwen3.6 (train or adopt when checkpoints appear; upstream llama.cpp already has dLLM inference primitives) | none (lossless verify) | weeks+, training | the 2x+ class win; supersedes A6/FastMTP as the training track |
| L4 | Serving throughput: -np > 1, prompt caching (production runs -np 1, cache-ram 0) | none | config | aggregate t/s for concurrent users |

Dead/closed: A3 (platform-parity proven by A2 control), A4 (this revision),
A6-as-FastMTP (superseded by L3 - block diffusion beats head distillation in
the current literature).

## Target

- Artifact: `/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf` (13.39 GiB)
  - TQ3_4S weights, MTP/nextn tensors at q4_K, output tensor at Q6_K
- Goals: improve prompt processing (PP/prefill) and decode (TG, including the MTP
  self-speculative path) without violating the master-plan guardrails
  (correctness > quality > size > TG > PP).

## Critical Context Shift: The Hardware Changed

All proven numbers in `proven/TQ3_4S_PERFORMANCE_DASHBOARD.md` (TG 22.89, PP 673-703)
were measured on an RTX 5060 Ti 16GB (SM120 Blackwell). The current local GPU is an
**RTX 3090 24GB (SM86 Ampere, 936 GB/s)**. Consequences:

1. Every tuning decision made for SM120 (VDR=8, nwarps=2 for ncols=1, MMQ tile shapes)
   is unvalidated on SM86. VDR=8 is active on the 3090 (`__CUDA_ARCH__ >= 800` gate)
   but was chosen by measurement on Blackwell only.
2. The bandwidth roofline doubles: 13.39 GiB/token at 936 GB/s gives a ~65 t/s
   theoretical non-speculative TG ceiling (vs ~32 on the 5060 Ti). If we are far below
   ~45 t/s baseline TG, there is kernel headroom, not just format overhead.
3. 24 GB removes the VRAM pressure that drove B=384 batch caps and KV compromises.
   Longer contexts become the realistic workload, which raises the value of KV-cache
   bandwidth work.
4. The June smoke logs say this model "failed to fit on a 3090 at ngl=99", which is
   inconsistent with a 13.39 GiB file on a 24 GB card. This must be reproduced and
   root-caused in Phase 0 (suspects: huge default ctx, MTP shadow buffers,
   another process holding VRAM).

The second context shift: `feature/gemma4-mtp-support` just rebased the fork onto a
recent upstream. Upstream brought CUDA kernel fusion, concurrent streams, MMVQ
improvements, and faster FA since the April measurements. Some speed arrives free;
some fork customizations (especially the `mmq.cuh` TQ3_4S tile loader) may have been
silently degraded in conflict resolution. Nothing can be claimed until re-measured.

## Phase 0 - Re-baseline (do first, ~1 day, highest information per hour)

1. Reproduce/diagnose the 3090 fit failure for the out6k artifact (`ngl=99`, then
   bisect ctx and MTP settings). Record the fix in this doc.
2. `llama-bench` matrix on the 3090, warmed up per BENCHMARK_PROTOCOL:
   - models: out6k winner, plain `Qwen3.6-27B-MTP-TQ3_4S.gguf`, `Q3_K_M` witness
   - `-p 2048 -n 128 -r 5 -fa 1`, default KV and `K=q4_0,V=tq3_0`
3. MTP on/off server timings: acceptance rate per category prompt set
   (reuse `ab_27b_mtp_*` harness; prior sweeps are 5060 Ti numbers and stale).
4. Two `nsys` profiles:
   - PP512: confirm `mul_mat_q<TQ3_4S>` is still dispatched (tile-loader rewrite
     survived the rebase; remember the touch-recompile pitfall for `mmq.cuh`)
   - MTP decode at nmax=3: confirm verify batches (ne11 = 2..6) hit MMVQ
     (`mul_mat_vec_q`), not the dequant+cuBLAS fallback
5. Quality gate witness: PPL chunks + KLD vs BF16 on the exact artifact, chat suite.

Exit criteria: a dated artifact set that is the single source of truth for "before".

## Track A - Decode (TG) Plan

Decode is bandwidth-bound; the levers are (1) amortize weight reads via MTP,
(2) read fewer bytes, (3) keep small-batch kernels on the roofline.

### A1. MTP operating-point sweep on the 3090 (config-only, cheap, do first)

Re-run the nmax {1,3,5} x p-min x batch sweep on the 3090. The optimal speculation
depth depends on the verify-batch cost curve, which changed with the GPU. Prior
acceptance evidence (0.77-0.94) supports deeper drafts than nmax=1.

### A2. Verify-batch kernel tuning (small, likely win)

MTP verification runs matmuls at ne11 = 2..6. Today `calc_nwarps` returns the generic
4 warps for ncols 2-4 and the TQ3_4S-specific tuning exists only for ncols=1.
Sweep nwarps and rows_per_cuda_block for TQ3_4S at ncols 2-6 on SM86. Upstream's
multi-column MMVQ work showed double-digit spec-decode gains on other backends, and
our vec_dot is heavier than q4/q8, so the generic table is unlikely to be optimal.

### A3. ncols=1 retune for SM86 (small)

Grid: nwarps {1,2,4} x VDR {4,8} on the 3090. One afternoon, possibly a few percent.
Keep the SM120 values behind the existing arch gates; add SM86 values only if the
measured win is over noise.

### A4. Bytes-per-token audit of the out6k recipe (medium)

Per-token weight reads decompose roughly as: TQ3_4S backbone + q4_K MTP layer
(read on every draft AND verify pass) + Q6_K output (~0.6 GiB read per token,
~5% of total). Tests, each gated on KLD:
- output Q6_K vs Q5_K vs Q4_K (speed delta vs KLD delta, pick by ratio)
- MTP layer q4_K vs TQ3_4S (the draft pass runs once per emitted token group;
  its byte cost multiplies with draft depth)
- confirm token_embd type does not matter for TG (row gather, not a matmul read)

### A5. KV-cache bandwidth at long context (medium, mostly validation of existing work)

- Validate the already-implemented sparse-V dequant
  (`GGML_FATTN_SPARSE_V_THRESHOLD`, branch `experiment/decode-speed-plan-x`)
  at 4K/8K/16K/32K on the 3090; PPL within noise at threshold 1e-4/1e-5.
  This was implemented and never GPU-validated; 24 GB now makes 32K realistic.
- Recheck the proven asymmetric KV witness (`K=q4_0, V=tq3_0`, FA) on the rebased
  branch; it was a +4% PP witness and should also help TG at long ctx.
- Park FP4/Turbo4 KV (new-type work) unless the above shows KV reads >15% of decode
  time at target contexts.

### A6. Acceptance-curve work, research-informed (large, gated)

FastMTP (arXiv 2509.18362) shows vanilla MTP heads collapse at depth >=2
(accept: 70%/11%/2% for tokens 1/2/3) and that self-distillation training of a
single recursive MTP head recovers 81%/56%/36%, yielding ~2x lossless speedup.
Phase 0 will give us our own per-position acceptance curve:
- If depth-2+ acceptance is already decent (Qwen3.6 trained its MTP head natively),
  skip this track; tune depth instead (A1).
- If depth-2+ collapses, the inference-side option is tree/parallel drafts
  (L-MTP-style) and the training-side option is FastMTP-style self-distill of the
  MTP head. Training is explicitly parked per the master plan storage constraints;
  re-open only with a measured ceiling from A1.

## Track B - Prefill (PP) Plan

### B1. Confirm the PP breakthrough survived the rebase (do in Phase 0)

The 673-703 t/s regime rests on two fragile pieces: the `tq3_1s_mmq_ok` dispatch and
the rewritten `load_tiles_tq3_4s` with scale baking. Both live in heavily-rebased
files. nsys + PPL witness will tell. If PP regressed to ~360-class numbers, fixing
the rebase damage IS the PP plan.

### B2. MMQ tile/occupancy tuning for SM86 (small-medium)

The tile loader rewrite was tuned on SM120 (larger smem, different scheduler).
Sweep MMQ nwarps / tile sizes for the TQ3_4S instance on Ampere. Bounded by the
known structural ceiling: 32-element blocks mean 8x more tile-loader iterations
than Q3_K; we already accepted that and still reached parity with Q3_K_S.

### B3. MTP prefill overhead (small)

With MTP enabled, prefill also runs the nextn layer and extracts nextn embeddings.
Measure the delta PP with/without MTP enabled; ensure masked-mode extraction
(only the rows the draft needs) and no full-batch embd copies. If overhead is >5%,
inspect the nextn output path before touching kernels.

### B4. Known-failed list (do not repeat)

- Fused rotate+quantize for the MMVQ path: -3% (measured)
- Rotation caching: rotation is 1.4% of PP, nothing to win
- Tile-loader micro-opts (smem pack, 8-lane pack): neutral to -2%
- Direct MMQ without scale baking: PPL 132-170, unusable
- QK_TQ3_0 32->128 redefinition: breaks WARP_SIZE static_assert and tile contracts;
  requires a new type (documented in DECODE_SPEED_PLAN_X_ANALYSIS)

### B5. Moonshot: LUT-centric tensor-core GEMM (parked, evidence-gated)

The 2025-2026 literature converged on table-lookup GEMM for non-uniform/codebook
formats: FLUTE (EMNLP'24), CodeGEMM (arXiv 2512.17970), LUT Tensor Core (ISCA'25),
T-MAC/T-MAN for CPU/NPU. The shared idea: keep weights coded, expand through a
shared-memory LUT into tensor-core fragments at >=128-element granularity, with
offline weight repacking to match the fragment layout. This is the principled fix
for our 32-element-block tile-loader starvation, and unlike the failed Marlin
attempt it does not fight the MMQ framework: it replaces it for TQ3_4S.
Open only if B1/B2 leave PP >30% behind Q3_K_S on the 3090, and treat as a
multi-week kernel project with the full correctness gate.

HadaCore-style tensor-core WHT is NOT worth it: our rotation is 1.4% of PP.

## Branching and Commit Workflow

All kernel/runtime work happens in the llama.cpp-tq3 worktree
(`/home/awee/code/worktrees/tan_llama-main-ref`). One branch per phase; never work
on a shared branch.

Rules:

1. WINNER pointer: the current winner starts as `feature/gemma4-assistant-final`
   (PR #27 head, post-rebase, CI-green). After each phase, if the phase passed its
   gates and beat the prior witness, its branch becomes the new WINNER.
2. Every phase branch is created FROM the current WINNER, never from another
   experiment branch.
3. No dirty state across phases: `git status --porcelain` must be empty before
   creating a phase branch and before declaring a phase done. Stray files are
   committed, stashed with a note, or deleted - never carried silently.
4. Breakthrough = the phase passes the master-plan gates (build, correctness/PPL,
   performance vs witness, artifacts saved). On breakthrough, COMMIT on the phase
   branch immediately with the measured numbers in the commit message
   (`Assisted-by:` trailer, ASCII only). Do not pile a second experiment on top of
   an uncommitted win.
5. Failed experiments stay on their branch (committed with a `experiment:` prefix
   or abandoned); record the negative result in this doc's do-not-repeat table.
6. Artifacts go to `tan_llama/artifacts/` with branch name + commit + date in the
   filename.

Phase branches:

| Phase | Branch | Scope |
|---|---|---|
| 0 | `perf/p0-baseline-3090` | Re-baseline on 3090 + rebased branch; fit bug; nsys audits |
| A1 | `perf/a1-mtp-sweep` | MTP nmax/p-min operating point (config + harness only) |
| A2 | `perf/a2-mmvq-verify-ncols` | MMVQ nwarps/rows tuning for ncols 2-6 (verify batches) |
| A3 | `perf/a3-mmvq-ncols1-sm86` | VDR x nwarps grid for ncols=1 on SM86 |
| A4 | `perf/a4-recipe-bytes` | outQ6K recipe byte audit (quantize recipes, KLD-gated) |
| A5 | `perf/a5-kv-bandwidth` | Sparse-V validation 4-32K; asymmetric KV recheck |
| B2 | `perf/b2-mmq-sm86` | MMQ tile/occupancy tuning for Ampere |
| B3 | `perf/b3-mtp-prefill` | MTP prefill overhead measurement + fixes |
| A6 | `perf/a6-mtp-acceptance` | (gated) tree drafts / head fine-tune prep |
| B5 | `perf/b5-lut-gemm` | (gated) LUT-centric tensor-core GEMM moonshot |

Phase 0 produces no kernel changes; its branch exists so that any rebase-damage
fixes it uncovers are committed in isolation and become the new WINNER base.

## Phase 0 Findings (2026-06-10, partial)

Branch: `perf/p0-baseline-3090` (from winner `feature/gemma4-assistant-final` @ c20558324)
Artifact: `artifacts/perf_p0_baseline_3090/production_server_probes_20260610.md`

1. FIT BUG SOLVED: the 3090 runs the production llama-server (llm-launch release
   build, port 8085) with this exact model resident at 18 GiB, 64K ctx,
   K=q4_0/V=tq3_0, draft-mtp n-max 2 p-min 1.0. The smoke "fit failure" was GPU
   occupancy by this server, not model size.
2. Production operating point measured through the live server (pre-rebase build):
   PP 633.8 t/s at 2.2K prompt; TG 47.2-53.2 t/s; MTP acceptance 95-96 percent.
3. KEY INSIGHT: acceptance 95-96 percent at n-max=2, p-min=1.0 with drafting on
   only ~68 percent of steps means the speculation config is leaving speed on the
   table. A1 (deeper drafts, lower p-min) is promoted to the first experiment and
   needs no kernel changes. FastMTP-style head training (A6) is likely unnecessary:
   the native Qwen3.6 MTP head is strong.
4. Exclusive bench matrix completed (user approved pausing the server; it is a
   user systemd unit `llama-server.service`, stop/start with `systemctl --user`).

## Phase 0 Results (2026-06-10, COMPLETE)

Artifacts: `artifacts/perf_p0_baseline_3090/bench_matrix_c20558324_20260610.log`,
`ppl10_out6k_newbuild_c20558324.log`

llama-bench, RTX 3090, ngl 99, fa 1, warmed up, r=5:

| Config | pp2048 t/s | tg128 t/s |
|---|---|---|
| NEW build (c20558324), out6k, default KV | 1103.74 +/- 6.08 | 33.73 +/- 0.06 |
| NEW build, out6k, K=q4_0 V=tq3_0 | 1103.53 +/- 7.58 | 32.60 +/- 0.17 |
| NEW build, plain TQ3_4S, default KV | 1095.06 +/- 9.71 | 33.17 +/- 0.31 |
| NEW build, Q3_K_M witness, default KV | 1172.78 +/- 13.98 | 33.75 +/- 0.12 |
| OLD build (llm-launch Jun 3), out6k, default KV | 968.04 +/- 6.61 | 8.64 +/- 0.08 |

PPL witness (NEW build, out6k, 10 chunks, c=2048): 5.9720 +/- 0.15305

Conclusions:

1. BREAKTHROUGH BY REBASE: the old production binary decodes at 8.64 t/s in
   plain tg128; the rebased build does 33.73 t/s (3.9x). Production survives on
   the old build only because MTP verify batches avoid the broken/slow ncols=1
   path. Swapping the production binary to the rebased build is the single
   biggest available decode win and costs zero kernel work.
2. The MMQ tile-loader rewrite SURVIVED the rebase: TQ3_4S PP is 1103 t/s vs
   Q3_K_M 1172 (within 6 percent) on Ampere.
3. TG parity with Q3_K_M (33.73 vs 33.75) despite +4 percent file size.
4. TG is at ~52 percent of the 936 GB/s roofline (~65 t/s at 13.38 GiB), so
   A2/A3 kernel headroom exists for both formats.
5. Asymmetric KV gives no PP/TG benefit at 2K ctx (KV reads too small to matter
   there); it remains a long-context lever only (A5).
6. WINNER unchanged: `feature/gemma4-assistant-final` @ c20558324, now
   bench-validated. Phase 0 produced no code changes.

## Phase A1 Results (2026-06-10, COMPLETE)

Branch: `perf/a1-mtp-sweep` (config-only, no code changes; winner unchanged)
Artifacts: `artifacts/perf_a1_mtp_sweep/results.jsonl` + per-config server logs
Setup: NEW build server, 3090, ngl 99, c 8192, fa 1, K=q4_0 V=tq3_0, essay probe
n_predict=256 temp=0, 2 runs per config.

| Config | TG t/s | drafted | accepted | accept rate |
|---|---|---|---|---|
| no spec (baseline) | 32.0-32.2 | - | - | - |
| n-max 2, p-min 1.0 | 45.4-46.2 | 179 | 165 | 92% |
| n-max 3, p-min 1.0 | 39.0-39.3 | 225 | 179 | 80% |
| n-max 4, p-min 1.0 | 37.7-37.8 | 263 | 188 | 71% |
| n-max 6, p-min 1.0 | 32.2-32.4 | 369 | 193 | 52% |
| n-max 2, p-min 0.9/0.75/0.5 | 45.3-45.7 | 179 | 165 | 92% |
| n-max 3, p-min 0.75 | 38.9-39.1 | 225 | 179 | 80% |

Conclusions:

1. n-max 2 is the optimal depth; the hypothesis that deeper drafts would win is
   REJECTED. Marginal acceptance collapses with depth (92 -> 80 -> 71 -> 52 pct),
   the classic vanilla-MTP curve from the FastMTP paper. Each rejected draft
   costs a wasted verify slot plus draft passes, so n-max 6 lands back at baseline.
2. p-min is a no-op on the draft-mtp path (identical draft counts and TG at
   1.0/0.9/0.75/0.5). Do not spend more time on this axis; check the
   implementation if p-min control is ever actually wanted.
3. Winner config = production config (n-max 2). The config axis is exhausted:
   45.8 t/s avg is the MTP ceiling without code or model changes.
4. New build matches the old build on the MTP path (45.8 vs 47.2 on the same
   probe) - the 3.9x ncols=1 win does not compound because MTP decode runs
   verify batches (ne11=2-3), not single-token passes. The production binary
   swap is still strictly safer: any drafting stall on the old build drops to
   8.6 t/s, on the new build to 32 t/s.
5. Implication for sequencing: with depth capped by acceptance collapse, the
   paths to >46 t/s are (a) A2 verify-batch kernel tuning (ncols 2-3 is THE hot
   path), (b) A4 fewer bytes per pass, (c) A6 FastMTP-style head improvement
   (training; parked). A3 (ncols=1) only helps the non-spec floor.

## Phase A2 Results (2026-06-10, COMPLETE - negative)

Branch: `perf/a2-mmvq-verify-ncols` (experiment committed at 0e9d00405, reverted
at branch tip 38ff51148; winner unchanged at c20558324)
Metric: llama-batched-bench, out6k, 3090, ngl 99, fa 1, npp 512, ntg 64

Aggregate TG t/s by decode batch size:

| Config | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| TQ3_4S baseline (winner) | 34.33 | 55.99 | 68.32 | 76.16 |
| TQ3_4S nwarps=2 for ncols 2-4 (3-run mean) | 34.06 | 54.2 | 69.4 | 77.7 |
| Q3_K_M control | 33.61 | 57.00 | 67.40 | 75.14 |

Conclusions:

1. The sub-linear batch scaling (B=3 at 68 aggregate vs ideal ~100) is
   PLATFORM-GENERIC: Q3_K_M shows the identical curve. There is no
   TQ3-specific verify-batch inefficiency to fix. A2 premise rejected.
2. The nwarps variant moved B3/B4 +1.6-2 pct and B2 -3 pct - within run
   drift, with a regression risk for single-draft verifies. Not kept.
3. TQ3_4S multi-column MMVQ is at full parity with Q3_K on Ampere.
4. Implication: A3 (ncols=1 retune) is likely dead for the same reason -
   B=1 parity is exact (34.33 vs 33.61) and both formats sit at ~52 pct of
   the bandwidth roofline, i.e. the gap to roofline is attention/KV/non-matmul
   overhead, not the weight kernels. Deprioritize A3; A5 (KV at long ctx)
   attacks the actual remaining decode cost. The speed levers left are
   A4 (bytes), A5 (KV), A6 (MTP head, training), B5 (PP moonshot).

## Phase A4 Results (2026-06-11, PARTIAL - benchmarking blocked)

Branch: `perf/a4-recipe-bytes` (committed at 5535b4052; winner unchanged at c20558324)
Artifacts: `artifacts/perf_a4_recipe_bytes/README.md`

Created 3 quantization variants of outQ6K baseline:

| Variant | Change | Expected Speed | Expected Quality |
|---------|--------|----------------|------------------|
| outQ5K | output at Q5_K | +1-2% | -0.5-1% KLD |
| outQ4K | output at Q4_K | +2-4% | -2-5% KLD |
| mtp-tq3_4s | MTP eh_proj at tq3_4s | +5-10% draft | -10-20% KLD |

**BENCHMARKING BLOCKED**: Disk space constraints (11G free) prevent loading
additional models for comparison. Production server running on port 8085 with
baseline outQ6K cannot be stopped without user permission.

Conclusions:

1. Variants created successfully but cannot be benchmarked until disk space
   is freed up (need at least 50G for proper KLD-gated benchmarking).
2. Output tensor is only ~5% of total model reads, so Q5_K/Q4_K variants
   likely have minimal speed impact (<4%) for significant quality loss.
3. MTP head (eh_proj) is critical for speculative decoding; tq3_4s variant
   may reduce acceptance rate from 92% to 70-80%, negating speed gains.
4. **Recommendation**: Defer Phase A4 until disk space available. Focus on
   Phase A5 (KV bandwidth at long context) which may have higher impact.

## Phase L1 Results (2026-06-11): MTP loop cost ledger (measured)

Branch: `perf/l1-mtp-loop-overhead` (commit 7db4710fe adds env-gated profiling,
TURBO_MTP_PROF=1; keepable, zero cost when off)

Per spec cycle (n-max 2, ~2.9 tokens emitted, 59-62ms total, 46-49 t/s):

| Component | ms | Notes |
|---|---|---|
| Target verify decode (B=3) | 39-42 | kernel-bound; CUDA graphs verified ACTIVE at B=3 (pp3 77.7 on vs 72.2 off) |
| Draft side (2x prefetch+decode+sample) | 5.8 | drafts are NOT the problem |
| Server spec block (ckpt save 0.86, smpl clone 0.3, sample_accept 0.6) | 1.8 | checkpoint theory falsified |
| Residual (server loop, process_token, queue) | 8-12 | next attribution target (perf/nsys) |

Falsified theories: handle_mtp_for_ubatch hook (never fires for qwen3.6 MTP -
t_h_pre_norm is null on the nextn path), hybrid checkpoint cost, CUDA-graph
batch gate. Found: p-min gating is DEAD CODE on this path (draft sampler is
top_k=1, so selected_p renormalizes to 1.0 and never trips the threshold) -
explains the A1 p-min no-op.

Honest ceiling: ~85 percent of the cycle is irreducible GPU verify work at the
current acceptance rate. Eliminating ALL draft-side syncs and the residual
would give roughly 59 -> 48-50ms, i.e. ~58-60 t/s (+20-25 percent). Remaining
concrete trims, in order: (1) attribute and trim the 8-12ms residual,
(2) overlap draft prefetch/sample syncs with verify launch, (3) beyond that,
only acceptance improvements (L1c tree drafts) or fewer verify passes can help.

## Phase M0 Results (2026-06-11, COMPLETE)

Branch: `perf/m0-pass-trace` (measurement only, no code; winner unchanged)
Artifacts: `artifacts/perf_m0_pass_trace/` (nsys reps + kernel summaries)
Method: nsys with `--cuda-graph-trace=node` (default graph tracing hides
per-kernel times inside CUDA graphs - 26 visible vs 416 real matvecs/pass).

B=1 decode pass (~28.5ms GPU busy of ~31ms wall, GPU ~93 pct utilized):

| Kernel group | ms/pass | Share |
|---|---|---|
| mul_mat_vec_q type46 FUSED (128/pass @ 119us) | 15.2 | 53 pct |
| mul_mat_vec_tq3_4s_ncols1 special (288/pass @ 26us) | 7.5 | 26 pct |
| Q6_K output head (1/pass @ 1.36ms = ~490 GB/s) | 1.4 | 5 pct |
| Q6_K layer tensors (16/pass @ 37us) | 0.6 | 2 pct |
| norms, quantize_q8_1_tq3 (416 launches!), get_rows, SSM, FA | ~3.9 | 14 pct |

B=3 verify pass (~36ms GPU): ALL weight matvecs collapse to ONE path -
`mul_mat_vec_q<type46, ncols=3, fusion OFF>` (480/pass @ 65us = 31.4ms).
The dispatch gates fusion to `ncols_dst == 1` (ggml-cuda.cu, "we only support
fusion for ncols_dst = 1") and the tuned TQ3 ncols1 kernel has no ncols>1
sibling.

Conclusions:

1. No scheduling bubbles: the pass is kernel-bound (93 pct GPU busy). The L1
   "residual" lives in the server layer, not between kernels.
2. Weight matvecs run at ~545 GB/s = 58 pct of effective bandwidth. This is
   the MBU frontier; Q3_K sits at the same plateau (A2 control), so it is an
   upstream MMVQ ceiling for many-small-tensor hybrid models, not TQ3 debt.
3. ACTIONABLE (new phase M3): the B1->B3 verify penalty is largely lost
   fusion + lost special kernel, not inherent batch cost. Extending the fused
   MMVQ path and/or the TQ3 special kernel to ncols 2-4 should bring verify
   from 37.8 toward ~33ms = +10-13 pct on MTP TG, multiplicative with M1/M2.
4. Hybrid confirmed: gated_delta_net + ssm_conv kernels (48 linear-attention
   layers, 16 full-attention with FA), explains the FULL-type seq_rm path.
5. quantize_q8_1_tq3 runs 416 tiny launches/pass (0.7ms) - minor fusion
   candidate, only worth it as part of M3.

## Sequencing

1. Phase 0 (re-baseline, fit bug, rebase audit) - everything else depends on it
2. A1 + A2 + A3 (cheap decode tuning) and B2/B3 if B1 shows the loader survived
3. A4 (recipe byte audit) + A5 (KV validation)
4. Re-run benchloop publish template; update dashboard with 3090 witnesses
5. Decide A6/B5 moonshots from measured acceptance curve / PP gap

## Expected Outcomes (to be replaced by measurements)

- Baseline TG on 3090 should land near 40-46 t/s (roofline ~65 at 13.39 GiB);
  if Phase 0 shows materially less, A2/A3 kernel work moves up in priority.
- MTP at 0.85 depth-1 acceptance with tuned depth: 1.5-1.8x effective TG.
- PP on 3090: if the tile loader survived, expect >1000 t/s class; the Q3_K_S
  witness on the same card defines parity.

## External Research Notes (2025-2026)

- FastMTP: self-distilled recursive MTP head, 2.03x lossless, acceptance
  70->81 / 11->56 / 2->36 at positions 1/2/3 (arXiv 2509.18362)
- L-MTP: leap multi-token prediction + tree decoding for higher accept rates
  (arXiv 2505.17505)
- FLUTE: LUT-quantized GEMM via tensor cores, offline repack + smem LUT
  (EMNLP Findings 2024)
- CodeGEMM: codebook-centric GEMM for quantized LLMs (arXiv 2512.17970)
- LUT Tensor Core (ISCA 2025): hardware-level case that LUT-GEMM is the right
  abstraction for sub-4-bit non-uniform formats
- HadaCore: tensor-core Hadamard transform, 3.5x vs prior FHT kernels
  (PyTorch blog / arXiv 2412.08832) - not applicable here (rotation is 1.4%)
- QuaRot/SpinQuant/FireQ line: rotation-based W4A4/INT4-FP8 with fused prefill
  kernels - validates the TQ3 rotated-activation design direction
- Upstream llama.cpp 2026: CUDA kernel fusion, concurrent streams, faster model
  load, native MXFP4 on Blackwell - mostly free via the rebase, re-baseline needed
