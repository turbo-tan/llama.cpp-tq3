# TurboQuant Benchloop, MTP, KV Cache, and Rebase Index

Date: 2026-06-10

This is the detailed navigation index. The shorter canonical handover is:

- [`2026-06-10_turboquant_recovery_handover.md`](file:///home/awee/code/tan_llama/docs/turboquant/active/handovers/2026-06-10_turboquant_recovery_handover.md)

## Purpose

Provide one recovery index for the late TurboQuant work:

- BenchLoop / publication evidence
- Gemma4 MTP implementation and testing
- Turbo3 / Turbo4 / TQ3_0 cache history
- Rebase workflow and branch hygiene
- Missing June handover search results

## June Handover Search

I searched the local repo, recovered docs, and surviving artifacts for:

- `2026-06-05_numtyped_template_beats_q3km_handover.md`
- `2026-06-06_quality_experiments_handover.md`
- `2026-06-06_refract_outq6k_comparison_handover.md`

No standalone files with those names were found in the accessible workspace.

## Winner Model Location

Recovered from the outQ6 smoke logs:

- `/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf`

Relevant evidence:

- [`artifacts/main_outq6k_smoke_20260606.log`](file:///home/awee/code/tan_llama/artifacts/main_outq6k_smoke_20260606.log)
- [`artifacts/main_outq6k_smoke_20260606_rerun.log`](file:///home/awee/code/tan_llama/artifacts/main_outq6k_smoke_20260606_rerun.log)

Note:

- the logs show this model failed to fit on a 3090 with `n_gpu_layers=99`
- the recovered winner publication path below is the one to keep in the handover, even if this outQ6K variant was a later experiment

## BenchLoop Evidence

### Full late publication run

Recovered artifact:

- [`benchloop_full_qwen36_27b_mtp_tq3_4s_rebuilt_publish_template_pmin10_20260605/benchloop.log`](file:///home/awee/code/tan_llama/artifacts/benchloop/benchloop_full_qwen36_27b_mtp_tq3_4s_rebuilt_publish_template_pmin10_20260605/benchloop.log)

Recovered result:

- `speed 69.0 9/9`
- `toolcall 96.7 14/15`
- `coding 100.0 12/12`
- `dataextract 89.1 12/15`

Interpretation:

- this is the strongest confirmed late-stage full run found in the surviving logs
- it used the rebuilt publish template and conservative `p-min=1.0`
- it is the right public anchor for the late `Qwen3.6-27B-MTP-TQ3_4S` story

### Side-by-side partial run

Recovered artifact:

- [`benchloop_partial_side_by_side_pr18rc_20260604_091021`](file:///home/awee/code/tan_llama/artifacts/benchloop/benchloop_partial_side_by_side_pr18rc_20260604_091021)

Recovered highlights:

- `tq3_4s`
  - `speed 67.0`
  - `toolcall 96.7`
  - `coding 100.0`
- `unsloth_q3km`
  - `speed 68.5`
  - `toolcall 96.7`
  - `coding 100.0`
- `unsloth_q4km`
  - `speed 70.2`
  - `toolcall 90.0`
  - `coding 93.8`

Interpretation:

- the public claim must stay tied to the exact template and exact model variant
- partial benchloop numbers alone are not enough to declare a winner

## MTP Implementation

### Current status

The Gemma4 MTP implementation is documented as in progress and still architecturally blocked.

Key recovered status docs:

- [`docs/experiments/gemma4-mtp-status.md`](file:///home/awee/code/tan_llama/docs/experiments/gemma4-mtp-status.md)
- [`docs/experiments/gemma4-mtp-test-report.md`](file:///home/awee/code/tan_llama/docs/experiments/gemma4-mtp-test-report.md)
- [`docs/experiments/gemma4-mtp-upstream-comparison.md`](file:///home/awee/code/tan_llama/docs/experiments/gemma4-mtp-upstream-comparison.md)
- [`docs/experiments/gemma4-mtp-upstream-alignment-status.md`](file:///home/awee/code/tan_llama/docs/experiments/gemma4-mtp-upstream-alignment-status.md)
- [`docs/procedure/test-gemma4-mtp.md`](file:///home/awee/code/tan_llama/docs/procedure/test-gemma4-mtp.md)

### What the docs say

- the embeddings-nextn infrastructure exists and compiles
- the actual gemma4-assistant model alignment is still the issue
- upstream uses `gemma4-assistant` and `ctx_other`
- our local port originally diverged from that structure

### Practical implication

For MTP work, use the upstream-alignment docs as the source of truth, not the older local architectural guess.

## Turbo3 / Turbo4 / TQ3_0 Cache

### TQ3_0 cache and KV path

Recovered sources:

- [`recover/git_docs_all/archive/exploration-legacy/TQ3_0_PROD_QJL_PROPOSAL.md`](file:///home/awee/code/tan_llama/recover/git_docs_all/archive/exploration-legacy/TQ3_0_PROD_QJL_PROPOSAL.md)
- [`recover/git_docs_all/procedures/BENCHMARK_PROTOCOL.md`](file:///home/awee/code/tan_llama/recover/git_docs_all/procedures/BENCHMARK_PROTOCOL.md)
- [`recover/git_docs_all/proven/TQ3_4S_PERFORMANCE_DASHBOARD.md`](file:///home/awee/code/tan_llama/recover/git_docs_all/proven/TQ3_4S_PERFORMANCE_DASHBOARD.md)
- [`docs/turboquant/archive/GPU_WEIGHT_PLAN.md`](file:///home/awee/code/tan_llama/docs/turboquant/archive/GPU_WEIGHT_PLAN.md)
- [`docs/turboquant/archive/CPU_SIMD_PLAN.md`](file:///home/awee/code/tan_llama/docs/turboquant/archive/CPU_SIMD_PLAN.md)

Recovered conclusions:

- `TQ3_0` is the KV-cache line, not the weight-format line
- `K=q4_0, V=tq3_0` was a confirmed asymmetric KV witness
- `tq3_0` cache was repeatedly treated as the smaller / bandwidth-saving path
- the best docs emphasize measuring it at long context, not just short-context toy runs

### Turbo3 / Turbo4 direction

Recovered source:

- [`recover/git_docs_all/archive/exploration-legacy/TQ3_WEIGHT_VARIANT_SKETCH.md`](file:///home/awee/code/tan_llama/recover/git_docs_all/archive/exploration-legacy/TQ3_WEIGHT_VARIANT_SKETCH.md)
- [`recover/git_docs_all/archive/exploration-legacy/TQ3_5_PRECOMPUTED_WHT.md`](file:///home/awee/code/tan_llama/recover/git_docs_all/archive/exploration-legacy/TQ3_5_PRECOMPUTED_WHT.md)
- [`recover/git_docs_all/active/TQ3_BLOCK128_PLAN.md`](file:///home/awee/code/tan_llama/recover/git_docs_all/active/TQ3_BLOCK128_PLAN.md)

Recovered interpretation:

- `turbo3` is the earlier KV-cache compression line
- `turbo4` is the later, higher-quality / higher-capacity follow-on concept
- these are cache-family ideas, not the same thing as TQ3_4S weight quantization

## Rebase Workflow

Current steering and workflow sources:

- [`docs/steering/testing-procedures.md`](file:///home/awee/code/tan_llama/docs/steering/testing-procedures.md)
- `.git/logs/HEAD` entries for `rebase: sync main with upstream master`

Recovered rule set:

- never merge into `master`
- keep `master` synced with upstream/master
- branch from `main` for work
- rebase main onto upstream master when syncing the fork

Practical note:

- the rebase workflow is currently documented in steering rather than in a single dedicated turboquant SOP
- if a standalone rebase guide is desired, it should be created as a new canonical doc under `docs/steering/` or `docs/procedure/`

## Most Useful File Map

### BenchLoop / publication

- [`docs/turboquant/active/handovers/2026-06-10_qwen36_27b_mtp_tq3_4s_recovery_handover.md`](file:///home/awee/code/tan_llama/docs/turboquant/active/handovers/2026-06-10_qwen36_27b_mtp_tq3_4s_recovery_handover.md)
- [`docs/turboquant/public/qwen36_27b_tq3_publication_20260529.html`](file:///home/awee/code/tan_llama/docs/turboquant/public/qwen36_27b_tq3_publication_20260529.html)
- [`artifacts/benchloop/benchloop_full_qwen36_27b_mtp_tq3_4s_rebuilt_publish_template_pmin10_20260605/benchloop.log`](file:///home/awee/code/tan_llama/artifacts/benchloop/benchloop_full_qwen36_27b_mtp_tq3_4s_rebuilt_publish_template_pmin10_20260605/benchloop.log)

### MTP implementation

- [`docs/experiments/gemma4-mtp-status.md`](file:///home/awee/code/tan_llama/docs/experiments/gemma4-mtp-status.md)
- [`docs/experiments/gemma4-mtp-test-report.md`](file:///home/awee/code/tan_llama/docs/experiments/gemma4-mtp-test-report.md)
- [`docs/experiments/gemma4-mtp-upstream-alignment-status.md`](file:///home/awee/code/tan_llama/docs/experiments/gemma4-mtp-upstream-alignment-status.md)
- [`docs/procedure/test-gemma4-mtp.md`](file:///home/awee/code/tan_llama/docs/procedure/test-gemma4-mtp.md)

### Cache and quantization

- [`recover/git_docs_all/archive/exploration-legacy/TQ3_0_PROD_QJL_PROPOSAL.md`](file:///home/awee/code/tan_llama/recover/git_docs_all/archive/exploration-legacy/TQ3_0_PROD_QJL_PROPOSAL.md)
- [`recover/git_docs_all/procedures/BENCHMARK_PROTOCOL.md`](file:///home/awee/code/tan_llama/recover/git_docs_all/procedures/BENCHMARK_PROTOCOL.md)
- [`recover/git_docs_all/proven/TQ3_4S_PERFORMANCE_DASHBOARD.md`](file:///home/awee/code/tan_llama/recover/git_docs_all/proven/TQ3_4S_PERFORMANCE_DASHBOARD.md)

### Rebase and steering

- [`docs/steering/testing-procedures.md`](file:///home/awee/code/tan_llama/docs/steering/testing-procedures.md)
- [`docs/procedure/compile-remote-gpu-server.md`](file:///home/awee/code/tan_llama/docs/procedure/compile-remote-gpu-server.md)

## Recommendation

If you want this turned into the next canonical doc, do it in this order:

1. keep the recovered handover as the late publication state
2. keep the benchloop / MTP / cache / rebase index as the navigation doc
3. create a dedicated rebase SOP only if you actually want a separate canonical workflow document
