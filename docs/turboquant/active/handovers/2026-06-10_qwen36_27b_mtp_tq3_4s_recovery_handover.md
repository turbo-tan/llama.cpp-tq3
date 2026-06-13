# Qwen3.6-27B-MTP-TQ3_4S Recovery Handover

Date: 2026-06-10

## Purpose

Recover the deleted `docs/turboquant` knowledge base and preserve the late Qwen3.6-27B-MTP-TQ3_4S publication state from surviving artifacts.

## What Was Recovered

### Git-backed docs

The historical TurboQuant markdown set was restored into:

- [`recover/git_docs_all`](file:///home/awee/code/tan_llama/recover/git_docs_all)

High-signal recovered originals:

- [`TQ3_MOONSHOT_MASTER_LOG.recovered.md`](file:///home/awee/code/tan_llama/recover/git_docs/TQ3_MOONSHOT_MASTER_LOG.recovered.md)
- [`TQ3_Q3KS_BOTTLENECK_MAP.recovered.md`](file:///home/awee/code/tan_llama/recover/git_docs/TQ3_Q3KS_BOTTLENECK_MAP.recovered.md)
- [`TQ3_NATIVE_PROMPT_KERNEL_DESIGN.recovered.md`](file:///home/awee/code/tan_llama/recover/git_docs/TQ3_NATIVE_PROMPT_KERNEL_DESIGN.recovered.md)
- [`TQ3_KV_PREROTATION_PROOF.recovered.md`](file:///home/awee/code/tan_llama/recover/git_docs/TQ3_KV_PREROTATION_PROOF.recovered.md)
- [`TQ3_4S_CONVERSION_GUIDE.recovered.md`](file:///home/awee/code/tan_llama/recover/git_docs/TQ3_4S_CONVERSION_GUIDE.recovered.md)
- [`BENCHMARK_RESULTS.recovered.md`](file:///home/awee/code/tan_llama/recover/git_docs/BENCHMARK_RESULTS.recovered.md)
- [`TQ3_RUNTIME_AND_RESULTS.recovered.md`](file:///home/awee/code/tan_llama/recover/git_docs/TQ3_RUNTIME_AND_RESULTS.recovered.md)
- [`TQ3_4S_PERFORMANCE_DASHBOARD.recovered.md`](file:///home/awee/code/tan_llama/recover/git_docs/TQ3_4S_PERFORMANCE_DASHBOARD.recovered.md)
- [`TQ3_4S_V2_RELEASE.recovered.md`](file:///home/awee/code/tan_llama/recover/git_docs/TQ3_4S_V2_RELEASE.recovered.md)

### Reconstruction notes

- [`tq3_4s_recovery_2026-06-10.md`](file:///home/awee/code/tan_llama/recover/tq3_4s_recovery_2026-06-10.md)
- [`tq3_4s_git_recovery_manifest_2026-06-10.md`](file:///home/awee/code/tan_llama/recover/tq3_4s_git_recovery_manifest_2026-06-10.md)
- [`qwen36_27b_mtp_tq3_4s_artifact_recovery_2026-06-10.md`](file:///home/awee/code/tan_llama/recover/qwen36_27b_mtp_tq3_4s_artifact_recovery_2026-06-10.md)

## Late Winner State

The late publication-era target was:

- `Qwen3.6-27B-MTP-TQ3_4S`

Known variants that showed up repeatedly:

- `Qwen3.6-27B-MTP-TQ3_4S.gguf`
- `Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k.gguf`
- `outQ6K`

Important winner path remembered in the artifacts:

- `/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k.gguf`

Later user guidance changed the winner story:

- after corrected template retesting, the plain non-`mtp-q4k` `Qwen3.6-27B-MTP-TQ3_4S.gguf` was treated as the winner

## BenchLoop Evidence

### Full run at `p-min=1.0`

Recovered from:

- [`artifacts/benchloop/benchloop_full_qwen36_27b_mtp_tq3_4s_rebuilt_publish_template_pmin10_20260605/benchloop.log`](file:///home/awee/code/tan_llama/artifacts/benchloop/benchloop_full_qwen36_27b_mtp_tq3_4s_rebuilt_publish_template_pmin10_20260605/benchloop.log)

Recovered summary:

- `speed 69.0 9/9`
- `toolcall 96.7 14/15`
- `coding 100.0 12/12`
- `dataextract 89.1 12/15`

Interpretation:

- this is the strongest confirmed late-stage full run found in the surviving logs
- it was run with the rebuilt publish template and conservative `p-min=1.0`
- coding was perfect, tool use was strong, data extraction remained the weaker category

### Partial side-by-side run

Recovered from:

- [`artifacts/benchloop/benchloop_partial_side_by_side_pr18rc_20260604_091021`](file:///home/awee/code/tan_llama/artifacts/benchloop/benchloop_partial_side_by_side_pr18rc_20260604_091021)

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
- `qwopus_q4km`
  - `speed 70.3`
  - `toolcall 96.7`
  - `coding 100.0`

Interpretation:

- `TQ3_4S` was competitive, but the exact leadership depended on template/runtime details
- the later public story should stay narrow and cite the exact benchloop log used

## Quality And KLD Line

Recovered working values:

- PPL overhead: `+3.95%`
- `tq3_4s: 6.840` vs `BF16: 6.580`
- top-1 token agreement: `87.57%`
- mean KLD: `0.146`
- median KLD: `0.036`
- 90th percentile KLD: `0.193`

This was already considered a strong central-agreement result with a tail that could be improved later by selective protection.

## MTP Acceptance Evidence

Recovered from:

- [`artifacts/q3km_vs_tq3_4s_pr18_20260604/logs/tq3_4s_mtp.log`](file:///home/awee/code/tan_llama/artifacts/q3km_vs_tq3_4s_pr18_20260604/logs/tq3_4s_mtp.log)

Observed draft acceptance values included:

- `0.85511`
- `0.77381`
- `0.87069`
- `0.93137`
- `0.85336`
- `0.90152`
- `0.83684`
- `0.93836`
- `0.93798`

Interpretation:

- MTP was active and not a fake no-spec path
- acceptance was usually high enough to support the later throughput claims

## Missing Publication HTML

I searched the repo, git history, and the accessible saved rollout summaries for:

- `qwen36_27b_tq3_publication_20260529.html`
- `qwen36_27b_tq3_publication_x.html`
- `packed_layout_graphic.html`
- `showcase`

No matching publication HTML was found in the current workspace or the saved history I could access.

So for now:

- the artifact-backed handover is the source of truth
- the publication HTML must be rebuilt later from the recovered artifact evidence

## Remaining Search Result

I also searched the accessible Codex rollout summaries for the June handover names and did not find a matching saved rollout entry for:

- `2026-06-05_numtyped_template_beats_q3km_handover.md`
- `2026-06-06_quality_experiments_handover.md`
- `2026-06-06_refract_outq6k_comparison_handover.md`

That means the only reliable source of truth for the late stage is:

- the recovered git docs
- the surviving artifacts
- the reconstructed notes in `recover/`

## Next Steps

1. Rebuild a clean canonical `docs/turboquant/active/` narrative from the recovered docs.
2. Rebuild the publication HTML from the recovered benchloop and KLD evidence.
3. Recover or recreate the missing June handover files under `docs/turboquant/active/handovers/`.
4. Keep the public claim narrow:
   - specify the exact model variant
   - specify the exact template
   - specify the exact `p-min`
   - specify the exact artifact path
