# TurboQuant Recovery Handover

Date: 2026-06-10

## Current State

The deleted TurboQuant docs have been recovered into git-backed and artifact-backed recovery files.

Recovered areas:

- historical `docs/turboquant` tree restored from git
- late Qwen3.6-27B-MTP-TQ3_4S winner evidence reconstructed from surviving benchloop logs
- Gemma4 MTP status and upstream-alignment notes preserved under `docs/experiments`
- TQ3_0 / turbo3 / turbo4 cache history preserved in the recovered archive set

## What Matters

- Public recovery anchor: `Qwen3.6-27B-MTP-TQ3_4S`
- Confirmed full benchloop anchor: `speed 69.0`, `toolcall 96.7`, `coding 100.0`, `dataextract 89.1`
- Conservative publication setting: `p-min=1.0`
- Winner model location recovered from the smoke logs:
  - `/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf`
- The refract comparison handover name was found in the index search, but no standalone file was recovered:
  - `2026-06-06_refract_outq6k_comparison_handover.md`
- Exact handover details live in:
  - [`2026-06-10_turboquant_benchloop_mtp_cache_rebase_index.md`](file:///home/awee/code/tan_llama/docs/turboquant/active/handovers/2026-06-10_turboquant_benchloop_mtp_cache_rebase_index.md)

## Next Actions

1. Use the detailed index as the navigation document for benchloop, MTP, cache, and rebase history.
2. Use the new rebase SOP for branch sync work.
3. Rebuild any missing publication HTML only from artifact-backed evidence.
