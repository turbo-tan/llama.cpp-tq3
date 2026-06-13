# Out6k BenchLoop SOP

Use this exact runtime shape for the flagship Qwen3.6-27B-MTP out6k model when running local BenchLoop checks.

## Rules

- Keep `-fa on`.
- Use `--spec-type draft-mtp`.
- Use `--spec-draft-n-min 1`.
- Use `--spec-draft-n-max 2`.
- Use `--spec-draft-p-min 0.0` on current `main`.
- Do not reuse older `p_min 1.0` guidance from the legacy hook-driven `mtp` path; on current `draft-mtp` it can suppress draft generation entirely.
- If the long-prompt repro shows `draft acceptance = 0.00000` or the draft path looks wrong, stop and investigate before running any further BenchLoop jobs.
- Keep BenchLoop local-only with `BENCHLOOP_NO_SUBMIT=1`.
- Run a fast dry run first to confirm speed is on par before running the full `partial` suite.
- For template-only quality isolation on the out6k artifact, use `-c 32768`.
- Do not use `-c 262144` for template A/B checks unless you have already confirmed VRAM headroom; on 2026-06-13 it produced a false negative via CUDA OOM before quality could be measured.
- Before starting the `-ngl 99` server, stop any other GPU consumers and verify the card is free enough for a full load; do not assume the model will auto-fit around a busy VRAM state.
- Quality guard: if the long-prompt path reaches `draft acceptance = 0.00000`, stop immediately and go back to the last known-good template/runtime before comparing anything else.

## Server launch

```bash
./build-sync-cuda/bin/llama-server \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --alias Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --host 127.0.0.1 --port 18124 \
  -c 32768 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 \
  --spec-type draft-mtp \
  --spec-draft-n-min 1 \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.0 \
  --reasoning off
```

Current recovery status:

- For `Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf`, the embedded GGUF template is not equivalent to the local publish template.
- Measured on 2026-06-13:
  - embedded template regressed `tc11` by calling `calculator`
  - `--chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja` restored `tc11` to a direct answer
- Until the GGUF metadata is rebuilt and revalidated, use the publish template override for local recovery checks.
- The embedded publish-template GGUF artifact is not the baseline for this recovery pass. The validated path is the original out6k GGUF plus the publish template override.

Use:

```bash
--chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja
```

Preflight for the `-ngl 99` launch:

```bash
pkill -f 'llama-server.*Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6'
nvidia-smi
```

Only launch the test server after confirming the GPU is clear enough for the full load.

Do not assume the embedded template is the publication winner for this artifact.

Verified 2026-06-13 template-isolation gate on current `build-current` with `-c 32768`:

- `toolcall 96.7 14/15`
- `coding 100.0 12/12`
- template lineage:
  - old stock-derived publish template from `25b98f6be`
  - plus scoped JSON-only numeric rule
  - plus direct-answer tool-avoidance rules from `artifacts/tc11-prompt-patched.txt`

Verified 2026-06-13 full local BenchLoop on the recovered publish template and current rebase runtime:

- run:
  - `/home/awee/.bench-loop/runs/20260613-131757-Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf-local-openai_compat/run.json`
- runtime shape:
  - `./build-current/bin/llama-server`
  - `-c 32768`
  - `-ctk q8_0`
  - `-ctv tq3_0`
- measured:
  - `speed 68.6 9/9`
  - `toolcall 96.7 14/15`
  - `coding 100.0 12/12`
  - `dataextract 91.0 12/15`
  - `instructfollow 74.5 9/15`
  - `reasonmath 73.3 11/15`
- `GEN TOK/S 42.73`

Interpretation:

- This full run does not show the earlier template-collapse behavior.
- Remaining misses are ordinary benchmark-quality misses, not the old out6k template regression.

Direct smoke on the original out6k GGUF plus this override is sane:

- a direct math prompt answers `30` for `15% of 200`
- a direct coding prompt returns a valid `parse_csv` implementation

## Dry run

Use a direct `llama-bench` check before BenchLoop:

```bash
./build-sync-cuda/bin/llama-bench \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  -p 128 -n 64 -pg 128,64 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 -r 1 -o md
```

If the dry run is materially below the expected local baseline, stop and fix the runtime shape before running BenchLoop.
Also stop immediately if server stats show `#gen drafts = 0` for `draft-mtp`; that means the speculative path is configured but not actually drafting.
If a long prompt collapses to `draft acceptance = 0.00000`, treat that as a hard stop and revert to the last known-good runtime before continuing.

## Partial suite

Run `partial` as one combined local-only invocation:

```bash
BENCHLOOP_NO_SUBMIT=1 benchloop run \
  --model Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --endpoint http://127.0.0.1:18124 \
  --provider openai_compat \
  --suites speed,toolcall,coding \
  --harness raw
```
