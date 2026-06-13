# SOP: BenchLoop, hard86, and GPQA

## Goal

Run the public-facing evaluation gates with the correct model, template, and artifact capture so results stay comparable.

## Shared Rules

- Use the exact runtime binary you intend to publish.
- Use the exact chat template you intend to publish.
- Record the model path, runtime commit, template path, and date in the artifact folder name or log header.
- Do not mix partial and full results in the same summary table.
- Do not compare against stale binaries in `/tmp` or other ephemeral build paths.
- Keep the winner model path explicit in the note header when comparing variants, especially if you are switching between the late public winner and an outQ6K experiment.

## Canonical Commands

### BenchLoop

Use the repo's benchloop wrapper or harness entrypoint that maps to these presets:

```bash
benchloop partial
benchloop full
```

If you are invoking the model checks manually, keep the exact model/template pair fixed for the whole run and save the artifacts under `artifacts/benchloop/<date>_<model>_<preset>/`.

### llama-server smoke start

Use the same runtime binary you intend to publish:

```bash
cd /home/awee/code/llama.cpp-tq3
nohup ./build/bin/llama-server \
  -m MODEL.gguf \
  --host 127.0.0.1 --port 18129 \
  -ngl 99 -c 2048 -np 1 -t 8 -fa on \
  --reasoning off --reasoning-budget 0 --reasoning-format deepseek \
  < /dev/null > /dev/null 2>&1 &
```

For the simpler benchmark-protocol smoke path:

```bash
pkill -9 -f llama-server 2>/dev/null; sleep 2
setsid ./build/bin/llama-server \
  -m MODEL.gguf \
  -ngl 99 -c 2048 --port 8090 -fa 1 \
  </dev/null >/tmp/llama-server.log 2>&1 &
```

## Model Location Notes

- Late public winner:
  - `/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k.gguf`
- Editable publish template:
  - `/home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja`
- Later outQ6K experiment:
  - `/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf`
- If a handover refers to the refract comparison, keep the filename as a note even if the standalone file is missing:
  - `2026-06-06_refract_outq6k_comparison_handover.md`

## BenchLoop Partial

### What It Covers

The partial preset is the fast gate used for publication work. In this repo it should mean:

- `speed`
- `toolcall`
- `coding`

### Procedure

1. Rebuild or confirm the runtime binary you want to evaluate.
2. Confirm the target model and template are the publication versions.
3. Run the benchloop harness with the `partial` preset.
4. Save the run under a dated artifact folder in `artifacts/benchloop/`.
5. Keep the raw log and the summary log together.

### Acceptance

- `speed`, `toolcall`, and `coding` must all be reported.
- The result is not a publication winner unless the exact model/template pair is clear.

## BenchLoop Full

### What It Covers

The full preset is the publication gate. It should include the full task mix used for the final score.

### Procedure

1. Start from the exact model/template pair that won or is under test.
2. Run the benchloop harness with the `full` preset.
3. Keep the runtime stable for the entire run. Do not swap binaries mid-run.
4. Save the log and summary artifact under a unique dated folder.

### Acceptance

- Use the full run for headline publication claims.
- Report the exact task breakdown and the exact `p-min` or other decoding policy.

## hard86

### Purpose

Use `hard86` as the strict correctness and robustness gate for the same release candidate.

### Procedure

1. Reuse the same runtime binary and template from the benchloop run.
2. Run the `hard86` harness on the same model variant.
3. Save the raw output and the pass/fail summary.

### Acceptance

- A failure in `hard86` blocks publication even if speed looks good.
- If the failure is prompt/template-sensitive, rerun only after documenting the template delta.

## GPQA

### Purpose

Use GPQA as the higher-signal reasoning gate when publication claims mention reasoning quality.

### Procedure

1. Use the same model/template pair as the winning benchmark run.
2. Run the GPQA harness with deterministic decoding.
3. Save the raw answers and the scored summary.

### Acceptance

- Do not publish reasoning claims from a GPQA run unless the model/template pair matches the published artifact.
- If GPQA regresses after a runtime or template change, treat that as a real quality regression until explained.

## Recommended Order

1. Run `benchloop partial` first.
2. If the candidate looks good, run `benchloop full`.
3. Run `hard86` on the same candidate.
4. Run `GPQA` last, after the candidate has passed the other gates.

## Artifact Naming

Use descriptive folder names such as:

- `artifacts/benchloop/<date>_<model>_<preset>/`
- `artifacts/hard86/<date>_<model>/`
- `artifacts/gpqa/<date>_<model>/`

## Notes

- If the model, template, or runtime changes, create a fresh artifact folder.
- If you need to compare against a previous winner, keep the older artifact folder intact and side by side.

## CI Log Extraction

When a GitHub Actions run fails, extract the job log before guessing at the root cause.

### 1. Check run status

```bash
gh run view RUN_ID -R turbo-tan/llama.cpp-tq3 --json status,conclusion
```

### 2. Read the job metadata

```bash
gh api --hostname github.com repos/turbo-tan/llama.cpp-tq3/actions/jobs/JOB_ID
```

This confirms whether the job is actually `completed`, `failed`, or still running.

### 3. Download the raw job log

```bash
gh api --hostname github.com repos/turbo-tan/llama.cpp-tq3/actions/jobs/JOB_ID/logs
```

Pipe the log through `rg` to find the first real error:

```bash
gh api --hostname github.com repos/turbo-tan/llama.cpp-tq3/actions/jobs/JOB_ID/logs \
  | rg -n "Traceback|AssertionError|FAILED|Error:|error:|fatal:|Exception"
```

### 4. Isolate the failing test

If CTest fails, search for the test name and the summary block:

```bash
gh api --hostname github.com repos/turbo-tan/llama.cpp-tq3/actions/jobs/JOB_ID/logs \
  | rg -n -C 12 "test-name|The following tests FAILED|LastTest.log"
```

### 5. Interpret the result

- If the log shows a missing downloaded model or artifact, treat it as a CI/environment failure first.
- If the log shows a compile or runtime assertion in the changed code path, treat it as a real regression.
- Do not call the PR merge-ready until the red job is either fixed or clearly exempted.
