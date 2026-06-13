# Upstream Sync Checklist - 2026-06-13

This is the current source-of-truth checklist for merging the public `tan_llama` branch back to `main` in a state that is both upstream-synced and preserves the required TurboQuant and MTP behavior.

## Goal

Merge to `main` with:

- upstream sync intact
- `tq3_4s` weight path intact
- `tq3_0` KV cache path intact
- `turbo3_0` KV cache alias intact
- `turbo4_0` KV cache alias intact
- out6k MTP quality restored
- no known large-prompt regression left untracked

## Current branch status

- repo: `/home/awee/code/tan_llama`
- working branch: `sync/main-upstream-catchup-20260613`
- sync base: `turbo/main`
- upstream comparison: `upstream/master...sync/main-upstream-catchup-20260613`
- ahead by 10 commits
- behind by 26 commits

Committed recovery deltas on top of `turbo/main`:

- `22693bd3f` `publish: restore out6k winner chat template`
- `703797b40` `docs: record out6k full suite recovery result`

Tracked local code change still not committed:

- `common/chat.cpp`

## What is already proven

### 1. Upstream sync base is in place

- the working branch is rebased onto the newer `turbo/main`
- the branch is now `10` ahead and `26` behind `upstream/master`
- this is the intended shape for continuing catch-up work, instead of the stale local `main` that had drifted far away

### 2. Out6k winner template is recovered

- committed template:
  - `/home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja`
- template-isolation gate recovered on 2026-06-13:
  - `toolcall 96.7 14/15`
  - `coding 100.0 12/12`

### 3. Full local BenchLoop is stable on the recovered template/runtime

- run:
  - `/home/awee/.bench-loop/runs/20260613-131757-Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf-local-openai_compat/run.json`
- measured:
  - `speed 68.6 9/9`
  - `toolcall 96.7 14/15`
  - `coding 100.0 12/12`
  - `dataextract 91.0 12/15`
  - `instructfollow 74.5 9/15`
  - `reasonmath 73.3 11/15`

Current comparison note:

- this is the last known-good local full-suite reference for the out6k recovery
- do not replace it with the later `20260613-143632` run
- `20260613-143632` is the broken long-prompt follow-up and shows `draft acceptance = 0.00000`, so it is a stop-and-investigate result, not a new baseline

### 4. KV-cache naming coverage exists in tree

Current tree documents:

- `tq3_0`
- `turbo3_0`
- `turbo4_0`

Reference:

- `/home/awee/code/tan_llama/README.md`

## Open checkpoints

### A. Commit the prompt-parity code delta

Reason:

- the tested binary may include the local `common/chat.cpp` change
- current working branch is therefore not fully source-reproducible yet

Checklist:

- [ ] review `common/chat.cpp`
- [ ] decide whether it is required for the recovered result
- [ ] if required, commit it as a separate prompt-parity chunk
- [ ] rebuild from committed source

### B. Revalidate from committed source only

Reason:

- current quality proof used a binary that may include an uncommitted code delta
- the validated draft-MTP contract on this branch uses `--spec-draft-p-min 0.0`, not the legacy `1.0` guidance
- the `ngl=99` validation path must start from a quiet GPU state; a busy VRAM state can prevent a fair source-only test

Checklist:

- [ ] rebuild `llama-server` from committed source only
- [ ] rerun narrow gate:
  - `toolcall`
  - `coding`
- [ ] rerun one speed check
- [ ] confirm the same quality holds without any floating local edits
- [ ] if `draft acceptance = 0.00000`, stop and investigate before any more benchmark runs
- [ ] before any `ngl=99` launch, stop other GPU consumers and verify the card is free enough for a full load

### C. Recheck the large-prompt failure directly

Reason:

- full BenchLoop stability is not the same as re-proving the old large-prompt failure is gone
- the historical failure was triggered in the real serving path with a repeated long prompt
- the current long-prompt repro has already shown `draft acceptance = 0.00000`, so this is a live stop-and-investigate condition

Checklist:

- [ ] rerun the repeated large-prompt repro
- [ ] record exact prompt size, context size, and server flags
- [ ] confirm whether draft acceptance or request stability still collapses
- [ ] stop immediately if the draft path returns `0` accepted drafts
- [ ] if fixed, document the exact proof run
- [ ] if not fixed, isolate it as a separate runtime issue from template recovery

### D. Recheck the intended KV-cache runtime shape

Reason:

- the merge goal explicitly includes `tq3_0` and `turbo3_0` / `turbo4_0`
- we need one explicit validation pass that the intended public runtime shape still works

Checklist:

- [ ] validate `-ctk q8_0 -ctv tq3_0`
- [ ] validate `-ctk q4_0 -ctv tq3_0` for the out6k speed experiment
- [ ] confirm `turbo3_0` remains equivalent to `tq3_0`
- [ ] confirm `turbo4_0` still loads and runs for KV cache use
- [ ] record any alias or parser mismatch if found

### E. Decide whether to re-embed the GGUF template

Reason:

- current proof uses `--chat-template-file`
- embedded GGUF template is older and not the recovered winner template
- the embedded publish-template GGUF artifact is not the validated baseline for this pass
- direct smoke on the original out6k GGUF plus the publish template override returns `30` for `15% of 200`

Checklist:

- [ ] keep override template for code/runtime validation
- [ ] decide whether publication requires GGUF template re-embed
- [ ] if yes, re-embed and rerun `toolcall` + `coding` once

## Required runtime references

### Quality gate

Use the recovered publish template and the current local-only SOP:

- `/home/awee/code/tan_llama/docs/turboquant/benchloop-sop.md`

Important current rule:

- use `-c 32768` for template/quality isolation on this out6k artifact

### Partial BenchLoop contract

In this repo, `partial` means:

- `speed,toolcall,coding`

### No publish rule

- keep BenchLoop local only with `BENCHLOOP_NO_SUBMIT=1`
- do not publish remote benchmark runs without explicit approval

## Artifact storage map

Keep these locations in mind so recipe/config history does not get lost again:

- generated turboquant recipes and tensor-type policies:
  - `docs/turboquant/generated/`
- out6k recipe and policy pair:
  - `docs/turboquant/generated/qwen36_27b_mtp_tq3_4s_out6k_v2b.recipe.md`
  - `docs/turboquant/generated/qwen36_27b_mtp_tq3_4s_out6k_v2b.tensor-types.txt`
- family-specific historical benchmark archives:
  - `artifacts/qwen36/`
  - `artifacts/qwopus/`
  - `artifacts/gemma4/`
- local BenchLoop run outputs:
  - `artifacts/benchloop/`
- model-specific conversion or quantization logs:
  - `artifacts/`

## Merge gate

Do not treat this work as complete until all of the following are true:

- [ ] the working branch remains `0` behind `turbo/main`
- [ ] the working branch stays within a small, understood gap to `upstream/master`
- [ ] all required recovery code is committed
- [ ] no critical recovery delta remains only in the working tree
- [ ] out6k `toolcall` is back at `14/15` or better
- [ ] out6k `coding` is back at `12/12`
- [ ] one speed check is recorded on committed source
- [ ] the large-prompt repro result is recorded explicitly
- [ ] `tq3_4s`, `tq3_0`, `turbo3_0`, and `turbo4_0` status is written down

## Next action

The next concrete step is:

- commit or discard `common/chat.cpp`, then rebuild and rerun `toolcall`, `coding`, and the large-prompt repro from committed source only on `sync/main-upstream-catchup-20260613`
