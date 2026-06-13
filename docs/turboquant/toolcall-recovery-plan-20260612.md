# Toolcall Recovery Plan - 2026-06-12

Superseded as the main execution checklist by:

- `docs/turboquant/upstream-sync-checklist-20260613.md`

This document records the active recovery plan for the Qwen3.6 27B out6k regression work so another engineer can pick it up directly.

## Scope

Primary target:

- Restore winner-grade behavior for the flagship out6k model on:
  - `toolcall`
  - `coding`

Secondary target:

- Keep the recovered MTP runtime shape intact while fixing the prompt-path regression.

Out of scope for this pass:

- Full BenchLoop
- Publication runs
- Wide refactors of chat/template code

## Working assumptions

- Current repo: `/home/awee/code/tan_llama`
- Current branch at handoff time: `main`
- Flagship model:
  - `/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf`
- Published template override:
  - `/home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja`
- Runtime SOP:
  - `docs/turboquant/out6k-benchloop-sop.md`

## Problem statement

Two distinct regressions were mixed together and must stay separated:

1. MTP runtime regression
2. Toolcall/coding behavior regression

The current MTP runtime appears largely recovered enough for narrow testing, but `toolcall` still looks wrong and prior investigation indicates the drift is upstream of MTP:

- disabling speculative decoding did not fix the bad toolcall case
- current prompt/rendering appears to differ from the historical winner path
- there is evidence of a 2-token prompt drift in the failing toolcall case

That means the next step is not random MTP tuning. The next step is prompt-path parity work.

## Known references

Use these first before doing new speculation:

- MTP restore handover:
  - `docs/turboquant/mtp-restore-handover-20260612.md`
- MTP expert review:
  - `docs/turboquant/mtp-expert-review-20260612.md`
- Out6k runtime SOP:
  - `docs/turboquant/out6k-benchloop-sop.md`
- Investigation notes mentioned in chat history:
  - branch `fix/toolcall-2token-prompt-drift`
  - commits:
    - `e71bc3307`
    - `8f2f2c112`

Historical winner markers found in local history:

- `main-b9267-c87bcb3`
- `main-b9268-25c3719`

Relevant upstream chat/prompt change window:

- `98d5e8ba8` common/chat: fix LFM2/LFM2.5 reasoning round-trip and `<think>` leak
- `d2462f8f7` chat: fix LFM2/LFM2.5 ignoring json_schema
- `1e912561d` server: log prompts to directory
- `0d250f6a0` input_tokens API
- `f5c6ae182` token counting changes

## Plan

### Phase 1: Recreate the baseline precisely

Goal:

- establish a known-good prompt/rendering baseline before editing code

Checklist:

- [ ] Identify the exact winner-era source state used for the passing out6k/toolcall run
- [ ] Confirm whether the winner used embedded GGUF template or override template
- [ ] Confirm whether reasoning was forced off in the passing run
- [ ] Confirm whether the passing run used speculative decoding on or off for toolcall
- [ ] Record exact binary, model, template, and server flags used by the winner

Preferred evidence sources:

- git tags/branches around `main-b9268-25c3719`
- existing benchmark artifacts under `artifacts/`
- prior SOP notes

Do not proceed to code edits until this phase has a written answer.

### Phase 2: Separate template-content drift from renderer drift

Goal:

- determine whether the regression is caused by:
  - changed template text
  - changed template application/rendering logic
  - changed server prompt assembly around the template

Checklist:

- [ ] Diff current published override template against historical template if available
- [ ] Compare current embedded model template behavior vs override template behavior
- [ ] Use a minimal prompt/apply-template probe to render the same toolcall input across:
  - winner-era source
  - current source
  - embedded template
  - override template
- [ ] Identify the exact byte/token difference, not just “looks different”

Success condition:

- a concrete divergence is captured, preferably with rendered prompt text and token counts

### Phase 3: Patch the smallest prompt-path regression

Goal:

- fix only the smallest identified drift first

Constraints:

- preserve the recovered MTP runtime shape
- avoid broad chat/template refactors
- prefer upstream behavior unless local history clearly documents a required deviation

Checklist:

- [ ] Make one focused code change in chat/server/template application
- [ ] Keep the patch small enough to revert independently
- [ ] Commit it separately from docs or benchmark changes

Likely files:

- `common/chat.cpp`
- `common/chat-diff-analyzer.cpp`
- `common/common.cpp`
- `common/arg.cpp`
- `src/llama-chat.cpp`
- `tools/server/*`

### Phase 4: Rebuild the intended binary

Goal:

- validate the exact CUDA server shape used for out6k checks

Build command required by user:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_PEER_MAX_BATCH_SIZE=512 -DGGML_NATIVE=ON -DGGML_CUDA_GRAPHS=ON
cmake --build build -j
```

Checklist:

- [ ] Build `llama-server`
- [ ] Build any helper binary needed for prompt/token comparison
- [ ] Confirm the rebuilt binary is the one being tested

### Phase 5: Narrow validation only

Goal:

- validate the fix without wasting cycles on full-suite runs

Required order:

1. direct dry run
2. toolcall check
3. coding check

Checklist:

- [ ] Run a dry run first and confirm speculative drafting is active
- [ ] Stop immediately if `draft_n == 0`
- [ ] Run the narrow toolcall repro
- [ ] Run the out6k coding gate
- [ ] Only after those pass, consider a larger `partial` run

Use the local-only rule:

```bash
BENCHLOOP_NO_SUBMIT=1
```

Do not publish BenchLoop runs remotely without explicit user approval.

## Required runtime shape

Use the current documented out6k shape unless the baseline phase proves the winner used something else:

```bash
./build/bin/llama-server \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --alias Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --host 127.0.0.1 --port 18124 \
  -c 262144 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 \
  --spec-type draft-mtp \
  --spec-draft-n-min 1 \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.0 \
  --reasoning off
```

Template rule:

- default to embedded template for the recovery run
- add `--chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja` only when explicitly testing template differences

## Decision tree

If prompt drift reproduces with speculative decoding off:

- the bug is not MTP-specific
- continue in chat/template/server prompt assembly

If prompt drift disappears with historical source but same template:

- the bug is in renderer/server code, not the template text

If prompt drift follows the template file across both old and new source:

- the template content itself regressed

If toolcall is fixed but coding still misses:

- treat coding as a separate evaluation problem after confirming prompt parity

If MTP speed drops again while prompt parity is fixed:

- return to the MTP handover and keep that work isolated from toolcall fixes

## Deliverables for each handoff

Before pausing, update this document or the handover note with:

- current branch
- current HEAD commit
- exact server command used
- exact model path used
- whether embedded or override template was used
- whether speculative decoding was on or off
- observed result for:
  - toolcall
  - coding
- next recommended action

## Current status

Completed:

- [x] Collected the relevant June 6-10 commit window
- [x] Located the out6k SOP and published template override
- [x] Confirmed the current published template file exists
- [x] Confirmed the last-known winner tags are present locally
- [x] Confirmed parser-side empty `<think></think>` cleanup already exists on current `main`
- [x] Identified the remaining prompt-shape drift in `common/chat.cpp` on the autoparser path
- [x] Restored the historical Qwen off-thinking assistant prefill blank line in `common/chat.cpp`

Implemented fix:

- File:
  - `common/chat.cpp`
- Change:
  - when autoparser detects Qwen-style thinking support and the generation prompt ends with `<think>\n`, restore the historical winner-era blank line so the assistant prefill becomes:

```text
<|im_start|>assistant
<think>

</think>

```

Reason:

- current parser cleanup was already stripping empty leading `<think></think>` blocks from model output parsing
- the missing piece was prompt generation parity
- the older winner-era toolcall baseline depended on the extra blank line after `<think>`

Validation run:

1. C++ parser test:

```bash
cmake --build /home/awee/code/tan_llama/build-test --target test-chat-peg-parser -j
/home/awee/code/tan_llama/build-test/bin/test-chat-peg-parser
```

Result:

- passed

2. Direct server prompt-shape probe:

```bash
/home/awee/code/tan_llama/build-test/bin/llama-server \
  --model /home/awee/models/tinyllama-1.1b-q4_0.gguf \
  --host 127.0.0.1 --port 19080 \
  --no-webui --jinja --reasoning off \
  --chat-template-file /home/awee/code/tan_llama/models/templates/Qwen-Qwen3-0.6B.jinja

curl -s http://127.0.0.1:19080/apply-template \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is today?"}],"tools":[]}'
```

Observed response:

```json
{"prompt":"<|im_start|>user\nWhat is today?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"}
```

This matches the expected historical Qwen off-thinking prompt suffix.

Remaining work:

- [x] Re-run the exact tc-11 toolcall probe against current source
- [x] Confirm prompt token count returns to a winner-grade direct-answer behavior on the corrected template path
- [x] Run out6k narrow validation for `toolcall`
- [x] Run out6k narrow validation for `coding`

Next concrete action:

- resolve the remaining template tradeoff between `tc11` and `coding-csv-parser` before claiming full quality recovery

## 2026-06-13 validation update

Server shape used for valid reruns:

```bash
./build-current/bin/llama-server \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --alias Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --host 127.0.0.1 --port 18124 \
  -c 262144 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 \
  --cache-ram 0 \
  --spec-type draft-mtp \
  --spec-draft-n-min 1 \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.0 \
  --chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja \
  --reasoning off --no-warmup
```

Important correction:

- The earlier `toolcall,coding` BenchLoop run that finished in about 2.2s with empty outputs was not a model-quality result.
- It was a bad runtime state caused by server/OOM failure; BenchLoop swallowed the provider error and scored empty outputs.
- The valid rerun on the stable no-cache server is the authoritative result.

Measured results on the stable rerun:

- BenchLoop narrow rerun:
  - `toolcall 96.7 14/15`
  - `coding 93.8 11/12`
  - run file:
    - `/home/awee/.bench-loop/runs/20260613-114951-Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf-local-openai_compat/run.json`
- Direct `tc11` repro on the same server:
  - direct answer `30`
  - `predicted_per_second ~= 53.82 tok/s`
- Direct coding repro for `coding-csv-parser` on the same server:
  - still fails
  - `predicted_per_second ~= 51.55 tok/s`

Template provenance findings:

- Current local publish template:
  - `/home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja`
  - untracked in the current worktree
  - `template_version = "qwen3.6-froggeric-v20"`
- Last committed publish template:
  - `git show 25b98f6be:publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja`
  - 154-line copy of `models/templates/Qwen3.5-4B.jinja` plus the tool-response guard fix
- Embedded GGUF template is not equivalent to the local publish template for this artifact.

Template tradeoff now proven:

1. Current untracked `froggeric-v20` publish template:
   - fixes `tc11`
   - restores `toolcall` to `14/15`
   - still misses `coding-csv-parser`, leaving `coding` at `11/12`

2. Last committed older publish template from `25b98f6be`:
   - improves `coding-csv-parser`
   - regresses `tc11` back to an unnecessary `calculator` tool call

3. `--spec-draft-p-min 1.0` on current `main`:
   - does not recover `coding-csv-parser`
   - does not break MTP on the direct probes
   - is not the missing fix for the final coding gap

Conclusion:

- MTP runtime is healthy enough for validation on the corrected server shape.
- Toolcall recovery is achieved on the publish-template override path.
- Full quality recovery is still blocked by a template-content tradeoff:
  - old template is better for at least one coding task
  - current untracked template is better for `tc11`
- The next useful task is to reconstruct or synthesize the intermediate "winner" template variant, likely from the `numtyped` / `tc11fix` lineage in the local artifacts.

## 2026-06-13 template verdict

The exact template family has now been reproduced on the current `build-current` binary with the out6k model.

Use this runtime shape for template isolation:

```bash
./build-current/bin/llama-server \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --alias Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --host 127.0.0.1 --port 18124 \
  -c 32768 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 \
  --cache-ram 0 \
  --spec-type draft-mtp \
  --spec-draft-n-min 1 \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.0 \
  --chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja \
  --reasoning off --no-warmup
```

Why `-c 32768`:

- `-c 262144` produced a CUDA OOM during template A/B testing on 2026-06-13.
- That failure was not a quality result.
- `-c 32768` matches the historical winner shape and yields valid quality measurements.

Verified results with the promoted publish template:

- `toolcall 96.7 14/15`
  - run: `/home/awee/.bench-loop/runs/20260613-123133-Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf-local-openai_compat/run.json`
- `coding 100.0 12/12`
  - run: `/home/awee/.bench-loop/runs/20260613-123229-Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf-local-openai_compat/run.json`
- combined local full suite on the same recovered publish template:
  - run: `/home/awee/.bench-loop/runs/20260613-131757-Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf-local-openai_compat/run.json`
  - `speed 68.6 9/9`
  - `toolcall 96.7 14/15`
  - `coding 100.0 12/12`
  - `dataextract 91.0 12/15`
  - `instructfollow 74.5 9/15`
  - `reasonmath 73.3 11/15`
  - `GEN TOK/S 42.73`

Long-context status from this full run:

- The old template-collapse behavior does not appear in the recovered publish-template path.
- This does not by itself prove `262144`-context behavior is fully fixed.
- It does prove the recovered template is stable on the validated `32768` quality gate and no longer the blocker for out6k recovery.

The working publish template lineage is:

1. old stock-derived publish template from `25b98f6be`
2. add scoped JSON-only numeric rule:
   - "If and only if your output is JSON ..."
3. add direct-answer tool-use rules from:
   - `/home/awee/code/tan_llama/artifacts/tc11-prompt-patched.txt`

This means the template is no longer the blocker.
Any remaining difference after this point is in code or runtime shape, not template selection.
