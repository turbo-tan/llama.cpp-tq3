# Jacobi Decode Release Checklist

**Branch:** `experiment/tq3-decode-next`

## Release Goal

Ship Jacobi decode as an **experimental** feature first.

Do **not** ship as default-on until:
- correctness is stable
- quality is acceptable
- speed wins hold on a representative prompt mix

## Current Status

What is already proven:
- `llama-jacobi` works on qwen35 hybrid models
- strong continuation / repetition prompts can give large decode wins
- full-accept windows are now handled efficiently without redundant second decode

What is not yet proven:
- broad real-world prompt win
- acceptable behavior on open-ended generation
- 27B validation

## Release Shape

First release should be:
- new example / experimental binary
- explicit opt-in
- documented as best for repetitive or strongly structured continuations

Do not present it as a universal decode accelerator yet.

## Required Test Areas

### 1. Correctness

Pass criteria:
- no crashes
- no KV / seq position errors
- no malformed decode loops
- EOS / stop handling matches baseline

Required checks:
- short smoke on qwen35 4B
- short smoke on qwen35 9B TQ3_4S
- short smoke on qwen35 27B TQ3_4S
- 100+ generated tokens per model on at least one safe prompt

### 2. Quality

Pass criteria:
- no obvious gibberish
- no obvious degradation on structured prompts
- no systematic failure on code / QA prompts

Required prompt classes:
- repetition / templated continuation
- structured factual prompt
- code generation prompt
- open-ended prose/chat prompt

Required outputs to review:
- baseline output
- Jacobi output
- acceptance rate
- speculative windows used

### 3. Speed

Pass criteria:
- clear win on repetition / strong continuation prompts
- no catastrophic regression on generic prompts
- fallback behavior works when speculation is not useful

Required metrics:
- decode tokens/s
- `n_windows`
- `n_accept`
- acceptance %
- speculative threshold used

## Required Models

### Minimum
- qwen35 4B BF16
- qwen35 9B TQ3_4S
- qwen35 27B TQ3_4S

### Nice to have
- one 35B qwen35-family model after the 27B path is stable

## Required Prompt Set

### P1: Strong repeat
Purpose:
- prove best-case speculative upside

### P2: Structured factual continuation
Purpose:
- test medium-strength local continuation

### P3: Code prompt
Purpose:
- test whether local syntax repetition is enough for useful acceptance

### P4: Open-ended chat/prose
Purpose:
- ensure fallback behavior is safe

## Required Baselines

For every test:
- same model
- same prompt
- same token count
- same binary tree

Compare:
- `llama-jacobi` with speculation effectively disabled
- `llama-jacobi` active policy

This is important because it isolates the speculative policy from unrelated binary/runtime differences.

## Current Known Good Policy

Current promising policy:
- one-pass verify
- no redundant second decode on full accept
- `draft-max = 8`
- confidence gate via `LLAMA_JACOBI_MIN_MARGIN`

Important:
- this is still a tuning point, not final release policy

## Must-Have Artifacts Before Release

- 4B correctness artifact
- 9B correctness artifact
- 27B correctness artifact
- quality comparison artifacts for P1-P4
- speed comparison artifacts for P1-P4
- one summary table with:
  - model
  - prompt class
  - baseline t/s
  - Jacobi t/s
  - speedup
  - acceptance %
  - qualitative verdict

## Release Decision Rules

### Release as experimental
Allowed if:
- all correctness checks pass
- quality is acceptable on reviewed prompts
- strong prompts show clear wins
- generic prompts fall back safely without major regressions

### Do not release yet
If any of these happen:
- qwen35 sequence-position failures
- crashes under normal use
- obvious gibberish on reviewed prompts
- broad regressions on generic prompts

## Next Immediate Work

1. Run the checklist on 4B, 9B, 27B
2. Record artifacts
3. Decide whether experimental release is justified
4. Only then consider commit / branch promotion
