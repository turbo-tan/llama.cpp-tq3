# Speculative Decode Benchmark Matrix

Date: 2026-04-14

## Purpose

Keep speculative-decode claims split into explicit contracts.

Do not mix:

- fast speed-oriented contract
- safe / conservative contract
- exact-output validation contract

## Shared Baseline

- Branch: `experiment/persistent-decode`
- Binary: `/home/awee/code/llama.cpp/build-360/bin/llama-lossy-ngram`
- Model: `/home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf`
- Cache: `/home/awee/code/tan_llama/artifacts/ngram_caches/ngram_decode.bin`
- GPU: RTX 5060 Ti 16GB

## Contract A: Fast

Goal:
- maximize decode speed

Settings:
- `-c 2048`
- `-n 128`
- `--temp 0`
- `--top-k 1`
- `MIN_MARGIN=0.0`

Question:
- can we reproduce the original ~`+37%` speedup?

Gate:
- speed only until exactness is checked separately

Measured recheck on 2026-04-14:

- prompt set: `/home/awee/code/tan_llama/artifacts/spec_prompts_breakthrough_10.json`
- cache: `ngram_decode.bin`
- mean baseline: `21.31 t/s`
- mean speculative: `26.58 t/s`
- speedup: `1.247x`
- exact matches: `4 / 10`

Interpretation:
- the fast contract is a real speed win
- it is not a strict exact-output contract

## Contract B: Safe

Goal:
- conservative speculation

Settings:
- `-c 4096`
- `-n 64`
- `--temp 0`
- `--seed 123`
- `MIN_MARGIN=4.0`

Question:
- what speedup survives under tighter gating?

Gate:
- speed + exact-output check

Measured recheck on 2026-04-14:

- prompt set: `/home/awee/code/tan_llama/artifacts/spec_prompts_breakthrough_10.json`
- cache: `ngram_decode.bin`
- mean baseline: `21.28 t/s`
- mean speculative: `23.07 t/s`
- speedup: `1.084x`
- exact matches: `9 / 10`

Interpretation:
- the safe contract retains a smaller but real speed win
- it still does not meet a strict `100% exact` gate on the current checked state

## Contract C: Exactness

Goal:
- prove output equality

Settings:
- deterministic decoding only
- compare baseline vs speculative on the same prompt set
- exact string match unless otherwise stated

Question:
- does the seq_cp restore path preserve exact output?

Current answer on the checked state:

- not under the fast contract
- not yet under the safe contract either
- exactness still needs more work before any “zero quality loss” headline is reused

## Reporting Rule

Every speculative result must state:

1. contract (`fast`, `safe`, or `exactness`)
2. model
3. cache source
4. prompt set
5. whether the claim is:
   - speed only
   - exact output
   - semantically correct but not byte-identical

## Contract D: Fast Exact

Goal:
- keep a meaningful fast-contract speedup
- also require strict exact output

Settings:
- `-c 2048`
- `-n 128`
- `--temp 0`
- `--seed 123`
- `--top-k 1`
- `MIN_MARGIN=1.0`
- `4-gram only`
- `dynamic speculation off`
- GPU full shadow buffer enabled

Measured on 2026-04-14:

- prompt set: `/home/awee/code/tan_llama/artifacts/spec_prompts_breakthrough_10.json`
- mean baseline: `21.734 t/s`
- mean speculative: `26.942 t/s`
- speedup: `1.240x`
- exact matches: `10 / 10`

Interpretation:
- this is the current production-quality speculative contract
- it gives a real speed win without reusing the old non-exact fast headline

Revalidated recovered branch on 2026-04-20:

- branch: `experiment/specdecode-next-v2-20260419`
- worktree: `/home/awee/code/worktrees/llama-specdecode-next`
- base: `experiment/persistent-decode` @ `b644bf478`
- fix: `LOSSY_NGRAM_BATCH` default `384` replaces hardcoded `512` to keep room for the 299 MB GPU shadow buffer
- prompt set: `/home/awee/code/tan_llama/artifacts/spec_prompts_breakthrough_10.json`
- mean baseline: `21.2203 t/s`
- mean speculative: `24.7688 t/s`
- speedup: `1.1672x`
- exact matches: `10 / 10`
- artifact: `/home/awee/code/tan_llama/artifacts/spec_exact_fast_contract_recheck_v3_20260419.json`
- artifact: `/home/awee/code/tan_llama/artifacts/specdecode_recovery_batch384_20260419.txt`

Interpretation:
- the recovered branch is correct and stable under the 16GB VRAM budget
- the historical `26.942 t/s` exact-contract mean is not recovered yet
- next speed work should focus on cache/gating policy rather than shadow-memory survival
