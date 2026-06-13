# Active Progress Plan (2026-04-20)

This is the single current execution plan for TurboQuant work.

It replaces scattered "next steps" notes across handovers. The current focus is:

1. speculative decode on the hybrid SSM path
2. Decode Speed Plan X validation and follow-up

EAGLE work is explicitly out of the active plan for now because it requires extra model and artifact storage that is not justified by the current constraints.

## Scope Rules

- Keep active work concentrated on branches that already have working code or measured wins.
- Do not spend time on EAGLE or draft-head storage-heavy tracks unless priorities change.
- Treat build-only success as incomplete until benchmark, correctness, and artifact gates are done.

## Current Branches In Scope

- `experiment/specdecode-next-v2-20260419` — recovered speculative decode branch with GPU shadow buffer
- `experiment/decode-speed-plan-x` — sparse V dequant + TQ3 AMD nwarps whitelist
- `feature/tq3-4s-quantize-gemma4` — lower priority, keep parked unless it directly helps the two main tracks

## Master Checklist

### A. Speculative Decode

- [x] GPU shadow save/restore implemented
- [x] N-gram speculation path implemented
- [x] 10-prompt benchmark run with positive speedup
- [x] Correctness checked: 10/10 answers correct, 8/10 token-exact
- [x] Known limitation documented: `pp2 != tg1+tg1` on hybrid SSM
- [x] Recover buildable spec-decode branch and re-run exact contract with saved artifacts
  - Working repo: `/home/awee/code/worktrees/llama-specdecode-next`
  - Branch: `experiment/specdecode-next-v2-20260419`
  - Base: `experiment/persistent-decode` at `b644bf478`
  - Fix: `lossy-ngram.cpp` now uses `LOSSY_NGRAM_BATCH`, default `384`, for `n_batch`, `n_ubatch`, and `llama_batch_init`
  - Root cause fixed: hardcoded `512` left too little VRAM for the 299 MB GPU shadow buffer on RTX 5060 Ti 16GB
  - `LOSSY_NGRAM_BATCH=512` remains the OOM boundary on this card
  - Full 10-prompt exact contract at B=384: baseline `21.2203 t/s`, spec `24.7688 t/s`, speedup `1.1672x`, exact `10/10`
  - Hard prompt single-case sweep showed B=128-448 all fit, B=512 OOM; B=384 was the best safe high-water mark in that prompt sweep
  - Artifacts: `artifacts/spec_exact_fast_contract_recheck_v3_20260419.json`, `artifacts/specdecode_recovery_batch384_20260419.txt`
  - Important: this restores correctness and VRAM stability, but does not recover the older `26.942 t/s` exact-contract mean
- [ ] Re-run chat smoke and code-style probe on current branch tip
- [ ] Decide whether current drift is acceptable for shipping or needs more mitigation
- [ ] Write one short "ship/no-ship" summary with exact commands, commit, and artifacts

### B. Decode Speed Plan X

- [x] Sparse V dequant implemented
- [x] Sparse V dequant build-verified
- [x] TQ3 AMD `nwarps` whitelist implemented
- [x] TQ3 AMD `nwarps` whitelist build-verified
- [x] Block-32-to-128 plan_x claim rejected as incorrect
- [x] Correct replacement direction documented: new `GGML_TYPE_TQ3_0_B128`
- [ ] GPU benchmark sparse V dequant at `4K`, `8K`, `16K`, `32K`
- [ ] GPU verify whether sparse V dequant preserves PPL within noise at practical threshold(s)
- [ ] Save benchmark and PPL artifacts with model, commit, date, and threshold in filename
- [ ] Decide whether sparse V dequant is ready to keep enabled as an experimental option
- [ ] Benchmark AMD `nwarps` whitelist on actual RDNA3/RDNA4 hardware, if available

### C. Plan X Follow-Up Design

- [ ] Confirm whether Enhancement 4 (`TURBO_LAYER_ADAPTIVE` / boundary V) is worth validating now
- [ ] Design `GGML_TYPE_TQ3_0_B128` as a new type rather than changing `QK_TQ3_0`
- [ ] Decide whether Turbo4 V-cache work is worth starting before block-128 type groundwork exists
- [ ] Keep Enhancements 1/2/5 parked until A and B have benchmark-backed outcomes

## Execution Order

1. Continue spec-decode speed search from the recovered exact branch.
2. Compare runtime-local cache, `universal_v1`, and combined cache policy.
3. Run chat/code probes before any ship/no-ship decision.
4. GPU-benchmark sparse V dequant after the spec-decode branch has a clear speed verdict.

## Execution Sheet

Use these as the default commands and artifact names. Replace placeholders before running.

### A. Speculative Decode Revalidation

Working repo:
- `/home/awee/code/worktrees/llama-specdecode-next`

Suggested artifact directory:
- `/home/awee/code/tan_llama/artifacts/spec_decode`

Suggested artifact stem:
- `specdecode_qwen35_27b_<branch>_<commit>_<date>`

Checklist:
- [ ] Record branch and commit before running
- [ ] Build required binaries on `experiment/persistent-decode-cow-shadow`
- [ ] Run benchmark set used for the shipping claim
- [ ] Run chat smoke and code-style probe
- [ ] Save raw outputs under `artifacts/spec_decode/`

Suggested commands:

```bash
git -C /home/awee/code/llama.cpp branch --show-current
git -C /home/awee/code/llama.cpp rev-parse --short HEAD
cmake --build /home/awee/code/llama.cpp/build --target lossy-ngram pp2-vs-tg1 llama-simple-chat llama-server -j8
```

Save benchmark output:

```bash
/home/awee/code/llama.cpp/build/bin/lossy-ngram ... \
  2>&1 | tee /home/awee/code/tan_llama/artifacts/spec_decode/specdecode_qwen35_27b_<branch>_<commit>_<date>_bench.txt
```

Save divergence diagnostic:

```bash
/home/awee/code/llama.cpp/build/bin/pp2-vs-tg1 ... \
  2>&1 | tee /home/awee/code/tan_llama/artifacts/spec_decode/specdecode_qwen35_27b_<branch>_<commit>_<date>_pp2_vs_tg1.txt
```

Save chat smoke summary:

```bash
/home/awee/code/tan_llama/scripts/chat_test_suite.sh ... \
  2>&1 | tee /home/awee/code/tan_llama/artifacts/spec_decode/specdecode_qwen35_27b_<branch>_<commit>_<date>_chat.txt
```

Ship/no-ship note target:
- `docs/turboquant/active/SPEC_DECODE_STATUS.md`

### B. Sparse V Dequant GPU Validation

Working repo:
- `/home/awee/code/llama.cpp`

Suggested artifact directory:
- `/home/awee/code/tan_llama/artifacts/plan_x`

Suggested artifact stem:
- `planx_sparsev_qwen35_27b_<branch>_<commit>_<date>`

Thresholds to test first:
- `0`
- `1e-5`
- `1e-4`

Contexts to test first:
- `4096`
- `8192`
- `16384`
- `32768`

Checklist:
- [ ] Record branch and commit before running
- [ ] Build `llama-bench` and `llama-perplexity`
- [ ] Run warmed-up TG benchmark at each context and threshold
- [ ] Run PPL witness at practical threshold(s)
- [ ] Save raw outputs under `artifacts/plan_x/`

Suggested commands:

```bash
git -C /home/awee/code/llama.cpp branch --show-current
git -C /home/awee/code/llama.cpp rev-parse --short HEAD
cmake --build /home/awee/code/llama.cpp/build --target llama-bench llama-perplexity -j8
```

Warmup:

```bash
GGML_FATTN_SPARSE_V_THRESHOLD=0 \
/home/awee/code/llama.cpp/build/bin/llama-bench -m MODEL.gguf -ngl 99 -fa 1 -p 128 -n 0 -r 1
```

Per-context TG benchmark:

```bash
GGML_FATTN_SPARSE_V_THRESHOLD=<threshold> \
/home/awee/code/llama.cpp/build/bin/llama-bench \
  -m MODEL.gguf -ngl 99 -fa 1 -p <context> -n 128 -r 3 \
  2>&1 | tee /home/awee/code/tan_llama/artifacts/plan_x/planx_sparsev_qwen35_27b_<branch>_<commit>_<date>_ctx<context>_thr<threshold>_bench.txt
```

PPL witness:

```bash
GGML_FATTN_SPARSE_V_THRESHOLD=<threshold> \
/home/awee/code/llama.cpp/build/bin/llama-perplexity \
  -m MODEL.gguf -ngl 99 -fa 1 -t 8 -c 512 --no-warmup \
  -f /home/awee/code/llama.cpp/wikitext-2-raw/wiki.test.raw --chunks 100 \
  2>&1 | tee /home/awee/code/tan_llama/artifacts/plan_x/planx_sparsev_qwen35_27b_<branch>_<commit>_<date>_thr<threshold>_ppl100ch.txt
```

Conclusion targets:
- `progress-freebuff.md`
- `docs/turboquant/active/DECODE_SPEED_PLAN_X_ANALYSIS.md`

## Definition of "On Track"

We are on track only if each active item ends with:

- a build result
- a correctness result
- a performance result
- a saved artifact
- a short conclusion in docs

Anything short of that is still in-progress.
