# TurboQuant TQ3_4S — Documentation Index

If you are resuming a TurboQuant task, start here and then follow the linked
SOPs below. This is the navigation page for agents and humans alike.

Canonical roots:

- Repo root: `/home/awee/code/tan_llama`
- Primary branch build: `/home/awee/code/tan_llama/build-current`
- Legacy branch build: `/home/awee/code/tan_llama/build`
- llm-launch root: `/home/awee/code/llm-launch`
- Shared models: `/home/awee/models/turboquant`

For the cleaner project entrypoint, also see [README.md](README.md).

## Entry Points

| Folder / Doc | Purpose |
|---|---|
| [active/README.md](active/README.md) | Live plans and current research |
| [proven/README.md](proven/README.md) | Validated results and release-facing docs |
| [procedures/README.md](procedures/README.md) | SOPs and checklists |
| [papers/weight-compression-tq3.md](papers/weight-compression-tq3.md) | Technical overview of the TQ3 compression family |
| [journey/tq3_1s_to_tq3_4s.md](journey/tq3_1s_to_tq3_4s.md) | Narrative history from `TQ3_1S` to `TQ3_4S` |

## Active

| Doc | Purpose |
|---|---|
| [active/ACTIVE_PROGRESS_PLAN.md](active/ACTIVE_PROGRESS_PLAN.md) | Single active execution checklist |
| [active/TQ3_MASTER_PLAN.md](active/TQ3_MASTER_PLAN.md) | Current roadmap and guardrails |
| [active/SPEC_DECODE_STATUS.md](active/SPEC_DECODE_STATUS.md) | Current speculative-decode status |
| [active/DECODE_SPEED_PLAN_X_ANALYSIS.md](active/DECODE_SPEED_PLAN_X_ANALYSIS.md) | Plan X analysis and pending validation |
| [active/TQ3_4F_FASTROT_PLAN.md](active/TQ3_4F_FASTROT_PLAN.md) | Fast-rotation `TQ3_4F` experiment |
| [active/FP4_KV_PLAN.md](active/FP4_KV_PLAN.md) | FP4 / HiFloat4-style KV exploration |

## Proven

| Doc | Purpose |
|---|---|
| [proven/TQ3_4S_PERFORMANCE_DASHBOARD.md](proven/TQ3_4S_PERFORMANCE_DASHBOARD.md) | Current validated speed/quality dashboard |
| [proven/BENCHMARK_RESULTS.md](proven/BENCHMARK_RESULTS.md) | Full PPL comparison tables |
| [proven/TQ3_PP_ANALYSIS.md](proven/TQ3_PP_ANALYSIS.md) | PP bottleneck analysis and failed paths |
| [proven/TQ3_PP_BREAKTHROUGH.md](proven/TQ3_PP_BREAKTHROUGH.md) | Revalidated PP breakthrough |
| [proven/TQ3_RUNTIME_AND_RESULTS.md](proven/TQ3_RUNTIME_AND_RESULTS.md) | Public-facing runtime explanation |

## Procedures

| Doc | Purpose |
|---|---|
| [procedures/BENCHMARK_PROTOCOL.md](procedures/BENCHMARK_PROTOCOL.md) | Benchmark protocol |
| [procedures/TEST_SUITE.md](procedures/TEST_SUITE.md) | Minimum guardrail test suite |
| [procedures/CHAT_TEST_SUITE.md](procedures/CHAT_TEST_SUITE.md) | Chat quality gate |
| [procedures/LLAMA_SIMPLE_CHAT_TEST_SOP.md](procedures/LLAMA_SIMPLE_CHAT_TEST_SOP.md) | Chat smoke test procedure |
| [procedures/HF_UPLOAD_SOP.md](procedures/HF_UPLOAD_SOP.md) | HuggingFace upload procedure |
| [procedures/NEW_TYPE_CHECKLIST.md](procedures/NEW_TYPE_CHECKLIST.md) | New quant-type checklist |
| [procedures/TQ3_4S_QUANTIZE_SOP.md](procedures/TQ3_4S_QUANTIZE_SOP.md) | TQ3_4S and outQ6K quantization SOP |
| [../procedure/eval-harness-sop.md](../procedure/eval-harness-sop.md) | Canonical BenchLoop and harness SOP |
| [benchloop-sop.md](benchloop-sop.md) | Detailed benchmark gate and runtime shape |
| [upstream-sync-checklist-20260613.md](upstream-sync-checklist-20260613.md) | Current sync checklist and recovery guardrails |
| [toolcall-recovery-plan-20260612.md](toolcall-recovery-plan-20260612.md) | Recovery plan and prompt-parity notes |

## Archive

Historical and parked docs are in `archive/`, including:
- `archive/handovers/` — dated handovers and one-off session state
- `archive/research-snapshots/` — exactness/debug snapshots kept for reference
- `archive/parked/` — ideas kept out of the active plan
- `archive/community/` — public/community draft material
- `archive/marlin-legacy/` — 10 docs from abandoned Marlin fused kernel approach
- `archive/exl-legacy/` — 5 docs from abandoned EXL2/EXL3 exploration
- `archive/exploration-legacy/` — format design rationale, math proofs, parked ideas
- `archive/superseded/` — speed plans/analyses replaced by `proven/TQ3_PP_ANALYSIS.md`
- `archive/moonshot-legacy/` — early speed optimization attempts
- `archive/session-legacy/` — session handover notes
- `archive/strategy-legacy/` — early strategy docs
