# TurboQuant Documentation

This folder records the TurboQuant work around custom GGUF low-bit formats,
runtime kernels, quality gates, and model release procedures.

The current production-facing format is `TQ3_4S`. `TQ3_1S` is retained as an
important historical and fit-first format, but most current work targets
`TQ3_4S` quality and runtime speed.

## Start Here

| Page | Purpose |
|---|---|
| [INDEX.md](INDEX.md) | Agent and human navigation index for TurboQuant docs |
| [papers/weight-compression-tq3.md](papers/weight-compression-tq3.md) | Public-style explanation of the TQ3 weight compression family |
| [journey/tq3_1s_to_tq3_4s.md](journey/tq3_1s_to_tq3_4s.md) | Development history from `TQ3_1S` to `TQ3_4S` |
| [active/README.md](active/README.md) | Current work only: active plans, decode/KV research, speculative-decode status |
| [proven/README.md](proven/README.md) | Validated results and release-facing docs only |
| [procedures/README.md](procedures/README.md) | SOPs, checklists, benchmark protocol, and upload steps |
| [../steering/rebase-sop.md](../steering/rebase-sop.md) | Canonical upstream/master -> master -> main sync workflow |

## Current Position

`TQ3_4S` is a custom low-bit GGUF weight format for the TurboQuant runtime fork.
It is intended to sit between aggressive `3`-bit-style compression and common
`4`-bit GGUF formats:

- much smaller than `Q8_0` and `BF16`
- closer to source quality than the earlier `TQ3_1S`
- CUDA runtime support for prompt and token-generation paths
- designed for real model releases, not just isolated kernels

The runtime is not stock `llama.cpp`. Models using `TQ3_*` tensor types require
the TurboQuant-enabled fork:

`https://github.com/turbo-tan/llama.cpp-tq3`

## Documentation Layout

| Folder | Contents |
|---|---|
| `active/` | Current work in progress, active plans, and research that is still live |
| `proven/` | Artifact-backed results, validated dashboards, and release-facing summaries |
| `procedures/` | SOPs, checklists, and repeatable benchmark/release workflows |
| `papers/` | More polished technical explanations suitable for public references |
| `journey/` | Chronological engineering notes and lessons learned |
| `generated/` | Generated tensor-type policies and model-specific policy files |
| `archive/` | Superseded plans, handovers, parked tracks, and historical notes |

## Engineering Rule

Every speed or quality claim should point to an artifact under `artifacts/` and
should name:

- model
- runtime build or commit
- command or benchmark type
- date
- exact result

Microbenchmarks are useful for direction, but they do not replace end-to-end
perplexity, chat, and `llama-bench` validation.
