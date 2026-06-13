# GGML Enum Reservation — Notes

**Date**: 2026-04-06  
**Status**: Planning, not yet submitted

## Problem

Upstream `ggml.h` has `GGML_TYPE_COUNT = 41`. Our TQ3_4S uses slot 46.
Next upstream type addition will collide with our TQ3_0 at 41.

## Our types

| Type | Slot | In GGUF? | Status |
|---|---|---|---|
| TURBO2_0 | 31 | No (KV cache only) | Can renumber |
| TURBO3_0 | 32 | No (KV cache only) | Can renumber |
| TURBO4_0 | 33 | No (KV cache only) | Can renumber |
| TQ3_0 | 41 | No (KV cache only) | Can renumber |
| TQ3_1S | 44 | Yes but dead | Quality 0.15 PPL worse than TQ3_4S at same size; speed regression claim was false alarm (see investigation 2026-04-17) |
| TQ3_4S | 46 | Yes, published | **Only type that matters** |

## Plan

- TQ3_1S is dead — quality is 0.15 PPL worse than TQ3_4S at identical size/bpw, no reason to keep
  - **Note**: A reported "40% speed regression" after the TQ3_4S quantizer commit was investigated and found to be **false** — caused by cold-start measurement without warmup. Clean rebuild shows TQ3_1S is actually faster than the reported baseline. The 5-line TQ3_4S commit never touched CUDA dispatch paths. See `artifacts/tq3_regression_investigation_20260417.md`.
- TQ3_4S is the only GGUF weight type that needs an official slot
- KV cache types (TURBO*, TQ3_0) can use high numbers or a separate flag

## When to submit PR

- After TQ3_4S speed is competitive enough to justify upstreaming
- Request **one slot** (e.g. 41) for TQ3_4S
- Re-quantize and re-upload all published TQ3_4S GGUFs with new enum value
- Include full implementation: quantizer, dequantizer, CUDA kernels

## Interim risk

- If upstream adds 2+ types before we submit, our slot 46 breaks
- Mitigation: monitor upstream releases, submit before collision
- Current pace: ~1 new type per month (MXFP4=39, NVFP4=40)
- Estimated time before collision with 46: ~6 months at current pace

## Action items

- [ ] Remove TQ3_1S from codebase (dead type)
- [ ] Move KV cache types (TURBO*, TQ3_0) to high enum values (200+)
- [ ] Get TQ3_4S speed to acceptable level
- [ ] Submit upstream PR with single slot reservation + full implementation
