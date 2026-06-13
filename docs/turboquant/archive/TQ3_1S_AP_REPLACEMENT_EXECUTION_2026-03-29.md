# TQ3_1S-AP Replacement Execution Plan

Date: 2026-03-29

This file is the durable execution plan for the active adaptive-weight line.

If session credit runs out, resume from here.

## Goal

Build a real replacement-style `TQ3_1S-AP` weight format that:

- stays near `4.05-4.10 bpw`
- beats plain `TQ3_1S` on the 9B dense witness
- remains materially smaller than native `Q4_0`
- is runtime-usable, first on CPU, then on CUDA

## Current Truth

### Main Witness

Use persistent 9B dense witness:

- `/home/awee/models/bartowski/Qwen_Qwen3.5-9B-GGUF/Qwen_Qwen3.5-9B-Q8_0.gguf`

### Active Baselines

- plain `TQ3_1S`:
  - file: `/home/awee/models/turboquant9b/Qwen_Qwen3.5-9B-TQ3_1S.gguf`
  - size: about `4.48 GiB`, `4.29 BPW`
- additive AP fallback:
  - file: `/home/awee/models/turboquant9b/Qwen_Qwen3.5-9B-TQ3_1S-AP-4p05-r2.gguf`
  - size: about `4.53 GiB`, `4.35 BPW`

### Model-Level Quality

9B short CPU PPL gate, `wiki.test.raw`, `2` chunks, `-ngl 0 -fa 0 -t 8 -c 2048`:

- plain `TQ3_1S`: `8.7477 +/- 0.5024`
- additive AP `4.05`: `8.7629 +/- 0.5048`

Interpretation:

- additive AP now loads and runs
- additive AP does not improve quality
- additive AP is a scaffold only, not the product path

### Offline Signal That Still Matters

Replacement-style adaptive promotion on the 9B witness:

- `TQ3_1S` base `4.00 bpw`:
  - `RMSE 0.002398`
  - `dot 0.013558`
- replacement adaptive mix, effective `4.08128 bpw`:
  - `RMSE 0.002324`
  - `dot 0.013138`
- ideal replacement reconstruction in the harness is exact

Interpretation:

- the quality gain is real
- the gain comes from replacement, not additive repair

## What Was Fixed Already

### Additive AP Runtime/Writing

Commit in `llama.cpp`:

- `1dad6d782` `feat: add tq3 ap cpu runtime scaffold`

This commit includes:

- CPU `get_rows` support for AP sidecars
- CPU `mul_mat` correctness-first AP path
- model loader sidecar attach path
- sidecar metadata/order handling
- writer bug fix for GGUF data order

### Writer Bug Root Cause

The broken additive AP model failed because:

- GGUF metadata order was:
  - all base tensors
  - then all AP sidecars
- writer output order was:
  - base tensor
  - sidecars
  - next base tensor

This made the loader read incorrect sidecar bytes.

The fix was:

- buffer sidecars during quantization
- write sidecars after base tensors, in metadata order

## Why Replacement Is The Real Line

Additive AP was only worth trying because it fit fixed-row constraints.

What we learned:

- additive mean-sidecar is runtime-friendly but quality-weak
- additive richer metadata is not worth it
- replacement-style AP preserves the compression story and the offline quality win

Therefore:

- additive AP stays as a debugging/runtime scaffold
- replacement AP is the product candidate

## Replacement Format Decision

### Chosen Design

Replacement-style `TQ3_1S-AP`:

- base blocks:
  - `TQ3_1S`
  - `16 bytes / 32 weights`
- promoted blocks:
  - shared-shift variant
  - `18 bytes / 32 weights`
- selector:
  - 1-bit bitmap sidecar
- promoted payload:
  - promoted blocks only, in block order

### Sidecar Naming

Keep:

- `<tensor>.tq3_ap_bitmap`
- `<tensor>.tq3_ap_promoted`

### Effective Budget Target

Primary target:

- `4.05 bpw`

Secondary target:

- `4.10 bpw`

Do not push a uniform `4.5 bpw` line.

## Execution Plan

### Phase 1: Freeze The Scaffold

1. Keep additive AP code compiling and runnable.
2. Do not spend more time tuning additive AP quality.
3. Do not use additive AP as the product benchmark.
4. Keep `Q4_*_TQ` work archived, not active.

### Phase 2: Define Replacement Contract

5. Write down the exact replacement block layout in code comments and doc notes.
6. Reuse the existing shared-shift promoted block math from the harness.
7. Keep bitmap semantics flat and global within each tensor.
8. Keep promoted payload ordered by promoted block index.

### Phase 3: Writer Implementation

9. Extend `llama-quant.cpp` so AP writer emits replacement promoted blocks instead of additive means.
10. Keep the base tensor as normal `TQ3_1S`.
11. Emit bitmap sidecar as packed bits.
12. Emit promoted sidecar as packed promoted block stream.
13. Start with the `4.05 bpw` target only.

### Phase 4: CPU Runtime First

14. Extend `ggml_tq3_ap_extra` to support replacement promoted block payload.
15. Add CPU helper to map block id -> promoted block pointer.
16. Add CPU dequant for replacement AP blocks.
17. Update CPU `get_rows` path to use replacement promoted blocks.
18. Update CPU `mul_mat` fallback path to use replacement promoted blocks.
19. Keep correctness ahead of speed on CPU.

### Phase 5: First Real Model Gate

20. Build:
  - `/home/awee/models/turboquant9b/Qwen_Qwen3.5-9B-TQ3_1S-AP-4p05-repl.gguf`
21. Run raw completion:
  - `The capital of France is`
22. Run coding sanity:
  - `two_sum`
23. Run short CPU PPL:
  - `wiki.test.raw`
  - `2` chunks
  - same baseline settings

### Phase 6: Decide If It Has Real Legs

24. Compare replacement AP against plain `TQ3_1S`.
25. If replacement AP does not beat plain `TQ3_1S`, stop and reassess.
26. If it wins, also build `4.10 bpw` and compare.
27. Choose the better of:
  - lower bpw
  - better PPL

### Phase 7: CUDA Runtime

28. Only after CPU quality is confirmed, add CUDA support.
29. Add replacement AP handling to CUDA dequant/convert path.
30. Add replacement AP handling to CUDA `GET_ROWS`.
31. Defer MMQ/MMVQ optimization until correctness is stable.

### Phase 8: Benchmark And Product Gate

32. Run 9B GPU VRAM measurements.
33. Run 9B short prompt/decode checks.
34. Compare against:
  - plain `TQ3_1S`
  - native `Q4_0` if available on the same witness
35. Record:
  - size
  - BPW
  - PPL
  - prompt tok/s
  - decode tok/s
  - VRAM

### Phase 9: Larger Confirmation

36. If 9B wins cleanly, carry the method to 27B.
37. Do not use 27B as the first debug witness.
38. Use 27B only after the 9B replacement path is stable.

## Success Criteria

Replacement AP is worth keeping only if it satisfies all of:

1. Loads and runs correctly.
2. Sane factual and coding smoke outputs.
3. Beats plain `TQ3_1S` on the 9B PPL gate.
4. Effective BPW remains around `4.05-4.10`.
5. Still materially below `Q4_0` density.

## Failure Criteria

Stop the line if any of these become true:

1. Replacement AP cannot beat plain `TQ3_1S` on model-level PPL.
2. Runtime complexity explodes without a quality win.
3. Effective BPW drifts too close to `Q4_0`.
4. CUDA support becomes much more expensive than the quality gain justifies.

## Immediate Next Steps

These are the exact next tasks after this document:

1. Implement replacement promoted payload writer in `llama-quant.cpp`.
2. Extend `ggml-tq-adaptive.h` for replacement payload metadata.
3. Update CPU sidecar attach to understand replacement promoted blocks.
4. Update CPU dequant/get_rows/mul_mat.

## Launch-Pad Note

Replacement AP remains the main adaptive-format execution line, but the cheapest new math work has moved
slightly earlier in the funnel:

1. screen transform improvements on top of plain `TQ3_1S`
2. screen better promotion objectives at the same adaptive budget
3. only then carry the winning idea into AP/replacement runtime work

Current 9B offline screening additions:

- diagonal second-moment pre-transform on plain `TQ3_1S`:
  - baseline: `RMSE 0.002398`, `dot 0.013558`
  - diagonal transform: `RMSE 0.002394`, `dot 0.013522`
- activation-aware promotion at `4.05 bpw`:
  - SSE-ranked mix: `RMSE 0.002324`, `dot 0.013138`
  - dot-ranked mix: `RMSE 0.002337`, `dot 0.013019`
- combined diagonal-transform + adaptive mix:
  - `4.05 bpw`: `RMSE 0.002323`, `dot 0.013115`
  - `4.10 bpw`: `RMSE 0.002304`, `dot 0.013019`

Interpretation:

- transform quality is still a live lever and is now the cheapest one
- sensitivity-aware promotion is also live, but it looks more like a ranking refinement than a complete objective swap
- the best current launch-pad candidate is now:
  - diagonal-transform base
  - plus adaptive promotion under the existing `4.05-4.10 bpw` budget
- this should be the next candidate moved from offline screening to the 9B chat-path quality gate
5. Build fresh `4.05` replacement AP 9B model.
6. Run short completion + short PPL.

## Notes For Resume

- Persistent models only. Never use `/tmp` for model files.
- If the model fails to load, check writer order vs GGUF metadata order first.
- If CPU quality does not improve, do not jump to CUDA.
- If replacement AP wins, add a rocket marker in the docs and summary.

## Late 2026-03-29 Correction

The original `4.081 bpw` bitmap-sidecar replacement result was useful as a quality signal, but its storage accounting was optimistic:

- the prototype dequant path used full `18`-byte promoted shared-shift blocks
- the reported effective BPW only charged the extra `2` bytes over base per promoted block
- that accounting is only valid if promoted blocks reuse the base payload and scales

The full promoted-sidecar writer proved the real file economics are much worse:

- 9B replacement-sidecar build landed at about `4.62 BPW`

So the sidecar replacement path is not the product path.

### Honest Fixed-Size Replacement Path

The real replacement line is now:

- `TQ3_1S_AP1`
- one promoted shared-shift block per `16` logical `TQ3_1S` blocks
- fixed `512`-weight superblock
- real local storage: `260` bytes per `512` weights
- honest local density: `4.0625 bpw`

Why this matters:

- it fits GGML's fixed-row model cleanly
- it preserves the replacement economics
- it maps directly onto the earlier row-quota prototype

9B prototype signal supporting `AP1`:

- row-mix quota `8` on `128`-block rows
- `eff_bpw = 4.05183`
- `RMSE = 0.002365`
- `dot = 0.013347`

### Current Runtime Status

Implemented in the `llama.cpp` working tree:

- new ggml type: `GGML_TYPE_TQ3_1S_AP1`
- new llama ftype: `LLAMA_FTYPE_MOSTLY_TQ3_1S_AP1`
- quantize/dequant helpers for the fixed `512`-weight superblock
- CPU-only correctness-first runtime path
- quantizer fallback to plain `TQ3_1S` for incompatible tensor widths

Real 9B dry-run result:

- file type: `TQ3_1S_AP1 - 4.06 bpw turbo promoted superblock`
- whole-model quant size: `4642.88 MiB`
- whole-model file density: `4.35 BPW`

Interpretation:

- local quantized density is in the intended `~4.06` band
- whole-model GGUF density is higher because of unquantized tensors and the mixed tensor set
- this remains materially smaller than native `Q4_0`, but not a literal whole-file `4.08 BPW`

### Honest 9B AP1 Checkpoint

Real generated model:

- `/home/awee/models/turboquant9b/Qwen_Qwen3.5-9B-TQ3_1S_AP1.gguf`

What is now verified:

- the model loads successfully
- raw completion is sane on the simple factual smoke:
  - prompt: `The capital of France is`
  - answer starts with: `Paris.`

Current CPU-only timing check, same raw factual prompt, `-ngl 0 -fa 0 -t 8 -c 512 -n 4`:

| Format | Size | Prompt tok/s | Decode tok/s | Elapsed |
|---|---:|---:|---:|---:|
| `TQ3_1S` | `4.48 GiB` | `0.36` | `0.35` | `23.08 s` |
| `TQ3_1S_AP1` | `4.53 GiB` | `0.21` | `0.05` | `89.82 s` |

Interpretation:

- `TQ3_1S_AP1` is functionally sane
- but the current CPU fallback path is still far too slow
- this fails the runtime side of the success criteria by a wide margin
- CPU-first AP1 is therefore only a correctness checkpoint, not a launchable runtime

Updated next step:

- do not treat the current CPU AP1 timing as meaningful product performance
- either finish a cheaper AP1 CPU hot path or move the type to CUDA support before judging tok/s
- keep the quality gate focused on correctness first, then re-evaluate speed on CUDA

### First Real GPU AP1 Checkpoint

The first CUDA mirror is now working correctly for `TQ3_1S_AP1`.

9B factual smoke, `-ngl 99 -fa 1 -t 8 -c 512 -n 4`:

- plain `TQ3_1S`:
  - output: `Paris.`
  - prompt: `10.75 tok/s`
  - decode: `3.02 tok/s`
  - model/self CUDA memory: about `4666 MiB`
- `TQ3_1S_AP1`:
  - output: `Paris.`
  - prompt: `9.56 tok/s`
  - decode: `2.53 tok/s`
  - model/self CUDA memory: about `4717 MiB`

9B coding smoke, `two_sum`, `-ngl 99 -fa 1 -t 8 -c 1024 -n 32`:

- plain `TQ3_1S`:
  - coherent code-start answer
  - prompt: `27.48 tok/s`
  - decode: `2.56 tok/s`
- `TQ3_1S_AP1`:
  - coherent code-start answer
  - prompt: `25.06 tok/s`
  - decode: `2.29 tok/s`

9B short GPU PPL, `wiki.test.raw`, `1` chunk, `-ngl 99 -fa 1 -t 8 -c 1024`:

- plain `TQ3_1S`: `4.9704 +/- 0.55456`
- `TQ3_1S_AP1`: `4.8638 +/- 0.54008`

Interpretation:

- `TQ3_1S_AP1` is now quality-positive on the 9B GPU gate
- but it is still slower than plain `TQ3_1S`
- the current gap is roughly:
  - prompt: about `9%` slower
  - decode: about `10-16%` slower depending on probe
- so the format now passes correctness and quality, but still fails the `within 5%` speed target

Current conclusion:

- the format line is alive
- the next work is runtime/dequant optimization, not more format math
- no rocket yet

### Honest 9B `Q4_0` Anchor And Fixed-Layout `AP1_v2`

Persistent 9B `Q4_0` baseline now exists:

- `/home/awee/models/turboquant9b/Qwen_Qwen3.5-9B-Q4_0.gguf`

Exact current 9B file sizes:

- `Q4_0`: `5.0G`
- `TQ3_1S`: `4.5G`
- `TQ3_1S_AP1_v2`: `4.6G`

Short GPU PPL, `wiki.test.raw`, `1` chunk, `-ngl 99 -fa 1 -t 8 -c 1024`:

- `Q4_0`: `4.5051 +/- 0.47563`
- `TQ3_1S`: `4.9704 +/- 0.55456`
- `TQ3_1S_AP1_v2`: `4.7741 +/- 0.52509`

Interpretation:

- `AP1_v2` materially improves over plain `TQ3_1S`
- `AP1_v2` is still clearly behind `Q4_0` on quality
- the remaining quality gap to `Q4_0` is now explicit and should be treated as the real bar

Fixed-layout change now in the runtime:

- `AP1` superblock keeps the same `260` bytes
- layout changed from mixed logical ordering to:
  - `mask`
  - `15` contiguous base `TQ3_1S` blocks
  - `1` fixed promoted shared-shift trailer

Reason:

- remove mixed-stream pointer walking from CPU/CUDA decode
- keep the same size class and avoid sacrificing the compression story

Longer plain factual decode probe, `-ngl 99 -fa 1 -t 8 -c 512 -n 32`:

- `TQ3_1S`:
  - prompt: `5.94 tok/s`
  - decode: `1.42 tok/s`
- `TQ3_1S_AP1_v2`:
  - prompt: `5.71 tok/s`
  - decode: `1.36 tok/s`

Interpretation:

- on the current build, fixed-layout `AP1_v2` is within about `4%` of plain `TQ3_1S` on both prompt and decode
- that satisfies the current “within 5% of the comparable non-AP path” discipline
- but it still does not solve the larger quality gap to `Q4_0`

Updated next target:

1. preserve the new near-parity speed versus `TQ3_1S`
2. close more of the `Q4_0` quality gap
3. only after that revisit broader launch criteria
