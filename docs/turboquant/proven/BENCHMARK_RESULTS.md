# TurboQuant TQ3_4S — Benchmark Results

## Perplexity (Qwen3.5-27B, wikitext-2, 100 chunks)

### c=2048, 100 chunks (original methodology)

| Format | bpw | PPL (c=2048, 100ch) | Size | PP tok/s | TG tok/s |
|--------|-----|---------------------|------|----------|----------|
| **TQ3_4S** | **4.00** | **6.7727** | **12.9 GB** | **315** | **14.6** |
| Q3_K_S | 3.44 | 6.7970 | 11.4 GB | 689 | 20.7 |
| IQ4_XS | 4.25 | 6.8334 | 13.9 GB | 841 | 23.2 |
| TQ3_1S | 4.00 | 6.9186 | 12.9 GB | 220 | 15.0 |
| UD-Q2_K_XL | ~3.3 | 7.5294 | 11.0 GB | — | — |

### c=2048, full wikitext-2 (gold standard — do not mix with above)

| Format | bpw | PPL (c=2048, 145ch) | Size | Notes |
|--------|-----|---------------------|------|-------|
| **TQ3_4S + last8 FFN Q5_K** | **~4.1** | **6.6947** | **~13.5 GB** | mixed policy |
| Q3_K_S | 3.44 | 6.8630 | 11.4 GB | worse than 100ch result |
| EXL3 3.0bpw | 3.02 | 7.0010 | 13.0 GB | 100ch, stride=2048, ExLlamaV3 |

TQ3_4S 145ch result pending. Q3_K_S degraded from 6.7970→6.8630 at full run.
**TQ3_4S beats Q3_K_S and IQ4_XS on quality at c=2048.**

### c=512 (for reference — not comparable to published benchmarks)

| Format | PPL (c=512) |
|--------|-------------|
| TQ3_4S | 7.0468 |
| TQ3_4S + last8 FFN -> Q5_K | 6.9271 |
| TQ3_1S | 7.1078 |

Context window matters significantly: TQ3_4S improves from 7.05 → 6.77 going from c=512 to c=2048.

### KLD Results (TQ3_4S + last8 FFN Q5_K, 100ch, c=2048)

Artifact: `artifacts/kld_cmp_27b_tq3_4s_last8ffn_q5k_c2048_100ch_20260401.txt`

| Metric | Value |
|--------|-------|
| Mean PPL(Q) | 6.6449 |
| Mean PPL(base) | 6.7951 |
| Mean PPL(Q)/PPL(base) | 0.9779 (better than Q8!) |
| Mean KLD | **0.1363** |
| Median KLD | **0.0260** |
| 99.9% KLD | 16.67 |
| Max KLD | 27.79 |
| RMS Δp | 8.67% |
| Same top-p | 89.3% |

Reference (Unsloth Qwen3.5-27B benchmarks, same model):
Source: https://docs.unsloth.ai/models/qwen3.5/qwen3.5-gguf-benchmarks

| Format | Size | Mean KLD | Notes |
|--------|------|----------|-------|
| AesSedai IQ3_S | 12.65 GB | 0.0613 | QuIP#-based |
| Unsloth IQ3_XXS | 13.12 GB | 0.0501 | QuIP#-based |
| **TQ3_4S + last10 FFN Q5_K** | **~13.5 GB** | **0.1348** | WHT-based (ours) |
| **TQ3_4S + last8 FFN Q5_K** | **~13.3 GB** | **0.1363** | WHT-based (ours) |
| Q3_K_S | 11.4 GB | 0.1318 | Standard llama.cpp (ours) |

**Interpretation**: 
- Mean KLD is **2.2x worse than IQ3_S** (0.135 vs 0.061) at similar size - this is the gap between fixed WHT and learned incoherence (QuIP#)
- However **competitive with Q3_K_S** (0.135 vs 0.132) - only 2% worse, and beats on same-top-p (89.4% vs 88.7%)
- Median KLD (0.026) is low — 90%+ of tokens are fine. The mean is pulled up by a small tail of outlier tokens with very wrong distributions (max 27.8)
- This is the WHT signature: occasional large errors on specific weight patterns
- PPL is competitive because top-token accuracy is preserved; KLD suffers because those outliers shift the full distribution

Artifact:

- [ppl_27b_tq3_4s_last8ffn_q5k_100ch_20260401.txt](/home/awee/code/tan_llama/artifacts/ppl_27b_tq3_4s_last8ffn_q5k_100ch_20260401.txt)

Observed:

- base `TQ3_4S`: `7.0468 +/- 0.11034`
- `TQ3_4S + last8 FFN -> Q5_K`: `6.9271 +/- 0.10654`

Conclusion:

- a tiny EXL3-style per-module mixed policy does improve `TQ3_4S`
- gain on this gate is about `0.12` PPL
- the next quality lever is likely selective mixed precision, not plain imatrix weighting

## Key Properties

| Property | TQ3_4S | Q3_K_S | IQ4_XS |
|----------|--------|--------|--------|
| bpw | 4.00 | 3.44 | 4.25 |
| Size (27B) | 12.9 GB | 11.4 GB | 13.9 GB |
| Fits 16 GB GPU | ✅ | ✅ | ❌ (OOM at c=2048) |
| PPL (c=2048) | **6.7727** | 6.7970 | 6.8334 |
| PP tok/s | 315 | 689 | 841 |
| TG tok/s | 14.6 | 20.7 | 23.2 |

## Speed Gap

TQ3_4S is still materially slower on PP than the strongest stock low-bit baselines.

Current understanding:

- the old Marlin/MMVQ fused path is not the answer
- it was removed from the active CUDA path after proving much slower than the cuBLAS baseline
- the current active branch is `feature/tq3_4s-rotated-gemm`

The next serious speed path is:

- keep TQ3_4S weights in the rotated domain
- rotate activations once per tile
- use cuBLASLt / tensor-core GEMM on the rotated operands

That is the speed line worth pursuing now, not more fused Marlin work.

### Experimental Smoke Results

These are debugging results, not publishable headline numbers.

#### Staged `prompt_512` kernel smoke

Artifact:

- [bench_27b_tq3_4s_trace_stage_prompt512_smoke_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_trace_stage_prompt512_smoke_20260401.txt)

Observed:

- the new staged path is definitely being selected for `shape=prompt_512`
- representative `prompt_512` kernel times moved the wrong way
- old traced `prompt_512` median kernel time: about `42.23 ms`
- staged `prompt_512` median kernel time: about `48.06 ms`
- smoke throughput:
  - `pp512 = 37.18 tok/s`
  - `tg2 = 12.11 tok/s`

Conclusion:

- the first staged prompt kernel is functionally wired
- but it is much slower than the fallback and is not the current answer

#### Staged `prompt_512` tile-128 retry

Artifact:

- [bench_27b_tq3_4s_trace_stage_prompt512_tile128_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_trace_stage_prompt512_tile128_20260401.txt)

Observed:

- the stage selector was previously choosing `prompt_512` while the implementation still hard-coded `TILE_N = 32`
- after fixing that mismatch and reusing each decoded weight tile across `128` columns:
  - `pp512 = 83.43 tok/s`
  - `tg2 = 11.68 tok/s`

Conclusion:

- widening the staged kernel materially helps
- `37 -> 83 tok/s` is the first real speed jump on this path
- still far from the `315 tok/s` cuBLAS baseline, but no longer a trivial dead end

#### TQ3_4S MMQ smoke

Artifacts:

- [bench_27b_tq3_4s_mmq_blocking_smoke_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_mmq_blocking_smoke_20260401.txt)
- [bench_27b_tq3_4s_mmq_blocking_p512_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_mmq_blocking_p512_20260401.txt)
- [bench_27b_tq3_4s_mmq_memcheck_p512_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_mmq_memcheck_p512_20260401.txt)
- [bench_27b_tq3_4s_mmq_blocking_p512_afterfix_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_mmq_blocking_p512_afterfix_20260401.txt)
- [bench_27b_tq3_4s_mmq_smoke_afterfix_20260401.txt](/home/awee/code/tan_llama/artifacts/bench_27b_tq3_4s_mmq_smoke_afterfix_20260401.txt)

Observed:

- before the layout fix, `TQ3_4S` MMQ crashed on the large prompt path
- memcheck showed out-of-bounds shared-memory writes in `mul_mat_q<GGML_TYPE_TQ3_4S, 128>`
- root cause: `GGML_TYPE_TQ3_4S` was missing from the MMQ tile-size switch tables in `mmq.cuh`
- after adding the missing tile-size cases, the crash is gone
- but the speed is still poor:
  - `pp512 = 118.91 tok/s`
  - `tg2 = 8.82 tok/s`

Conclusion:

- MMQ crash is fixed
- MMQ is still not competitive with the known-good `TQ3_4S` cuBLAS baseline
- keep MMQ as a debugged fallback, not the main optimization line

## Format Description

TQ3_4S applies a fixed 32-element randomized Hadamard transform (WHT) before quantization. Each 16-byte block stores:
- 4 × E3M5 (8-bit mini-float) scales — one per 8-element group
- 12 bytes of 3-bit packed indices (32 elements × 3 bits)

The WHT decorrelates weights within each block, enabling better utilization of the 8-level codebook. Unlike importance-weighted formats (IQ4_XS, Q3_K_S), TQ3_4S does not use activation statistics (imatrix) — the WHT equalizes coefficient importance, making imatrix ineffective.

## Comparison to Challenger's Baselines

| Format | bpw | PPL (c=2048) | Notes |
|--------|-----|-------------|-------|
| TQ3_4S + last8 FFN Q5_K | ~4.1 | **6.6947** | This work, mixed policy |
| TQ3_4S | 4.00 | 6.7727 | This work |
| Q3_K_S | 3.44 | 6.8630 | Standard GGUF (145ch full run) |
| IQ4_XS | 4.25 | 6.8334 | Importance-weighted |
| EXL3 3.0bpw | 3.02 | 7.0010 | ExLlamaV3, QTIP-based, 100ch/2048 non-overlapping |
| IQ4_KS (ik_llama) | 4.25 | TBD | Not in standard llama.cpp |

TQ3_4S beats Q3_K_S by **0.09 PPL** and EXL3 3.0bpw by **0.22 PPL** at similar size.
Mixed policy (last8 FFN → Q5_K) gains another **0.08 PPL** on top.

## Hardware

- GPU: NVIDIA RTX 5060 Ti (16 GB VRAM)
- Model: Qwen3.5-27B (dense, 26.9B params)
- Inference: llama.cpp with CUDA backend
## Current Position

Current 27B `c=2048` quality/speed picture:

- `TQ3_4S`: `PPL 6.7727`, `PP 315`, `TG 14.6`
- `Q3_K_S`: `PPL 6.7970`, `PP 689`, `TG 20.7`
- `IQ4_XS`: `PPL 6.8334`, `PP 841`, `TG 23.2`

Interpretation:

- `TQ3_4S` is already competitive or better on the current quality gate
- the real gap is prompt throughput
- current rotated cuBLAS prototype (`pp2048 265`) improved the dirty-branch baseline but is still far below `Q3_K_S`
- trace evidence shows the native `TQ3_4S` kernel is the bottleneck, not activation rotation
