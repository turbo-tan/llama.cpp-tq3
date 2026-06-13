# Recipe: Qwen3.6-27B-MTP-TQ3_4S-out6k-v2b

Reconstructed 2026-06-13 from session memory (original doc was never committed).

## Goal

Beat UD-Q3_K_XL on **both size AND quality**.

| Metric | Target | v2b actual |
|---|---|---|
| Size | < 13.77 GiB (UD) | **13.52 GiB** ✅ |
| bpw | < 4.34 (UD) | **4.25 bpw** ✅ |
| Quality | ≥ out6k baseline | **unmeasured** — gates pending |

## Artifact

```
/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-out6k-v2b.gguf
14G  (13.52 GiB / 4.25 bpw)  rebuilt 2026-06-13 20:05 (ssm_alpha/beta F32 → Q8_0; previous F32 build crashed bench-loop)
```

## Source

Quantized from the BF16 master:

```
/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4L2.gguf
```

Source repo commit: `main @ 2ecf73bc8` (Merge pull request #27 — gemma4-mtp-support).

Policy file: `docs/turboquant/generated/qwen36_27b_mtp_tq3_4s_out6k_v2b.tensor-types.txt`

## Tensor mix vs out6k winner

| Tensor pattern | out6k winner | v2b | Rationale |
|---|---|---|---|
| `blk.*.ssm_out` | Q4_K | **Q5_K** | Recurrent output projection; upping to Q5_K captures more of the state dynamics |
| `blk.*.ssm_alpha` | Q3_K | **Q8_0** | Recurrent decay param — Q8_0 preserves near-full precision vs 3-bit while staying on GPU (F32 tried first but dispatched to CPU via mmvq fallback → 2× speed regression + bench-loop crash) |
| `blk.*.ssm_beta` | Q3_K | **Q8_0** | Same as ssm_alpha — Q8_0 keeps it on the fast mmvq path |
| `blk.*.attn_v` (full-attn layers) | Q5_K | **Q6_K** | Value projection; one step up reclaims quality in the attention layers that matter most |
| `blk.*.attn_output` (full-attn) | Q6_K | **Q4_K** | Claw-back: fund the above upgrades by dropping attn_output one step |
| Everything else (bulk, output, embd, MTP) | TQ3_4S | TQ3_4S (unchanged) | |

The ssm_alpha/ssm_beta precision hypothesis: 3-bit on recursive decay parameters is likely what
caused the template collapse in out6k. Q8_0 raises precision dramatically vs TQ3_4S/Q3_K while
staying on the fast CUDA mmvq path (F32 was tried first, but `ggml_cuda_should_use_mmvq` returns
true for F32 by default, dispatching to `mul_mat_vec_q_switch_type` → `GGML_ABORT` on the missing
F32 case, causing bench-loop crash + CPU fallback → ~18 t/s vs expected ~43 t/s).

## Open A/B

- `ssm_out` Q5_K (v2b) vs Q6_K — is the extra cost worth it vs just claw-backing further?
- v2c idea: add imatrix-guided quantization for the bulk TQ3_4S tensors.

## Build command

```bash
# Source is already a GGUF (BF16), so use llama-quantize.
# Note: llama-quantize --tensor-type-file uses name=type format (no spaces),
# so pass overrides via --tensor-type flags instead of the .tensor-types.txt file.
/home/awee/code/llama.cpp-tq3/build/bin/llama-quantize \
  --tensor-type 'blk.*.ssm_alpha=Q8_0' \
  --tensor-type 'blk.*.ssm_beta=Q8_0' \
  --tensor-type 'blk.*.ssm_out=Q5_K' \
  --tensor-type 'blk.*.attn_v=Q6_K' \
  --tensor-type 'blk.*.attn_output=Q4_K' \
  /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4L2.gguf \
  /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-out6k-v2b.gguf \
  TQ3_4S
```

## Pending gates

Run on desktop-play (192.168.1.77) when the GPU is free.
See `docs/turboquant/out6k-benchloop-sop.md` for the runtime shape.

1. **PPL/KLD witness** — v2b vs out6k baseline vs UD-Q3_K_XL.
2. **BenchLoop partial** — `speed,toolcall,coding` suites, reasoning-off, `-c 32768`.
   Compare v2b vs out6k (`…mtp-q4k-outq6.gguf`) and UD (`Qwen3.6-27B-UD-Q3_K_XL.gguf`).
3. Note: use `-c 32768` (not `-c 262144`) for A/B quality checks — the 262144 path caused
   a CUDA OOM false negative on 2026-06-13.
4. If v2b wins both: promote to flagship; update publish/ alias.
5. If ssm_out Q5_K is the bottleneck: build v2b-alt with Q6_K and re-gate.
6. If quality still lags UD: proceed to v2c (add imatrix).
