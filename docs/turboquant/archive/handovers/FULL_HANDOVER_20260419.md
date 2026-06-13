# Speculative Decode & TQ3_4S Quantization — Full Session Handover

Date: 2026-04-14 (multi-session)
Updated: 2026-04-20

---

## 1. Speculative Decode for Qwen3.5-27B Hybrid SSM

### What Ships Today

**GPU shadow buffer + n-gram speculation with gating.**

- **+28.7% average speed** (22.7 -> 29.3 tok/s) on relaxed 10-prompt benchmark, but only 8/10 token-exact
- **+16.7% average speed** (21.2203 -> 24.7688 tok/s) on recovered exact contract, 10/10 token-exact
- **+60% peak** (36.3 tok/s) on high-acceptance prompts under relaxed gating
- **GPU shadow save/restore**: ~0.5ms via cudaMalloc D2D (was 26ms CPU path)
- **N-gram cache**: 3-4 gram exact-match lookup from model's own decode patterns

Current recovered branch:
- Branch: `experiment/specdecode-next-v2-20260419`
- Worktree: `/home/awee/code/worktrees/llama-specdecode-next`
- Base: `experiment/persistent-decode` @ `b644bf478`
- Fix: `LOSSY_NGRAM_BATCH=384` default avoids the 299 MB GPU shadow buffer OOM caused by the old hardcoded `512`
- Artifacts: `artifacts/spec_exact_fast_contract_recheck_v3_20260419.json`, `artifacts/specdecode_recovery_batch384_20260419.txt`

### Key Innovation: GPU Shadow Buffer

File: `ggml/src/ggml-cuda/ssm-shadow.cu`

- Persistent cudaMalloc buffer (~150MB) allocated on first speculation
- All 64 R/S tensor copies via cudaMemcpyAsync D2D + single cudaDeviceSynchronize
- Save: ~0.5ms. Restore: ~0.5ms. Total rollback: ~1ms.

### Known Limitation: pp2 ≠ tg1 on Hybrid SSM

- Fused GatedDeltaNet kernel produces ~0.1 logit difference between pp2 and tg1+tg1
- Confirmed across ALL quants (TQ3_4S, Q3_K_S, Q4_0) — NOT a TQ3_4S issue
- Drift only affects continuation phrasing, never the primary answer
- Upstream hybrid recurrent-state path consistency issue

### Files

| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/ssm-shadow.cu` | GPU D2D shadow buffer |
| `examples/jacobi/lossy-ngram.cpp` | Main speculation tool |
| `src/llama-memory-recurrent.h/cpp` | Shadow save/restore API |
| `src/llama-context.cpp` | llama_memory_attn_seq_rm, shadow APIs |
| `examples/jacobi/pp2-vs-tg1.cpp` | Per-layer divergence diagnostic |

---

## 2. EAGLE3 Status

EAGLE is no longer part of the active execution plan.

Reason:
- it increases storage requirements
- it is not on the shortest path to current benchmarkable wins

Keep prior EAGLE notes as historical context only. Do not treat them as next-step guidance unless priorities change.

---

## 3. TQ3_4S Quantization (Fixed)

Branch: `feature/tq3-4s-quantize-gemma4`

Three fixes applied:
1. `ggml/src/ggml.c`: quantize dispatch + from_float_ref
2. `src/llama-quant.cpp`: ftype mapping + token_embd guard
3. `tools/quantize/quantize.cpp`: CLI entry

Results:
- Qwen3.5-9B Q8_0 → TQ3_4S: ✅ 4.35 BPW, 64.8 tok/s
- Gemma4 26B-A4B Q4_K_M → TQ3_4S: ✅ quantizing (background)
- Gemma4 31B Q8_0: download started from unsloth

---

## 4. Current Focus

Use [ACTIVE_PROGRESS_PLAN.md](../../active/ACTIVE_PROGRESS_PLAN.md) as the current source of truth.

Current focus:
1. speculative decode revalidation and ship/no-ship decision
2. Decode Speed Plan X GPU benchmarking for sparse V dequant

---

## 5. Next Steps

1. Continue spec-decode speed search from the recovered exact branch
2. Compare runtime-local cache, `universal_v1`, and combined cache policy
3. Run chat/code probes before any ship/no-ship decision
4. GPU-benchmark sparse V dequant after spec-decode has a clear speed verdict

---

## 6. Key Branches

| Branch | Content |
|--------|---------|
| `main` | TQ3_4S type + KV cache (7 ahead of public) |
| `feature/tq3-4s-quantize-gemma4` | Quantizer fixes |
| `experiment/persistent-decode` | Base for recovered GPU-shadow speculation |
| `experiment/specdecode-next-v2-20260419` | Active recovered spec-decode branch |
| `backup-all-work` | Full TQ3_4S CUDA kernels |
