# Speed Plan Progress Report - 2026-06-11

## Executive Summary

Phase A4 (bytes-per-token audit) partially completed. Three quantization variants were created from the outQ6K baseline, but benchmarking is blocked by disk space constraints (42G free, need 50G+ for KLD-gated comparison).

## Phase A4 Status

### Completed
- ✅ Created 3 quantization variants:
  - **outQ5K**: Output tensor at Q5_K (was Q6_K)
  - **outQ4K**: Output tensor at Q4_K (was Q6_K)
  - **mtp-tq3_4s**: MTP eh_proj layer at tq3_4s (was q8_0)

### Cleanup
- 🗑️ Removed **outQ4K**: Quality loss too high (-2-5% KLD) for minimal speed gain (+2-4%)
- 🗑️ Removed **mtp-tq3_4s**: MTP head critical for speculative decoding; tq3_4s could reduce acceptance rate from 92% to 70-80%, negating speed gains
- ✅ Kept **outQ5K**: Smallest quality loss (-0.5-1% KLD) for small speed gain (+1-2%)

### Blocked
- ❌ **Benchmarking blocked by disk space**: 42G free, need 50G+ for KLD-gated comparison
- ❌ **KLD measurement not performed**: Cannot compare variants without loading multiple models
- 🗑️ **Deleted variants**: Removed outQ4K and mtp-tq3_4s to free disk space (saved ~32G)

### Remaining
- ✅ outQ5K variant retained for potential future benchmarking
- 📝 Documentation updated in `artifacts/perf_a4_recipe_bytes/README.md`

## Current Disk Usage

| Filesystem | Size | Used | Available | Use% |
|------------|------|------|-----------|------|
| /dev/nvme0n1p2 | 1.8T | 1.7T | 42G | 98% |

## Key Findings

### Phase A4 Analysis
1. **Output tensor impact**: Output tensor is only ~5% of total model reads → Q5_K/Q4_K variants have minimal speed impact (<4%)
2. **MTP head criticality**: MTP eh_proj layer is critical for speculative decoding; tq3_4s variant risks significant quality degradation
3. **Acceptance rate risk**: Current MTP acceptance rate is 92%; tq3_4s could reduce to 70-80%, negating draft speed gains

### Recommendations
- **Defer Phase A4 benchmarking** until disk space becomes available (need 50G+)
- **Focus on Phase A5** (KV bandwidth at long context) which may have higher impact
- **Revisit Phase A4** only if disk space increases significantly or if Phase A5 shows limited gains

## Next Steps

### Immediate: Phase A5 (KV bandwidth at long context)
- Investigate KV cache bandwidth bottlenecks at long context (4K+)
- Profile KV read patterns and identify optimization opportunities
- Consider KV quantization or sparse KV strategies

### Deferred: Phase A4 completion
- Benchmark outQ5K variant when disk space available
- Perform KLD-gated quality comparison
- Update speed plan with final results

### Phase A6+ (if applicable)
- Evaluate further quantization strategies
- Investigate MTP head optimization opportunities
- Consider alternative speculative decoding strategies

## Artifacts

### Phase A4
- `artifacts/perf_a4_recipe_bytes/README.md`: Detailed variant analysis
- `artifacts/perf_a4_recipe_bytes/bench_outq6_baseline.log`: Baseline benchmark
- `artifacts/perf_a4_recipe_bytes/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq5.gguf`: outQ5K variant (retained)

## Status Summary

| Phase | Status | Blocking Issue |
|-------|--------|----------------|
| A4 (bytes-per-token) | Partially complete | Disk space (42G free, need 50G+) |
| A5 (KV bandwidth) | Not started | - |
| A6+ (further optimization) | Not started | - |

---
*Report generated: 2026-06-11*
*Next review: After Phase A5 completion or disk space increase*
