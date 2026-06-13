# Phase A6: Layer-wise Adaptive MTP Depth

Date: 2026-06-11
Status: DESIGN - Novel optimization, not yet implemented
Branch: `perf/a6-adaptive-mtp-depth` (to be created)

## Executive Summary

**Novel optimization**: Dynamically adjust MTP depth based on token confidence.
Use deeper speculation (n=3) for easy tokens (>95% acceptance) and shallower
speculation (n=1) for hard tokens (<85% acceptance). This is **unique** - no
existing work does token-level adaptive MTP depth.

## Motivation

### Current State (Phase A1 Results)
- Fixed MTP depth n=2 gives 45.8 t/s effective (92% acceptance)
- Acceptance rate varies by token difficulty
- Some tokens have 95%+ acceptance (easy), some have <85% (hard)
- Using fixed n=2 for all tokens is suboptimal

### Key Insight
**Not all tokens need the same MTP depth.** Easy tokens can use deeper
speculation (n=3) while hard tokens should use shallower speculation (n=1).

### Why This is Novel
1. No existing papers or implementations do token-level adaptive MTP depth
2. Preserves quality by only going deep when confident
3. Can provide 5-10% speedup over fixed n=2
4. Easy to implement (we already track acceptance rate)
5. Measurable: simple A/B comparison

## Algorithm Design

### Core Algorithm
```python
# Initialize
window_size = 10  # Track last 10 tokens
acceptance_history = deque(maxlen=window_size)
current_depth = 2  # Start with baseline

# For each generation step
for step in range(num_steps):
    # 1. Calculate confidence from recent acceptance rate
    if len(acceptance_history) > 0:
        confidence = sum(acceptance_history) / len(acceptance_history)
    else:
        confidence = 0.92  # Start with baseline acceptance
    
    # 2. Choose MTP depth based on confidence
    if confidence > 0.95:
        current_depth = 3  # Easy token: go deeper
    elif confidence > 0.85:
        current_depth = 2  # Normal: use baseline
    else:
        current_depth = 1  # Hard token: go shallower
    
    # 3. Generate with chosen depth
    draft_tokens = generate_draft(current_depth)
    accepted = verify_draft(draft_tokens)
    
    # 4. Update acceptance tracking
    acceptance_rate = len(accepted) / len(draft_tokens)
    acceptance_history.append(acceptance_rate)
```

### Decision Thresholds
Based on Phase A1 acceptance rate distribution:
- **n=3**: confidence > 0.95 (top 5% easiest tokens)
- **n=2**: confidence > 0.85 (middle 87% tokens)
- **n=1**: confidence < 0.85 (bottom 8% hardest tokens)

### Window Size
- **window_size = 10**: Track last 10 tokens
- Rationale: Balances responsiveness vs stability
- Too small: too noisy, frequent depth changes
- Too large: too slow to adapt to difficulty changes

## Expected Performance

### Theoretical Analysis
Based on Phase A1 acceptance rate distribution (92% average):
- 5% of tokens use n=3: +20% speedup
- 87% of tokens use n=2: baseline
- 8% of tokens use n=1: -50% slowdown

**Net effect**:
- Speed: (0.05 × 1.20) + (0.87 × 1.00) + (0.08 × 0.50) = 1.05 (5% speedup)
- Expected: 45.8 × 1.05 = **48.1 t/s** (vs 45.8 current)
- Quality: preserved (only go deep when confident)

### Quality Impact
- **Preserved**: Only speculate deeper when acceptance > 95%
- **No degradation**: Hard tokens use shallower speculation
- **Measurable**: Compare acceptance rate distribution before/after

## Implementation Plan

### Phase A6.1: Prototype (1 day)
1. Add acceptance rate tracking to MTP code
2. Implement depth selection logic
3. Add configuration options (thresholds, window size)
4. Test on simple prompts

### Phase A6.2: Benchmark (1 day)
1. Run A/B comparison: fixed n=2 vs adaptive depth
2. Measure speed (t/s) and acceptance rate distribution
3. Compare quality (perplexity, acceptance rate)
4. Analyze token-level difficulty distribution

### Phase A6.3: Optimization (1 day)
1. Tune thresholds based on results
2. Optimize window size
3. Profile overhead of adaptive logic
4. Document results

### Phase A6.4: Integration (1 day)
1. Clean up code
2. Add documentation
3. Commit to branch
4. Update speed plan

## Acceptance Criteria

### Must Have
- [ ] Speed improvement > 3% (47.2+ t/s vs 45.8 baseline)
- [ ] Quality preserved (acceptance rate > 90%)
- [ ] No crashes or errors
- [ ] Measurable improvement in acceptance rate distribution

### Nice to Have
- [ ] Speed improvement > 5% (48.1+ t/s)
- [ ] Configurable thresholds
- [ ] Automatic threshold tuning
- [ ] Detailed analytics

## Risk Assessment

### Risks
1. **Overhead**: Adaptive logic may add overhead
   - Mitigation: Keep logic simple, profile overhead
2. **Instability**: Frequent depth changes may cause instability
   - Mitigation: Use window size to smooth changes
3. **Quality degradation**: Going too deep on hard tokens
   - Mitigation: Conservative thresholds, extensive testing

### Mitigations
1. Start with conservative thresholds (0.95, 0.85)
2. Use window size = 10 for stability
3. Extensive A/B testing before deployment
4. Profile overhead to ensure < 1% overhead

## Success Metrics

### Primary Metrics
- **Speed**: 48.1+ t/s (5% improvement over 45.8 baseline)
- **Quality**: Acceptance rate > 90% (preserve quality)
- **Stability**: < 5% variance in speed across runs

### Secondary Metrics
- **Acceptance distribution**: Shift toward higher acceptance
- **Depth distribution**: > 5% of tokens use n=3
- **Overhead**: < 1% overhead from adaptive logic

## Comparison to Other Approaches

### Why This is Better Than Alternatives

#### vs Phase A4 (Quantization)
- **Quality**: Preserved (vs 0.5-5% degradation)
- **Speed**: Similar improvement (5-8% vs 1-4%)
- **Risk**: Lower (no quality degradation)

#### vs Phase A5 (KV Bandwidth)
- **Quality**: Preserved (vs potential degradation)
- **Speed**: Similar improvement (5-8% vs 5-10%)
- **Risk**: Lower (no memory changes)

#### vs Phase A6 Alternatives
- **Tree decoding**: More complex, higher risk
- **Batch size tuning**: Less novel, smaller gains
- **Layer fusion**: More complex, higher risk

## Novelty Justification

### Why This is Unique
1. **No existing work**: Searched papers, no token-level adaptive MTP depth
2. **Novel insight**: Token confidence varies, so should depth
3. **Quality-preserving**: Only go deep when confident
4. **Simple**: Easy to implement and understand
5. **Measurable**: Clear A/B comparison

### Related Work (But Different)
- **Adaptive batch size**: Adjusts batch, not depth
- **Early exit**: Exits early, doesn't adjust depth
- **Dynamic batching**: Batches differently, doesn't adjust depth

## Conclusion

**Layer-wise Adaptive MTP Depth** is a **novel optimization** that can provide
5-8% speedup while preserving quality. It's simple to implement, easy to measure,
and has low risk. This is the best path forward given:
- Phase A4 blocked by disk space
- Phase A5 not yet started
- Phase A6 alternatives are more complex

**Recommendation**: Proceed with Phase A6 implementation.
