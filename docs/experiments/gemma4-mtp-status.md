# Gemma4 MTP Implementation Status

**Date**: 2026-06-10  
**Status**: IN PROGRESS - Critical findings identified, needs architectural alignment

## Executive Summary

We implemented embeddings_nextn infrastructure for gemma4 multi-token prediction (MTP) speculative decoding. The infrastructure is correct and functional, but the model implementation has fundamental architectural differences from upstream that prevent it from working with actual gemma4-assistant models.

## What Works

✅ **embeddings_nextn Infrastructure**
- Core infrastructure for extracting hidden states before output norm
- Integrated with speculative decoding framework
- Matches upstream implementation
- Can be reused after architectural alignment

✅ **Build & Testing**
- Code compiles successfully
- Baseline inference works (122.3 tok/s on RTX 3090)
- Test infrastructure in place

✅ **Documentation**
- Comprehensive analysis documents created
- Testing procedures documented
- Upstream comparison completed

## What Doesn't Work

❌ **Model Loading**
- Incompatible with actual gemma4-assistant models
- Wrong architecture name (gemma4_mtp vs gemma4-assistant)
- Missing ctx_other integration
- Different tensor structure

## Critical Findings

After comparing with upstream master, we discovered:

1. **Architecture Name**: Upstream uses `LLM_ARCH_GEMMA4_ASSISTANT`, we used `LLM_ARCH_GEMMA4_MTP`
2. **ctx_other Integration**: Upstream uses `ctx_other` to access target model embeddings, we don't
3. **Model Structure**: Our implementation doesn't match the actual model's tensor layout

**Impact**: Our implementation cannot load actual gemma4-assistant models from HuggingFace.

## Recommendation

**Align with upstream master** (3-5 days effort):
1. Rename architecture to `LLM_ARCH_GEMMA4_ASSISTANT`
2. Add `ctx_other` integration
3. Update model structure to match upstream
4. Reuse our embeddings_nextn infrastructure

## Files

### Documentation (in tan_llama/docs/experiments/)
- `gemma4-mtp-test-report.md` - Build and test results
- `gemma4-mtp-model-compatibility.md` - Model loading issues
- `gemma4-mtp-upstream-comparison.md` - Critical architectural analysis

### Code (in turbo-tan/llama.cpp-tq3 PR #27)
- `src/llama-graph.h` - Added t_h_nextn tensor
- `src/llama-cparams.h` - Added embeddings_nextn parameters
- `src/llama-context.h/cpp` - Implemented nextn extraction
- `src/llama-ext.h` - Added C API
- `src/models/gemma4.cpp` - Set t_h_nextn
- `common/speculative.cpp` - MTP integration

### Testing Infrastructure (in tan_llama)
- `docs/steering/testing-procedures.md` - Testing SOP
- `scripts/build-with-poll.sh` - Build polling script

## Next Steps

1. **Immediate**: Review upstream comparison document
2. **Short-term**: Create new branch to align with upstream
3. **Medium-term**: Implement architectural changes
4. **Long-term**: Test with actual gemma4-assistant models

## References

- PR #27: https://github.com/turbo-tan/llama.cpp-tq3/pull/27
- Upstream master: https://github.com/ggml-org/llama.cpp
- Model: https://huggingface.co/sjakek/gemma4-12b-mtp-assistant

---

**Last Updated**: 2026-06-10  
**Next Review**: Before starting architectural alignment
