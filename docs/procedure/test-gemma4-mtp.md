# SOP: Testing Gemma4 MTP Support

## Overview

This document describes the standard operating procedure for testing gemma4 multi-token prediction (MTP) speculative decoding support.

**Purpose**: Verify that the embeddings_nextn infrastructure correctly enables gemma4 MTP speculative decoding.

---

## Prerequisites

### 1. Required Models

You need two gemma4 models:
- **Backbone model**: gemma4 base model (e.g., `gemma-4-12b-it.gguf`)
- **Draft model**: gemma4 assistant/MTP model (e.g., `gemma-4-12b-it-assistant.gguf`)

**Where to get them**:
- Check `/models/` directory
- Check `~/models/` directory
- Download from Hugging Face: `google/gemma-4-12b-it` and convert with `convert_hf_to_gguf.py`

### 2. Built Binaries

Ensure you have built the following targets:
```bash
cmake --build build --target llama-speculative -j4
cmake --build build --target llama-cli -j4
```

**Expected binaries**:
- `build/bin/llama-speculative` - Main speculative decoding binary
- `build/bin/llama-cli` - Standard inference binary (for baseline)

### 3. GPU Availability

Check GPU memory:
```bash
nvidia-smi
```

**Requirements**:
- For 12B model: ~8 GB VRAM
- For 27B model: ~16 GB VRAM
- For MTP: Add ~2 GB for draft model

---

## Test Procedure

### Test 1: Baseline Inference (No MTP)

**Purpose**: Verify the backbone model works correctly without MTP.

```bash
./build/bin/llama-cli \
  -m /models/gemma-4-12b-it.gguf \
  -p "The capital of France is" \
  -n 50 \
  --temp 0.7 \
  -ngl 99
```

**Expected output**:
- Model loads successfully
- Generates coherent text
- Reports tokens/sec (should be ~40-80 tok/s on RTX 3090)

**Record**:
- Load time
- Generation speed (tok/s)
- Output text (first 100 tokens)

### Test 2: MTP Speculative Decoding

**Purpose**: Verify MTP speculative decoding works with gemma4.

```bash
./build/bin/llama-speculative \
  -m /models/gemma-4-12b-it.gguf \
  -md /models/gemma-4-12b-it-assistant.gguf \
  -p "The capital of France is" \
  -n 50 \
  --temp 0.7 \
  -ngl 99
```

**Expected output**:
- Both models load successfully
- No errors about missing `t_h_nextn` or `embeddings_nextn`
- Generation speed should be **1.5-2.5x faster** than baseline
- Output text should be coherent and similar to baseline

**Record**:
- Load time (both models)
- Generation speed (tok/s)
- Acceptance rate (if reported)
- Output text (first 100 tokens)
- Speedup factor vs baseline

### Test 3: Long Context MTP

**Purpose**: Verify MTP works with longer contexts.

```bash
./build/bin/llama-speculative \
  -m /models/gemma-4-12b-it.gguf \
  -md /models/gemma-4-12b-it-assistant.gguf \
  -p "Write a detailed explanation of how neural networks work, covering the following topics: $(cat /dev/urandom | tr -dc 'a-zA-Z' | head -c 1000)" \
  -n 200 \
  -c 4096 \
  --temp 0.7 \
  -ngl 99
```

**Expected output**:
- No memory errors
- Stable generation speed
- No crashes or segfaults

**Record**:
- Peak VRAM usage
- Generation speed over time
- Any errors or warnings

### Test 4: Batch Processing

**Purpose**: Verify MTP works with batched prompts.

```bash
# Create a batch file
cat > /tmp/batch.txt << 'EOF'
The capital of France is
The largest planet in our solar system is
The chemical formula for water is
The author of Romeo and Juliet is
EOF

./build/bin/llama-speculative \
  -m /models/gemma-4-12b-it.gguf \
  -md /models/gemma-4-12b-it-assistant.gguf \
  -f /tmp/batch.txt \
  -n 20 \
  --temp 0.7 \
  -ngl 99 \
  -b 4
```

**Expected output**:
- All prompts processed
- No errors
- Reasonable speed

**Record**:
- Total processing time
- Average speed per prompt
- Any errors

---

## Validation Checklist

### Build Validation
- [ ] Code compiles without errors
- [ ] No warnings about unused variables or functions
- [ ] Binary size is reasonable (~50-100 MB)

### Runtime Validation
- [ ] Backbone model loads successfully
- [ ] Draft model loads successfully
- [ ] No errors about missing tensors (`t_h_nextn`, `embeddings_nextn`)
- [ ] No segmentation faults
- [ ] No CUDA errors

### Functional Validation
- [ ] Baseline inference works (Test 1)
- [ ] MTP inference works (Test 2)
- [ ] MTP provides speedup (1.5-2.5x expected)
- [ ] Output quality is maintained (coherent text)
- [ ] Long context works (Test 3)
- [ ] Batch processing works (Test 4)

### Performance Validation
- [ ] Generation speed meets expectations
- [ ] Memory usage is reasonable
- [ ] No memory leaks (check with `nvidia-smi` over time)
- [ ] No performance degradation over long runs

---

## Troubleshooting

### Problem: "t_h_nextn not found" Error

**Symptom**:
```
error: tensor 't_h_nextn' not found in model
```

**Cause**: The gemma4 model doesn't set `t_h_nextn` in the graph.

**Solution**: 
- Verify you're using the correct gemma4 model
- Check that `src/models/gemma4.cpp` has `res->t_h_nextn = cur;`
- Rebuild the code

### Problem: "embeddings_nextn not supported" Error

**Symptom**:
```
error: embeddings_nextn not supported by this model
```

**Cause**: The model architecture doesn't support nextn embeddings.

**Solution**:
- Verify you're using gemma4 (not gemma3 or other variants)
- Check model metadata: `./build/bin/llama-cli -m model.gguf --verbose-prompt`

### Problem: MTP Slower Than Baseline

**Symptom**: MTP speed is less than baseline speed.

**Cause**: 
- Draft model is too large
- Acceptance rate is too low
- Network overhead (if using RPC)

**Solution**:
- Try a smaller draft model
- Check acceptance rate (should be >50%)
- Verify GPU is not memory-bound (`nvidia-smi`)

### Problem: Out of Memory

**Symptom**:
```
CUDA error: out of memory
```

**Cause**: Not enough VRAM for both models.

**Solution**:
- Use smaller models
- Reduce context length (`-c`)
- Reduce batch size (`-b`)
- Use CPU offloading (`-ngl 50` instead of `-ngl 99`)

### Problem: Segmentation Fault

**Symptom**: Process crashes with segfault.

**Cause**: Memory corruption or null pointer dereference.

**Solution**:
- Run with `gdb` to get stack trace
- Check for null pointers in the code
- Verify tensor dimensions match expectations
- Enable debug build: `cmake -DCMAKE_BUILD_TYPE=Debug ..`

---

## Performance Benchmarks

### Expected Performance (RTX 3090)

| Model | Baseline (tok/s) | MTP (tok/s) | Speedup |
|-------|------------------|-------------|---------|
| gemma-4-12b-it | 60-80 | 120-160 | 2.0x |
| gemma-4-27b-it | 30-40 | 60-80 | 2.0x |

**Notes**:
- Actual performance depends on prompt length, context size, and GPU
- MTP speedup is typically 1.5-2.5x for gemma4
- Longer contexts may reduce speedup due to memory bandwidth

### Acceptance Rate

**Expected**: 50-70% for gemma4 MTP

**How to measure**:
```bash
./build/bin/llama-speculative ... 2>&1 | grep "acceptance rate"
```

**If acceptance rate is low (<30%)**:
- Draft model may not be compatible
- Try a different draft model
- Check that models are from the same family

---

## Reporting Results

After running tests, create a report with:

```markdown
# Gemma4 MTP Test Report

**Date**: YYYY-MM-DD
**Commit**: abc123
**GPU**: RTX 3090 (24 GB)
**Models**: gemma-4-12b-it + gemma-4-12b-it-assistant

## Results

### Test 1: Baseline
- Load time: 12.3s
- Generation speed: 68 tok/s
- Output: "The capital of France is Paris, which is located..."

### Test 2: MTP
- Load time: 18.7s (both models)
- Generation speed: 142 tok/s
- Acceptance rate: 62%
- Speedup: 2.09x
- Output: "The capital of France is Paris, which is located..."

### Test 3: Long Context
- Peak VRAM: 14.2 GB
- Generation speed: 135 tok/s (stable)
- No errors

### Test 4: Batch Processing
- Total time: 8.4s
- Average speed: 128 tok/s per prompt
- No errors

## Conclusion

✓ All tests passed
✓ MTP provides 2.09x speedup
✓ No errors or crashes
✓ Output quality maintained
```

---

## Quick Reference

### Common Commands

```bash
# Check GPU
nvidia-smi

# Build
cmake --build build --target llama-speculative -j4

# Run baseline
./build/bin/llama-cli -m model.gguf -p "prompt" -n 50 -ngl 99

# Run MTP
./build/bin/llama-speculative -m backbone.gguf -md draft.gguf -p "prompt" -n 50 -ngl 99

# Check model info
./build/bin/llama-cli -m model.gguf --verbose-prompt 2>&1 | head -50

# Monitor GPU during run
watch -n 1 nvidia-smi
```

### File Locations

```
build/bin/llama-speculative    # MTP binary
build/bin/llama-cli            # Baseline binary
/models/                       # Model directory
~/models/                      # Alternative model directory
```

---

**Last Updated**: 2026-06-10
**Maintained By**: turbo-tan
**Git Commit**: (update after merge)
