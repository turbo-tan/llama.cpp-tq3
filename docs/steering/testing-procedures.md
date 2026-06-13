# Testing Procedures and Build Guidelines

## Build Procedures

### Local Build with Polling

For long builds, use the polling script to avoid timeouts:

```bash
# Start build in background with polling
cd /home/awee/code/tan_llama
./scripts/build-with-poll.sh llama-speculative 4

# Or manually start and poll
cd build
nohup cmake --build . --target llama-speculative -j2 > build.log 2>&1 &
echo $! > build.pid

# Poll status
for i in {1..12}; do
  sleep 30
  if ps -p $(cat build.pid) > /dev/null 2>&1; then
    echo "[$i] Still building... $(grep -oP '\[\s*\d+%\]' build.log | tail -1)"
  else
    echo "[$i] Build finished"
    tail -30 build.log
    break
  fi
done
```

### Remote GPU Server Build

For building on 192.168.1.77:

```bash
# SSH to remote
ssh awee@192.168.1.77

# Navigate to build directory
cd ~/code/llama.cpp-tq3/build-rpc-cuda12

# Configure and build
cmake .. -DGGML_RPC=ON -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build . --target ggml-rpc rpc-server -- -j12
```

## Test Procedures

### Gemma4 MTP Testing

**Prerequisites**:
- Base model: `gemma-4-E2B-it-F16.gguf` or `gemma-4-12b-it-Q4_K_M.gguf`
- Assistant model: `gemma-4-E2B-it-assistant-BF16.gguf` (must be complete)

**Test 1: Baseline Inference**
```bash
./build/bin/llama-cli \
  -m /home/awee/models/google/gemma-4-E2B-it/gemma-4-E2B-it-F16.gguf \
  -p "Hello" \
  -n 50 \
  --temp 0.7 \
  -ngl 99
```

Expected: ~120 tok/s, coherent output

**Test 2: MTP Speculative Decoding**
```bash
./build/bin/llama-speculative \
  -m /home/awee/models/google/gemma-4-E2B-it/gemma-4-E2B-it-F16.gguf \
  -md /home/awee/models/google/gemma-4-E2B-it-assistant/gemma-4-E2B-it-assistant-BF16.gguf \
  -p "Hello" \
  -n 50 \
  --temp 0.7 \
  -ngl 99
```

Expected: 1.5-2.0x speedup over baseline

### Common Issues

**Issue: Missing tensor 'rope_freqs.weight'**
- Cause: Incomplete or corrupted model file
- Solution: Re-download or re-convert the model

**Issue: Build timeout**
- Cause: CUDA compilation takes 5-10 minutes
- Solution: Use polling script or increase timeout to 600s

**Issue: Out of memory**
- Cause: Model too large for GPU
- Solution: Use quantized model (Q4_K_M) or reduce context size

## Model Locations

### Local Models
```
/home/awee/models/
├── google/
│   ├── gemma-4-E2B-it/gemma-4-E2B-it-F16.gguf (8.7G)
│   └── gemma-4-E2B-it-assistant/gemma-4-E2B-it-assistant-BF16.gguf (163M) ⚠️
├── turboquant/tq3_4s/
│   ├── gemma-4-12b-it-Q4_K_M.gguf (6.9G)
│   └── gemma-4-12b-it-BF16.gguf (23G)
└── eagle3-gemma4/eagle3-gemma4-f16.gguf (1.8G)
```

⚠️ The gemma-4-E2B-it-assistant model is incomplete and cannot be loaded.

### Remote Models (192.168.1.77)
```
~/models/
└── (check for available models)
```

## Git Workflow

### Branch Management

**Important**: Never push to `upstream` or merge into `master`.

For the exact sync sequence between `upstream/master`, `master`, and `main`, see:

- [`rebase-sop.md`](rebase-sop.md)

```bash
# Correct workflow
git checkout main
git checkout -b feature/my-feature
# ... make changes ...
git commit -m "feat: add my feature"
git push origin feature/my-feature

# Create PR against main (not master)
gh pr create --base main --head feature/my-feature
```

### Remote Repositories

```bash
origin  git@github.com:charpdev/tan_llama.git (your fork)
turbo   git@github.com-turbotan:turbo-tan/llama.cpp-tq3.git (turbo's fork)
upstream https://github.com/ggml-org/llama.cpp (upstream, read-only)
```

**Rules**:
- ✅ Push to `origin` (your fork)
- ✅ Push to `turbo` (turbo's fork, with permission)
- ❌ NEVER push to `upstream`
- ❌ NEVER create PRs against `upstream`
- ❌ NEVER merge into `master` (must stay synced with upstream/master)

## Documentation

### Test Reports

After testing, create a report in `docs/experiments/`:

```markdown
# [Feature] Test Report

**Date**: YYYY-MM-DD
**Commit**: abc123
**Branch**: feature/xxx

## Summary
Brief overview of test results.

## Test Results
### Test 1: [Name]
- Status: PASSED/FAILED
- Command: ...
- Results: ...

## Conclusions
What works, what doesn't, next steps.
```

### SOP Updates

When discovering new procedures, update the relevant SOP:
- `docs/procedure/compile-remote-gpu-server.md` - Remote build
- `docs/procedure/test-gemma4-mtp.md` - Gemma4 MTP testing
- `docs/steering/testing-procedures.md` - This file

## Performance Benchmarks

### Expected Performance (RTX 3090)

| Model | Baseline (tok/s) | MTP (tok/s) | Speedup |
|-------|------------------|-------------|---------|
| gemma-4-E2B-it (F16) | 120-140 | 180-280 | 1.5-2.0x |
| gemma-4-12b-it (Q4_K_M) | 140-160 | 210-320 | 1.5-2.0x |
| supergemma4-26b (TQ3_4S) | 60-80 | 90-160 | 1.5-2.0x |

### Memory Requirements

| Model | VRAM | Notes |
|-------|------|-------|
| gemma-4-E2B-it (F16) | 9 GB | Fits in RTX 3090 |
| gemma-4-12b-it (Q4_K_M) | 8 GB | Fits in RTX 3090 |
| supergemma4-26b (TQ3_4S) | 14 GB | Fits in RTX 3090 |
| MTP overhead | +0.5-2 GB | For assistant model |

## Troubleshooting

### Build Issues

**Problem**: Build times out
- Solution: Use polling script or increase timeout to 600s

**Problem**: CUDA compilation fails
- Solution: Check CUDA version (need 12.9), verify GPU architecture

**Problem**: Out of memory during build
- Solution: Reduce parallel jobs (`-j2` instead of `-j4`)

### Runtime Issues

**Problem**: Model fails to load
- Solution: Check model file integrity, verify all tensors present

**Problem**: MTP slower than baseline
- Solution: Check acceptance rate, verify assistant model quality

**Problem**: Segmentation fault
- Solution: Run with gdb, check for null pointers, enable debug build

## Quick Reference

### Common Commands

```bash
# Build
cmake --build build --target llama-speculative -j4

# Test baseline
./build/bin/llama-cli -m model.gguf -p "prompt" -n 50 -ngl 99

# Test MTP
./build/bin/llama-speculative -m base.gguf -md draft.gguf -p "prompt" -n 50 -ngl 99

# Check GPU
nvidia-smi

# Monitor build
tail -f build/build.log

# Create PR
gh pr create --base main --head feature/xxx
```

### File Locations

```
build/bin/llama-speculative    # MTP binary
build/bin/llama-cli            # Baseline binary
/home/awee/models/             # Model directory
docs/experiments/              # Test reports
docs/procedure/                # SOPs
scripts/build-with-poll.sh     # Build polling script
```

---

**Last Updated**: 2026-06-10
**Maintained By**: turbo-tan
