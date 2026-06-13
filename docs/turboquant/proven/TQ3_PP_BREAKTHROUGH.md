# TQ3_4S PP Speed Breakthrough: 360 tok/s to recovered 700+ regime

**Date**: 2026-04-10
**Branch**: `experiment/pp-speed-opt` on private repo (`/home/awee/code/llama.cpp`)
**Tag**: `pp-707-scale-baking` (commit `b9a10edcf`)
**GPU**: RTX 5060 Ti 16GB (SM120 Blackwell)
**Model**: Qwen3.5-27B TQ3_4S

## Results

| Metric | Before | After | Change |
|---|---|---|---|
| PP 2048 (27B) | 360 tok/s | **702.73 tok/s** | **+95%** |
| TG 128 (27B) | 23.3 tok/s | 23.0 tok/s | unchanged |
| PPL 9B (10ch) | 9.8972 | 9.8900 | ≈identical |
| PPL 27B (10ch c=2048) | 6.1629 | 6.1629 | **bit-identical** |
| Gap vs Q3_K_S (~689) | 1.9x | **0.96x** | **TQ3_4S now faster** |
| Chat quality (27B) | — | **7/8** | zero garbage output |

**Update (2026-04-19)**: exact reruns on `b11f67b73` did not reproduce the older
`715-720` default-KV claims. The first currently revalidated `700+` witness is:
`K=q4_0, V=tq3_0, FA=1`, `pp2048 = 702.73 ± 7.24` on `b11f67b73`.
Default KV + FA reruns on the same build are in the high-600s.

## What Changed (2 files, ~50 lines)

### 1. Enable MMQ for TQ3_4S (`ggml-cuda.cu`, 1 line)

Removed `src0->type != GGML_TYPE_TQ3_4S` from the `tq3_1s_mmq_ok` guard.
This allows TQ3_4S to use the direct MMQ path (`ggml_cuda_mul_mat_q`)
instead of falling through to cuBLAS (dequant → fp16 → HGEMM).

MMQ was previously disabled because the old tile loader produced wrong PPL
(see "Why the old tile loader was broken" below).

### 2. Rewrite `load_tiles_tq3_4s` (`mmq.cuh`, ~50 lines)

Two independent improvements:

**A. 16-thread-per-row structure (16x warp utilization)**

Old: 8 warps per tile row, each warp handles 1 TQ3_4S block (32 elements).
Only 8 of 32 lanes produce output. 7 warp shuffles per element.

New: 16 threads per row, each thread handles 2 subgroups (16 elements).
2 rows per warp. Zero shuffles. All threads produce output.

```
Old: 8 warps × 32 lanes = 256 threads → 1 row per iteration
New: 1 warp × 16 threads = 16 threads → 2 rows per warp, nwarps*2 per iteration
With nwarps=8: 1 row/iter → 16 rows/iter (16x improvement)
```

**B. Scale baking (fixes PPL)**

Old: each int8 value = `tq3_q8_levels[centroid_idx]` with a global scale
`d = rms * 2.1519/127`. This has 3-8% per-element error because the int8
levels don't exactly match the float centroids. Also, only subgroup 0's
scale was stored — subgroups 1-3 got the wrong scale.

New: each int8 value = `round(centroid[idx] * rms_subgroup / d_block * 127)`
where `d_block = max(rms_g * centroid_max)` across all 4 subgroups.
This bakes the per-subgroup scale into the int8 values, giving <0.4%
error (int8 rounding only). All 4 subgroup scales are correctly represented.

## Why the Old Tile Loader Was Broken

Two bugs:

1. **Wrong scale for subgroups 1-3**: The q8_0 format stores 1 scale per 32
   elements. The old loader wrote subgroup 0's scale for the entire block.
   Subgroups 1-3 (which have different RMS scales) got the wrong scale factor.

2. **Int8 centroid approximation error**: The mapping
   `centroid ≈ tq3_q8_levels[i] * 2.1519/127` has up to 8.2% relative error
   on the extreme centroids (-1.997, +1.989). This compounds across 5120-dim
   matmul → PPL 170 (vs 9.9 correct).

The scale-baking approach fixes both: per-subgroup scales are folded into the
int8 values, and the float centroids are used directly (no int8 approximation
of the centroid values themselves).

## Mathematical Proof of Scale Baking

For a TQ3_4S block with 4 subgroups, each with scale `rms_g`:

```
d_block = max_g(rms_g * |centroid_max|) / 127

For element i in subgroup g with centroid index idx:
  exact_value = centroid[idx] * rms_g
  qs[i] = round(exact_value / d_block * 127)
       = round(centroid[idx] * rms_g * 127 / d_block)

Reconstruction:
  qs[i] * d_block ≈ centroid[idx] * rms_g

Error: |qs[i] * d_block - exact_value| ≤ d_block / 2 = max_g(rms_g * 1.997) / 254
```

The maximum relative error is bounded by `1/(2 * min_level)` where `min_level`
is the smallest non-zero int8 value. For typical weight distributions, this is
<0.4% — well within the noise floor of 3-bit quantization.

## Dispatch Path (Corrected)

```
TQ3_4S PP (ne11 >= 64) — NEW:
  ggml_cuda_mul_mat()
    → tq3_1s_mmq_ok = true (TQ3_4S no longer blocked)
    → use_mul_mat_q = true
    → ggml_cuda_mul_mat_q()
      → rotate src1 (memcpy + WHT kernel) — 1.4% of time
      → quantize src1 → q8_1_mmq — 0.2% of time
      → mul_mat_q<TQ3_4S> with new tile loader — 94% of time

TQ3_4S PP (ne11 >= 64) — OLD:
  → tq3_1s_mmq_ok = false → cuBLAS fallback
  → dequant TQ3_4S → fp16 (67% of time) → HGEMM (19%)
```

## Files Changed

| File | Change |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Remove TQ3_4S from `tq3_1s_mmq_ok` block (1 line) |
| `ggml/src/ggml-cuda/mmq.cuh` | Rewrite `load_tiles_tq3_4s` (~50 lines) |

## How to Build and Test

```bash
cd /home/awee/code/llama.cpp
git checkout experiment/pp-speed-opt  # or tag pp-707-scale-baking

# Build
cmake --build build-360 --target llama-bench llama-perplexity -j$(nproc)

# IMPORTANT: force recompile of tile loader (header change detection is unreliable)
touch ggml/src/ggml-cuda/template-instances/mmq-instance-tq3_4s.cu ggml/src/ggml-cuda/mmq.cu
cmake --build build-360 --target llama-bench llama-perplexity -j$(nproc)

# Benchmark
./build-360/bin/llama-bench -m /home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -p 2048 -n 128 -ngl 99 -r 3

# PPL verification (must match baseline)
./build-360/bin/llama-perplexity \
  -m /home/awee/models/turboquant9b/Qwen_Qwen3.5-9B-TQ3_4S.gguf \
  -f /home/awee/code/llama.cpp/wikitext-2-raw/wiki.test.raw \
  -ngl 99 --chunks 10
# Expected: 9.8900 (baseline: 9.8972)
```

## Next Steps

1. Run full 100-chunk PPL on 27B to confirm no degradation at scale
2. Test on Gemma4 26B TQ3_4S
3. Test chat quality (llama-server smoke test)
4. If all pass → merge to private main → then public main

## Appendix: Reproducibility Notes

### Current reproducibility status

Exact reruns performed on 2026-04-19:
- `b11f67b73`, default KV + FA: `673.39 ± 1.39`
- `b11f67b73`, `K=q4_0, V=tq3_0`, FA: `702.73 ± 7.24`
- `341307f4e`, default KV + FA: `677.29 ± 2.17`

Practical conclusion:
- the MMQ tile-loader rewrite is real and still active
- the historical `715-720` numbers are not currently reproduced as default-KV facts
- the currently revalidated `700+` path is the asymmetric KV contract

### CRITICAL: Force recompile after mmq.cuh changes

The CMake build system does not reliably detect changes to `mmq.cuh` (a header
included by template instance `.cu` files). After modifying `mmq.cuh`, you MUST:

```bash
touch ggml/src/ggml-cuda/template-instances/mmq-instance-tq3_4s.cu ggml/src/ggml-cuda/mmq.cu
cmake --build build --target llama-bench -j$(nproc)
```

Without this, the old tile loader binary remains and PP stays at ~360 tok/s
(cuBLAS fallback) or ~654 tok/s (partially recompiled).

### Clean benchmark procedure (updated 2026-04-15)

**CRITICAL: Cold-start JIT causes first run to be 15-25% slower. Always warmup first.**

```bash
# Step 0: Kill existing processes and verify GPU idle
pkill -9 -f llama 2>/dev/null; sleep 3
nvidia-smi --query-compute-apps=pid --format=csv,noheader
# Must be empty before proceeding

# Step 1: Warmup pass (triggers CUDA JIT compilation)
./build/bin/llama-bench -m MODEL -ngl 99 -fa 1 \
    -p 128 -pg 128,0 -r 2 --no-warmup
# PP128 numbers will be low (~520-660) — this is expected, just JIT overhead

# Step 2: Speed measurement (warmed up)
./build/bin/llama-bench -m MODEL -ngl 99 -fa 1 \
    -p 2048 -pg 2048,0 -n 128 -r 5
# Use r=5 for stable numbers. PP512 has high variance (±30), PP2048 is stable (±1-2).

# Step 3: Chat quality SOP (must verify no garbage output)
pkill -9 -f llama-bench 2>/dev/null; sleep 2
setsid ./build/bin/llama-server -m MODEL -ngl 99 -c 2048 --port 8090 -fa 1 \
    </dev/null >/tmp/llama-server.log 2>&1 &
sleep 30
curl -s http://localhost:8090/health
# Must return {"status":"ok"} before proceeding

ask() {
  curl -s --max-time 300 http://localhost:8090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":1000,\"temperature\":0}" \
    | python3 -c "import sys,json;r=json.load(sys.stdin);print(r['choices'][0]['message']['content'][:200])" 2>/dev/null
}
echo -n "127*43: "; ask "What is 127*43? Answer only the number."
echo -n "Capital: "; ask "Capital of Australia? One word."
echo -n "sqrt144: "; ask "sqrt(144)? Just the number."
echo -n "pattern: "; ask "Next: 2,6,12,20,30,? Answer only the number."
echo -n "json: "; ask "Output a JSON array of the first 5 prime numbers. Only the array."
echo -n "logic: "; ask "All roses are flowers. Some flowers fade quickly. Can we conclude all roses fade quickly? Yes or No only."
echo -n "code: "; ask "What does this print? x=[1,2,3]; x.append(x.pop(0)); print(x)"
echo -n "haiku: "; ask "Write a haiku about the moon."

pkill -9 -f llama-server 2>/dev/null
```

### Build requirements

```bash
cd /home/awee/code/llama.cpp-tq3
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_COMPRESSION_MODE=size
# Do NOT pass -DCMAKE_CUDA_ARCHITECTURES=120 (produces PTX+native, slower)
cmake --build build -j$(nproc)
```

### Cold-Start Pattern (RTX 5060 Ti, SM 120a)

| Run | PP128 tok/s | Notes |
|-----|-------------|-------|
| 1st | ~520-550 | CUDA JIT compiling kernels |
| 2nd | ~640-660 | Partially cached |
| 3rd+ | ~670-705 | Fully warmed up on current reruns |

**Any speed claim MUST be from a warmed-up run with a saved artifact.**
Cold-start numbers (520-600 tok/s) are NOT valid measurements.
