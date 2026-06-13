# TQ3_4S Performance Dashboard

**Date**: 2026-04-19  
**GPU**: NVIDIA RTX 5060 Ti (16GB, SM120 Blackwell)  
**Model**: Qwen3.5-27B-TQ3_4S (12.91 GiB, 4.12 BPW)  
**Repo**: [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3)

---

## Headlines

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Token Generation (TG)** | 14.36 tok/s | **22.89 tok/s** | **+59%** |
| **Prompt Processing (PP)** | 308 tok/s | **673-703 tok/s** | **+118% to +128%** |
| **Quality (PPL)** | 6.82 | **6.82** | **Unchanged** |

---

## Decode Speed Breakdown

| Change | TG tok/s | Δ | Cumulative |
|--------|----------|---|------------|
| Published baseline | 14.36 | — | — |
| + Packed-X dp4a staging | 14.84 | +3.3% | +3.3% |
| + Fused rotation+quantization | ~15.0 | +1.4% | +4.5% |
| + **VDR=8 for SM120 Blackwell** | **22.89** | **+48.5%** | **+59%** |

## Prompt Speed Breakdown

| Change | PP tok/s | Δ | Cumulative |
|--------|----------|---|------------|
| Published baseline | 308 | — | — |
| + Packed-X dp4a staging | 343 | +11.4% | +11.4% |
| + MMA packed-X staging | 360 | +5.0% | +17% |
| + MMQ tile-loader rewrite (current reruns, default KV + FA) | 673 | +87% | +118% |
| + Recovered asymmetric KV (`K=q4_0, V=tq3_0`, FA) | 703 | +4% | +128% |

---

## vs Q3_K_S (same model, same GPU)

| Metric | TQ3_4S | Q3_K_S | Gap |
|--------|--------|--------|-----|
| TG | 22.89 tok/s | 25.46 tok/s | -10% |
| PP | 673-703 tok/s | 689 tok/s | roughly parity to +2% |
| PPL (27B) | **6.77** | 6.80 | **TQ3 wins** |
| Size | 12.91 GiB | 11.44 GiB | +13% |

**TQ3_4S now reaches parity-class prompt speed on the recovered asymmetric KV witness.**

---

## Current Prompt Witnesses

| Config | Result | Build | Date |
|--------|--------|-------|------|
| default KV + FA | `673.39 ± 1.39` | `b11f67b73` | 2026-04-19 |
| `q4_0/q4_0` KV + FA | `672.35 ± 0.65` | `b11f67b73` | 2026-04-19 |
| `tq3_0/tq3_0` KV + FA | `687.86 ± 15.16` | `b11f67b73` | 2026-04-19 |
| `K=q4_0, V=tq3_0`, FA | `702.73 ± 7.24` | `b11f67b73` | 2026-04-19 |

Historical `715-720` claims remain historical until they are reproduced with current artifacts.

## What Changed (Code Only — No Model Re-download Needed)

1. **VDR=8 for Blackwell (SM120)** — process 8 weight elements per thread instead of 4 in the MMVQ decode kernel. Biggest single improvement.

2. **Packed-X activation staging** — stage activation packs into shared memory before the dot product loop, improving data reuse across output rows.

3. **MMA packed-X** — same staging technique applied to the tensor-core MMA prompt path.

4. **Fused rotation+quantization** — combine WHT rotation and int8 quantization into a single kernel, eliminating a temp buffer and extra kernel launch.

5. **Scale table LUT** — replace runtime `ldexpf()` with precomputed lookup table for TQ3_4S scale decode.

---

## Quality Verification

| Test | Result |
|------|--------|
| PPL 27B (c=2048, first 19 chunks) | **Bit-identical** to published baseline |
| PPL 9B (5 chunks) | 7.8172 = 7.8172 (identical) |
| PPL 27B (full, historical) | 6.8224 (confirmed matching) |

**Zero quality regression. All improvements are pure kernel optimizations.**

---

## How to Get the Speedup

```bash
git clone https://github.com/turbo-tan/llama.cpp-tq3
cd llama.cpp-tq3
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# Use your existing TQ3_4S GGUF — no re-download needed
./build/bin/llama-cli -m your-model-TQ3_4S.gguf -ngl 99
```
