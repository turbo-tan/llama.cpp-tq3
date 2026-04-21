# Ampere Deployment Showcase

This page records one measured deployment example for `llama.cpp-tq3` on Ampere GPUs.

It is intentionally separate from the main README because the exact topology, model, and numbers are machine-specific.

## Scope

- hardware example: `2 x RTX 3090`
- interconnect: PCIe / PHB, no NVLink
- runtime focus: `TQ3_4S`
- benchmark style: `llama-bench` and warm `llama-server` requests

## Build Preset

Example preset for a clean Ampere-only build:

```bash
cmake --fresh --preset x64-linux-cuda-ampere-release
cmake --build build-x64-linux-cuda-ampere-release -j
```

Recommended options in that preset:

| Flag | Value | Why |
|------|-------|-----|
| `CMAKE_CUDA_ARCHITECTURES` | `86` | target Ampere directly |
| `GGML_CUDA_FA` | `ON` | enable Flash Attention |
| `GGML_CUDA_FA_ALL_QUANTS` | `ON` | allow FA on broader KV cache types |
| `GGML_CUDA_GRAPHS` | `ON` | reduce launch overhead |
| `GGML_CUDA_PEER_MAX_BATCH_SIZE` | `512` | improve multi-GPU peer traffic on this setup |

## Model And Contract

Measured model:

- `Qwopus3.5-27B-v3-Abliterated-TQ3_4S.gguf`

Measured contract:

- `-ctk q8_0 -ctv q8_0`
- `-fa 1`
- `-b 2048`
- `-ub 512`

## Measured Results

`llama-bench`, 3 reps:

| Configuration | `pp512` | `tg128` |
|---------------|--------:|--------:|
| single GPU RTX 3090 | `713.75 ± 16` | `21.81 ± 0.1` |
| dual GPU 0+4, `--split-mode layer`, `--tensor-split 0.5/0.5` | `729.48 ± 5` | `27.81 ± 0.0` |

Observed effect:

- decode improved by about `27.5%`
- prompt processing improved only slightly

This matches the expected shape for PCIe-connected GPUs:

- decode benefits from splitting weight traffic
- prompt processing remains more compute-bound
- `--split-mode layer` is the correct choice on PHB/PCIe
- `--split-mode row` is not recommended without NVLink-class bandwidth

## Example Server Commands

Dual GPU example:

```bash
CUDA_VISIBLE_DEVICES=0,4 \
./build-x64-linux-cuda-ampere-release/bin/llama-server \
  -m /path/to/Qwopus3.5-27B-v3-Abliterated-TQ3_4S.gguf \
  --mmproj /path/to/Qwopus3.5-27B-v3-Abliterated-mmproj.gguf \
  -ngl 99 \
  --split-mode layer \
  --tensor-split 0.5/0.5 \
  --main-gpu 0 \
  -c 112640 \
  -b 512 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  -fa on \
  --host 0.0.0.0 \
  --port 1234
```

Single GPU fallback:

```bash
CUDA_VISIBLE_DEVICES=4 \
./build-x64-linux-cuda-ampere-release/bin/llama-server \
  -m /path/to/Qwopus3.5-27B-v3-Abliterated-TQ3_4S.gguf \
  --mmproj /path/to/Qwopus3.5-27B-v3-Abliterated-mmproj.gguf \
  -ngl 99 \
  -c 112640 \
  -b 512 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  -fa on \
  --host 0.0.0.0 \
  --port 1234
```

## Practical Notes

- Keep deployment notes separate from format-level claims.
- Compare `pp` and `tg` independently.
- Keep custom KV-cache settings separate from base weight-format numbers.
- When publishing benchmark notes, always state topology and cache settings.
