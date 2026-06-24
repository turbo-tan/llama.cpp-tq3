# Running TurboQuant `TQ3_4S` on Apple Silicon (Metal)

`TQ3_4S` (ggml type 46) was originally CUDA-only. This documents the Metal GPU
support and how to run a `TQ3_4S` model on an Apple Silicon Mac.

## Build

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build-metal --target llama-completion llama-server -j
```

### Troubleshooting: `unknown type name 'uuid_string_t'`

If the build fails compiling Accelerate/CoreServices users (`ggml-cpu/vec.cpp`,
`ggml-blas.cpp`) with `error: unknown type name 'uuid_string_t'` in the macOS SDK
`hfs/hfs_format.h`, a Homebrew package (commonly `util-linux`) has force-linked a
`uuid/uuid.h` into `/usr/local/include`, which clang searches before the SDK and
which lacks `uuid_string_t`. Fix by unlinking it:

```bash
brew unlink util-linux      # keg-only; reversible with: brew link --force util-linux
# verify it now resolves to the Xcode SDK header:
echo '#include <uuid/uuid.h>' | clang -E -x c - | grep -m1 uuid/uuid.h
```

## Run

```bash
build-metal/bin/llama-completion \
  -m model-TQ3_4S.gguf \
  -ngl 99 -c 4096 \
  -p "your prompt"
```

Notes:
- This fork's CLI is `llama-completion` (it rejects `-no-cnv`; use it directly).
- Some `qwen35`/`qwen36` `TQ3_4S` quants drop the MTP/`nextn` layer while still
  declaring `nextn_predict_layers > 0`. If loading fails with
  `missing tensor 'blk.N.attn_norm.weight'`, add:
  `--override-kv llama.nomtp_trunk_only=bool:true`.

## What is accelerated

| Op | Metal path |
| --- | --- |
| `MUL_MAT`, decode (`ne11 == 1`) | coalesced mat-vec (`kernel_mul_mv_tq3_4s_f32`) |
| `MUL_MAT`, batch (`ne11 > 1`, prefill / spec verify) | `simdgroup_matrix` GEMM (`kernel_mul_mm_tq3_4s_f32`) |
| `GET_ROWS` | `kernel_get_rows_tq3_4s` |

TQ3_4S dequant applies a per-32-element randomized Hadamard transform (RHT). To
keep the weight matmul coalesced, the RHT is applied once to the activation in a
pre-pass (`kernel_tq3_4s_rht_f32`); by RHT orthogonality the weights then need
only a local codebook lookup. No MoE `_id` kernels yet (dense models only).

Indicative throughput on an M3 Pro (150 GB/s) for a dense 27B `TQ3_4S`:
decode ~6.4 tok/s, prefill ~32 tok/s (short) up to ~85 tok/s (longer prompts).
Decode is fundamentally bounded by reading the weights once per token
(~12.5 GB / 150 GB/s).

## Speculative decoding (draft model)

The GEMM makes batched verification cheap, so draft-model speculation works. The
draft must share the target's tokenizer (e.g. a small Qwen3.5 model for a Qwen3.5
target — same 248320 vocab). Tuned example:

```bash
build-metal/bin/llama-speculative-simple \
  -m target-TQ3_4S.gguf -ngl 99 \
  -md draft-Qwen3.5-small.gguf -ngld 99 \
  --spec-type draft-simple \
  --spec-draft-n-max 12 --spec-draft-n-min 3 --spec-draft-p-min 0.6 \
  -fa on -ctk q8_0 -ctv q8_0 \
  -c 8192 --override-kv llama.nomtp_trunk_only=bool:true \
  -p "your prompt"
```

Caveats for hybrid (SSM / gated-delta-net) targets such as `qwen35`: the
recurrent cache does not support partial sequence removal, so speculation falls
back to full-state checkpoints. The gain is therefore modest and very sensitive
to draft acceptance — on an M3 Pro, tuned speculation reaches ~7 tok/s (73%
acceptance) vs ~6.4 plain. High absolute numbers reported elsewhere (>100 tok/s)
are high-bandwidth GPUs / Ultra-class Macs, not an M3 Pro.
