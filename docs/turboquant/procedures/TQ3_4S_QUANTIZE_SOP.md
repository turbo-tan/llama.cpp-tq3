# TurboQuant Quantization SOP

This SOP covers two repeatable paths:

- the standard `TQ3_4S` quantization path for the public winner family
- the selective-tensor `outQ6K` variant, where chosen tensors are kept at `Q6_K`

Use this as a runbook. If the exact tensor names differ for a model variant,
check the `--dry-run` output first and adjust the tensor policy file before
running the full conversion.

## 1. Build the quantizer

```bash
cd /home/awee/code/tan_llama
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build build --target llama-quantize -j"$(nproc)"
```

## 2. Pick the source GGUF

Use the highest-quality source you already have on disk.

- Preferred source for normal production quantization: `Q8_0`
- Use `BF16` only when you need a diagnostic or reference source for analysis
- Do not requantize an already quantized file unless you are intentionally
  testing a chained format

If the source GGUF is split, keep the shard layout stable with `--keep-split`
or merge it first, but do not change the source and target recipe at the same
time.

## 3. Standard TQ3_4S recipe

This is the normal public format recipe for dense 27B-style models:

```bash
./build/bin/llama-quantize --allow-requantize \
  --output-tensor-type q6_K \
  --token-embedding-type q6_K \
  source-Q8_0.gguf \
  Qwen3.6-27B-MTP-TQ3_4S.gguf \
  TQ3_4S
```

If the model is MoE and already has a documented expert policy, use the
specialized tensor overrides from the conversion guide instead of inventing a
new recipe.

## 4. Selective outQ6K recipe

The outQ6K variant keeps the model close to the TQ3_4S footprint but protects a
small set of high-impact tensors at `Q6_K`.

### 4.1 Create a tensor policy file

Create the policy file under `docs/turboquant/generated/` so the tensor mix is
explicit and easy to reuse:

- [`docs/turboquant/generated/qwen36_27b_mtp_tq3_4s_outq6k.tensor-types.txt`](../generated/qwen36_27b_mtp_tq3_4s_outq6k.tensor-types.txt)

Starter policy:

```text
token_embd.weight=q6_K
output.weight=q6_K
blk\.0\.attn_output\.weight=q6_K
blk\.0\.ffn_down(_exps)?\.weight=q6_K
```

Notes:

- The first two lines protect the embedding and output paths explicitly.
- The `blk\.0\...` lines give a concrete first-block override for the earliest
  block.
- If the model architecture uses different tensor names, update the file after
  checking `--dry-run`.
- If the tail KLD still looks bad after the first run, extend the policy file
  with wider block-wide overrides and rerun `--dry-run` before quantizing.

### 4.2 Dry run first

Always check the final size before writing the output GGUF:

```bash
./build/bin/llama-quantize --dry-run \
  --tensor-type-file docs/turboquant/generated/qwen36_27b_mtp_tq3_4s_outq6k.tensor-types.txt \
  --output-tensor-type q6_K \
  --token-embedding-type q6_K \
  source-Q8_0.gguf \
  Qwen3.6-27B-MTP-TQ3_4S-outq6k.gguf \
  TQ3_4S
```

If the dry-run size is too large or the tensor matches are not what you expect,
edit the policy file before the real quantization run.

### 4.3 Build the outQ6K file

```bash
./build/bin/llama-quantize --allow-requantize \
  --tensor-type-file docs/turboquant/generated/qwen36_27b_mtp_tq3_4s_outq6k.tensor-types.txt \
  --output-tensor-type q6_K \
  --token-embedding-type q6_K \
  source-Q8_0.gguf \
  Qwen3.6-27B-MTP-TQ3_4S-outq6k.gguf \
  TQ3_4S
```

## 5. Validate the result

Use the same runtime and template that will be published.

1. Run a quick size check on the output GGUF.
2. Run `benchloop partial` against the exact model/template pair.
3. If the candidate is still good, run `benchloop full`.
4. If the change is meant for public release, follow with `hard86` and `GPQA`
   using the same binary and template.

If the outQ6K file is only an experiment, keep the artifact folder separate so
it is not confused with the public winner.

## 6. Rules of thumb

- Quantize from the best available source, not from a previous low-bit output.
- Keep the standard TQ3_4S recipe stable unless you have a measured reason to
  change it.
- Use the tensor policy file for selective variants so the exact tensor mix is
  reviewable.
- Keep `BF16` around for KLD/reference analysis, but do not make it the default
  production source unless there is a specific reason.
