# Qwen Witness Investigation

## Purpose

We needed to answer a specific question:

- did current `TQ3_0` or `Q4_0_TQ` regress relative to the older "known good" TurboQuant path?

The immediate working hypothesis was:

- old `TQ3_0` was known-good
- current `Q4_0_TQ` is broken
- therefore current behavior on Qwen should differ from old `TQ3_0`

This note records the result of testing that hypothesis directly.

## Historical Claim We Checked

The strongest local history claim is commit:

- `ea3eec15a` `feat: TQ3_0 weight quantization via llama-quantize`

That commit message says:

- `Output: correct ('Paris') on both CPU and GPU`

Important detail:

- at `ea3eec15a`, `LLAMA_FTYPE_MOSTLY_TQ3_0 -> GGML_TYPE_TQ3_0`
- `Q4_0_TQ` did not exist yet as a separate runtime type

So `ea3eec15a` is the best documented "known good" weight-quant baseline.

## What We Tested

We tested the same source model under:

1. current tree, true `TQ3_0`
2. current tree, true `Q4_0_TQ`
3. historical worktree at `ea3eec15a`, true `TQ3_0`

Model used:

- `/tmp/quantize/Qwen3-0.6B-Q8_0.gguf`

Prompt used:

- `The capital of France is`

Command shape:

```bash
./build/bin/llama-completion \
  --model <model.gguf> \
  -no-cnv \
  -p "The capital of France is" \
  --n-predict 8 \
  --seed 42 \
  --no-warmup \
  -ngl 0 \
  -c 1024
```

## Current Tree Setup

We restored separate file/runtime identities:

- true `TQ3_0` remains `ftype 40`
- true `Q4_0_TQ` is now `ftype 41`

This matters because earlier in the branch:

- `TQ3_0` had been redirected to emit/load `Q4_0_TQ`

That prevented a clean A/B.

## Current Tree Results

### True TQ3_0

Built with:

```bash
/home/awee/code/llama.cpp/build/bin/llama-quantize \
  --allow-requantize \
  /tmp/quantize/Qwen3-0.6B-Q8_0.gguf \
  /tmp/quantize/Qwen3-0.6B-TQ3_0-direct.gguf \
  TQ3_0
```

Observed output:

- `in the border of the border of the`

Model size:

- `389.17 MiB (4.34 BPW)`

### True Q4_0_TQ

Built with:

```bash
/home/awee/code/llama.cpp/build/bin/llama-quantize \
  --allow-requantize \
  /tmp/quantize/Qwen3-0.6B-Q8_0.gguf \
  /tmp/quantize/Qwen3-0.6B-Q4_0_TQ-direct.gguf \
  Q4_0_TQ
```

Observed output:

- `. Capital of France is. Capital of`

Model size:

- `370.63 MiB (4.14 BPW)`

## Historical Commit Reproduction

We created a separate worktree:

```bash
git worktree add /tmp/llama-ea3eec15a ea3eec15a
cmake -S /tmp/llama-ea3eec15a -B /tmp/llama-ea3eec15a/build -DGGML_CUDA=ON
cmake --build /tmp/llama-ea3eec15a/build -j 16 --target llama-quantize llama-completion
```

Then built the historical `TQ3_0` Qwen model:

```bash
/tmp/llama-ea3eec15a/build/bin/llama-quantize \
  --allow-requantize \
  /tmp/quantize/Qwen3-0.6B-Q8_0.gguf \
  /tmp/quantize/Qwen3-0.6B-TQ3_0-ea3eec15a.gguf \
  TQ3_0
```

Then ran the same prompt with the historical binary:

```bash
/tmp/llama-ea3eec15a/build/bin/llama-completion \
  --model /tmp/quantize/Qwen3-0.6B-TQ3_0-ea3eec15a.gguf \
  -no-cnv \
  -p "The capital of France is" \
  --n-predict 8 \
  --seed 42 \
  --no-warmup \
  -ngl 0 \
  -c 1024
```

Observed output:

- `in the border of the border of the`

Model size:

- `389.17 MiB (4.34 BPW)`

## Forced Old Runtime Proof

We also tested one targeted runtime rollback in the current tree:

- forced current `TQ3_0` CUDA `mul_mat` back toward the older plain fp32 `cublasSgemm` fallback
- removed the newer activation-rotation special case from the `TQ3_0` `ggml_cuda_op_mul_mat_cublas` path

Result:

- output remained bad on the same Qwen witness
- it briefly started `The capital of France is in...`
- but completed to the same bad pattern:
  - `in the border of the border of the`

Conclusion from this subtest:

- the newer activation-rotation special case is not the sole reason Qwen is bad

## Key Conclusion

Qwen 0.6B is **not** a valid regression witness for the historical "TQ3_0 says Paris" claim.

Why:

1. current true `TQ3_0` on Qwen is bad
2. historical `ea3eec15a` true `TQ3_0` on the same Qwen model is also bad
3. therefore current Qwen failure does **not** prove a regression relative to the historical known-good commit

This is the most important outcome of the investigation.

## What This Means

Do **not** use Qwen 0.6B to answer:

- "did we regress from the old working `TQ3_0` path?"

It cannot answer that question because the old commit is already bad on this witness.

## Better Regression Witnesses

Use the models actually referenced in the original working claims and test docs:

- TinyLlama
- Llama-2-7B

Relevant local evidence:

- commit `53b4dd575`:
  - `Verified: 'The capital of France is Paris' (GPU, ngl=99, -fa off)`
- commit `ea3eec15a`:
  - `Output: correct ('Paris') on both CPU and GPU`
- `/home/awee/code/tan_llama/STEERING.md`
  - quality sanity requires the prompt to contain `Paris`
- `/home/awee/code/tan_llama/docs/turboquant/procedures/TEST_SUITE.md`
  - TinyLlama/Qwen3.5-9B/TQ3 guardrails are documented there

## Practical Next Step

The next regression-proof comparison should be:

1. choose TinyLlama or another model explicitly used in the historical "Paris" claim
2. quantize with:
   - historical `ea3eec15a` binary, `TQ3_0`
   - current binary, true `TQ3_0`
   - current binary, true `Q4_0_TQ` if needed
3. run the same deterministic `Paris` sanity prompt
4. only call it a regression if:
   - old commit says `Paris`
   - current tree does not

Until that comparison is done, Qwen 0.6B should be treated as:

- useful for smoke and stress
- **not** valid as a historical correctness witness
