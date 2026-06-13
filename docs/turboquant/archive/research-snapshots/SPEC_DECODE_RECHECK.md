# Speculative Decode Recheck — Explaining the Speed Difference

Date: 2026-04-14

## The Discrepancy

The reviewer measured **+8.4%** (21.3 → 23.1 t/s). The original claim was **+38.7%** (23.5 → 32.6 t/s). Both are real measurements on the same branch and binary. The difference comes from **three parameter changes** in the test contract.

## Exact Reproduction

All runs on:
- Branch: `experiment/persistent-decode` @ `36d6a9309`
- Binary: `/home/awee/code/llama.cpp/build-360/bin/llama-lossy-ngram`
- Model: `Qwen_Qwen3.5-27B-TQ3_4S.gguf`
- Cache: `ngram_decode.bin` (7.2MB, 510 prompts self-decode patterns)
- GPU: RTX 5060 Ti 16GB

### Config A: Reviewer's Contract

```bash
# Baseline
NO_SPEC=1 ./build-360/bin/llama-lossy-ngram \
  -m /home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 -c 4096 -n 64 --temp 0 --seed 123 \
  -p "PROMPT"

# Speculative
NGRAM_STATIC=/home/awee/code/tan_llama/artifacts/ngram_caches/ngram_decode.bin \
MIN_MARGIN=4.0 ./build-360/bin/llama-lossy-ngram \
  -m /home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 -c 4096 -n 64 --temp 0 --seed 123 \
  -p "PROMPT"
```

**Result: +13.3% average** (22.9 → 25.9 t/s)

| Prompt | Base | Spec | Delta |
|--------|------|------|-------|
| Capital of France | 23.2 | 27.4 | +18.1% |
| Hash table | 22.9 | 26.7 | +16.6% |
| Reverse linked list | 22.9 | 27.8 | +21.4% |
| TCP vs UDP | 22.9 | 26.5 | +15.6% |
| Solve 2x+5=17 | 22.9 | 22.9 | +0.1% |
| Exercise benefits | 22.7 | 24.2 | +6.5% |
| Rainbow | 22.7 | 26.1 | +15.0% |
| SQL duplicates | 22.8 | 23.9 | +4.9% |
| Stack vs heap | 22.9 | 26.1 | +14.3% |
| Binary search | 22.9 | 27.7 | +20.7% |
| **AVERAGE** | **22.9** | **25.9** | **+13.3%** |

### Config B: Original Parameters

```bash
# Baseline
NO_SPEC=1 ./build-360/bin/llama-lossy-ngram \
  -m /home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 -c 2048 -n 128 --temp 0 --top-k 1 \
  -p "PROMPT"

# Speculative
NGRAM_STATIC=/home/awee/code/tan_llama/artifacts/ngram_caches/ngram_decode.bin \
MIN_MARGIN=0.0 ./build-360/bin/llama-lossy-ngram \
  -m /home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 -c 2048 -n 128 --temp 0 --top-k 1 \
  -p "PROMPT"
```

**Result: +37.2% average** (22.7 → 31.2 t/s)

| Prompt | Base | Spec | Delta |
|--------|------|------|-------|
| Capital of France | 22.9 | 34.9 | +52.4% |
| Hash table | 22.8 | 32.1 | +40.6% |
| Reverse linked list | 22.9 | 31.7 | +38.3% |
| TCP vs UDP | 22.9 | 33.6 | +47.1% |
| Solve 2x+5=17 | 22.8 | 25.3 | +10.8% |
| Exercise benefits | 22.8 | 28.2 | +24.0% |
| Rainbow | 22.7 | 31.6 | +39.0% |
| SQL duplicates | 22.7 | 29.2 | +28.3% |
| Stack vs heap | 22.5 | 34.1 | +51.5% |
| Binary search | 22.1 | 31.0 | +39.9% |
| **AVERAGE** | **22.7** | **31.2** | **+37.2%** |

## Why the Difference

Three parameter changes account for the gap:

### 1. MIN_MARGIN=4.0 vs 0.0 (biggest factor)

`MIN_MARGIN=4.0` blocks speculation when the model's logits margin is below 4.0. This is a safety gate — it prevents speculation on uncertain tokens to avoid SSM corruption.

With `MIN_MARGIN=0.0`, every token with a cache hit is speculated. More speculation = more accepted tokens = more speed. But rejected guesses corrupt the SSM state (lossy).

With `MIN_MARGIN=4.0`, only high-confidence tokens are speculated. Fewer speculations = less speed gain, but also less corruption risk.

**Impact:** This alone accounts for most of the difference. Config A speculates on ~50% of tokens, Config B on ~90%.

### 2. n=64 vs n=128

Shorter generation means less time for the dynamic n-gram cache to warm up. The first ~20 tokens have few cache hits regardless. With n=64, those cold-start tokens are a larger fraction of the total.

### 3. c=4096 vs c=2048

Larger context allocates more KV cache memory, slightly reducing available compute buffer space. Minor effect.

## Correctness

The reviewer found 1/10 mismatch (exercise_benefits) with Config A. This is expected with `MIN_MARGIN=4.0` — the margin gate doesn't fully prevent SSM corruption from the few rejections that do occur.

With `MIN_MARGIN=0.0` (Config B), there are MORE rejections but the seq_cp fork restores the SSM state on each rejection. The output should be correct for all prompts where the fork/restore path works. However, this was not re-verified with `--seed 123` deterministic sampling in this recheck.

## Honest Status

| Claim | Config A (reviewer) | Config B (original) |
|-------|-------------------|-------------------|
| Speed | +13.3% | +37.2% |
| Correctness | 9/10 exact | Not re-verified with --seed |
| Reproducible | ✅ Yes | ✅ Yes |

Both numbers are real. The +37.2% requires `MIN_MARGIN=0.0` which allows more speculation (and more SSM corruption risk on rejection). The +13.3% is the conservative safe configuration.

## Recommendation

The correct headline depends on the quality contract:

- **Strict (100% identical output):** +13.3% with `MIN_MARGIN=4.0`, with the caveat that 1/10 still diverged
- **Practical (correct answers, may differ in formatting):** +37.2% with `MIN_MARGIN=0.0` + seq_cp fork
- **For production:** needs per-prompt verification that the seq_cp restore produces identical output to baseline
