# Lossy 2-Token Speculative Decode — Handover

Date: 2026-04-14
Branch: `experiment/persistent-decode` @ `5934e7155` on private repo

## What We Built

A speculative decode system for Qwen3.5-27B hybrid (SSM + attention) that batches 2 tokens per forward pass. The model reads weights once for both tokens, giving up to 2x throughput when guesses are correct.

### Tools Created

| File | Purpose |
|------|---------|
| `examples/jacobi/lossy-decode.cpp` | Original lossy decode with hand-rolled bigram/trigram cache |
| `examples/jacobi/lossy-ngram.cpp` | **Best version** — uses llama.cpp's proper `ngram-cache.h` with static + dynamic caches |
| `examples/jacobi/tree-decode.cpp` | Tree decode with `seq_cp` (no SSM corruption but too slow) |
| `examples/jacobi/shadow-decode.cpp` | Shadow state prototype (incomplete) |
| `src/llama-memory-recurrent.h/cpp` | Shadow save/restore API (`shadow_save()`, `shadow_restore()`) |
| `src/llama-context.cpp` | Public API: `llama_memory_shadow_save/restore()` |
| `include/llama.h` | API declarations |

### Build

```bash
cd /home/awee/code/llama.cpp
git checkout experiment/persistent-decode
cmake -B build-360 -DGGML_CUDA=ON
cmake --build build-360 --target llama-lossy-ngram llama-lossy-decode llama-tree-decode -j$(nproc)
```

## The Core Problem

Qwen3.5 hybrid has 32 SSM (GatedDeltaNet) + 32 attention layers. The SSM state is **sequential and cumulative** — processing a wrong token corrupts the state for ALL future tokens. Unlike attention (KV cache can be partially removed), SSM state cannot be partially corrected without re-processing every token from the point of corruption.

This makes standard speculative decode impossible: every rejected guess requires a full re-decode of the accepted prefix through the SSM, which costs a full forward pass (~44ms on 27B), eliminating any speed gain.

## The Solution: Lossy Decode + Margin Gate

### Algorithm

```
1. Look up guess from n-gram cache (4-gram → 3-gram → 2-gram → 1-gram fallback)
2. Check: is model confident? (logits margin from previous decode > MIN_MARGIN)
3. If no guess or low confidence → normal 1-token decode
4. If guess available and confident → batch decode [cur, guess] as pp2
5. Verify: does argmax(logits[0]) == guess?
   YES → accept both tokens, 2 tokens for 1 batch cost
   NO  → accept cur only, SSM corruption at guess position (lossy)
```

### Why It Works

On 27B TQ3_4S, a 2-token batch (pp2) costs 63ms vs 44ms for 1-token decode — only 1.44x overhead. So even moderate acceptance rates are profitable:
- 50% accept → 1.5 tokens per 63ms = 23.8 tok/s (break-even)
- 70% accept → 1.7 tokens per 63ms = 27.0 tok/s (+15%)
- 95% accept → 1.95 tokens per 63ms = 31.0 tok/s (+32%)

### The Margin Gate

The logits margin (difference between top-1 and top-2 logit values) is a free confidence signal:
- **High margin** → model is confident → output is predictable → n-gram guess likely correct → safe to speculate
- **Low margin** → model is uncertain → output is novel → n-gram guess likely wrong → skip speculation

`MIN_MARGIN=4.0` is the sweet spot: prevents speculation on uncertain tokens, avoiding SSM corruption while allowing speculation on confident/repetitive tokens.

## Results

### 10 Random Questions (MIN_MARGIN=4.0, wiki static cache)

| Prompt | Baseline | Lossy-ngram | Delta | Quality |
|--------|----------|-------------|-------|---------|
| Capital of France | 23.5 | 30.7 | +30.5% | ✅ |
| Hash table | 23.5 | 24.1 | +2.4% | ✅ |
| Reverse linked list | 23.5 | 23.2 | -1.1% | ✅ |
| TCP vs UDP | 23.5 | 23.2 | -1.1% | ✅ |
| Solve 2x+5=17 | 23.5 | 24.9 | +6.1% | ✅ |
| Exercise benefits | 23.5 | 24.6 | +4.7% | ✅ |
| Rainbow | 23.5 | 23.4 | -0.2% | ✅ |
| SQL duplicates | 23.5 | 23.3 | -0.7% | ✅ |
| Stack vs heap | 23.5 | 23.5 | -0.0% | ✅ |
| Binary search | 23.5 | 30.2 | +28.4% | ✅ |
| **AVERAGE** | **23.5** | **25.1** | **+6.9%** | **10/10 ✅** |

### Repetitive Text (100% acceptance)

41.9 tok/s (+78%), output identical to baseline (diff exit 0).

## N-gram Cache

The guess quality depends entirely on the static n-gram cache.

### Cache Sources Tested

| Source | Size | Avg Speed | Notes |
|--------|------|-----------|-------|
| No cache | — | 23.5 | Baseline |
| Wiki (wikitext-2) | 3.9MB | 25.1 (+6.9%) | Best overall |
| Chat (hand-written QA) | 15KB | 26.1 (+10.9%) | Good on matching prompts, too small |
| Alpaca (GPT-3.5 outputs) | 3.8MB | 24.7 (+5.0%) | Wrong model's patterns, adds noise |
| Wiki + Chat merged | 4.0MB | 26.2 (+11.3%) | Best on 5-prompt subset |

### Key Finding

**Size matters more than style.** The wiki cache wins because it has 250x more n-grams than the chat cache. A large corpus matching the model's own output patterns would be ideal.

### Overnight Generation (IN PROGRESS)

Running 500 diverse prompts through the 27B itself to build a self-generated n-gram cache:

```bash
# Check progress
tail -1 /tmp/27b_gen.log

# Results when done
ls -lh /tmp/ngram_27b_best.bin   # merged wiki + self-generated

# Test
NGRAM_STATIC=/tmp/ngram_27b_best.bin MIN_MARGIN=4.0 ./build-360/bin/llama-lossy-ngram \
  -m /home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 -c 2048 -n 128 --temp 0 --top-k 1 \
  -p "your prompt"
```

The raw output is at `/tmp/27b_raw.txt`. Think blocks and special tokens need stripping:

```bash
python3 /tmp/cleanup_corpus.py /tmp/27b_raw.txt /tmp/27b_corpus_clean.txt
```

If the script's built-in cleanup missed anything, re-run the cache build:

```bash
cd /home/awee/code/llama.cpp
./build-360/bin/llama-lookup-create \
  -m /home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -f /tmp/27b_corpus_clean.txt \
  -lcs /tmp/ngram_27b_self.bin

./build-360/bin/llama-lookup-merge /tmp/ngram_static.bin /tmp/ngram_27b_self.bin /tmp/ngram_27b_best.bin
```

## Environment Variables

| Var | Default | Purpose |
|-----|---------|---------|
| `NO_SPEC` | unset | Disable all speculation (pure baseline) |
| `MIN_MARGIN` | 4.0 | Minimum logits margin to allow speculation |
| `MAX_REJECTS` | 999 | Max consecutive rejections before stopping speculation |
| `COOLDOWN` | 4 | Clean decodes after rejection before retrying (not very useful) |
| `NGRAM_STATIC` | unset | Path to static n-gram cache file |

## What Didn't Work

### Shadow State + Re-decode (correct but slow)
- `shadow_save()` copies 150MB SSM state to host memory
- On reject: `shadow_restore()` + `seq_rm` + re-decode accepted prefix
- The re-decode costs a full forward pass (44ms), eliminating the speed gain
- Code works correctly, just not profitable

### Tree Decode with seq_cp (correct but slow)
- Fork SSM state via `seq_cp(0, 1)`, decode both branches
- On reject: `seq_rm` the bad branch, keep the good one — no SSM corruption
- Problem: 3-token batch (cur×2 seqs + guess) costs 74ms vs 44ms baseline
- Need >76% acceptance to break even, rarely achieved

### Unlimited Lossy (fast but broken quality)
- No rollback, no margin gate — just batch and accept corruption
- 29-42 tok/s (+25-78%) but output degrades into repetition loops
- SSM corruption cascades: even 15 rejections out of 57 speculations produces completely different output
- Only safe at 100% acceptance (repetitive text)

## Quality Analysis

SSM corruption from rejected guesses:
- **100% acceptance**: identical output (verified with diff)
- **95%+ acceptance**: first answer correct, minor repetition later
- **70-90% acceptance**: first answer usually correct, degradation in continuation
- **<50% acceptance**: output diverges significantly from baseline

The margin gate (`MIN_MARGIN=4.0`) keeps most speculation in the 90%+ acceptance zone.

## Next Steps

### 1. Test Overnight N-gram Cache
The 27B self-generated cache should give better matches than wikitext. Expected improvement: +8-12% average (up from +6.9%).

### 2. Larger Static Cache
Build from a much larger corpus (10MB+ raw text). Options:
- Generate 5000+ prompts from the 27B itself (overnight job)
- Use a large HuggingFace dataset tokenized with Qwen tokenizer
- Key: must match the model's own generation patterns

### 3. TriAttention Integration
The `experiment/triattention-rework` branch has a per-layer attention scorer. Could provide:
- **Better confidence signal** than logits margin (per-layer attention entropy)
- **Attention-guided n-gram lookup** — find which previous positions the model attends to, look up what followed those positions
- The scorer is 763 lines but we'd only need the confidence signal

### 4. Adaptive Margin
Instead of fixed `MIN_MARGIN=4.0`, adapt based on recent acceptance rate:
- High recent acceptance → lower margin threshold → speculate more
- Low recent acceptance → raise threshold → speculate less

### 5. Server Integration
The lossy-ngram decode could be integrated into `llama-server` as an optional decode mode for hybrid SSM models. The static cache would be loaded at server startup.

## Key Files on Disk

```
/tmp/ngram_static.bin          # Wiki n-gram cache (3.9MB)
/tmp/ngram_chat.bin            # Hand-written QA cache (15KB)
/tmp/ngram_alpaca.bin          # Alpaca outputs cache (3.8MB)
/tmp/ngram_merged.bin          # Wiki + chat + alpaca merged
/tmp/ngram_27b_self.bin        # 27B self-generated (building overnight)
/tmp/ngram_27b_best.bin        # Wiki + 27B self merged (building overnight)
/tmp/27b_raw.txt               # Raw 27B outputs (building overnight)
/tmp/27b_corpus.txt            # Cleaned 27B outputs (built after generation)
/tmp/gen_prompts_500.txt       # 500 diverse prompts from alpaca
/tmp/cleanup_corpus.py         # Script to strip think blocks + special tokens
/tmp/gen_ngram_corpus.sh       # Overnight generation script
/tmp/27b_gen.log               # Generation progress log
```

## Batch Scaling Data (27B TQ3_4S, RTX 5060 Ti)

| Batch Size | tok/s | Time | Per-token | Ratio vs tg1 |
|-----------|-------|------|-----------|---------------|
| tg1 | 22.8 | 43.8ms | 43.8ms | 1.00x |
| pp2 | 31.6 | 63.3ms | 31.6ms | 1.44x |
| pp3 | 40.7 | 73.8ms | 24.6ms | 1.68x |
| pp4 | 49.0 | 81.6ms | 20.4ms | 1.86x |

The 1.44x ratio for pp2 is why this works — even 44% acceptance breaks even.
