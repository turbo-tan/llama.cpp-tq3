# TQ3 Format Benchmark Protocol

## The Only Reliable PPL Gate

**100 chunks, c=512, wiki.test.raw, 27B witness.**

No format is declared better or worse without this test. Period.

### Why Not Fewer Chunks

| Chunks | Reliability | Known failures |
|--------|-------------|----------------|
| 1-3 | Useless | Hid MMQ NaN bug, gave wrong TQ3_4S ranking |
| 10 | Misleading | Showed TQ3_4S worse than TQ3_1S (wrong) |
| 100 | Reliable | Correctly showed TQ3_4S beats TQ3_1S |
| 580 (full) | Gold standard | Takes 40+ min, use for final publication only |

### Command

```bash
./build/bin/llama-perplexity \
    -m MODEL.gguf \
    -ngl 99 -fa 1 -t 8 -c 512 --no-warmup \
    -f wikitext-2-raw/wiki.test.raw --chunks 100 \
    2>&1 | tee artifacts/ppl_27b_FORMAT_100ch_DATE.txt
```

Always pipe to file. Always include date in filename.

## Trial Run (before 100-chunk)

1 chunk, confirm no crash/NaN:
```bash
./build/bin/llama-perplexity -m MODEL.gguf \
    -ngl 99 -fa 1 -t 8 -c 512 --no-warmup \
    -f wiki.test.raw --chunks 1
```

If chunk 1 gives NaN or PPL > 50, the format is broken. Fix before running 100 chunks.

## Speed Benchmark

### Build (clean, native arch only — no PTX)

```bash
cd /home/awee/code/llama.cpp-tq3
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_COMPRESSION_MODE=size
cmake --build build -j$(nproc)
```

**Do NOT pass `-DCMAKE_CUDA_ARCHITECTURES=120`** — this produces PTX+native (slower).
Without it, cmake auto-detects and produces `sm_120a` (native only), which is faster.

### ⚠️ Cold-Start Warning (CRITICAL)

CUDA JIT compilation causes the first run to be 15-25% slower. This is NOT a real
speed regression — it's kernel compilation overhead. **Always do a warmup pass first.**

Cold-start pattern observed on RTX 5060 Ti (SM 120a):
- 1st run: ~520-600 tok/s (JIT compiling kernels)
- 2nd run: ~640-660 tok/s (partially cached)
- 3rd+ run: ~700-720 tok/s (fully warmed up)

**Any speed claim must be from a warmed-up run.**

### Step 1: Warmup pass (cold-start JIT)

```bash
pkill -9 -f llama 2>/dev/null; sleep 3
nvidia-smi --query-compute-apps=pid --format=csv,noheader
# Must be empty before proceeding

./build/bin/llama-bench -m MODEL.gguf -ngl 99 -fa 1 \
    -p 128 -pg 128,0 -r 2 --no-warmup
```
This triggers CUDA JIT compilation. The PP128 numbers will be low — that's expected.

### Step 2: Speed measurement (warmed up)

```bash
./build/bin/llama-bench -m MODEL.gguf -ngl 99 -fa 1 \
    -p 2048 -pg 2048,0 -n 128 -r 5
```

Use **`-r 5`** for stable numbers. PP512 has high variance (±30), PP2048 is stable (±1-2).

### Step 3: With quantized KV cache (optional)

For models with TQ3_0 KV support, test with compressed KV:
```bash
./build/bin/llama-bench -m MODEL.gguf -ngl 99 -fa 1 \
    -ctk tq3_0 -ctv tq3_0 \
    -p 2048 -pg 2048,0 -n 128 -r 5
```

Or with mixed KV types:
```bash
./build/bin/llama-bench -m MODEL.gguf -ngl 99 -fa 1 \
    -ctk q4_0 -ctv q4_0 \
    -p 2048 -pg 2048,0 -n 128 -r 5
```

Recovered asymmetric KV witness:
```bash
./build/bin/llama-bench -m MODEL.gguf -ngl 99 -fa 1 \
    -ctk q4_0 -ctv tq3_0 \
    -p 2048 -n 0 -r 5
```

### Step 4: Record all conditions

Before claiming a speed number, capture ALL of:
1. Exact command (including FA flag, KV types, reps)
2. Exact model path and format
3. Build commit (`git rev-parse --short HEAD`)
4. FA state confirmed (check log for "flash_attn" messages)
5. Whether warmup was done before measurement
6. PP and TG separately — never combine them

### 27B TQ3_4S Reference Numbers (RTX 5060 Ti 16GB, post-warmup)

| Config | PP2048 | TG128 | Date | Build |
|--------|--------|-------|------|-------|
| Default KV + FA | **673.39 ± 1.39** | — | 2026-04-19 | b11f67b73 |
| q4_0 KV + q4_0 KV + FA | **672.35 ± 0.65** | — | 2026-04-19 | b11f67b73 |
| tq3_0 KV + tq3_0 KV + FA | **687.86 ± 15.16** | — | 2026-04-19 | b11f67b73 |
| **q4_0 K + tq3_0 V + FA** | **702.73 ± 7.24** | — | **2026-04-19** | **b11f67b73** |

Historical, not currently reproduced on 2026-04-19:
- `715 / 23.0` default KV + FA
- `720` q4_0 KV + FA
- `710` tq3_0 KV + FA

Cold-start (no warmup): 520-600 tok/s — NOT a valid measurement.

## Chat Quality Gate

### Quick SOP (8 prompts, ~2 min total)

```bash
# Kill existing processes
pkill -9 -f llama-server 2>/dev/null; sleep 2

# Launch server
setsid ./build/bin/llama-server \
    -m MODEL.gguf \
    -ngl 99 -c 2048 --port 8090 -fa 1 \
    </dev/null >/tmp/llama-server.log 2>&1 &
sleep 30
curl -s http://localhost:8090/health
# Must return {"status":"ok"} before proceeding

# Run test prompts
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

# Cleanup
pkill -9 -f llama-server 2>/dev/null
```

### Expected answers

| Prompt | Expected | Notes |
|--------|----------|-------|
| 127*43 | 5461 | Math |
| Capital | Canberra | Factual |
| sqrt(144) | 12 | Math |
| Pattern | 42 | Reasoning (n²+n) |
| JSON primes | [2, 3, 5, 7, 11] | Format compliance |
| Logic | No | Formal logic |
| Code | [2, 3, 1] | Code tracing |
| Haiku | Any coherent haiku | Creativity |

### Pass criteria
- **7/8 correct** = clean (no output corruption)
- **5-6/8 correct** = acceptable for 4.0 bpw quant (math/reasoning degrade)
- **<5/8 correct** = investigate for corruption/garbage output
- **Zero garbage output** = mandatory (no NaN, no mixed-script corruption)

**`max_tokens=1000` is needed** for Qwen3.5 thinking models — shorter budgets
produce empty responses. `temperature=0` for deterministic comparison.

## Current Ladder (27B, 100 chunks, c=512, RTX 5060 Ti 16GB)

### Perplexity Ladder

| Format | bpw | PPL | Size | Date |
|--------|-----|-----|------|------|
| Q4_K_M | 4.83 | 6.8086 | 15.5 GB | 2026-03-31 |
| IQ4_XS | 4.25 | 6.8065 | 13.9 GB | 2026-03-31 |
| Q3_K_S | 3.44 | 6.9622 | 11.4 GB | 2026-03-31 |
| TQ3_4SE | 4.50 | 7.0216 | 14.5 GB | 2026-03-31 |
| Q4_0 | 4.00 | 7.0350 | 14.4 GB | 2026-03-30 |
| TQ3_4S | 4.00 | 7.0468 | 12.9 GB | 2026-03-31 |
| TQ3_1S | 4.00 | 7.1078 | 12.9 GB | 2026-03-30 |
| UD-Q2_K_XL | 2.?? | 7.3352 | 10.5 GB | 2026-03-31 |

### Speed Ladder (post-warmup, PP2048, default KV + FA)

| Format | bpw | PP2048 tok/s | TG128 tok/s | Fits 16GB | Date |
|--------|-----|-------------|------------|-----------|------|
| IQ4_XS | 4.25 | 841 | 23.2 | ❌ (OOM at c=2048) | 2026-03-31 |
| Q3_K_S | 3.44 | 689 | 20.7 | ✅ | 2026-03-31 |
| **TQ3_4S** | **4.00** | **673** default / **703** asym KV | — | **✅** | **2026-04-19** |
| TQ3_1S | 4.00 | — | — | ✅ | not retested post-MMQ |
| Q4_K_M | 4.83 | — (OOM) | — | ❌ | — |
| Q4_0 | 4.00 | — (OOM) | — | ❌ | — |

Notes:
- TQ3_4S PP speed updated from 327 to the high-600s on current exact reruns
- Current recovered `700+` witness is asymmetric KV: `K=q4_0, V=tq3_0, FA=1`
- TQ3_4SE and TQ3_4SV removed (dead formats, enum cleanup 2026-04-15)
- Cold-start (no warmup) gives ~520-600 tok/s — NOT a valid measurement
- Historical `715-720` claims remain unverified until reproduced with current artifacts

### Speed Gap Analysis (2026-03-31)

TQ3_4S is 3x slower on PP and 30% slower on TG vs Q3_K_S:

| | Q3_K_S | TQ3_4S | Gap |
|---|---|---|---|
| PP (tok/s) | 689 | 219 | **3.1x slower** |
| TG (tok/s) | 20.7 | 14.7 | **1.4x slower** |

Root causes:
1. **PP**: TQ3 uses cuBLAS dequant→GEMM fallback. Q3_K_S has optimized MMQ kernels.
   Fix: implement MMQ kernel for TQ3_4S (load_tiles + dp4a accumulation).
2. **TG**: TQ3 MMVQ kernel exists but is unoptimized. Q3_K_S has hand-tuned MMVQ.
   Fix: optimize vec_dot_tq3_4s_q8_1 (memory access patterns, shared memory).
3. **WHT overhead**: inverse WHT butterfly in dequant adds ~5 stages of shuffle ops.
   This is inherent to the format but only ~10% of total dequant time.

### Priority: Speed over Quality

TQ3_4S quality (PPL 7.05) is close to Q4_0 (7.04) at 1.5 GB smaller.
Can't beat Q3_K_S on quality (WHT can't use imatrix — fundamental limit).
**Must beat on speed to justify the format.**

Target: match Q3_K_S speed (689 PP, 20.7 TG) at 12.9 GB.
Stretch: beat Q3_K_S speed (smaller model = less memory bandwidth).

### Honest Assessment (2026-03-31)

**Q3_K_S beats all TQ3 variants in BOTH size and quality.**

- Q3_K_S: 3.44 bpw, PPL 6.96, 11.4 GB
- TQ3_4S: 4.00 bpw, PPL 7.05, 12.9 GB — 16% more bits, 0.09 worse PPL
- TQ3_4SE: 4.50 bpw, PPL 7.02, 14.5 GB — 31% more bits, 0.06 worse PPL

TQ3 uses more bits per weight and still gets worse quality. The WHT + 3-bit
centroid approach is fundamentally less efficient than K-quant's mixed-precision
super-blocks with importance matrices.

### Where the Bits Go: Q3_K_S vs TQ3_4S

Q3_K_S (3.44 bpw, 256-element super-block):
- 3 bits × 256 weights = 768 bits
- 12 × 6-bit scales + 12 × 4-bit mins = 120 bits (for 16-element sub-blocks)
- 1 × fp16 super-scale + 1 × fp16 super-min = 32 bits
- Total: ~920 bits / 256 = ~3.59 bpw actual
- KEY: 26 scale/min parameters for 256 elements + importance matrix guidance

TQ3_4S (4.00 bpw, 32-element block):
- 3 bits × 32 weights = 96 bits
- 4 × 8-bit E3M5 scales = 32 bits (for 8-element groups)
- Total: 128 bits / 32 = 4.00 bpw
- KEY: 4 scale parameters for 32 elements, no importance matrix

### Why Q3_K_S Wins With Fewer Bits

1. **Importance matrix**: Q3_K_S uses activation-aware quantization — salient
   weights get more precision. TQ3 treats all weights equally.
2. **Mins (asymmetric)**: Q3_K_S stores per-sub-block minimums, allowing
   asymmetric ranges. TQ3 centroids are symmetric around zero.
3. **Larger super-blocks**: 256-element blocks amortize overhead better.
   TQ3's 32-element blocks waste 25% of bits on scales.
4. **WHT overhead**: The Hadamard transform decorrelates but doesn't compress.
   The 3-bit centroids in WHT domain lose precision when transformed back.

### Possible Improvements to Investigate

1. ~~**Importance-weighted quantization**~~: DOES NOT WORK with WHT.
   The Hadamard transform spreads each weight's importance equally across
   all 32 coefficients (H[k,j]^2 = 1/32 for all k,j). imatrix weighting
   becomes uniform in WHT domain → no effect. Confirmed: PPL 7.049 with
   imatrix vs 7.047 without (identical within noise).
   **This is the fundamental limitation of WHT-based quantization.**
2. ~~**Larger blocks (64-element)**~~: Only 1% RMSE improvement in testing.
   Not worth the implementation complexity.
3. **Asymmetric centroids / shifts**: TQ3_4SE adds shifts but only per-16.
   Per-8 shifts might help more. But adds bytes.
4. **Mixed precision**: Use 4-bit centroids for important weights, 2-bit for
   unimportant ones (like IQ4_XS approach). But can't identify "important"
   weights in WHT domain (see point 1).
5. **Better WHT utilization**: The transform concentrates energy in fewer
   coefficients — could use variable-rate coding instead of fixed 3-bit.
   This is the only remaining path that leverages WHT's actual strength.
6. **Abandon WHT**: Quantize in original domain like K-quants, use imatrix.
   Would lose the decorrelation benefit but gain importance weighting.

### The Core Dilemma

WHT decorrelation and importance weighting are fundamentally incompatible:
- WHT makes coefficients uncorrelated → better for fixed-rate coding
- WHT makes coefficients equally important → can't do importance weighting
- K-quants skip decorrelation → coefficients are correlated but importance-weightable
- K-quants win because importance weighting > decorrelation at 3-4 bpw

## Formats To Retest at 100 Chunks

These were declared dead based on short evaluations. Must retest before closing:

| Format | Short-test result | Reason to retest |
|--------|-------------------|------------------|
| TQ3_K_v1 Candidate A (shared-shift) | 15% worse at tensor RMSE | RMSE doesn't predict PPL |
| TQ3_1S_K (amortized 4-bit scales) | 19% worse at tensor RMSE | Same — RMSE unreliable |
| TQ3_4S+E (4 scales + 2 shifts, 4.5 bpw) | 7.4% better RMSE | Never model-tested |

## Rules

1. **NEVER declare a format dead without 100-chunk PPL**
2. **NEVER project PPL from tensor RMSE** — the relationship is nonlinear and model-dependent
3. **NEVER compare PPL numbers from different chunk counts**
4. **Trial (1 chunk) is only for crash/NaN detection, not quality judgment**
5. **10-chunk PPL is for development iteration only — not for decisions**
6. **All final claims must cite 100-chunk results with artifact file path**
