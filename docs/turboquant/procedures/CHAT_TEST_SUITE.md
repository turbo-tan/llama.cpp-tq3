# Chat Test Suite — Quantized Model Quality Gate

Compare TQ3_1S vs Q4_0 on Qwen3.5-27B (or 9B).
Run via `llama-server` with `--reasoning off --reasoning-budget 0 --reasoning-format deepseek`.

---

## 1. Strict Instruction Following & Formatting

### 1a. Code — No Fences, No Explanation

**Prompt:**
```
Write a Python function called `merge_sorted` that merges two sorted lists into one sorted list in O(n) time. Output ONLY the function. No explanation. No markdown fences. No comments.
```

**Good:** Raw Python function, nothing else. Correct two-pointer merge.
**Bad:** Markdown fences, explanation text, comments, or brute-force sort.
**Checks:** Instruction adherence, code correctness, format discipline.

### 1b. Strict JSON — Nothing But JSON

**Prompt:**
```
List the 5 largest countries by area. Output ONLY a JSON array of objects with keys "name" and "area_km2". No text before or after the JSON.
```

**Good:** `[{"name":"Russia","area_km2":17098242}, ...]` — valid JSON, no wrapper text.
**Bad:** Markdown fences, preamble like "Here are...", invalid JSON, wrong countries.
**Checks:** Format compliance, factual accuracy, JSON validity.

---

## 2. Reasoning & Math

### 2a. Bat and Ball (Cognitive Reflection)

**Prompt:**
```
A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost? Show your work step by step, then give the final answer on its own line as: Answer: $X.XX
```

**Good:** Sets up equation, arrives at $0.05. Final line: `Answer: $0.05`
**Bad:** Says $0.10 (the intuitive wrong answer), or correct answer without work.
**Checks:** Reasoning under cognitive bias, format compliance.

### 2b. Multi-Step Logic

**Prompt:**
```
Five people (A, B, C, D, E) sit in a row. B is not next to A. C is at one end. D is next to E. A is in the middle. List ALL valid arrangements, one per line.
```

**Good:** Enumerates valid permutations systematically. A is position 3, C is position 1 or 5, D-E adjacent, B not adjacent to A.
**Bad:** Missing arrangements, includes invalid ones, or gives up.
**Checks:** Constraint satisfaction, exhaustive enumeration, logical rigor.

---

## 3. Knowledge & Factual Accuracy

### 3a. Established Knowledge

**Prompt:**
```
What is the capital of Kazakhstan, and when was it renamed? Answer in exactly two sentences.
```

**Good:** Astana, renamed from Nur-Sultan back to Astana in 2022 (originally renamed from Astana to Nur-Sultan in 2019). Exactly two sentences.
**Bad:** Says Almaty, wrong dates, more or fewer than two sentences.
**Checks:** Factual recall, sentence count constraint.

---

## 4. Creativity & Style Control

### 4a. Constrained Creative Writing

**Prompt:**
```
Write a 4-line poem about a GPU running out of memory. Each line must have exactly 8 syllables. Do not use the word "memory" or "GPU".
```

**Good:** 4 lines, ~8 syllables each, about OOM without forbidden words, creative.
**Bad:** Uses "memory" or "GPU", wrong line count, wildly wrong syllable count.
**Checks:** Creative ability under hard constraints, counting, vocabulary control.

---

## 5. Long-Context Coherence

### 5a. Instruction Recall After Filler

**Prompt:**
```
I will give you a secret code at the start, then ask you to do something else. Remember the code.

The secret code is: BLUE-TIGER-42

Now, please explain how photosynthesis works in 3 sentences.

After your explanation, on a new line write: "Code: " followed by the secret code I gave you.
```

**Good:** 3-sentence photosynthesis explanation, then `Code: BLUE-TIGER-42` on its own line.
**Bad:** Forgets the code, mangles it, or puts it in the wrong place.
**Checks:** Instruction tracking across distraction, working memory.

---

## Scoring

For each test, score 0-2:
- **2** = Perfect (correct answer + correct format)
- **1** = Partially correct (right answer wrong format, or vice versa)
- **0** = Wrong answer and/or wrong format

**Max score: 12** (6 tests × 2 points)

| Test | Q4_0 | TQ3_1S | Notes |
|------|------|--------|-------|
| 1a Code strict | | | |
| 1b JSON strict | | | |
| 2a Bat & ball | | | |
| 2b Logic puzzle | | | |
| 3a Kazakhstan | | | |
| 4a Poem | | | |
| 5a Code recall | | | |

---

## Running the Suite

```bash
# Start server (stable TQ path uses llama.cpp-tq3, not plain llama.cpp)
cd /home/awee/code/llama.cpp-tq3
nohup ./build/bin/llama-server \
    -m MODEL.gguf \
    --host 127.0.0.1 --port 18129 \
    -ngl 99 -c 2048 -np 1 -t 8 -fa on \
    --reasoning off --reasoning-budget 0 --reasoning-format deepseek \
    < /dev/null > /dev/null 2>&1 &

# Run automated version
../tan_llama/scripts/chat_test_suite.sh
```
