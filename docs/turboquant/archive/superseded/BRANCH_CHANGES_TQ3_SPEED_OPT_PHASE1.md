# Branch: `feature/tq3-speed-opt-phase1`

**Date**: 2026-04-05  
**Base**: `research/tq3-decode-24tok` (VDR=8 + scale table checkpoint)  
**Repo**: `charpdev/llama.cpp`

---

## Summary

This branch implements Phase 1 of the TQ3_4S decode speed optimisation plan
(`TQ3_SPEED_OPTIMIZATION_PLAN.md`). The primary change is **KV cache
pre-rotation for TQ3_4S models** — the single highest-impact remaining
optimisation (+15–20% decode speed expected on 27B).

---

## Commits

### 1. `f751429e5` — kv cache: add optional rotation before quantization *(cherry-pick)*

Cherry-picked from `feature/kv-cache-rotation`. Adds a general-purpose
random orthogonal rotation applied to K at cache write time and to Q at
attention time, enabled via `LLAMA_KV_ROTATION=1`.

**Files changed:**
- `src/CMakeLists.txt` — add `kv-rotation.cpp` to build
- `src/kv-rotation.h` / `src/kv-rotation.cpp` — new: random orthogonal
  matrix generator (Gram-Schmidt QR of Gaussian noise)
- `src/llama-kv-cache.cpp` — allocate `kv_rotation` tensor; apply
  `ggml_mul_mat(kv_rotation, k_cur)` in `cpy_k` before `ggml_set_rows`
- `src/llama-kv-cache.h` — add `kv_rotation` member; expose
  `get_kv_rotation()` on both `llama_kv_cache` and
  `llama_kv_cache_context`
- `src/llama-graph.cpp` — rotate Q at all three `build_attn` call sites
  when `get_kv_rotation()` is non-null

**Conflict resolved:** `llama-kv-cache.h` had a conflict between the
existing `set_input_k_rot` / `set_input_v_rot` (TQ3 branch) and the new
`get_kv_rotation()`. Both are kept.

---

### 2. `7615d6063` — tq3: auto-enable WHT KV pre-rotation for TQ3_4S models

**The main new work.** Extends the cherry-picked infrastructure to use the
Walsh-Hadamard Transform (WHT) instead of a random matrix when the loaded
model uses TQ3_4S weights.

#### Why WHT, not random?

TQ3_4S stores weights pre-rotated with the WHT:

```
w_stored = R_WHT(w_original)
dot(w_stored, R_WHT(x)) = dot(w, x)   [orthogonality]
```

The same rotation must be applied to K at cache write time so that
attention scores remain correct:

```
score = dot(R_WHT(Q), R_WHT(K))  =  dot(Q, K)   [orthogonality]
```

Storing `R_WHT(K)` in the cache means the per-access rotation of K is
eliminated. Only Q needs rotating once per decode step (already done in
the MMVQ path). This is mathematically proven in
`TQ3_KV_PREROTATION_PROOF.md`.

#### Files changed

**`src/kv-rotation.h`**
- Added declaration: `float * kv_rotation_generate_wht(int head_dim)`

**`src/kv-rotation.cpp`**
- Added `kv_rotation_generate_wht(int head_dim)`:
  - Builds a `head_dim × head_dim` block-diagonal matrix
  - Each 32×32 block is the normalised WHT with TQ3's sign pattern
    (`sign(i) = ((i * 0x9E3779B9u) >> 31) & 1 ? -1 : 1`)
  - Butterfly construction matches `tq3_rotate_act_kernel` exactly
  - Returns `R^T` (transposed) so `ggml_mul_mat(R^T, x)` computes `R·x`
  - `head_dim` must be a multiple of 32

**`src/llama-kv-cache.cpp`** — two changes:

*Allocation block* — auto-detect TQ3_4S and enable `kv_rotation`:
```cpp
const bool is_tq3_4s = !model.layers.empty() && model.layers[0].wq &&
                       model.layers[0].wq->type == GGML_TYPE_TQ3_4S;
const bool want_kv_rotation = (kv_rot_env && atoi(kv_rot_env)) || is_tq3_4s;
```
- TQ3_4S: auto-enabled, no env var needed
- Other quant types: still opt-in via `LLAMA_KV_ROTATION=1`
- Log message now says `WHT/TQ3` vs `random`

*Fill block* — use WHT matrix for TQ3_4S, random for everything else:
```cpp
float * rot = is_tq3_4s_fill ? kv_rotation_generate_wht(d)
                              : kv_rotation_generate(d, 42);
```

---

## What is already in the base branch (`research/tq3-decode-24tok`)

These optimisations were already present before this branch was created:

| Optimisation | Status | Expected gain |
|---|---|---|
| VDR=8 for SM120 (MMVQ) | ✅ done | +20–30% MMVQ |
| Scale table lookup (no `ldexpf`) | ✅ done | eliminates expensive decode |
| SM120 2-row blocking (MMA path) | ✅ done (`2*granularity`) | +47% on SM120 |
| Packed-X activation staging (dp4a) | ✅ done | +5–10% PP |
| Scale collapse (MMA path) | ✅ done | +0.8% |

---

## What this branch adds

| Optimisation | Status | Expected gain |
|---|---|---|
| WHT KV pre-rotation (TQ3_4S auto) | ✅ this branch | **+15–20% TG** |
| Random KV rotation (any quant, opt-in) | ✅ cherry-pick | quality improvement |

---

## Expected performance after this branch

| Metric | Before (baseline) | After (projected) |
|---|---|---|
| TG tok/s (27B) | 14.35–14.48 | ~16.5–17.2 |
| PP tok/s (27B) | 315 | ~315 (unchanged) |
| PPL (27B) | 6.77 | 6.77 (exact, WHT is lossless) |

---

## How to test

```bash
# Build
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc) --target llama-bench llama-cli

# Benchmark decode (TQ3_4S auto-enables WHT pre-rotation)
./build/bin/llama-bench -m qwopus35-27b-v3-tq3_4s.gguf -p 0 -n 128 -t 1

# Disable pre-rotation to compare baseline
LLAMA_ATTN_ROT_DISABLE=1 ./build/bin/llama-bench -m qwopus35-27b-v3-tq3_4s.gguf -p 0 -n 128 -t 1

# Verify correctness (PPL should be identical)
./build/bin/llama-perplexity -m qwopus35-27b-v3-tq3_4s.gguf -f wikitext-2-raw/wiki.test.raw
```

---

## Next steps (Phase 2+)

Per `TQ3_SPEED_OPTIMIZATION_PLAN.md`:

1. **Fused Q-projection + rotation kernel** (+8–12% TG) — collapse
   `proj → rotate → quantize → gemv` into one kernel launch
2. **Hybrid Q8 attention policy** (+8–15% TG) — store attn Q/K as Q8_0,
   keep FFN as TQ3_4S
3. **CUDA Graphs** (+5–10%) — amortise kernel launch overhead

---

## Files changed (total across both commits)

```
src/CMakeLists.txt          |   1 +
src/kv-rotation.h           |  10 +++
src/kv-rotation.cpp         | 120 ++++++++++++++++++++++++++++++++++
src/llama-kv-cache.cpp      |  55 ++++++++++++++++
src/llama-kv-cache.h        |  12 ++++
src/llama-graph.cpp         |  24 +++++++
```
