# Rebase SOP

## Goal

Keep `main` aligned with `upstream/master` without rewriting or polluting the public work branches.

## Rules

- Never merge into `master`.
- Never push to `upstream`.
- Branch feature work from `main`.
- Rebase `main` onto `upstream/master` when syncing the fork.

## Standard Flow

### 1. Refresh remotes

```bash
git fetch upstream
git fetch origin
```

### 2. Update local `master` from upstream

```bash
git checkout master
git merge --ff-only upstream/master
```

If `master` cannot fast-forward cleanly, stop and inspect the divergence before proceeding.

### 3. Rebase `main` onto the refreshed `master`

```bash
git checkout main
git rebase master
```

Resolve conflicts in the worktree, then continue:

```bash
git rebase --continue
```

### 4. Verify the branch state

```bash
git status --short
git log --oneline --decorate -n 10
git rev-list --left-right --count upstream/master...main
```

Target:

- `master` should match `upstream/master`
- `main` should contain only the intended fork-local work

### 5. Recreate feature branches from `main`

```bash
git checkout main
git checkout -b feature/my-new-work
```

## Rebase Protection (Post-Rebase Verification)

**After every rebase onto `upstream/master`, verify each item below. These are fork-local changes that are silently dropped during rebase — no merge conflict, no error, just missing code.**

Every fix below was lost at least once during the TQ3_4S MTP rebase saga and had to be manually restored. If any are missing, performance collapses, the build breaks, or MTP silently produces zero drafts without any error message.

### 1. TQ3_4S dp4a dot-product — prefill drops 950→200 tok/s (CRITICAL)

**Files**: `ggml/src/ggml-cuda/vecdotq.cuh`, `ggml/src/ggml-cuda/mmvq.cu`, `ggml/src/ggml-cuda/mmq.cuh`

**What**: TQ3_4S uses dp4a (4-element dot-product accumulate) instructions for MMVQ and MMQ matrix-vector kernels. Without it, the TQ3_4S type falls back to a generic q8_1 path that is ~3x slower.

**Symptoms**:
- Prefill drops from ~950 tok/s to ~200 tok/s
- Generation drops from ~35 tok/s to ~10 tok/s
- `nvidia-smi` shows lower GPU utilization (MMVQ bottleneck)
- No crash, no error log — just silent slowdown

**Verification**: Run `llama-bench -m <tq3_4s_model> -p 128 -n 64 -pg 128,64 -ngl 99 -fa on -ctk q8_0 -ctv tq3_0`. Expected: >800 tok/s prefill (measured: 952 tok/s). If <300 tok/s, dp4a is missing.

**Key files to guard**:
- `ggml/src/ggml-cuda/vecdotq.cuh` — `vec_dot_tq3_4s_q8_1()` must use dp4a (`__dp4a`) not generic centroid lookup
- `ggml/src/ggml-cuda/mmvq.cu` — TQ3_4S MMVQ quantization path (`quantize_row_q8_1_tq3_cuda`)
- `ggml/src/ggml-cuda/mmq.cuh` — TQ3_4S MMQ tile loader (`decode_tq3_4s_dp4a` kernel)

### 2. TQ3_4S MMQ prefill threshold — narrow prefill regression (CRITICAL)

**File**: `ggml/src/ggml-cuda/mmq.cu`

**What**: The TQ3_4S MMQ threshold must be `ne11 >= 16` (not upstream's `ne11 >= 64`). Upstream's wider threshold forces narrow prefill shapes (common in speculative decoding and batch inference) onto the slow cuBLAS path.

**Symptoms**: Speculative decoding prefill drops from ~950 tok/s to ~200 tok/s. `nvidia-smi` shows high GPU utilization but low tok/s.

**Verification**: Check `ggml/src/ggml-cuda/mmq.cu` near `GGML_TYPE_TQ3_4S`: should return `ne11 >= 16`.

### 3. TQ3_4S CUDA quantization rotation — missing rotation = wrong results

**Files**: `ggml/src/ggml-cuda/mmq.cu`, `ggml/src/ggml-cuda/mmvq.cu`

**What**: TQ3_4S needs WHT (Walsh-Hadamard Transform) rotation applied to activations before quantization. Two implementations exist:
- Our path: `quantize_mmq_q8_1_tq3_cuda` / `quantize_row_q8_1_tq3_cuda` — specialized CUDA kernels that bake in the rotation
- Upstream path: `ggml_cuda_tq3_rotate_act` + generic `quantize_mmq_q8_1_cuda` — separate rotation + generic quantize

**Risk**: If both are dropped during a rebase, TQ3_4S silently uses unrotated quantization → quality regression without crashes.

### 4. Fattn-vec.cuh GGML_UNUSED_VARS parameter names — CUDA build failure

**File**: `ggml/src/ggml-cuda/fattn-vec.cuh`

**What**: The `GGML_UNUSED_VARS` calls (lines 93 and 545) must reference `Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr` (with `_ptr` suffix) to match the actual kernel function parameters. Upstream/master uses the non-`_ptr` names which fail on CUDA with "undeclared identifier" errors.

**Symptoms**: CUDA build fails with `error: identifier "Q" is undeclared` in `fattn-vec.cuh`.

**Verification**: Grep for `GGML_UNUSED_VARS` in `fattn-vec.cuh` — both occurrences must use `_ptr` suffixes.

### 5. MTP hook infrastructure — zero draft acceptance (CRITICAL)

**Files**: `src/llama-context.cpp`, `src/llama-context.h`, `src/llama-ext.h`, `src/llama-graph.cpp`, `src/llama-graph.h`, `src/llama-mtp.h`, `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`, `common/speculative.cpp`

**What**: The MTP (Multi-Token Prediction) speculative decoding hook requires:
- `llama_context::set_mtp()` and `handle_mtp_for_ubatch()` in `llama-context.cpp`
- `t_h_pre_norm` and `t_mtp_out` tensor exports in `llm_graph_result` (llama-graph.cpp/h)
- `h_pre_norm` / `t_mtp_out` registration in `qwen35.cpp` and `qwen35moe.cpp` graph builders
- `t_h_pre_norm` point to pre-norm (before `output_norm`), not post-norm (NaN risk)
- `llama_set_mtp()` must be called BEFORE `llama_set_embeddings_nextn()` (call-order bug)
- `need_n_rs_seq()` in `common/common.h` must return 0 for MTP (rollback RS snapshots)

**Symptoms**:
- Zero draft acceptance (`0.00000`)
- MTP slower than no speculation
- `NaN` in logits (pre-norm vs post-norm error)
- SEGFAULT during server startup (call-order bug)
- Build passes but MTP doesn't function

**Verification**: Run server with `--spec-type draft-mtp` and a capped request. If `#gen drafts = 0` in server stats, MTP hook is missing.

### 6. server-context.cpp prompt_load decoupling — KV cache restore failure

**File**: `tools/server/server-context.cpp`

**What**: When `cache_idle_slots` saves slot KV to cache-ram and clears the slot, reusing that slot must still attempt `prompt_load` even when `prompt_save` returns false. The upstream code guards `prompt_load` behind `saved_prompt`, which skips the load entirely when the slot was already cleared.

**Symptoms**: Server test `test_kv_keep_only_active.py` fails with `assert res.body["timings"]["cache_n"] > 0`. KV cache is never restored from cache-ram.

**Code pattern to verify**:
```cpp
// Must be: prompt_load always attempted
if (!ret->prompt_load(*prompt_cache, task.tokens)) {
    if (saved_prompt) {
        ret->prompt_clear(false);
    }
}
// NOT: prompt_load guarded by saved_prompt
// if (saved_prompt && !ret->prompt_load(*prompt_cache, task.tokens)) { ... }
```

### 7. Windows `_BitScanForward` build fix — Windows CI failure

**File**: `ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/ggml-cpu.c`

**What**: The `_BitScanForward` intrinsic is MSVC-only. Portable fallback with `__builtin_ctz` for GCC/Clang is needed. Use `#if defined(_MSC_VER)` guard.

**Symptoms**: Windows CI fails with linker error for `_BitScanForward`.

### 8. `common_chat_split_by_role` — link error after merge

**File**: `common/chat.cpp`

**What**: The function `common_chat_split_by_role()` was removed by upstream refactoring in `turbo/main` but is still referenced by some template parsers (Ministral, etc.). If dropped, the linker fails with undefined reference.

**Symptoms**: Linker error: `undefined reference to 'common_chat_split_by_role(...)'`.

### 9. Output reorder layer input stride fix — tensor stride corruption

**File**: `src/llama-graph.cpp` (or as committed in `fdc89eb`)

**What**: Fixes the output reorder function to use the correct layer input stride. Without this, the output reorder corrupts tensor strides when `n_embd` differs from expected layer dimensions.

### 10. Pre-norm export for MTP (NaN root cause) — zero drafts

**File**: `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`

**What**: The `t_h_nextn` tensor must point to the PRE-norm hidden state (before `output_norm`), not the post-norm state. If it points post-norm, the MTP head receives doubly-normalized hidden states → NaN → zero drafts.

**Historical fix** (commit `3ab921f`): Changed `res->t_h_nextn = cur` (post-norm) → `res->t_h_nextn = h_pre_norm` (pre-norm) in both `qwen35.cpp` and `qwen35moe.cpp`.

## When To Stop

Stop and ask for human review if:

- the rebase introduces widespread conflict
- a branch unexpectedly rewrites published history
- `master` diverges from `upstream/master` in a way that is not a fast-forward
- a feature branch depends on unrecovered local work in `/tmp` or another ephemeral directory
- any of the [Rebase Protection](#rebase-protection-post-rebase-verification) items cannot be verified

## Notes

- This SOP is the canonical branch-sync workflow for this repo.
- If a task needs a one-off exception, document it in a handover file before proceeding.
- After every rebase, run through the verification checklist above before pushing.
- For highest-impact items (#1, #2, #5), consider adding CI regression tests that detect silent regression.
