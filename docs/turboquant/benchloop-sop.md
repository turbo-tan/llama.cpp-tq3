# BenchLoop SOP

Use this exact runtime shape for the flagship Qwen3.6-27B-MTP release path when running local BenchLoop checks.
If you are orienting yourself, start with [INDEX.md](INDEX.md) first and then use this page as the detailed benchmark gate.

## Rules

- Keep `-fa on`.
- Use `--spec-type draft-mtp`.
- Use `--spec-draft-n-min 1`.
- Use `--spec-draft-n-max 2`.
- Use `--spec-draft-p-min 0.0` on current `main`.
- Do not reuse older `p_min 1.0` guidance from the legacy hook-driven `mtp` path; on current `draft-mtp` it can suppress draft generation entirely.
- If the long-prompt repro shows `draft acceptance = 0.00000` or the draft path looks wrong, stop and investigate before running any further BenchLoop jobs.
- If `common/speculative.cpp::common_speculative_impl_draft_mtp::process()` is already using the `llama_set_mtp()` hook path, do not try to remove a second catch-up decode there again; the remaining cost is the repeated seed/advance decodes in `draft()`.
- Do not skip the initial `draft()` seed decode based only on `pending_pos == dp.n_past - 1`; that leaves no draft logits for `common_sampler_sample()` and aborts on `batch.logits[0] != true`.
- For MTP prompt prefill, `tools/server/server-context.cpp` must reserve `n_batch` outputs for `draft-mtp` and mark prompt rows with `slot.need_embd() || slot.need_embd_nextn()`; otherwise `ctx_dft pos_max` stays `-1` after prefill or `output_reserve()` aborts.
- If `begin(): ctx_dft pos_max=-1` appears after prompt prefill, check whether `common_speculative_impl_draft_mtp::process()` is pruning draft KV from prompt `pos_start=0`; that deletes the hook catch-up. Removing that prune fixes prompt catch-up retention but is not sufficient by itself to recover speed.
- Do not repeat these as standalone speed fixes: enabling backend sampling, explicitly setting draft KV to `q8_0/tq3_0`, restoring the old single-sequence `8ad718007` MTP driver only in `common/speculative.cpp`, or restoring pinned hook buffers only in `llama_context::set_mtp()`. They were tested on 2026-06-15 and remained about `8 tok/s` on the capped tc-15 smoke.
- Keep BenchLoop local-only with `BENCHLOOP_NO_SUBMIT=1`.
- Run a fast dry run first to confirm speed is on par before running the full `partial` suite.
- For template-only quality isolation on the out6k artifact, use `-c 32768`.
- Do not use `-c 262144` for template A/B checks unless you have already confirmed VRAM headroom; on 2026-06-13 it produced a false negative via CUDA OOM before quality could be measured.
- Before starting the `-ngl 99` server, stop any other GPU consumers and verify the card is free enough for a full load; do not assume the model will auto-fit around a busy VRAM state.
- Quality guard: if the long-prompt path reaches `draft acceptance = 0.00000`, stop immediately and go back to the last known-good template/runtime before comparing anything else.
- Build guard: before any BenchLoop run, verify `./build-current/bin/llama-server --version` matches the current `git rev-parse HEAD`. If it does not, rebuild `build-current` first.

## Rebase Protection (Critical Fixes)

**These are fixes that can be silently dropped during a rebase from `upstream/master`. Verify each one after every rebase. If any are missing, performance collapses or the build breaks without obvious error messages.**

Every fix below was lost at least once during the TQ3_4S MTP rebase saga and had to be manually restored. Add CI regression tests for the highest-impact items where feasible.

### 1. TQ3_4S dp4a dot-product — prefill drops 950→200 tok/s (CRITICAL)

**Files**: `ggml/src/ggml-cuda/vecdotq.cuh`, `ggml/src/ggml-cuda/mmvq.cu`, `ggml/src/ggml-cuda/mmq.cuh`

**What**: TQ3_4S uses dp4a (4-element dot-product accumulate) instructions for MMVQ and MMQ matrix-vector kernels. Without it, the TQ3_4S type falls back to a generic q8_1 path that is ~3x slower.

**Symptoms**:
- Prefill drops from ~950 tok/s to ~200 tok/s (if dp4a missing, measured baseline: 952 tok/s)
- Generation drops from ~35 tok/s to ~10 tok/s (if dp4a missing, measured baseline: 35 tok/s)
- `nvidia-smi` shows lower GPU utilization (MMVQ bottleneck)
- No crash, no error log — just silent slowdown

**Verification test idea**: Run `llama-bench -m <tq3_4s_model> -p 128 -n 64 -pg 128,64 -ngl 99 -fa on -ctk q8_0 -ctv tq3_0`. Expected: >800 tok/s prefill with TQ3_4S (measured: 952 tok/s). If <300 tok/s, dp4a is missing.

**Key files to guard**:
- `ggml/src/ggml-cuda/vecdotq.cuh` — `vec_dot_tq3_4s_q8_1()` must use dp4a (`__dp4a`) not generic centroid lookup
- `ggml/src/ggml-cuda/mmvq.cu` — TQ3_4S MMVQ quantization path (`quantize_row_q8_1_tq3_cuda`)
- `ggml/src/ggml-cuda/mmq.cuh` — TQ3_4S MMQ tile loader (`decode_tq3_4s_dp4a` kernel)

### 2. TQ3_4S MMQ prefill threshold — narrow prefill regression (CRITICAL)

**File**: `ggml/src/ggml-cuda/mmq.cu`

**What**: The TQ3_4S MMQ (matrix-multiply quantized) prefill threshold must be `ne11 >= 16` (not upstream's `ne11 >= 64`). Upstream's wider threshold forces narrow prefill shapes (common in speculative decoding and batch inference) onto the slow cuBLAS path.

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

**Verification test idea**: Run server with `--spec-type draft-mtp` and a capped request. If `#gen drafts = 0` in server stats, MTP hook is missing.

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

### 7. Windows \_BitScanForward build fix — Windows CI failure

**File**: `ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/ggml-cpu.c`

**What**: The `_BitScanForward` intrinsic is MSVC-only. Portable fallback with `__builtin_ctz` for GCC/Clang is needed. Use `#if defined(_MSC_VER)` guard.

**Symptoms**: `llama.cpp-tq3` Windows CI fails with linker error for `_BitScanForward`.

### 8. common_chat_split_by_role — link error after merge

**File**: `common/chat.cpp`

**What**: The function `common_chat_split_by_role()` was removed by upstream refactoring in `turbo/main` but is still referenced by some template parsers (Ministral, etc.). If dropped, the linker fails with undefined reference.

**Symptoms**: Linker error: `undefined reference to 'common_chat_split_by_role(...)'`.

### 9. Output reorder layer input stride fix — tensor stride corruption

**File**: `src/llama-graph.cpp` (or as committed in `fdc89eb`)

**What**: Fixes the output reorder function to use the correct layer input stride. Without this, the output reorder corrupts tensor strides when `n_embd` differs from expected layer dimensions.

### 10. tc-15 pre-norm export (NaN root cause) — zero drafts

**File**: `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`

**What**: The `t_h_nextn` tensor must point to the PRE-norm hidden state (before `output_norm`), not the post-norm state. If it points post-norm, the MTP head receives doubly-normalized hidden states → NaN → zero drafts.

**Historical fix** (commit `3ab921f`): Changed `res->t_h_nextn = cur` (post-norm) → `res->t_h_nextn = h_pre_norm` (pre-norm) in both `qwen35.cpp` and `qwen35moe.cpp`.

## BenchLoop Results (Rebased Branch)

Latest run on `perf/tq3-4s-dp4a-mmvq-rebased` (commit `d127241a3`, rebased on upstream/master):

| Metric | Original Branch | **Rebased Branch** | Delta |
|--------|----------------|-------------------|-------|
| **Gen tok/s** | 42.73 | **54.23** | **+27%** |
| **Speed score** | 68.63 | **73.1** | +6.5% |
| **Toolcall** | 96.67% (14/15) | **96.67%** (14/15) | Unchanged |
| **Coding** | 100% (12/12) | **100%** (12/12) | Unchanged |
| **Overall** | — | **93.0** | — |

The +27% speedup comes from upstream/master improvements (61 new commits) applied on top of the same TQ3_4S MTP code. Quality is identical.

**Run configuration**:
- Model: `Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf`
- Binary: `build-current/bin/llama-server` (version `9712 (d127241a3)`)
- Template: `--chat-template-file publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja`
- Server: `-c 32768 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv tq3_0`
- Spec: `--spec-type draft-mtp --spec-draft-n-min 1 --spec-draft-n-max 2 --spec-draft-p-min 0.0`
- Suites: speed, toolcall, coding (partial)

## Canonical Roots

Use these paths as the benchmark defaults for this repo:

- Repo root: `/home/awee/code/tan_llama`
- Primary runtime binary: `/home/awee/code/tan_llama/build-current/bin/llama-server`
- Primary bench binary: `/home/awee/code/tan_llama/build-current/bin/llama-bench`
- Legacy build tree: `/home/awee/code/tan_llama/build`
- Model root: `/home/awee/models/turboquant`

## Server launch

```bash
cd /home/awee/code/tan_llama
./build-current/bin/llama-server \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --alias Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --host 127.0.0.1 --port 18124 \
  -c 32768 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 \
  --spec-type draft-mtp \
  --spec-draft-n-min 1 \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.0 \
  --reasoning off
```

Current recovery status:

- For `Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf`, the embedded GGUF template is not equivalent to the local publish template.
- Measured on 2026-06-13:
  - embedded template regressed `tc11` by calling `calculator`
  - `--chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja` restored `tc11` to a direct answer
- Until the GGUF metadata is rebuilt and revalidated, use the publish template override for local recovery checks.
- The embedded publish-template GGUF artifact is not the baseline for this recovery pass. The validated path is the original out6k GGUF plus the publish template override.

Use:

```bash
--chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja
```

Preflight for the `-ngl 99` launch:

```bash
pkill -f 'llama-server.*Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6'
nvidia-smi
```

Only launch the test server after confirming the GPU is clear enough for the full load.

Do not assume the embedded template is the publication winner for this artifact.

Verified 2026-06-13 template-isolation gate on current `build-current` with `-c 32768`:

- `toolcall 96.7 14/15`
- `coding 100.0 12/12`
- template lineage:
  - old stock-derived publish template from `25b98f6be`
  - plus scoped JSON-only numeric rule
  - plus direct-answer tool-avoidance rules from `artifacts/tc11-prompt-patched.txt`

## Dry run

Use a direct `llama-bench` check before BenchLoop:

```bash
cd /home/awee/code/tan_llama
./build-current/bin/llama-bench \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  -p 128 -n 64 -pg 128,64 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 -r 1 -o md
```

If the dry run is materially below the expected local baseline, stop and fix the runtime shape before running BenchLoop.
Also stop immediately if server stats show `#gen drafts = 0` for `draft-mtp`; that means the speculative path is configured but not actually drafting.
If a long prompt collapses to `draft acceptance = 0.00000`, treat that as a hard stop and revert to the last known-good runtime before continuing.

### tc-15 regression probe

Use this exact build-and-test sequence when checking the long-context MTP regression:

```bash
cd /home/awee/code/tan_llama
cmake --build build-current -j 16 --target llama-server
```

```bash
cd /home/awee/code/tan_llama
./build-current/bin/llama-server \
  --host 127.0.0.1 --port 18124 \
  -m /home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja \
  --ctx-checkpoints 0 --cache-ram 0 -np 1 \
  -c 32768 -ngl 99 -fa on -ctk q8_0 -ctv tq3_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
  --no-spec-draft-backend-sampling
```

Replay the exact tc-15 request:

```json
{
  "model": "Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant with access to tools..." },
    { "role": "user", "content": "Search for the population of Iceland and calculate what 2% of it would be." }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
          "type": "object",
          "properties": { "query": { "type": "string" } },
          "required": [ "query" ]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Calculate expressions",
        "parameters": {
          "type": "object",
          "properties": { "expression": { "type": "string" } },
          "required": [ "expression" ]
        }
      }
    }
  ],
  "temperature": 0,
  "stream": false
}
```

## Partial suite

Run `partial` as one combined local-only invocation:

```bash
BENCHLOOP_NO_SUBMIT=1 benchloop run \
  --model Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf \
  --endpoint http://127.0.0.1:18124 \
  --provider openai_compat \
  --suites speed,toolcall,coding \
  --harness raw
```
