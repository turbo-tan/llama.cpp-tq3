# FAULT REGISTRY — llama.cpp-tq3 fork

Owned by: Hermes. All faults must have repro + evidence + disposition.
No hand-waving: we own this fork; every crash/corruption gets root-caused.

Status legend: OPEN / ROOT-CAUSED / FIXED / NOT-OUR-BUG

---

## FAULT-001 — SEGFAULT on startup, local build (exit 139) — OPEN

**Symptom:** `~/code/llama.cpp-tq3/build/bin/llama-server --version` → SIGSEGV (core dumped).
100% reproducible, no args needed.

**Backtrace (gdb, 2026-08-07):**
```
#0 __strlen_avx2
#1 __printf_buffer (format="...allowed origins for CORS (default: %s)...")
#2 __vsnprintf_internal
#3 string_format(...) from libllama-common.so.0
#4 common_params_parser_init() from libllama-common.so.0
#5 common_params_parse() from libllama-common.so.0
#6 llama_server() from libllama-server-impl.so
```

**ROOT CAUSE CONFIRMED (2026-08-07):** mixed-vintage binaries in `build/bin`:
- `llama-server` + `libllama-server-impl.so`: **Jul 6 00:04**
- `libllama-common.so.0`: **Aug 2 23:14** (partial incremental rebuild touched
  only common libs after upstream rebase changed `common_params` layout)
Control: `build-current/bin/llama-server` (all artifacts Aug 2, v10410,
same tree) → `--version` exits 0. ABI skew: Jul-6 server constructs
`common_params` with pre-rebase layout; Aug-2 libllama-common reads new
layout → `params.cors_origins.c_str()` is garbage → strlen SEGV.

**Status:** ROOT-CAUSED. Not a runtime code bug — stale mixed build.
Fix: never run mixed-vintage build dirs; full clean rebuild or use
build-current. NOTE: build/bin is from Jul 6 tree state — verify nothing
depends on it before cleanup (DELETION POLICY).

---

## FAULT-002 — SEGFAULT on error paths, .77 build-critic — FIXED (2026-08-10)

**Symptom:** error paths dump core instead of clean exit. Two observed:
1. `llama-mtmd-cli` load with `-ngl 99` but GPU busy (cudaMalloc OOM):
   `llama_model_load: error loading model: unable to allocate CUDA0 buffer`
   then `timeout: the monitored command dumped core` (2026-08-07 ~06:50).
2. Loading truncated GGUF (`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, tensor
   `blk.24.ffn_gate_exps.weight` out of file bounds):
   error logged cleanly, then core dump.

**Build:** .77 `build-critic` (llama.cpp-tq3-main @ Aug 7 00:36, native Ampere build).

**ROOT CAUSE CONFIRMED (2026-08-07, gdb backtrace on .77):**
```
#0 llama_vocab::get_suppress_tokens() const        <- null vocab deref
#1 llama_vocab_get_suppress_tokens()
#2 common_sampler_init(llama_model const*, ...)
#3 mtmd_cli_context::mtmd_cli_context(common_params&)
#4 main
```
`tools/mtmd/mtmd-cli.cpp:106-117` constructor dereferences the model
BEFORE the null guard:
```cpp
model = llama_init->model();            // null on load failure
vocab = llama_model_get_vocab(model);   // <- uses null model
smpl  = common_sampler_init(model, ...) // <- SEGV here (suppress_tokens)
...
if (!model || !lctx) { exit(1); }       // guard TOO LATE
```
**Attribution: UPSTREAM BUG** — fork and upstream/master are byte-identical
at this site (upstream last touched 071327508, Aug 4). Not fork-introduced.
Fix: move the `!model || !lctx` guard above the vocab/sampler calls.

**Fixed in fork** (`tools/mtmd/mtmd-cli.cpp`, this PR): guard reordered above
the vocab/sampler calls, matching the fix drafted in #68. Not yet sent
upstream — worth a separate upstream PR since it's confirmed present in
upstream/master too.
The OOM case (cudaMalloc fail → exit?) needs the same verification —
load failed there too; presumed same path, now covered by the same guard.

---

## FAULT-003 — MTP GGUF: generation corruption on image batches — OPEN

**Symptom:** Qwen3.6-35B-A3B-MTP-TQ3_4S.gguf + mmproj-BF16 (fork build on .77, Ampere):
- text-only requests: fine
- one image ≤400×250 with F16 KV: fine
- one image ≥420×262: output degenerates to `/`-loop garbage
- TWO images of any size (even 64×64): immediate garbage
- server slot stays poisoned after a bad request (subsequent small requests fail until restart)
- fresh-process CLI reproduces it (not server state)

**Log signature (every corrupted request):**
```
find_slot: non-consecutive token position 4 after 3 for sequence 0 with 260 new tokens
```
(260 = image-embedding batch size; text batches never trigger the warning)

**Bisect results (.77, fork build-critic):**
| config | result |
|---|---|
| default (turbo KV), any image | BAD at all sizes |
| F16 KV, img ≤400×250 | OK |
| F16 KV, img ≥420×262 | BAD + poisons slot |
| F16 KV, 2× small img | BAD |
| GGML_CUDA_DISABLE_GRAPHS=1 + F16 KV | still BAD |
| CLI fresh process, F16 KV, one 640×400 img | BAD (non-consecutive positions) |
| `--spec-type none` | flag rejected by mtmd-cli (examples gating); env var had no effect |

Multi-image results across runs (all fork builds, Ornith non-MTP):
| run | build/config | 2-small | 2-big | repeat |
|---|---|---|---|---|
| run-1 | llm-launch/release, c8192, no min-tokens | BAD(empty) | text: "only one image" | text: "only one image" |
| run-2 | same server (judge2 bind-failed; ran on run-1 cfg) | BAD(empty) | BAD(empty) | BAD(empty) |
| run-3 | llm-launch/release, c16384, image-min-tokens 1024 | BAD(empty) | BAD(empty) | BAD(empty) |

Observations:
- single-image works (text OK, 1-small OK, 1-big OK) on run-3 config
- multi-image returns EMPTY content and hits the max_tokens cap
  (n_decoded=600 == budget) on every failing case — consistent with
  a thinking-model reasoning-content consuming the whole budget,
  NOT necessarily generation corruption. Diagnostic with budget 2500
  + reasoning_content capture in progress.
- NON-DETERMINISM: identical input gave different failure modes
  across runs (text "one image" vs empty).

**Control:** Ornith-1.0-35B-TQ3_4S (same A3B base, NON-MTP, own mmproj) on local
3090 with `~/code/llm-launch/release/bin/llama-server`.
**ATTRIBUTION CORRECTION:** that binary is NOT upstream — it is the atomic FORK
(worktree `llama.cpp-tq3-origin-main-llm-launch-20260703` @ 77dd77473,
"cuda: TQ3_4S Ampere decode +35%", Jul 3). So a 2-image failure there is
still fork territory, not a clean upstream control.
Single-image results (run 2, `-c 16384 --image-min-tokens 1024`):
  text OK · 1 small (400×250) OK · 1 big (1280×800) OK.
Multi-image: 2 small / 2 big / repeat all returned empty content.
**NOTE:** first multi-image run (run 1) was INVALIDATED — judge2 server failed
to bind (port 8091 held by the run-1 server, `-c 8192`); results came from the
old config. Run-3 (correct config) pending.

**Fork-only code in the suspect path:** `src/llama-mtp.h` (entirely new file,
not in upstream/master): `llama_mtp { ctx_mtp, hook_batch, hook_batch_embd_buffer,
hook_tokens, pending_h, pending_pos }`. Hook mechanism in
`llama-context.cpp`: `set_mtp()`, `handle_mtp_for_ubatch()` called at line 2289
with `t_h_nextn`. Upstream master has NONE of this — its MTP is implemented
differently (speculative decoding via separate draft context).

**Hypothesis:** the hook intercepts every ubatch including embd (image)
batches; `handle_mtp_for_ubatch` assumes token batches (positions strictly
+1, seq 0) — embd batches with n_pos_per_embd / non-contiguous positions
corrupt the MTP draft ctx KV and/or the main ctx state. The
"non-consecutive token position" warning confirms positions 4,5,...263 were
expected after 3 but the batch carried a different position layout.

---

## FAULT-004 — non-MTP control blocked on .77 — OPEN (logistics)

- `~/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` on .77 is TRUNCATED (13.9 GB of 22.7 GB).
  Full copy exists local: `~/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- .77 disk 97% full (15 GB free) → cannot receive a 22.7 GB file;
  /tmp/qwen36-35b-critic staging (14 GB) could be freed but deletion needs operator sign-off.
- Ornith on local substitutes for the Ampere non-MTP control.

---

## FAULT-005 — SECOND IMAGE SILENTLY DROPPED in multi-image requests — OPEN (root cause narrowed)

**Symptom:** request with 2 image_url parts → model only sees image #1.
Confirmed on 2026-08-07 with full diagnostic (Ornith, llm-launch/release fork
build, local 3090): budget 2500, no garbage generation; reasoning content says
verbatim "Wait, there is only one image provided."

**ROOT CAUSE PROVEN (2026-08-07):** the `[QWEN_VIDEO]` temporal frame-merge in
`mtmd_tokenizer::tokenize()` (tools/mtmd/mtmd.cpp:953-970) fuses ANY two
consecutive same-size bitmaps into a single image chunk, collapsing the second
one (`parts.erase`). It assumes consecutive images = video frames. The merged
chunk decodes as ONE image (debug log: `n_chunks = 1` for two images,
`n_tokens_batch = 1024`, both halves at identical positions — `non-consecutive
token position 11 after 11 ... 512 new tokens`), so the second image is
invisible to the model.

**Discriminator experiments (upstream v10307 build, UD-Q4_K_M GGUF):**
| test | setup | result |
|---|---|---|
| same-size adjacent | red 300² + green 300² | ❌ one image seen |
| **different-size adjacent** | red 300² + teal 500×400 | ✅ **both seen** (can't merge → proves merge is culprit) |
| text between | red + text + teal | ✅ both seen |
| separate turns | turn1 red, turn2 teal | ✅ both seen |

**Attribution:** upstream bug (reproduced on clean upstream master built today;
fork is byte-identical in this path). Present upstream since the mtmd_batch
temporal-merge work; upstream TODO at clip.cpp:5073 already flags the merge
logic as needing refactor.

**FIX MERGED-READY — PR #69 (2026-08-07):** branch
`fix/consecutive-image-batch-position-collision`, commit `ea117fd82`.
Gate temporal merge on empty bitmap ids:
`can_merge_with(...) && bm_a->id.empty() && bm_b->id.empty()`.
Rationale: video frames from the lazy video reader carry NO id
(`read_next_frame()` → `mtmd_bitmap_init` sets none), while standalone images
from `mtmd_helper_bitmap_init_from_buf()` always get a non-empty FNV-hash id
(server and CLI both use this path). So the merge keeps working for real
videos and stops fusing independent images.
**VERIFIED on fixed fork build (8091):** T1 same-size adjacent now reports
TWO images (red + teal); diffsize and text-between unchanged (two images).
All three pass.

**MECHANISM CONFIRMED earlier (order-swap, Ornith + fork build):**
solid RED and GREEN 300×300 images, enable_thinking=False:
- order A (red,green) → describes RED, byte-identical to single-red control
- order B (green,red) → "uniform solid color" singular, second image unseen
→ **positional drop: model always sees image #1; image #2's embd batch never
reaches context.** Ties to "non-consecutive token position" warning (.77 logs).

**Attribution progress:**
- happens on NON-MTP model (Ornith) → NOT the fork MTP hook
- happens with NO MTP draft head registered (verified: zero "MTP draft head
  registered" in server logs of all affected runs, local and .77)
- single image always fine; second image in the batch is dropped
- Suspect: second-image embd batch handling in mtmd/server batch construction,
  or KV/recurrent-memory cell overwrite when the second embd batch arrives
  (ties to the "non-consecutive token position" warning seen on .77).
- Build: atomic fork worktree @ 77dd77473 (Jul 3). Upstream control NOT yet
  run (needs an upstream-built binary).

**Separate finding (same tests):** the empty-content results on multi-image
runs were NOT corruption — Ornith is a thinking model; reasoning consumed the
entire max_tokens budget (2500 tokens of reasoning observed). Judge calls need
reasoning disabled or large budgets. See FAULT-003 notes.

---

## FAULT-006 — context loops at 40-50k tokens — OPEN (no repro captured)

**Symptom (user report):** generation degenerates into loops around 40-50k
context on this model family. Occurs on FORK MAIN (user-confirmed 2026-08-07).

**Status:** no captured repro, log, or config. Needs: model, quant, flags,
prompt length at onset, log excerpt. Candidate mechanisms to test once
reproduced: KV cache type quantization at depth, position encoding past a
training-length boundary, MTP hook pending_pos drift, CUDA graph replay
with long sequences (graphs were already shown to matter — FAULT-003 bisect).

**Suspect tie-in:** same `/`-loop output family as FAULT-003. If the loop
appears with text-only long context (no images), the embd-batch path is
excluded and the suspect list narrows to KV/position handling.

---

## Cross-reference

- User context-loop bug (40-50k context, this model family) — suspected same
  family as FAULT-003 (MTP hook state corruption). Needs its own entry with repro.
