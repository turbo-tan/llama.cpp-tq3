# Performance Campaign SOP (branch-per-phase, ledger-driven)

Status: ACTIVE steering. Applies to all TurboQuant speed/perf work.
Born from the 27B MTP outQ6K campaign (2026-06-10/11); see
[SPEED_PLAN_27B_MTP_OUT6K_20260610.md](../turboquant/active/SPEED_PLAN_27B_MTP_OUT6K_20260610.md).

Current canonical workspace roots for this campaign are:

- Docs and artifacts: `/home/awee/code/tan_llama`
- Primary branch build: `/home/awee/code/tan_llama/build-current`
- Legacy branch build: `/home/awee/code/tan_llama/build`
- Shared models: `/home/awee/models/turboquant`

If an older phase refers to a separate worktree, treat that as historical context
only unless the phase note says otherwise.

## 1. The ledger rule (why this SOP exists)

Every campaign has ONE living plan doc under `docs/turboquant/active/`. It must
always answer: what did we try, on which branch, what did we measure, and what
is the verdict. Update it at two moments, no exceptions:

- BEFORE starting a phase: one paragraph stating the hypothesis and the metric
  that will decide it.
- AFTER finishing a phase: measured numbers, verdict (WIN / NEGATIVE / BLOCKED),
  and what the result implies for the remaining phases.

Negative results are first-class: they are committed with the same care as wins
(they prevent re-trying dead ends). The plan doc keeps a do-not-repeat table.

## 2. Branch workflow

All runtime/kernel work happens in the llama.cpp-tq3 worktree
(`/home/awee/code/worktrees/tan_llama-main-ref`). Docs and artifacts live in
tan_llama.

1. WINNER pointer: the commit that all new work branches from. It only moves
   when a phase passes all gates (build, correctness, performance vs witness,
   artifacts saved).
2. One branch per phase, named `perf/<phase-id>-<slug>`, created FROM the
   current WINNER. Never branch from another experiment.
3. No dirty state: `git status --porcelain` must be empty before creating a
   phase branch and before declaring a phase done. Stray files are committed,
   stashed with a note, or deleted - never carried silently.
4. Breakthrough = gates passed. Commit IMMEDIATELY on the phase branch with the
   measured numbers in the commit message. Never stack a second experiment on
   an uncommitted win.
5. Failed experiments: commit with an `experiment:` prefix, then commit the
   revert so the branch tip equals winner content. The branch stays as the
   record; the plan doc gets the verdict.
6. Commit trailer `Assisted-by: <assistant>` (never Co-authored-by), ASCII only.
7. Pushes: user's own fork PR branches are pre-approved; NEVER push to
   upstream, never create upstream PRs, never merge into master.

## 3. Quality rules (hard, from the 2026-06-11 reset)

1. Only two classes of speed work are allowed:
   - LOSSLESS BY CONSTRUCTION: draft-then-verify with greedy verification;
     output is bit-identical, no quality gate needed.
   - MEASURED-NEUTRAL: runtime/kernel/KV changes that must pass the witness
     PPL/KLD gate with MEASURED numbers before any claim.
2. Estimated quality numbers are FORBIDDEN in decision-making. If it was not
   measured, it does not exist.
3. No recipe (quantization policy) change ships on a speed argument.

## 4. GPU session protocol (RTX 3090 shared with production)

The production server is the user systemd unit `llama-server.service`
(port 8085). User has standing approval to take the GPU, with obligations:

```bash
# take the GPU
systemctl --user stop llama-server.service
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # must be empty

# ... bench session ...

# give it back (ALWAYS, same session)
systemctl --user reset-failed llama-server.service   # old build segfaults on exit
systemctl --user start llama-server.service
sleep 45 && curl -s http://localhost:8085/health     # must return ok
```

- Capture the running command line (`pgrep -af llama-server`) before killing
  anything unfamiliar.
- pkill pitfall: `pkill -f` matches its own wrapper shell; use a
  self-excluding pattern like `pkill -f '[l]lama-server -m'`.
- Never end a session with the production server down.

## 5. Measurement protocol

- Warmup first; never claim from a cold-start run (JIT inflates 15-25 pct).
- `llama-bench -r 5` for headline numbers; pp2048 is stable, pp512 is noisy.
- Always run a CONTROL (e.g. Q3_K_M witness) before attributing an effect to
  TQ3_4S - the A2 phase died to a missing control.
- Useful probes discovered in this campaign:
  - `llama-batched-bench -npl 1,2,3,4` = decode cost at MTP verify shapes
  - `llama-bench -p 3` = single-sequence batch-3 verify pass cost
  - `TURBO_MTP_PROF=1` on llama-server = per-cycle spec-decode cost split
    (draft prefetch/decode/sample, hook, ckpt, sampler, target verify)
  - `GGML_CUDA_DISABLE_GRAPHS=1` = isolate CUDA-graph contribution
- Artifacts: `tan_llama/artifacts/perf_<phase>/` with branch, commit, model,
  date in filenames. Server probes: same prompt, temp 0, cache_prompt false,
  2 runs minimum.

## 6. Session delegation (avoid 8-hour Claude sessions)

Long interactive Claude sessions are expensive. Before starting GPU-intensive or
multi-step work, decide: can this run unattended?

### What to delegate (run unattended, read results in a fresh short session)

| Task | Tool |
|---|---|
| Build + bench loops (M-phases) | Write a `bench_<phase>.sh` script, launch with `! bash bench_<phase>.sh 2>&1 \| tee artifacts/perf_<phase>/bench.log &` |
| PPL / KLD sweeps | Same pattern — script it, pipe to log |
| nsys / ncu profiling runs | Capture to `.nsys-rep` / `.ncu-rep`, open in UI later |
| Analysis of bench output | Pipe log to local llama-server or ModelScope API |

### Available endpoints for delegation

- **desktop-play** (`192.168.1.77`, SSH as `awee`): second RTX 3090 24GB,
  same `llama.cpp-tq3` repo at `/home/awee/code/llama.cpp-tq3`, models in
  `/home/awee/models/27b-mtp/` and `35b-mtp/`. **Preferred target for all
  bench/build work** — keeps local production server (:8085) untouched.
  Service: `llama-server.service` (port 8080, systemd user unit). Kill before
  benching, restore after (same protocol as local §4).
- **Local llama-server** (port 8085, `llm-launch flagship`): use for reading
  bench logs and suggesting next steps; no Claude quota consumed.
- **ModelScope Qwen3.7-Plus** (`llm-launch --profile modelscope-qwen37-plus`):
  remote OpenAI-compatible API; good for code review, analysis, planning tasks
  that don't need file edits or git.
- **opencode** (`llm-launch opencode`): agentic code editor that can handle
  read/edit/commit cycles in a separate process; use for self-contained branches
  where the scope is clear (e.g. "apply this diff and verify it builds").

### When an interactive Claude session IS required

- File edits that need judgment or context from this conversation.
- git operations (commit, push, branch management).
- Interpreting novel results and deciding the next phase.

**Rule**: at the start of each phase, write the bench script FIRST. If the
script fully captures the work, hand it off and close the session. Only keep
the session open if there are decisions that can't be scripted.

## 7. Ops constraints

- Disk: check `df` before any phase that writes model-sized artifacts; keep
  >=100G free or archive variants first (HF_UPLOAD_SOP). Disk pressure has
  already corrupted one phase (A4).
- After editing `mmq.cuh`: touch the template instances and `mmq.cu` before
  rebuilding (header dependency detection is unreliable).
- Build dir must match the branch tip at phase close (rebuild after reverts).
