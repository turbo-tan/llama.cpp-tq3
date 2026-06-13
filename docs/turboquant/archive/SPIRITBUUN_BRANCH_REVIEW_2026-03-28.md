# Spiritbuun Branch Review

Repo reviewed:
- `https://github.com/spiritbuun/llama-cpp-turboquant-cuda`

Branches reviewed:
- `feature/turboquant-kv-cache`
- `experiment/asymmetric-kv`
- `experiment/layer-adaptive`
- `experiment/decode-speed-parity`

## What Is Worth Copying

### 1. Layer-adaptive KV policy

Relevant file:
- `src/llama-kv-cache.cpp`

Useful idea:
- per-layer KV type promotion controlled by `TURBO_LAYER_ADAPTIVE`
- concrete modes already encoded for:
  - first/last layers in `q8_0`
  - last-N layers only
  - asymmetric `K-only` / `V-only` promotions

Why it matters for us:
- this matches our current finding that not all layers are equally sensitive
- we already saw family sensitivity on weights (`ffn_down`)
- this is the cleanest reusable KV-side policy scaffold

What to copy:
- mode plumbing pattern
- per-layer `layer_type_k` / `layer_type_v` override at KV allocation time
- logging and env-controlled experimentation

What not to copy blindly:
- their exact mode defaults
- their quality claims without reproducing on our witnesses

### 2. CPU fallback for partial offload

Relevant file:
- `src/llama-kv-cache.cpp`

Useful idea:
- if a KV layer lands on CPU and turbo types have no CPU vec-dot kernel, downgrade that layer to `q8_0`

Why it matters for us:
- we repeatedly hit bad partial-offload behavior and misleading CPU-heavy runs
- this is a pragmatic guardrail for `--fit on` / mixed offload cases

What to copy:
- detect host buffer type per layer
- fall back to `q8_0` only for CPU-bound KV layers
- warn once, not per layer

### 3. Their benchmark framing

Relevant files:
- `benchmark-results.md`
- `TURBOQUANT_CUDA_IMPLEMENTATION.md`

Useful ideas:
- report PPL, prefill, decode, and context-fit separately
- treat long-context fit as a first-class outcome
- benchmark asymmetric KV and layer-adaptive modes, not just uniform modes

Why it matters for us:
- this matches our current product story better than chasing a single microbench

## What Is Not Worth Copying Directly

### 1. Hybrid memory scaffolding

Relevant files:
- `src/llama-memory-hybrid*`
- large parts of `src/llama-graph.cpp`

Why not:
- mostly unrelated to our immediate TQ3/TQ3_1S quality work
- high integration cost
- likely to create noise rather than a direct win

### 2. Their current Flash Attention auto behavior

Relevant diff:
- `feature/turboquant-kv-cache..experiment/asymmetric-kv`
- `feature/turboquant-kv-cache..experiment/layer-adaptive`

Observation:
- those branches remove the automatic "turbo KV requires FA" forcing in `llama-context.cpp`

Why not:
- we already learned that FA gating is fragile and important
- removing safety forcing is the opposite of what we need

### 3. Their weight-path implications

Observation:
- this fork is much more mature on KV than on a new weight-format line
- it does not give us a shortcut for `TQ3_1S` quality

Why not:
- our current bottleneck is weight quality, not missing basic KV scaffolding

## Most Relevant Claimed Results

From `benchmark-results.md`:
- `turbo3` with rotation and norm correction is competitive with `q8_0`
- layer-adaptive modes improve quality further
- prefill dequant+MMA is their big speed unlock for KV prefill
- asymmetric layer-adaptive (`K-only` or `V-only`) reportedly did not help in their setup

Important caution:
- these numbers are from their fork, their hardware, and their model choices
- useful as design signal, not as proof for our branch

## Best Copy Candidates For Our Branch

Priority order:
1. add layer-adaptive KV mode plumbing
2. add CPU fallback-to-`q8_0` for CPU-bound turbo KV layers
3. mirror their benchmark matrix structure in our docs
4. study their prefill dequant+MMA idea only after our current quality line is stable

## Relevance To Our Current State

For weights:
- no direct shortcut for `TQ3_1S`
- the branch does reinforce the general idea that quality wins come from selective precision, not uniform aggression

For KV:
- strong signal to adopt layer-adaptive KV experiments
- strong signal to keep long-context fit as an explicit target
- useful implementation reference for partial-offload guardrails
