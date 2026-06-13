# TurboQuant Master Plan

## Purpose

This is the current working plan for TurboQuant.

It replaces fragmented "moonshot" strategy notes with one execution order that balances:

- quality
- model size
- decode speed
- prompt processing speed

The plan is built from the measured state in `artifacts/`, not from older speculative designs.

## Current Baseline

Use the public `llama.cpp-tq3` mainline as the quality baseline unless a newer branch is proven with:

- successful build of the required binaries
- perplexity parity or improvement
- coherent chat output
- reproducible `llama-bench` numbers

Current measured anchors:

- 9B base `TQ3_4S`: `PPL = 7.7486 +/- 0.05116`
- 9B Qwopus `TQ3_4S`: `PPL = 7.6181 +/- 0.04897`
- 9B `Q3_K_S`: `PPL = 7.6021 +/- 0.05064`
- 9B `Q4_K_M`: `PPL = 7.2915 +/- 0.04797`
- 27B base `TQ3_4S` has historically shown strong quality at the normal evaluation gate, but recent prompt-path work introduced correctness and long-context concerns that must be treated as unresolved until revalidated.

Implication:

- `TQ3_4S` is near `Q3_K_S` on quality in the better variants
- `TQ3_4S` is still behind `Q4_K_M`
- quality is no longer hopeless, but it is not yet strong enough to justify trading away correctness for speed

## Hard Guardrails

Every optimization branch must pass these gates before it is allowed to replace the public baseline:

1. Build gate

- `llama-bench`
- `llama-server`
- `llama-simple`
- `llama-simple-chat`

2. Correctness gate

- perplexity on a fixed witness model
- strict chat smoke with thinking disabled where supported
- one code-generation style chat probe
- one server inference probe

3. Performance gate

- `pp2048`
- `tg128`
- model-specific prompt probe if that is the target path

4. Artifact gate

- every run saved under `artifacts/`
- artifact names include model, build, date, and test type

No kernel rewrite should be treated as a success based on microbench alone.

## Priority Order

Execution order matters.

1. Restore and lock correctness
2. Improve quality at nearly fixed size
3. Reduce memory size without collapsing quality
4. Improve decode speed
5. Improve prompt processing speed

This order is deliberate:

- a faster broken runtime is not useful
- a smaller model with a bad quality cliff is not useful
- prompt-speed work is the easiest place to accidentally introduce silent corruption

## Track 1: Correctness Recovery

### Objective

Establish one known-good runtime state and one known-good model state for repeatable comparisons.

### Immediate tasks

- confirm the last known coherent public build on:
  - `Qwen3.5-9B-TQ3_4S`
  - `Qwopus3.5-27B-v3-TQ3_4S`
- isolate the prompt-path regression introduced by the recent MMQ tile-loader rewrite
- compare current public main against the pre-rewrite commit on:
  - `pp2048`
  - `pp16000` or the highest context that still fits safely
  - chat coherence

### Success criteria

- public build compiles all required binaries
- public build passes chat smoke without gibberish
- long-context prompt path either:
  - works correctly, or
  - is reverted to the last safe implementation

### Stop condition

Do not continue prompt-path optimization until the long-context failure is explained.

## Track 2: Quality Improvement

### Objective

Push `TQ3_4S` quality from "near `Q3_K_S`" toward a more defensible lead at similar density.

### What the evidence says

- plain global transforms are not enough
- selective mixed precision does help
- the current KLD tail is the main weakness
- `Q4_K_M` still wins on total quality at modestly larger size
- TheTom's `turboquant_plus` suggests the missing ingredients are not just rotation, but better per-block error shaping and protection of sensitive paths

### Best next quality levers

1. Small mixed-precision policies

- keep the base as `TQ3_4S`
- selectively upgrade the most sensitive tensors or late layers
- start with:
  - late FFN up/down
  - attention output
  - token embedding / output head when needed

2. Layer-family policies rather than one-off patches

- dense 9B and dense 27B should each get a compact, reusable tensor-type policy
- MoE needs a separate policy family

3. Tail-error reduction

- use KLD and top-token disagreement to identify error tails
- optimize for the worst offending modules, not average weight MSE

4. Borrow from stronger quantizers selectively

- do not copy a whole foreign stack blindly
- evaluate:
  - learned or adaptive protection for sensitive tensors
  - better scale/codebook shaping
  - MoE-aware protection policy

### Concrete experiments

- dense 9B:
  - base `TQ3_4S`
  - `TQ3_4S + late FFN q5_k`
  - `TQ3_4S + late FFN q5_k + attn_out q5_k`
- dense 27B:
  - repeat the known best mixed policies on full `c2048`
  - compare against `Q3_K_S` and `Q4_K_M`
- MoE 35B:
  - run the existing APEX-lite policy branch and compare against plain `q3_k_s`

### Success criteria

- beat base `TQ3_4S` quality clearly
- approach or beat `Q3_K_S` at similar or smaller size
- narrow the gap to `Q4_K_M`

## Track 3: Memory Size Reduction

### Objective

Reduce model footprint while preserving the new quality floor.

### Principle

Do not shrink uniformly.

Uniform shrinking is what causes the current quality tails. Size reduction should come from structure-aware policy, not a blunt global downgrade.

### Best next size levers

1. Keep `TQ3_4S` as the default compact-quality format

- it is the best current tradeoff line
- `TQ3_1S` should only be used where a real fit constraint exists

2. Build compact mixed policies

- protect a small fraction of tensors
- push everything else to the cheapest acceptable format

3. MoE-specific storage policy

- routers, shared experts, and frequently used paths deserve higher precision than cold experts

4. Remove dead experimental surface area

- do not keep unused enum types and stale experimental formats in public branches
- every extra format has maintenance cost

### Success criteria

- produce one "balanced" policy and one "compact" policy per model family
- document expected size, PPL, and speed for each

## Track 4: Decode Speed

### Objective

Lift token generation speed without harming correctness.

### What already worked

- `VDR=8` style vec-dot/MMVQ decode changes were the clearest decode win on Blackwell when the path was correct
- packed activation staging helped
- dp4a conversion on the decode path helped

### What did not generalize safely

- many prompt-kernel rewrites
- register-scale broadcast
- overfitting microbench kernels without end-to-end proof

### Best next decode levers

1. Recover the last known good fast decode path

- identify the exact source state that produced coherent chat and strong TG
- lock it with tests

2. Make decode specialization dynamic

- select vec-dot contract by architecture and validated settings
- keep a safe fallback path

3. Pre-rotation and fused staging only if correctness is locked

- do not merge fusion work until quality and chat gates pass

### Success criteria

- decode speed above the safe public baseline
- no gibberish in `llama-simple-chat` or `llama-server`
- perplexity unchanged within noise

## Track 5: Prompt Processing Speed

### Objective

Improve `pp2048` and longer prompt throughput without reintroducing silent output corruption.

### Current diagnosis

- prompt-side kernel work can produce very large synthetic wins
- those wins do not matter if:
  - long context breaks
  - chat becomes incoherent
  - only a narrow microbench improves

### Best next prompt levers

1. Separate safe and experimental paths

- safe public prompt path must stay correctness-first
- experimental loader rewrites stay private until fully validated

2. Reuse-first MMQ work

- multi-row reuse is the only convincing arithmetic-intensity direction
- but it must be proven with:
  - correctness
  - prompt probe
  - chat
  - long-context stability

3. Activation-side tax reduction

- fused rotate + quant/staging is still worth pursuing
- it is orthogonal to weight-tile reuse

### Success criteria

- end-to-end prompt win on the real model, not just a microbench
- no long-context regressions
- no context creation or allocator crashes at the tested prompt lengths

## Model-Family Strategy

### Dense 9B

Use as the fast iteration witness:

- cheapest PPL loop
- easiest chat regression loop
- best place to test mixed policies

### Dense 27B

Use as the main product witness:

- primary public benchmark target
- primary prompt/decode optimization target

### MoE 26B / 35B

Use as the policy-design witness:

- test whether APEX-style tensor protection helps more than it costs
- separate router/shared-expert/expert policies

## Recommended Next 10 Actions

1. Rebuild the public runtime completely and record binary coverage in `artifacts/`.
2. Re-run coherent chat smoke on public main for 9B and 27B.
3. Re-run a pre-rewrite commit against current public main for long-context prompt processing.
4. Freeze one known-good public baseline commit for all future comparisons.
5. Re-run full `c2048` PPL on the best 27B mixed `TQ3_4S` policy.
6. Run the 35B MoE APEX-lite quant policy and compare against plain `q3_k_s`.
7. Create one balanced and one compact tensor-type policy per active model family.
8. Recover the last known good Blackwell decode fast path and revalidate with chat plus PPL.
9. Keep prompt-path loader rewrites private until they pass long-context and chat gates.
10. Replace stale dashboard numbers with "safe public" and "experimental private" sections.

## Decision Rules

- If a change improves microbench but fails chat, reject it.
- If a change improves speed but worsens PPL meaningfully, treat it as a new format tier, not a silent replacement.
- If a mixed policy improves quality for less than a small size penalty, keep it.
- If a feature only works on one GPU architecture, gate it explicitly.
- If a branch cannot be reproduced from source and artifacts, it is not a baseline.

## Near-Term Deliverables

1. Safe public runtime

- full build
- coherent chat
- benchmark artifacts

2. Quality-focused release candidates

- 9B balanced
- 27B balanced
- 35B MoE APEX-lite experimental

3. Speed-focused private branch

- decode specialization only after the safe baseline is locked
- prompt-path work only behind stronger correctness gates
