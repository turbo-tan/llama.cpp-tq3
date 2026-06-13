# From TQ3_1S To TQ3_4S

## Why This Page Exists

The TurboQuant work has many experiment logs. This page is the short narrative:
what changed, why it changed, and what should not be forgotten.

## Phase 1: Fit First With `TQ3_1S`

The first goal was simple: prove that a custom GGUF type could store and run
very compact transformed weights.

`TQ3_1S` established:

- custom tensor type plumbing in GGUF and llama.cpp
- quantization and load paths
- CUDA decode support
- initial model releases

It also exposed the main weakness: quality was not stable enough across all
models and tensor families. Some layers tolerated the compact format; others
created noticeable perplexity and chat-quality tails.

## Phase 2: Tensor Policies And Mixed Precision

The next step was not a new kernel. It was measurement.

We started tracking:

- perplexity deltas by model
- tensor-family sensitivity
- late-layer sensitivity
- attention vs FFN behavior
- policy files for selected tensor upgrades

This led to the first stable rule: do not compress every tensor the same way.
Uniform compression is easy to ship but creates quality cliffs.

## Phase 3: `TQ3_4S`

`TQ3_4S` was introduced to keep the TQ3 runtime idea while reducing the quality
tail.

The important change was more local scale flexibility. Instead of treating a
block as if one scale decision was always enough, `TQ3_4S` gives the block more
room to represent local variation.

That improved the practical tradeoff:

- still much smaller than `Q8_0` or source fp16/bf16
- materially better quality than `TQ3_1S`
- more defensible as a default model-release format

## Phase 4: Runtime Correctness Before Speed

Once `TQ3_4S` worked, much of the work moved into CUDA runtime speed.

The biggest lesson was that speed changes can silently break output:

- VDR changes improved or regressed depending on hardcoded assumptions
- prompt-path kernel rewrites could pass microbenchmarks but fail chat
- long-context tests exposed issues that short tests missed
- build/link coverage mattered because examples such as `llama-simple-chat`
  caught missing runtime symbols

The operating rule became:

1. compile all required binaries
2. verify perplexity
3. verify chat coherence
4. verify server/simple paths
5. only then record speed

## Phase 5: Prompt-Processing Breakthroughs

The prompt path became the main runtime battleground.

Useful techniques included:

- staging activation packs
- reusing weight tiles across rows
- reducing hot-loop decode work
- testing real `llama-bench` prompt processing instead of relying only on CTA
  microbenchmarks

The major lesson: weight reuse and dataflow changes matter more than endlessly
tuning the same single-row CTA structure.

## Phase 6: Decode Speculation Work

The decode-speed track moved beyond weight kernels into speculative execution.

The current lesson from that work:

- GPU shadow state makes speculation viable
- CPU shadow state is too expensive
- exactness is fragile and must be trace-tested
- model-self n-gram caches are useful upper bounds but are not production proof

This work is promising, but it remains separate from the TQ3 weight format
itself. It should not be mixed into TQ3 quality claims until exactness and model
coverage are proven.

## What We Should Preserve

The project should keep:

- `TQ3_4S` as the default release format
- artifact-backed claims
- tensor policy files for reproducibility
- strict chat and perplexity gates
- public docs that distinguish proven results from active experiments

## What We Should Avoid

Avoid:

- claiming microbench wins as model wins
- changing public runtime defaults without a rollback path
- keeping experimental tensor types in public branches without a release plan
- requantizing from already-low-bit sources without labeling the quality caveat

## Current Working Rule

For new model releases:

1. Prefer BF16/F16/Q8 sources.
2. If only `Q4_K_M` or MLX-4bit exists, label the result as requantized.
3. Quantize to `TQ3_4S`.
4. Run perplexity and chat checks.
5. Compare against the source quant if possible.
6. Store all commands and outputs in `artifacts/`.
