# TQ3_0_prod QJL Proposal

## Scope

This note is about the **KV-cache path** only:

- `TQ3_0` / TurboQuant-style cache compression
- attention quality
- inner-product accuracy in `Q·K`

It is **not** a proposal for the current public `TQ3_4S` weight format.

## Why This Matters

Our current weight-side story is already clear:

- `TQ3_4S` works as a public runtime format
- mixed tensor policy improves quality
- the remaining gap is mostly a distribution-fidelity problem, not a missing kernel trick

But the KV-cache path is different.

For KV cache, the thing we care about most is:

- preserving the attention score
- i.e. accurate inner products

That is exactly where the paper's QJL residual idea is most relevant.

## What The Paper Adds That We Do Not

### 1. QJL residual correction

Pipeline:

1. do MSE-oriented quantization
2. compute the residual
3. apply 1-bit QJL on that residual
4. use the corrected estimator at dot-product time

Effect:

- the inner-product estimator becomes unbiased
- plain MSE quantization alone is not enough for that

This is likely the most important missing paper idea for our KV line.

### 2. Outlier channel separation

The paper analytically splits channels into:

- outlier channels
- regular channels

and assigns different effective bit budgets.

This is conceptually similar to what our mixed-policy work discovered empirically:

- some parts matter much more than others

For KV, this likely means:

- selectively protecting the most attention-sensitive channels
- instead of treating every key/value channel equally

### 3. Entropy coding of indices

The quantized index distribution is non-uniform, so entropy coding can reduce file size below nominal bpw.

This matters more for:

- storage / transfer size

than for:

- online decode simplicity

So it is useful, but not the first thing to implement for production CUDA KV.

### 4. Distortion framing

The paper gives a stronger theoretical story around the MSE floor.

That is useful as a validation target, but not the first engineering lever.

## Most Actionable Insight For Us

The key paper idea for our codebase is:

- **add 1-bit QJL residual correction to the KV path**

Why this is a better fit for KV than weights:

- KV quality is dominated by dot-product behavior
- QJL is designed exactly for dot-product correction
- our earlier weight-side work showed that QJL is not the obvious missing piece there

So the right framing is:

- **weights:** keep focusing on tensor policy / mixed precision / format modeling
- **KV:** QJL residual is a plausible next production step

## Proposed `TQ3_0_prod` Design

### Base idea

Keep the current `TQ3_0` KV cache as the base representation, then add:

- compact residual sketch
- 1-bit QJL metadata on that residual

At attention time:

1. use the normal compressed `K` / `V`
2. apply the QJL correction when estimating inner products

Target:

- better attention fidelity
- less quality loss at the same nominal cache budget

### Intended product framing

`TQ3_0_prod` should mean:

- same deployment goal as the current TurboQuant KV path
- but with the paper's dot-product correction restored where it actually matters

Not:

- a new weight format
- a replacement for `TQ3_4S`

## Suggested Implementation Order

### Phase 1. Design + accounting

Write down:

- exact residual definition
- exact extra bits per token
- whether correction is for `K`, `V`, or both
- expected compute overhead in the attention kernel

Likely answer:

- prioritize `K` first, because `Q·K` sensitivity dominates

### Phase 2. CPU reference

Implement a tiny correctness prototype:

- no CUDA optimization
- just validate math and attention error

Metrics:

- attention-score error
- downstream perplexity / KLD on long context

### Phase 3. CUDA path

Only after the CPU reference is clearly useful:

- add the residual/QJL path into the CUDA KV implementation
- benchmark memory overhead vs quality gain

### Phase 4. Product decision

If the gain is real, expose it as:

- `TQ3_0_prod`

Otherwise:

- keep the current public recommendation
  - `K = q8_0`
  - `V = turbo3_0`

## Risks

### 1. Overhead may erase the win

If the residual metadata is too large or too slow, the idea stops being a production improvement.

### 2. K/V asymmetry may dominate

Our current evidence already says:

- `V` is the cheap side
- `K` is the sensitive side

So the design should not assume symmetric benefit.

### 3. The win may be theoretical but not practical

It is possible to improve inner-product faithfulness while gaining little on real perplexity or chat quality.

That has to be tested, not assumed.

## Recommendation

Yes, this is worth documenting and exploring, but with a strict boundary:

- **KV proposal:** yes
- **weight-format proposal:** no

The right next artifact is exactly this:

- a `TQ3_0_prod` design proposal focused on QJL residual correction for KV cache

And the right standard for advancing it is:

- first prove the math in a reference implementation
- then prove the quality/performance tradeoff
- only then make it part of the public runtime story
