# TQ3_4F Fastrot Plan

**Status:** private research format proposal  
**Goal:** replace fixed 32-point WHT weight rotation with a magnitude-selected pairwise transform that keeps most of the quality benefit while reducing decode cost  
**Do not ship as:** `TQ3_4S`  

---

## Decision

`TQ3_4F` is the right name for the experiment.

Meaning:

- `TQ3`: same 3-bit/4-scale TurboQuant family
- `4`: four-scale block contract remains the baseline
- `F`: fast transform / fast rotation, not the current fixed WHT contract

This must be a new format because the stored weights are transformed differently. Existing `TQ3_4S` GGUF files cannot be decoded with the new inverse transform.

---

## Why This Has Better ROI Than More Kernel Tuning

The current `TQ3_4S` quality comes from an orthogonal 32-element WHT-style transform:

```text
y = H x
x = H^T y
```

The benefit is incoherence: large local outliers are spread across the block before 3-bit quantization. The cost is decode-side inverse transform work and awkward 32-element block handling.

A magnitude-selected pairwise transform uses sparse orthogonal 2x2 rotations:

```text
[u] = [ c  s] [a]
[v]   [-s  c] [b]
```

where `(a, b)` is selected from high-risk coordinates in the 32-value block. The inverse is the transpose:

```text
[a] = [c -s] [u]
[b]   [s  c] [v]
```

If we restrict coefficients to cheap constants, for example `c = s = 1/sqrt(2)`, the transform becomes:

```text
u = (a + b) / sqrt(2)
v = (a - b) / sqrt(2)
```

This is much cheaper than a full WHT butterfly and only touches selected pairs.

---

## Selection Rule

For each 32-value block, choose `P` disjoint pairs.

Start with `P = 4`.

Simple first rule:

```text
score(i, j) = max(abs(x_i), abs(x_j)) + lambda * abs(x_i - x_j)
```

Pick disjoint pairs with highest score, subject to:

```text
i != j
each index used at most once
```

Better second rule:

```text
score(i, j) = quant_error_before(i, j) - quant_error_after_pair_rotation(i, j)
```

That directly optimizes the local quantization objective and is still cheap offline.

---

## Metadata

For a 32-value block with `P = 4`:

```text
pair_index[4] : 4 pairs, each pair encoded as two 5-bit indices
sign/mode     : optional 1-bit or 2-bit mode per pair
```

Raw metadata cost:

```text
4 pairs * 10 bits = 40 bits = 5 bytes per 32 weights
```

That is too expensive if stored naively.

Therefore the first viable version should use a small static pairbook:

```text
pairbook_id : 4-6 bits per block
```

The pairbook contains common pair patterns learned offline per tensor class.

Candidate classes:

- attention projection tensors
- FFN gate/up/down tensors
- MoE expert tensors
- embedding/output tensors, likely excluded or protected

---

## Format Sketch

Current `TQ3_4S` block:

```text
qs      : packed 3-bit values
scales  : 4 subgroup scales
signs   : TQ sign / transform contract
```

Experimental `TQ3_4F` block:

```text
qs          : packed 3-bit values
scales      : 4 subgroup scales
pairbook_id : static fast-rotation pattern
```

The block must stay close to the current `TQ3_4S` size. If metadata pushes the format materially above 4.0 bpw, it loses the point.

Hard budget:

```text
target <= 4.10 bpw
stretch <= 4.20 bpw only if PPL improves materially
```

---

## Quantization Objective

For each block:

```text
x       = original fp weights
R       = selected sparse orthogonal transform
z       = R x
q       = quantize_tq3_4scale(z)
zhat    = dequantize_tq3_4scale(q)
xhat    = R^T zhat
loss    = ||x - xhat||_2^2
```

If calibration activations are available:

```text
loss = (x - xhat)^T A (x - xhat)
```

where `A` is a diagonal or block-diagonal activation second moment estimate.

The first prototype should use plain L2 to avoid imatrix dependency. KLD/activation-guided selection can come later.

---

## Runtime Decode

Decode path per block:

1. unpack 3-bit values
2. apply existing 4 subgroup scales
3. apply inverse sparse pairwise transform
4. dot with activation

For `P = 4`, inverse transform adds roughly:

```text
4 pairs * (2 add/sub + 2 scale)
```

If `1/sqrt(2)` is folded into scales or approximated with fixed int8 factors, this becomes mostly add/sub plus one shared scale correction.

Expected decode advantage over WHT:

- fewer shuffle stages
- no full 32-lane butterfly
- less register pressure
- simpler packed lane formation

---

## Validation Gates

Do not add a ggml type until the standalone quantization proof passes.

### Gate 1: Block Reconstruction

Use sampled tensors from 4B and 9B models.

Compare:

- `TQ3_4S`
- `TQ3_4F` with static pairbook
- `TQ3_4F` with oracle per-block pairs

Metrics:

- mean squared error
- max block error
- per-tensor error distribution

Pass condition:

```text
TQ3_4F static pairbook MSE <= TQ3_4S MSE * 1.05
```

### Gate 2: PPL

Models:

- Qwen3.5 4B
- Qwen3.5 9B

Tests:

- 10 chunk quick PPL
- 145 chunk full PPL if quick gate passes

Pass condition:

```text
TQ3_4F PPL <= TQ3_4S PPL + 0.05 on quick gate
```

### Gate 3: Decode Speed

Tests:

- `llama-bench -p 0 -n 128`
- chat smoke with deterministic prompt
- compare with exact same KV settings

Pass condition:

```text
TG speed >= TQ3_4S * 1.10
```

If the speed win is under 10%, the new format is not worth the compatibility cost.

---

## Prototype Order

1. Write a standalone tensor/block simulator in private repo.
2. Implement `P = 4` oracle-pair search for quality ceiling.
3. Learn a small pairbook from sampled blocks.
4. Compare reconstruction against `TQ3_4S`.
5. Only then add a private `GGML_TYPE_TQ3_4F`.
6. Quantize 4B and 9B.
7. Run PPL and speed gates.

---

## Stop Conditions

Stop if any of these happen:

- static pairbook cannot get within 5% of `TQ3_4S` block MSE
- PPL loses more than 0.05 on 4B/9B quick gates
- metadata exceeds 4.20 bpw
- decode speed gain is under 10%

---

## Current Recommendation

Proceed as a private branch experiment.

Do not change public `TQ3_4S`.
Do not modify public README.
Do not allocate enum slots until Gate 1 and Gate 2 pass.

---

## Prototype State: 2026-04-20

Private worktree:

```text
/home/awee/code/worktrees/llama-tq3-4f
```

Private branch:

```text
experiment/tq3-4f-fastrot-prototype
```

Experimental enum reservation:

```c
GGML_TYPE_TQ3_4F = 202
LLAMA_FTYPE_MOSTLY_TQ3_4F = 202
GGML_TYPE_COUNT = 203
```

This is intentionally marked as unimplemented:

```text
tq3_4f_experimental_unimplemented
```

It is not an alias to `TQ3_4S`.

Build check:

```text
cmake --build build --target llama-quantize -j$(nproc)
```

Result: build passes.

### 9B Offline Probe

Probe script:

```text
/home/awee/code/tan_llama/scripts/tq3_4f_fastrot_probe.py
```

Source model:

```text
/home/awee/models/bartowski/Qwen_Qwen3.5-9B-GGUF/Qwen_Qwen3.5-9B-Q8_0.gguf
```

Artifact:

```text
/home/awee/code/tan_llama/artifacts/tq3_4f_fastrot_probe_9b_pairbook16_candidate12_512blocks_20260420.json
```

Configuration:

```text
8 tensors
64 sampled blocks per tensor
512 total 32-value blocks
4 selected pair rotations
candidate-oracle top-12 coordinates
pairbook size 16
held-out pairbook evaluation on 256 blocks
```

Results:

| Variant | Mean MSE Ratio vs `TQ3_4S` WHT | Gate |
|---|---:|---|
| `TQ3_4S` WHT | `1.000` | baseline |
| direct no-transform | `1.017` | near baseline, but no speed/quality proof |
| naive magnitude-pairs | `1.208` | fail |
| candidate oracle-pairs | `0.505` | pass |
| pairbook16 held-out | `0.777` | pass |

Interpretation:

- simple magnitude pairing is not viable
- pairwise transform has a real quality ceiling
- a small pairbook or learned selector may beat WHT reconstruction error on sampled 9B Q8 blocks
- exact pair patterns rarely repeat, so literal most-common pattern IDs are probably not the right final selector
- next step should be a learned deterministic pair scoring rule or tensor-class pairbook, not full GGUF implementation yet

### Next Step

Implement one of these selectors in the offline probe:

1. tensor-category pairbook: separate books for `attn`, `ffn`, and `ssm`
2. learned pair score over cheap block features: `abs(x_i)`, `abs(x_j)`, `abs(x_i-x_j)`, same-group flag, index distance
3. limited candidate search at quantization time, with only a compact `pairbook_id` stored in the final block

Only add real quant/dequant code after this larger probe remains below `1.05x` WHT MSE and metadata stays near the `4.10 bpw` target.

---

## Config-I FFN Probe: 2026-04-20

Reason for this test:

Config-I should not convert every tensor. The first realistic target is FFN bulk tensors only, while attention/SSM/output-sensitive tensors remain protected by the existing known-good policy.

Probe command shape:

```text
python3 scripts/tq3_4f_fastrot_probe.py \
  --tensor-filter ffn \
  --pairbook-mode category \
  --max-tensors 12 \
  --blocks-per-tensor 64 \
  --oracle-candidates 12
```

Source model:

```text
/home/awee/models/bartowski/Qwen_Qwen3.5-9B-GGUF/Qwen_Qwen3.5-9B-Q8_0.gguf
```

Sample:

```text
12 FFN tensors from first 4 layers
768 sampled 32-value blocks
384 held-out pairbook evaluation blocks
4 sparse pair rotations per block
```

### Results

| Variant | Pairbook ID bits/block | Estimated bpw | MSE ratio vs `TQ3_4S` WHT | Artifact |
|---|---:|---:|---:|---|
| category pairbook16 | 4 | `4.12500` | `0.725x` | `tq3_4f_config_i_ffn_pairbook16_candidate12_9b_20260420.json` |
| category pairbook8 | 3 | `4.09375` | `0.783x` | `tq3_4f_config_i_ffn_pairbook8_candidate12_9b_20260420.json` |
| category pairbook4 | 2 | `4.06250` | `0.853x` | `tq3_4f_config_i_ffn_pairbook4_candidate12_9b_20260420.json` |
| category pairbook2 | 1 | `4.03125` | `0.940x` | `tq3_4f_config_i_ffn_pairbook2_candidate12_9b_20260420.json` |
| random pairbook4 | 2 | `4.06250` | `0.847x` | `tq3_4f_config_i_ffn_pairbook4_random_9b_20260420.json` |
| fixed random pairbook1 | 0 | `4.00000` | `1.025x` | `tq3_4f_config_i_ffn_pairbook1_random_9b_20260420.json` |

### Interpretation

This changes the hypothesis.

The benefit is not mainly that exact learned pair patterns repeat. The top learned patterns mostly appear once. The benefit appears to come from giving the quantizer a small menu of cheap sparse orthogonal transforms and storing the chosen transform ID.

That means a very small static codebook may be enough:

```text
4 transform patterns
2-bit transform id per 32-value block
estimated block cost: 4.0625 bpw
```

This is better than the original pairbook16 idea:

```text
16 transform patterns
4-bit transform id per 32-value block
estimated block cost: 4.125 bpw
```

The current best next format target is therefore:

```text
TQ3_4F Config-I v0:
  FFN bulk only
  static 4-entry sparse transform codebook
  4 pair rotations per transform
  2-bit transform id per block
  keep attention/SSM/output protected
```

### Important Caveat

Offline block MSE is only the first gate. It does not prove PPL or speed.

The next implementation should add CPU quant/dequant for this restricted `TQ3_4F` block and quantize a 9B Config-I model. CUDA should wait until 9B quick PPL passes.

---

## Cross-Model Sanity: 35B and 27B

### Qwen3.6-35B-A3B Q8 Source

Source:

```text
/home/awee/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf
```

Probe:

```text
FFN-only
random static pairbook4
4 pair rotations
2-bit transform id per 32-value block
estimated 4.0625 bpw
12 FFN tensors
768 sampled blocks
```

Artifact:

```text
tq3_4f_config_i_ffn_pairbook4_random_qwen36_35b_a3b_20260420.json
```

Result:

| Variant | MSE ratio vs `TQ3_4S` WHT |
|---|---:|
| direct no-transform | `1.035x` |
| naive magnitude-pairs | `1.244x` |
| candidate oracle-pairs | `0.512x` |
| random pairbook4 | `0.813x` |

Conclusion:

The 35B MoE Q8 source supports the same truth as the 9B Q8 source. A tiny 4-entry sparse-transform menu improves reconstruction error versus WHT on sampled FFN tensors.

### Qwen3.5-27B Local Source Limitation

Available local 27B files are quantized (`TQ3_4S`, `Q3_K`, `Q4_K`, `IQ4`) or mmproj-only. No local 27B Q8/BF16 weight GGUF was found.

The local sanity run used:

```text
/home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf
```

Artifact:

```text
tq3_4f_config_i_ffn_pairbook4_random_qwen35_27b_from_tq3_4s_20260420.json
```

This is not a valid quality comparison, because the source is already `TQ3_4S`. Re-quantizing it through the WHT path compares the format against its own dequantized output and makes the `TQ3_4S` baseline artificially near-zero.

Observed ratio:

```text
random pairbook4 vs TQ3_4S self-baseline = 19.6x
```

Interpretation:

Do not use this as evidence against `TQ3_4F`. It only proves that a TQ3_4S-dequantized source is unsuitable for this offline MSE gate. A real 27B verdict requires a Q8/BF16/F16 27B source.

Current evidence:

| Source | Valid for MSE gate? | Result |
|---|---|---|
| Qwen3.5-9B Q8 | yes | pairbook4 `0.847x` WHT MSE |
| Qwen3.6-35B-A3B Q8 | yes | pairbook4 `0.813x` WHT MSE |
| Qwen3.5-27B TQ3_4S | no | invalid self-baseline |

Decision:

The idea survives the larger-family check on 35B. Before spending CUDA time, either:

1. get a 27B Q8/BF16 source and repeat the same offline gate, or
2. proceed with the 9B CPU quant/dequant prototype and let PPL decide.

## Runnable 9B GGUF Proof: Negative Speed Result

Date: 2026-04-20.

Branch:

```text
/home/awee/code/worktrees/llama-tq3-4f
experiment/tq3-4f-fastrot-prototype
```

Experimental files:

```text
/home/awee/models/tq3_4f_exp/Qwen3.5-9B-TQ3_4F-EXP-u8id-broad.gguf
/home/awee/models/tq3_4f_exp/Qwen3.5-9B-TQ3_4F-ConfigI-EXP-u8id.gguf
```

Artifacts:

```text
tq3_4f_9b_quantize_exp_u8id_allow_requant_v2_20260420.log
tq3_4f_9b_quantize_config_i_exp_u8id_20260420.log
tq3_4f_9b_config_i_tensor_inventory_20260420.txt
tq3_4f_9b_config_i_completion_smoke_fixed_prefill_20260420.txt
tq3_4f_9b_config_i_cuda_bench_after_prefill_fix_20260420.txt
tq3_4s_9b_reference_cuda_tg32_20260420.txt
```

What was implemented:

- Reserved experimental `GGML_TYPE_TQ3_4F = 202`.
- Added `block_tq3_4f` with a 1-byte transform id plus four scale bytes plus 12 quant bytes.
- Added CPU quant/dequant and a CUDA load/decode proof path.
- Added true Config-I quantizer policy: only `ffn_down`, `ffn_gate`, and `ffn_up` use `TQ3_4F`; other quantized tensors stay `TQ3_4S`.
- Added minimal CUDA MMVQ and cuBLAS dequant support so the model can load and generate.

Config-I tensor inventory:

```text
F32:     177
TQ3_4S:  152
TQ3_4F:   96
Q6_K:      1
Q4_K:      1
size: 5,031,945,312 bytes
```

Smoke result:

```text
Prompt: What is 2+2?
Output: 2+2 equals 4.
```

Speed result:

| Model | Test | Speed |
|---|---:|---:|
| Qwen3.5-9B `TQ3_4S` reference | `tg32` | `62.20 tok/s` |
| Qwen3.5-9B `TQ3_4F` broad | `tg32` | `8.40 tok/s` |
| Qwen3.5-9B `TQ3_4F` Config-I | `tg32` | `11.33 tok/s` |
| Qwen3.5-9B `TQ3_4F` Config-I | `pp16` | `46.61 tok/s` |

Conclusion:

The offline MSE signal does not translate into runtime speed with the current `u8id` design. The per-block dynamic pair transform adds too much decode overhead in FFN matvecs. This path should not receive more offline tuning until there is a packed CUDA kernel that makes the transform effectively free.

Current decision:

- Keep the branch and artifacts as a proof record.
- Do not promote `TQ3_4F-EXP-u8id` to public or production.
- If revisiting fastrot, the next experiment must remove the runtime transform id overhead. Candidate directions are fixed transform only, pre-rotated activation staging per FFN, or a dp4a-ready packed layout that fuses pair transforms into predecode.

## Next Plan: Fixed Transform vs Packed/dp4a

The `u8id` result proves the failure mode: dynamic transform selection improves offline MSE, but paying per-block pair logic in the runtime matvec destroys decode speed. The next work must remove the variable transform from the hot loop.

There are two viable branches. They should be tested in this order.

### Path A: Fixed-Transform `TQ3_4F0`

Goal:

Prove whether a single global sparse pair transform can preserve most of the quality benefit while making runtime almost as simple as `TQ3_4S`.

Format:

```c
typedef struct {
    uint8_t d[4];
    uint8_t qs[12];
} block_tq3_4f0;
```

Properties:

- Same physical block size as `TQ3_4S`: `16` bytes per 32 values.
- No `pid`.
- No transform lookup per block.
- Transform is compile-time constant.
- Should preserve the existing `4.00 bpw` contract.

Quantization:

Use one fixed pairbook entry for all blocks. Start with the best static winner from the 9B/35B offline probes.

Candidate fixed transform:

```text
pid 0:
(29,16), (21,22), (31,6), (7,17)
```

If offline MSE is poor, test the four existing pids and choose the best single pid by aggregate MSE. Do not add per-block ids.

Runtime decode options:

1. `TQ3_4F0` direct rotated-domain dot:
   - Keep weights in transformed domain.
   - Apply the fixed inverse/forward relation to activations once in the activation staging path.
   - Then dot exactly like `TQ3_4S`.

2. `TQ3_4F0` pre-rotated activation staging:
   - For FFN matvecs only, stage `x_rot = F x` before q8 quantization.
   - Weight decode stays identical to `TQ3_4S` except it reads `TQ3_4F0` scales/qs.
   - Inner loop remains bit-unpack plus dp4a, no pair math.

Expected result:

If the fixed transform is viable, decode should return close to `TQ3_4S` speed. The only extra cost should be a tiny activation transform for FFN layers, not per-weight-block transform logic.

Fail gate:

Stop this path if either condition is true:

- Offline FFN MSE is worse than `0.98x` of `TQ3_4S` on both 9B and 35B.
- Runtime `tg32` is below `0.90x` of `TQ3_4S` after activation pre-rotation.

Success gate:

Continue only if:

- `tg32 >= 0.95x` of `TQ3_4S`, and
- PPL is equal or better than `TQ3_4S` on at least 5 chunks, and
- file size does not exceed `TQ3_4S`.

### Path B: Packed/dp4a `TQ3_4FP`

Goal:

Make the variable-transform idea fast by moving all transform work into producer/predecode, so the consumer loop is dp4a-only.

This path exists only if Path A loses too much quality.

Format direction:

Keep the storage close to `u8id`, but do not let the consumer see the transform id.

```c
typedef struct {
    uint8_t pid;
    uint8_t d[4];
    uint8_t qs[12];
} block_tq3_4fp_disk;
```

Runtime shared tile:

```c
struct tq3_4fp_blockslot {
    uint32_t packed4[8];
};
```

Producer phase:

- Load one 32-value block.
- Decode 3-bit indices.
- Apply `pid` pair transform while assembling int8 lanes.
- Emit final dp4a-ready `packed4[8]` into shared memory.
- Do not write intermediate `qs`, scales, or pair state into shared.

Consumer phase:

- Same rule as the existing successful cooperative path:
  - load `packed4`
  - load/stage `q8`
  - `__dp4a`
  - no shifts
  - no masks
  - no branches
  - no float pair math

Important constraint:

The transform must be algebraically pushed onto either:

- the decoded weight lanes during producer predecode, or
- the activation lanes during staging.

It must not appear inside the repeated dot loop.

Expected result:

Packed/dp4a can only win if each transformed weight block is reused enough to amortize predecode. For single-row decode, it may still lose. It should be evaluated first on:

- prompt/prefill MMQ,
- multi-row FFN tiles,
- then TG/MMVQ.

Fail gate:

Stop if the first real 9B runtime test is below `0.90x` of `TQ3_4S` after the consumer loop is confirmed dp4a-only.

Success gate:

Continue if:

- `pp2048` improves over `TQ3_4S`, or
- `tg128` is at least equal to `TQ3_4S` while PPL improves.

### Recommended Execution Order

1. Implement offline fixed-pid sweep for 9B and 35B using the existing probe script.
2. If fixed-pid MSE passes, implement `TQ3_4F0` as a new experimental enum above `220` to avoid format collision.
3. Quantize 9B as `Qwen3.5-9B-TQ3_4F0-ConfigI-EXP.gguf`.
4. Run:

```text
llama-bench -p 0 -n 32 -r 1
llama-bench -p 16 -n 0 -r 1
llama-completion "What is 2+2?"
llama-perplexity --chunks 5 -c 2048
```

5. Only if fixed-pid quality fails, start `TQ3_4FP` packed/dp4a.

### Current Recommendation

Do `TQ3_4F0` first.

Reason:

The current `u8id` proof already shows that runtime transform ids are too expensive. A fixed transform removes the byte overhead, avoids dynamic branches, keeps the `4.00 bpw` block size, and can reuse most of the `TQ3_4S` kernel structure. If fixed transform quality is acceptable, it is a much cleaner production candidate than packed variable-transform `TQ3_4FP`.

## Fixed-Pid Reuse Test

Date: 2026-04-20.

Branch:

```text
experiment/tq3-4f-fixedpid-reuse
commit 6e9bf12da experiment: reuse tq3_4f with fixed transform id
```

Purpose:

Test the user's shortcut: reuse the existing experimental `TQ3_4F` enum and block layout instead of creating a new format immediately.

Change:

- Keep `GGML_TYPE_TQ3_4F`.
- Keep `block_tq3_4f` with the existing `pid` byte.
- Force quantization to `pid = 0`.
- Replace dynamic CUDA MMVQ transform-id lookup with fixed-pid lane logic.
- Keep Config-I policy: only FFN down/gate/up use `TQ3_4F`; other quantized tensors remain `TQ3_4S`.

Model:

```text
/home/awee/models/tq3_4f_exp/Qwen3.5-9B-TQ3_4F-ConfigI-FIXEDPID0-EXP.gguf
```

Artifacts:

```text
tq3_4f_9b_quantize_config_i_fixedpid0_reuse_20260420.log
tq3_4f_9b_fixedpid0_reuse_bench_20260420.txt
tq3_4f_9b_fixedpid0_reuse_completion_smoke_20260420.txt
```

Speed:

| Variant | `tg32` | `pp16` |
|---|---:|---:|
| `TQ3_4S` reference | `62.20 tok/s` | not rerun here |
| `TQ3_4F` variable-pid Config-I | `11.33 tok/s` | `46.61 tok/s` |
| `TQ3_4F` fixed-pid0 Config-I | `26.59 tok/s` | `46.68 tok/s` |

Smoke:

```text
Prompt: What is 2+2?
Output: 2 + 2 = 4
```

Interpretation:

Fixed-pid reuse is materially better than variable-pid (`26.59` vs `11.33 tok/s`), so the idea is not completely dead. However it is still only `0.43x` of `TQ3_4S` decode speed. The remaining bottleneck is not the transform-id branch; it is applying the fixed pair transform to activation lanes inside every FFN dot.

Decision:

Do not create a new disk format yet. Reuse `TQ3_4F` for the next proof, but move the fixed transform out of the MMVQ inner dot.

Next required test:

Implement fixed-pid activation staging for FFN `TQ3_4F`:

- before q8 quantization, apply the fixed `pid=0` pair transform to the activation vector for FFN tensors;
- then use the same dp4a-style dot pattern as `TQ3_4S`;
- the hot loop must not call `tq3_4f_forward_q8_at`.

Fail gate:

If fixed activation staging does not reach at least `0.90x` of `TQ3_4S` TG speed on 9B, stop this line and move to packed/dp4a predecode only.

## Fixed-Pid0 Activation Staging Result: 2026-04-20

Branch:

```text
/home/awee/code/worktrees/llama-tq3-4f
experiment/tq3-4f-fixedpid-reuse
```

Change:

- Keep the same experimental `GGML_TYPE_TQ3_4F = 202` format.
- Keep the existing `block_tq3_4f` layout, including the stored `pid` byte.
- Do not change `TQ3_4S`.
- Add a fixed-pid0 activation transform before q8 staging for `TQ3_4F` only.
- Replace the `TQ3_4F` MMVQ dot with the same dp4a-style subgroup dot shape used by `TQ3_4S`.

Model:

```text
/home/awee/models/tq3_4f_exp/Qwen3.5-9B-TQ3_4F-ConfigI-FIXEDPID0-EXP.gguf
```

Artifacts:

```text
tq3_4f_9b_fixedpid0_staged_bench_20260420.txt
tq3_4f_9b_fixedpid0_staged_completion_smoke_20260420.txt
tq3_4s_9b_reference_after_tq3_4f_staging_20260420.txt
```

Speed:

| Variant | `tg32` | `pp16` |
|---|---:|---:|
| `TQ3_4F` variable-pid Config-I | `11.33 tok/s` | `46.61 tok/s` |
| `TQ3_4F` fixed-pid0, transform inside dot | `26.59 tok/s` | `46.68 tok/s` |
| `TQ3_4F` fixed-pid0, staged activation | `61.06 tok/s` | `46.68 tok/s` |
| `TQ3_4S` reference, same build | `62.48 tok/s` | not rerun |

Smoke:

```text
Prompt: What is 2+2?
Output: 2 + 2 = 4
Generation speed in smoke: 59.59 tok/s
```

Decision:

This passes the fail gate. `TQ3_4F` fixed-pid0 staged activation reaches `0.98x` of the `TQ3_4S` 9B `tg32` reference while preserving coherent output in the quick smoke. The useful variant is not a new disk format yet: it is the same `TQ3_4F` format with fixed-pid0 quantization and runtime activation staging.

## Fixed-Pid0 Quality Gate: 2026-04-20

Artifacts:

```text
tq3_4s_9b_ppl_10ch_quality_reference_20260420.txt
tq3_4f_9b_fixedpid0_staged_ppl_10ch_20260420.txt
```

Size:

| Variant | File size | Loader BPW |
|---|---:|---:|
| `TQ3_4S` 9B | `4.48 GiB` | `4.29 BPW` |
| `TQ3_4F` fixed-pid0 Config-I | `4.68 GiB` | `4.49 BPW` |

PPL gate:

```text
Dataset: wikitext-2-raw/wiki.test.raw
Command: llama-perplexity -ngl 99 --chunks 10 -c 2048
```

| Variant | PPL | Prompt eval speed |
|---|---:|---:|
| `TQ3_4S` 9B reference | `6.9432 +/- 0.17199` | `2167.25 tok/s` |
| `TQ3_4F` fixed-pid0 staged | `7.2456 +/- 0.17956` | `1353.82 tok/s` |

Decision:

This fixed-pid0 `TQ3_4F` variant is not worth promoting. It is slightly slower than `TQ3_4S` for TG, materially slower for PP/PPL, larger on disk because the old `pid` byte is still present, and worse on the 10-chunk quality gate. Keep the staged activation implementation as a useful runtime proof, but do not spend more time on fixed-pid0 as a production candidate unless a new learned transform improves PPL first.
