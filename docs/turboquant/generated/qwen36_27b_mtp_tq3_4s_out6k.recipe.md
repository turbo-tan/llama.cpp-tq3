# Recipe: Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6 (out6k)

The current flagship — the model in active production at `:8085`.

## Artifact

```
/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6.gguf
14G  (13.39 GiB)   built 2026-06-06 00:09
```

A second copy with the recovered publish template baked in was produced 2026-06-13:

```
…/Qwen3.6-27B-MTP-TQ3_4S-mtp-q4k-outq6-publish-template.gguf
14G  built 2026-06-13
```

Policy file: `docs/turboquant/generated/qwen36_27b_mtp_tq3_4s_out6k.tensor-types.txt`

> **Provenance note:** The exact original build command was never committed.
> The tensor mix below is reconstructed from the artifact name and the speed plan
> doc (`6a9e0541b`). The policy file only contains the one override confirmed from
> both sources (`output Q6_K`). If you need to rebuild, verify with
> `llama-gguf-inspect` against the original artifact.

## Source

```
/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4L2.gguf
```

(BF16 master, same source as v2b)

## Tensor mix

The artifact name encodes the mix: `TQ3_4S-mtp-q4k-outq6`.

| Tensor group | Type | Notes |
|---|---|---|
| Backbone (bulk) — all non-special layers | TQ3_4S | Main quality+size budget |
| MTP / nextn layers (`blk.N.*` where N ≥ n_layer) | Q4_K | Draft-head tensors; upquanted vs bulk for better MTP accept rate |
| Output projection (`output.weight`) | Q6_K | "outq6" — upquanted for final-token quality; ~0.6 GiB, ~5% of total bytes |
| Token embeddings (`token_embd.weight`) | TQ3_4S | Row gather, not a matmul; type doesn't significantly affect TG |
| SSM params (`ssm_alpha`, `ssm_beta`) | TQ3_4S / Q3_K | Not explicitly overridden — fell through to bulk type. Suspected quality weakness vs v2b (see recipe notes) |

## Size vs UD

| Model | Size | bpw |
|---|---|---|
| UD-Q3_K_XL (reference) | 13.77 GiB | 4.34 |
| **out6k (this)** | **13.39 GiB** | **~4.22** |
| out6k-v2b | 13.63 GiB | 4.28 |

out6k is the smallest of the three; v2b is a quality-focused refinement that
trades ~240 MB for (hypothetically) better recurrent-layer fidelity.

## Gated results (Jun 13 2026)

Full local BenchLoop on the recovered publish template,
`build-current`, `-c 32768`, `-ctk q8_0 -ctv tq3_0`, `--spec-type draft-mtp`:

| Suite | Score | Pass |
|---|---|---|
| speed | 68.6 | 9/9 |
| toolcall | 96.7 | 14/15 |
| coding | **100.0** | 12/12 |
| dataextract | 91.0 | 12/15 |
| instructfollow | 74.5 | 9/15 |
| reasonmath | 73.3 | 11/15 |
| **GEN TOK/S** | **42.73** | — |

Run: `/home/awee/.bench-loop/runs/20260613-131757-…-local-openai_compat/run.json`

Remaining misses are ordinary benchmark-quality misses; the earlier
template-collapse regression no longer appears on the recovered publish template.

## Template status

The embedded GGUF template diverges from the production winner.
**Always use the publish template override for local runs:**

```bash
--chat-template-file /home/awee/code/tan_llama/publish/qwen36-27b-mtp-tq3_4s/chat_template.jinja
```

Template lineage (winner):
- stock-derived publish template from commit `25b98f6be`
- + scoped JSON-only numeric rule
- + direct-answer tool-avoidance rules from `artifacts/tc11-prompt-patched.txt`

See `docs/turboquant/benchloop-sop.md` for the full server launch command.

## Known issues / open questions

- `ssm_alpha` / `ssm_beta` are at bulk (3-bit) precision. These are recurrent
  decay parameters; low precision on them is the suspected quality weakness that
  motivated **v2b** (which moves them to F32). Until v2b's quality gates are run,
  this remains a hypothesis.
- Long-context (`-c 262144`) quality is **not validated** — the 262144 test on
  2026-06-13 produced a CUDA OOM false negative before quality could be measured.
  All validated results are at `-c 32768`.
- Output Q6_K vs Q5_K/Q4_K A/B not run (would reclaim ~60-120 MB; in speed plan
  as Track A4).

## What beat it on size

- UD-Q3_K_XL: larger, 13.77 GiB. out6k wins on size.

## What this is the baseline for

- out6k-v2b: quality-focused refinement (ssm F32, ssm_out Q5_K, attn_v Q6_K,
  attn_output Q4_K claw-back). Recipe at `qwen36_27b_mtp_tq3_4s_out6k_v2b.recipe.md`.
- Speed plan Track A4 (output quant A/B, MTP layer quant A/B).
