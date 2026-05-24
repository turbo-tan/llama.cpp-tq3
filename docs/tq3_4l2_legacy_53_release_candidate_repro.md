# TQ3_4L2 Legacy 53 Tok/s Release-Candidate Reproduction

Date: 2026-05-24

## Purpose

This is the release-candidate reproduction note for the Qwen3.6 27B MTP
`TQ3_4L2` legacy quantized line.

The critical lesson from the recovery work is that the 53 tok/s witness is not
defined only by the `llama-server` arguments. The benchmark request contract is
also part of the release candidate.

The request-level setting that must be preserved is:

```text
QWEN35_HARD_MAX_TOKENS=5000
```

Without that setting, the same server binary, same GGUF, and same draft-MTP
server arguments reproducibly landed around `51.3 tok/s`.

## Release-Candidate Identity

Runtime binary:

- path: `/home/awee/code/tan_llama/.work/llama-rotatek-private-main/build/bin/llama-server`
- version: `9278 (74977c26a)`
- commit: `74977c26a2db0866f362a3198c89fdc9d5965345`
- sha256: `6dca0d053f04a97dd21c59070362c4faaca8c2621166a76b638ac3edd319aa9b`

Model:

- path: `/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4L2-legacy-qTQ3_4S.gguf`
- sha256: `38304c17bb2afc9fbb6970006921322b6a7d1987eda2c6edb6703c7076aa7e1e`

Quantizer used to generate the model:

- path: `/home/awee/code/tan_llama/.work/llama.cpp-tq3-9240/build/bin/llama-quantize`
- version: `9240 (12b9130d2)`
- sha256: `c3d108ca19c0b130f4eb7ce8ccc9abde11e67f2930be3090d0824f1b943c54f8`

Legacy source artifact:

- path: `/home/awee/models/turboquant/tq3_4l2/unsloth_27b_mtp/Qwen3.6-27B-MTP-TQ3_4L2-legacy.gguf`
- sha256: `f993d6479437ba687b7cd632dafdd88984cf6389bbe7d3e1e2546284c18393f4`

## Required Server Arguments

Model-specific server arguments:

```text
-ngl 99
-ctk q4_0
-ctv tq3_0
--spec-type draft-mtp
--spec-draft-n-min 1
--spec-draft-n-max 2
--spec-draft-p-min 1.0
--spec-draft-ngl 99
--spec-draft-type-k q4_0
--spec-draft-type-v tq3_0
```

Shared server arguments from the hard-coder benchmark harness:

```text
--host 0.0.0.0
--port ${BENCHLOOP_PORT:-18123}
-c 32768
-fa on
--reasoning off
--reasoning-budget 0
--reasoning-format none
--metrics
--jinja
-np 1
```

## Required Request-Level Setting

The benchmark request must use:

```text
QWEN35_HARD_MAX_TOKENS=5000
```

This value is consumed by
`/home/awee/code/tan_llama/scripts/run_qwen35_hard_coder_bench.py` when it
builds the `/v1/chat/completions` payload. It is not visible in the server
preset JSON.

## No-Copy Witness Command

From this repository:

```bash
scripts/tq3_4l2_legacy_53_rc_witness.sh
```

The runner:

- verifies the expected server binary hash
- verifies the expected GGUF hash
- writes only a temporary preset JSON
- points the benchmark directly at the canonical GGUF
- sets `QWEN35_HARD_MAX_TOKENS=5000`
- runs only `async_rate_limiter`
- does not copy or quantize the GGUF

## Confirmed Reproduction

Final local pre-publish check:

- artifact: `/home/awee/code/tan_llama/artifacts/tq3_4l2_legacy_53_rc_witness_final_20260524/summary.md`
- task: `async_rate_limiter`
- pass: `2/2`
- speed: `53.20943732868691 tok/s`
- server sha256: `6dca0d053f04a97dd21c59070362c4faaca8c2621166a76b638ac3edd319aa9b`
- model path: canonical legacy q-file, used in place
- GGUF copy in output directory: no

Recovered no-copy check:

- artifact: `/home/awee/code/tan_llama/artifacts/direct6dca0d_nocopy_maxtok5000_20260524/summary.md`
- task: `async_rate_limiter`
- pass: `2/2`
- speed: `53.18483142898966 tok/s`

Historical primary witness:

- artifact: `/home/awee/code/tan_llama/.work/tq3-speedtrack-phase1/artifacts/tq3_4l2_legacy_power350/summary.md`
- task: `async_rate_limiter`
- pass: `2/2`
- speed: `53.49847715210959 tok/s`

## Known Failure Modes

`51.3 tok/s` with the same binary and GGUF:

- cause: request cap drifted to the current hard-coder harness default
- fix: set `QWEN35_HARD_MAX_TOKENS=5000`
- artifact: `/home/awee/code/tan_llama/artifacts/direct6dca0d_nocopy_20260524/summary.md`

Large duplicate artifact directories:

- cause: using `run_tq3_53_matrix.py` for a direct binary check
- fix: use `scripts/tq3_4l2_legacy_53_rc_witness.sh`
- note: `run_tq3_53_matrix.py` is for full source/quantize reconstruction and
  snapshots GGUFs by design

Power-limit failure:

- symptom: `nvidia-smi -pl 350` fails with insufficient permissions
- the no-copy runner does not set the power limit
- if a 350W limit is required, apply it separately before running the witness

## Release-Candidate Gate

Before pushing or publishing:

```bash
scripts/tq3_4l2_legacy_53_rc_witness.sh
```

Accept the RC only if:

- server sha256 is `6dca0d053f04a97dd21c59070362c4faaca8c2621166a76b638ac3edd319aa9b`
- GGUF sha256 is `38304c17bb2afc9fbb6970006921322b6a7d1987eda2c6edb6703c7076aa7e1e`
- `async_rate_limiter` passes `2/2`
- speed is in the 53-class range
- no GGUF snapshot copy is created in the output directory

