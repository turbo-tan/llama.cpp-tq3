# TQ3 Native Prefill Result

Date: 2026-04-02

Branch:
- `llama.cpp-tq3: research/tq3-native-prefill`
- commit `2056ed554`

Scope:
- restore the old native `TQ3_0` prefill path
- keep it gated behind `GGML_CUDA_TQ3_NATIVE_PREFILL=1`
- do not change default policy

What changed:
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- for `GGML_TYPE_TQ3_0`, when `GGML_CUDA_TQ3_NATIVE_PREFILL=1`:
  - disable the normal MMQ path
  - use native prefill for prompt batches `>= 8`
  - keep the fallback fp32 `sgemm` path for tiny prompt batches

Correctness:
- standalone unit test passes:
  - `tests/test-tq3-prefill.cu`
  - max relative error: `0.0132%`

Measured result:
- model: `/home/awee/models/tinyllama-1.1b-tq3_0.gguf`
- hardware: RTX 5060 Ti 16GB

Short prompt-only probes:
- baseline `pp32`: `438.79 ± 205.84 tok/s`
- gated native prefill `pp32`: `463.73 ± 189.90 tok/s`
- gated native prefill `pp64`: `345.87 tok/s`
- gated native prefill `pp512`: `800.56 tok/s`

Important blocker:
- ungated baseline still crashes on larger prompt batches before the experiment path matters
- repro:
  - baseline `pp64`
  - aborts in `mmq.cuh:93`
  - stack goes through `quantize_mmq_q8_1_cuda`

Conclusion:
- the gated native prefill path is alive and produces real prompt-processing numbers
- the current default MMQ bridge for `TQ3_0` is still unstable on larger prompt batches
- next useful work is not broader benchmarking
- next useful work is fixing or bypassing the `TQ3_0` MMQ bridge so A/B at larger prompt sizes is clean
