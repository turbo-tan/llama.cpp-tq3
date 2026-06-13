# TQ3_4S Rot-Dot Result

Date: 2026-04-02

Repo:
- `/home/awee/code/llama.cpp`
- branch: `research/tq3-1s-4s-kernel`

Scope:
- use the existing local `TQ3_4S` rotated-dot CUDA kernel
- wire it into `ggml-cuda.cu`
- keep it gated behind `GGML_CUDA_TQ3_4S_ROT_DOT=1`
- only for:
  - `GGML_TYPE_TQ3_4S`
  - `src1->type == F32`
  - contiguous weights
  - tiny prompt width `n <= 4`

What was wired:
- rotate activation in float
- convert rotated activation to fp16
- call `ggml_cuda_tq3_4s_rot_dot(...)`

Measured probe:
- model: `/home/awee/models/turboquant27/Qwen_Qwen3.5-27B-TQ3_4S.gguf`
- hardware: RTX 5060 Ti 16GB
- command shape:
  - `llama-bench`
  - `pp4`
  - `-ngl 99`
  - `-fa 0`
  - `-n 0`

Result:
- baseline `pp4`: `0.41 tok/s`
- gated `GGML_CUDA_TQ3_4S_ROT_DOT=1` `pp4`: `9.53 tok/s`

Interpretation:
- this is the first real positive moonshot signal on the actual public weight format line
- it is a narrow win:
  - `TQ3_4S`
  - tiny `n`
  - prompt path only
- but it proves the local rotated-dot kernel has legs

Next step:
- expand from `pp4` to slightly wider tiny-`n` prompt cases
- add a correctness sanity prompt outside `llama-bench`
- then decide whether to generalize or keep it as a tiny-`n` fast path
