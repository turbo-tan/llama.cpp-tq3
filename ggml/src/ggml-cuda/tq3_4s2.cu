#include "ggml-cuda.h"
#include "ggml-cuda/tq3_4s2.cuh"

// Placeholder for the dedicated TQ3_4S2 execution path.
// The first concrete kernel will live here so the generic MMQ/MMVQ code stays untouched.
void ggml_cuda_tq3_4s2_probe(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    ggml_cuda_tq3_4s2_log_candidate(src0, src1, dst);
}
