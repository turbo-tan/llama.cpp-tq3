#pragma once

#include "common.cuh"
#include "ggml-cuda/tq3_4s2_layout.h"

static inline bool ggml_cuda_tq3_4s2_enabled() {
    static bool env_enabled = getenv("GGML_CUDA_TQ3_4S2") != nullptr;
    return env_enabled || ggml_cuda_tq3_4s2_layout_enabled();
}

static inline bool ggml_cuda_tq3_4s2_is_decode_shape(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!ggml_cuda_tq3_4s2_enabled()) {
        return false;
    }

    if (src0->type != GGML_TYPE_TQ3_4S || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous(src0)) {
        return false;
    }

    if (dst->ne[1] != 1) {
        return false;
    }

    if (src0->ne[0] != 5120) {
        return false;
    }

    switch (src0->ne[1]) {
        case 10240:
        case 12288:
        case 17408:
            return true;
        default:
            return false;
    }
}

static inline void ggml_cuda_tq3_4s2_log_candidate(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!ggml_cuda_tq3_4s2_is_decode_shape(src0, src1, dst)) {
        return;
    }

    GGML_LOG_INFO("TQ3_4S2 candidate: %s x %s -> %s (%lld x %lld)\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(dst->type),
        (long long) src0->ne[0], (long long) src0->ne[1]);
}

void ggml_cuda_tq3_4s2_probe(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_tq3_4s2_launch_dot(
        const block_tq3_4s * in,
        const block_q8_1    * act,
        float               * out,
        int nblocks,
        cudaStream_t         stream);
