#include "ggml-cuda.h"
#include "ggml-cuda/tq3_4s2.cuh"
#include "ggml-cuda/tq3_4s2_layout.h"
#include "ggml-cuda/vecdotq.cuh"

static std::string g_tq3_4l2_layout;

bool ggml_cuda_tq3_4l2_layout_enabled() {
    return g_tq3_4l2_layout == "tq3_4l2";
}

void ggml_cuda_tq3_4l2_set_layout(const char * layout) {
    g_tq3_4l2_layout = layout != nullptr ? layout : "";
}

// Shape-specific TQ3_4L2 kernel: one warp computes one q8_1 dot against one TQ3_4S block.
// This is the first fused primitive for the dedicated runtime path.
static __global__ void ggml_cuda_tq3_4l2_dot_kernel(
        const block_tq3_4s * __restrict__ in,
        const block_q8_1    * __restrict__ act,
        float               * __restrict__ out,
        int nblocks) {

    const int blk = blockIdx.x;
    if (blk >= nblocks) {
        return;
    }

    out[blk] = vec_dot_tq3_4s_q8_1(in + blk, act + blk, 0, 0);
}

static bool ggml_cuda_tq3_4l2_enabled_runtime() {
    static bool enabled = getenv("GGML_CUDA_TQ3_4L2") != nullptr;
    return enabled;
}

void ggml_cuda_tq3_4l2_probe(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!ggml_cuda_tq3_4l2_is_decode_shape(src0, src1, dst)) {
        return;
    }

    GGML_LOG_INFO("TQ3_4L2 candidate: %s x %s -> %s (%lld x %lld)\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(dst->type),
        (long long) src0->ne[0], (long long) src0->ne[1]);
}

void ggml_cuda_tq3_4l2_launch_dot(
        const block_tq3_4s * in,
        const block_q8_1    * act,
        float               * out,
        int nblocks,
        cudaStream_t         stream) {

    if (!ggml_cuda_tq3_4l2_enabled_runtime()) {
        return;
    }

    ggml_cuda_tq3_4l2_dot_kernel<<<nblocks, WARP_SIZE, 0, stream>>>(in, act, out, nblocks);
}
