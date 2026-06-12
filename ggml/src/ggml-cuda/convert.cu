#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
            y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d * (q[l] & 0xF) + dm);
        y[l+16] = ggml_cuda_cast<dst_t>(d * (q[l] >>  4) + dm);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d.x * (q[l] & 0xF) + d.y);
        y[l+16] = ggml_cuda_cast<dst_t>(d.x * (q[l] >>  4) + d.y);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0_tq(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int64_t i = blockIdx.x;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0_tq_v0 * x = (const block_q4_0_tq_v0 *) vx + ib;
    const float scale0 = exp2f(((int) x->s0 - 127) / 16.0f);
    const float scale1 = scale0 * exp2f((float) x->ds1 / 16.0f);

    static constexpr int8_t levels[8] = { -7, -5, -3, -1, 1, 3, 5, 7 };

    for (int l = 0; l < 4; ++l) {
        const int idx0 = 4*il + l;
        const int idx1 = idx0 + 16;

        const int bit0 = idx0 * 3;
        const int byte0 = bit0 / 8;
        const int shift0 = bit0 % 8;
        const uint32_t cur0 = (uint32_t) x->qs[byte0]
            | ((uint32_t) x->qs[byte0 + 1] << 8)
            | ((uint32_t) (byte0 + 2 < 12 ? x->qs[byte0 + 2] : 0) << 16);

        const int bit1 = idx1 * 3;
        const int byte1 = bit1 / 8;
        const int shift1 = bit1 % 8;
        const uint32_t cur1 = (uint32_t) x->qs[byte1]
            | ((uint32_t) x->qs[byte1 + 1] << 8)
            | ((uint32_t) (byte1 + 2 < 12 ? x->qs[byte1 + 2] : 0) << 16);

        y[l + 0]  = ggml_cuda_cast<dst_t>(scale0 * (float) levels[(cur0 >> shift0) & 0x7]);
        y[l + 16] = ggml_cuda_cast<dst_t>(scale1 * (float) levels[(cur1 >> shift1) & 0x7]);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1_tq(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int64_t ib = blockIdx.x;
    const int64_t tid = threadIdx.x;
    if (ib >= nb32 || tid >= 32) {
        return;
    }

    const block_q4_0_tq_v1 * x = (const block_q4_0_tq_v1 *) vx + ib;
    static constexpr int8_t levels[8] = { -7, -5, -3, -1, 1, 3, 5, 7 };
    dst_t * y = yy + 32*ib;

    const int qg = tid / 8;
    const float scale = exp2f(((int) x->scales[qg] - 127) / 16.0f);
    const int bit = tid * 3;
    const int byte = bit / 8;
    const int shift = bit % 8;

    uint32_t word = x->qs[byte];
    if (byte + 1 < 12) {
        word |= (uint32_t) x->qs[byte + 1] << 8;
    }
    if (byte + 2 < 12) {
        word |= (uint32_t) x->qs[byte + 2] << 16;
    }

    y[tid] = (dst_t) (scale * (float) levels[(word >> shift) & 0x7]);
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q2_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q3_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q4_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q5_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q6_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_m(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_nl(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_mxfp4(vx, i, yy + i*QK_K, threadIdx.x);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_0_tq_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0_tq<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_tq_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    dequantize_block_q4_1_tq<<<nb32, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

// TQ3_0 CUDA dequantize: unpack indices, centroid lookup, inverse WHT, scale
__constant__ static const float tq3_0_centroids_cuda[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};
__constant__ static const float tq3_0_signs_cuda[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

static __device__ __forceinline__ uint8_t tq3_idx_from_packed_cuda(const uint8_t * qp, int r);

template<typename dst_t>
static __global__ void dequantize_block_tq3_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb) {
    const int i = blockIdx.x;
    if (i >= nb) return;

    const block_tq3_0 * x = (const block_tq3_0 *)vx + i;
    const float d = __half2float(x->d);
    dst_t * y = yy + i * QK_TQ3_0;
    const int j = threadIdx.x; // 0..31

    // No-WHT: direct centroid * d (V-only KV cache, stored without rotation)
    const int g = j / 8;
    const int r = j % 8;
    const uint8_t * qp = x->qs + g * 3;
    uint8_t idx;
    switch (r) {
        case 0: idx =  qp[0]       & 7; break;
        case 1: idx = (qp[0] >> 3) & 7; break;
        case 2: idx = ((qp[0] >> 6) | (qp[1] << 2)) & 7; break;
        case 3: idx = (qp[1] >> 1) & 7; break;
        case 4: idx = (qp[1] >> 4) & 7; break;
        case 5: idx = ((qp[1] >> 7) | (qp[2] << 1)) & 7; break;
        case 6: idx = (qp[2] >> 2) & 7; break;
        default: idx = (qp[2] >> 5) & 7; break;
    }
    y[j] = (dst_t)(tq3_0_centroids_cuda[idx] * d);
}

template<typename dst_t>
static void dequantize_row_tq3_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_TQ3_0;
    dequantize_block_tq3_0<<<nb, 32, 0, stream>>>(vx, y, nb);
}

template<typename dst_t>
static __global__ void dequantize_block_tq3_1s(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb) {
    const int i = blockIdx.x;
    if (i >= nb) return;

    const block_tq3_1s * x = (const block_tq3_1s *)vx + i;
    const float d0 = __half2float(x->d0);
    const float d1 = __half2float(x->d1);
    dst_t * y = yy + i * QK_TQ3_0;
    const int j = threadIdx.x;

    const int g = j / 8;
    const int r = j % 8;
    const uint8_t * qp = x->qs + g * 3;
    uint8_t idx;
    switch (r) {
        case 0: idx =  qp[0]       & 7; break;
        case 1: idx = (qp[0] >> 3) & 7; break;
        case 2: idx = ((qp[0] >> 6) | (qp[1] << 2)) & 7; break;
        case 3: idx = (qp[1] >> 1) & 7; break;
        case 4: idx = (qp[1] >> 4) & 7; break;
        case 5: idx = ((qp[1] >> 7) | (qp[2] << 1)) & 7; break;
        case 6: idx = (qp[2] >> 2) & 7; break;
        default: idx = (qp[2] >> 5) & 7; break;
    }

    float val = tq3_0_centroids_cuda[idx] * (j < 16 ? d0 : d1);
    for (int step = 1; step < 32; step <<= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        if (j & step) {
            val = other - val;
        } else {
            val = other + val;
        }
    }

    y[j] = (dst_t)(val * (tq3_0_signs_cuda[j] / sqrtf(32.0f)));
}

template<typename dst_t>
static void dequantize_row_tq3_1s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_TQ3_0;
    dequantize_block_tq3_1s<<<nb, 32, 0, stream>>>(vx, y, nb);
}

__device__ static inline float tq3_4s_decode_scale_cuda(uint8_t byte) {
    if (byte == 0) return 0.0f;
    const int exp = (byte >> 5) - 9;
    const float mantissa = 1.0f + (float)(byte & 31) / 32.0f;
    return ldexpf(mantissa, exp);
}

template<typename dst_t>
static __global__ void dequantize_block_tq3_4s(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb) {
    const int i = blockIdx.x;
    if (i >= nb) return;

    const block_tq3_4s * x = (const block_tq3_4s *)vx + i;
    const float ds[4] = {
        tq3_4s_decode_scale_cuda(x->d[0]),
        tq3_4s_decode_scale_cuda(x->d[1]),
        tq3_4s_decode_scale_cuda(x->d[2]),
        tq3_4s_decode_scale_cuda(x->d[3]),
    };

    dst_t * y = yy + i * QK_TQ3_0;
    const int j = threadIdx.x;
    const int g = j / 8;
    const int r = j % 8;
    const uint8_t * qp = x->qs + g * 3;
    const uint8_t idx = tq3_idx_from_packed_cuda(qp, r);

    float val = tq3_0_centroids_cuda[idx] * ds[g];
    for (int step = 1; step < 32; step <<= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        if (j & step) {
            val = other - val;
        } else {
            val = other + val;
        }
    }

    y[j] = (dst_t)(val * (tq3_0_signs_cuda[j] / sqrtf(32.0f)));
}

template<typename dst_t>
static void dequantize_row_tq3_4s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_TQ3_0;
    dequantize_block_tq3_4s<<<nb, 32, 0, stream>>>(vx, y, nb);
}

template<typename dst_t>
static __global__ void dequantize_block_tq3_4se(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb) {
    const int i = blockIdx.x;
    if (i >= nb) return;

    const block_tq3_4se * x = (const block_tq3_4se *)vx + i;
    const float ds[4] = {
        tq3_4s_decode_scale_cuda(x->d[0]),
        tq3_4s_decode_scale_cuda(x->d[1]),
        tq3_4s_decode_scale_cuda(x->d[2]),
        tq3_4s_decode_scale_cuda(x->d[3]),
    };
    // Decode shifts
    float max_s = fmaxf(fmaxf(ds[0], ds[1]), fmaxf(ds[2], ds[3]));
    float quantum = max_s / 8.0f;
    float shifts[2] = {
        ((int)x->s[0] - 128) / 127.0f * quantum,
        ((int)x->s[1] - 128) / 127.0f * quantum,
    };

    dst_t * y = yy + i * QK_TQ3_0;
    const int j = threadIdx.x;
    const int g = j / 8;
    const int h = j / 16;
    const int r = j % 8;
    const uint8_t * qp = x->qs + g * 3;
    const uint8_t idx = tq3_idx_from_packed_cuda(qp, r);

    float val = tq3_0_centroids_cuda[idx] * ds[g] + shifts[h];
    for (int step = 1; step < 32; step <<= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        if (j & step) {
            val = other - val;
        } else {
            val = other + val;
        }
    }

    y[j] = (dst_t)(val * (tq3_0_signs_cuda[j] / sqrtf(32.0f)));
}

template<typename dst_t>
static void dequantize_row_tq3_4se_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_TQ3_0;
    dequantize_block_tq3_4se<<<nb, 32, 0, stream>>>(vx, y, nb);
}

// 4-bit demoted scale lookup table (must match CPU TQ3_V_SCALE_TABLE)
__constant__ static const float tq3_v_scale_table_cuda[16] = {
    0.002f, 0.00289f, 0.004176f, 0.006034f, 0.008719f, 0.012599f, 0.018206f, 0.026307f,
    0.038013f, 0.054928f, 0.07937f, 0.114688f, 0.165723f, 0.239466f, 0.346025f, 0.5f,
};
// 2-bit centroids
__constant__ static const float tq3_v_c2_cuda[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };
// 4-bit centroids
__constant__ static const float tq3_v_c4_cuda[16] = {
    -2.733f, -2.069f, -1.618f, -1.256f, -0.942f, -0.656f, -0.386f, -0.126f,
     0.126f,  0.386f,  0.656f,  0.942f,  1.256f,  1.618f,  2.069f,  2.733f,
};

template<typename dst_t>
static __global__ void dequantize_block_tq3_4sv(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb) {
    const int i = blockIdx.x;
    if (i >= nb) return;

    const block_tq3_4sv * x = (const block_tq3_4sv *)vx + i;

    // Decode pattern
    const int promoted = (x->d[0] >> 6) & 3;
    const int demoted  = (x->d[0] >> 4) & 3;
    const float demoted_scale = tq3_v_scale_table_cuda[x->d[0] & 0xF];

    // Decode 3 E3M5 scales for non-demoted groups
    float scales[4];
    int si = 1;
    for (int g = 0; g < 4; g++) {
        if (g == demoted) { scales[g] = demoted_scale; continue; }
        scales[g] = tq3_4s_decode_scale_cuda(x->d[si++]);
    }

    // Slot-to-group mapping
    int slot_to_group[4];
    slot_to_group[0] = promoted;
    slot_to_group[3] = demoted;
    int si2 = 1;
    for (int g = 0; g < 4; g++) {
        if (g != promoted && g != demoted) slot_to_group[si2++] = g;
    }

    dst_t * y = yy + i * QK_TQ3_0;
    const int j = threadIdx.x;  // 0-31
    const int g = j / 8;        // WHT group 0-3
    const int r = j % 8;        // element within group

    // Determine which slot this group was packed into
    int slot;
    if (g == promoted) slot = 0;
    else if (g == demoted) slot = 3;
    else {
        // Normal groups: find which slot (1 or 2)
        int norm_idx = 0;
        for (int gg = 0; gg < g; gg++) {
            if (gg != promoted && gg != demoted) norm_idx++;
        }
        slot = 1 + norm_idx;
    }

    float val;
    if (slot == 0) {
        // 4-bit from qs[0-3]
        const int byte_idx = r / 2;
        const int nibble = (r & 1) ? (x->qs[byte_idx] >> 4) : (x->qs[byte_idx] & 0xF);
        val = tq3_v_c4_cuda[nibble] * scales[g];
    } else if (slot == 3) {
        // 2-bit from qs[10-11]
        const int byte_idx = 10 + r / 4;
        const int shift = (r % 4) * 2;
        const int idx2 = (x->qs[byte_idx] >> shift) & 3;
        val = tq3_v_c2_cuda[idx2] * scales[g];
    } else {
        // 3-bit from qs[4-6] (slot 1) or qs[7-9] (slot 2)
        const uint8_t * qp = x->qs + 4 + (slot - 1) * 3;
        const uint8_t idx3 = tq3_idx_from_packed_cuda(qp, r);
        val = tq3_0_centroids_cuda[idx3] * scales[g];
    }

    // Inverse WHT butterfly
    for (int step = 1; step < 32; step <<= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        if (j & step) {
            val = other - val;
        } else {
            val = other + val;
        }
    }

    y[j] = (dst_t)(val * (tq3_0_signs_cuda[j] / sqrtf(32.0f)));
}

template<typename dst_t>
static void dequantize_row_tq3_4sv_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_TQ3_0;
    dequantize_block_tq3_4sv<<<nb, 32, 0, stream>>>(vx, y, nb);
}

static __device__ __forceinline__ uint8_t tq3_idx_from_packed_cuda(const uint8_t * qp, int r) {
    switch (r) {
        case 0: return  qp[0]       & 7;
        case 1: return (qp[0] >> 3) & 7;
        case 2: return ((qp[0] >> 6) | (qp[1] << 2)) & 7;
        case 3: return (qp[1] >> 1) & 7;
        case 4: return (qp[1] >> 4) & 7;
        case 5: return ((qp[1] >> 7) | (qp[2] << 1)) & 7;
        case 6: return (qp[2] >> 2) & 7;
        default: return (qp[2] >> 5) & 7;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_tq3_1s_ap1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb_super) {
    const int super_idx = blockIdx.x;
    if (super_idx >= nb_super) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (warp >= 16) {
        return;
    }

    const block_tq3_1s_ap1 * sb = (const block_tq3_1s_ap1 *) vx + super_idx;
    __shared__ int promoted_slot_shared;
    if (tid == 0) {
        promoted_slot_shared = __ffs((int) sb->mask) - 1;
    }
    __syncthreads();

    const int slot = warp;
    const int promoted_slot = promoted_slot_shared;
    const bool is_promoted = promoted_slot == slot;
    const uint8_t * base_region = sb->qs;
    const uint8_t * promo_region = sb->qs + 15 * sizeof(block_tq3_1s);
    const int base_slot = slot - (slot > promoted_slot ? 1 : 0);
    const uint8_t * base = base_region + base_slot * sizeof(block_tq3_1s);

    dst_t * y = yy + (super_idx * 16 + slot) * QK_TQ3_0;

    if (!is_promoted) {
        const block_tq3_1s * x = (const block_tq3_1s *) base;
        const int g = lane / 8;
        const int r = lane % 8;
        const uint8_t * qp = x->qs + g * 3;
        const uint8_t idx = tq3_idx_from_packed_cuda(qp, r);
        const float d = lane < 16 ? __half2float(x->d0) : __half2float(x->d1);
        float val = tq3_0_centroids_cuda[idx] * d;
        for (int step = 1; step < 32; step <<= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
            val = (lane & step) ? (other - val) : (other + val);
        }
        y[lane] = (dst_t) (val * (tq3_0_signs_cuda[lane] / sqrtf(32.0f)));
        return;
    }

    const block_tq3_1s_shift * x = (const block_tq3_1s_shift *) promo_region;
    const int g = lane / 8;
    const int r = lane % 8;
    const uint8_t * qp = x->qs + g * 3;
    const uint8_t idx = tq3_idx_from_packed_cuda(qp, r);
    const float d = lane < 16 ? __half2float(x->d0) : __half2float(x->d1);
    float val = tq3_0_centroids_cuda[idx] * d + __half2float(x->m);
    for (int step = 1; step < 32; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (lane & step) ? (other - val) : (other + val);
    }
    y[lane] = (dst_t) (val * (tq3_0_signs_cuda[lane] / sqrtf(32.0f)));
}

template<typename dst_t>
static void dequantize_row_tq3_1s_ap1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ3_1S_AP1 == 0);
    const int nb_super = k / QK_TQ3_1S_AP1;
    dequantize_block_tq3_1s_ap1<<<nb_super, 512, 0, stream>>>(vx, y, nb_super);
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        case GGML_TYPE_TQ3_0:
            return dequantize_row_tq3_0_cuda;
        case GGML_TYPE_TQ3_1S:
            return dequantize_row_tq3_1s_cuda;
        case GGML_TYPE_TQ3_4S:
            return dequantize_row_tq3_4s_cuda;
        case GGML_TYPE_TQ3_4SE:
            return dequantize_row_tq3_4se_cuda;
        case GGML_TYPE_TQ3_4SV:
            return dequantize_row_tq3_4sv_cuda;
        case GGML_TYPE_TQ3_1S_AP1:
            return dequantize_row_tq3_1s_ap1_cuda;
        case GGML_TYPE_Q4_0_TQ:
            return dequantize_row_q4_0_tq_cuda;
        case GGML_TYPE_Q4_1_TQ:
            return dequantize_row_q4_1_tq_cuda;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        case GGML_TYPE_TQ3_0:
            return dequantize_row_tq3_0_cuda;
        case GGML_TYPE_TQ3_1S:
            return dequantize_row_tq3_1s_cuda;
        case GGML_TYPE_TQ3_4S:
            return dequantize_row_tq3_4s_cuda;
        case GGML_TYPE_TQ3_4SE:
            return dequantize_row_tq3_4se_cuda;
        case GGML_TYPE_TQ3_4SV:
            return dequantize_row_tq3_4sv_cuda;
        case GGML_TYPE_TQ3_1S_AP1:
            return dequantize_row_tq3_1s_ap1_cuda;
        case GGML_TYPE_Q4_0_TQ:
            return dequantize_row_q4_0_tq_cuda;
        case GGML_TYPE_Q4_1_TQ:
            return dequantize_row_q4_1_tq_cuda;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
