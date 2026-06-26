#include <ggml.h>
#include <ggml-backend.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cinttypes>
#include <cstring>

int ggml_cuda_flash_attn_ext_kernel_id(int device, const ggml_tensor * dst);

static constexpr int BEST_FATTN_KERNEL_VEC = 100;

static ggml_tensor * build_flash_attn_ext(
        ggml_context * ctx,
        ggml_type      type_k,
        ggml_type      type_v,
        int64_t        nb) {
    const int64_t hsk = 128;
    const int64_t hsv = 128;
    const int64_t nh  = 4;
    const int64_t kv  = 512;

    const int64_t hsk_padded = GGML_PAD(hsk, ggml_blck_size(type_k));
    const int64_t hsv_padded = GGML_PAD(hsv, ggml_blck_size(type_v));

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk_padded, nb, nh, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, type_k,        hsk_padded, kv, nh, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, type_v,        hsv_padded, kv, nh, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, nb, 1, 1);

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf((float) hsk), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    return out;
}

static ggml_tensor * build_set_rows(
        ggml_context * ctx,
        ggml_type      dst_type) {
    const int64_t ne0 = dst_type == GGML_TYPE_TURBO4_0 ? 128 : 256;
    const int64_t rows = 5;
    const int64_t set_rows = 2;

    ggml_tensor * dst      = ggml_new_tensor_4d(ctx, dst_type,      ne0, rows, 1, 1);
    ggml_tensor * src      = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, set_rows, 1, 1);
    ggml_tensor * row_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, set_rows);

    return ggml_set_rows(ctx, dst, src, row_idxs);
}

static bool cuda_supports(ggml_backend_t backend, ggml_tensor * op) {
    return ggml_backend_supports_op(backend, op);
}

static bool check_flash_case(ggml_backend_t backend, int cuda_device, ggml_type type_k, ggml_type type_v, int64_t nb) {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_tensor * out = build_flash_attn_ext(ctx, type_k, type_v, nb);
    const bool ok = cuda_supports(backend, out);
    const int kernel_id = ggml_cuda_flash_attn_ext_kernel_id(cuda_device, out);
    std::printf("FLASH_ATTN_EXT type_K=%s type_V=%s nb=%" PRId64 ": %s, kernel_id=%d\n",
            ggml_type_name(type_k), ggml_type_name(type_v), nb, ok ? "supported" : "not supported", kernel_id);

    ggml_free(ctx);
    return ok && kernel_id == BEST_FATTN_KERNEL_VEC;
}

static bool check_set_rows_case(ggml_backend_t backend, ggml_type dst_type) {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /* .mem_base = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_tensor * out = build_set_rows(ctx, dst_type);
    const bool ok = cuda_supports(backend, out);
    std::printf("SET_ROWS dst_type=%s: %s\n", ggml_type_name(dst_type), ok ? "supported" : "not supported");

    ggml_free(ctx);
    return ok;
}

int main(void) {
    ggml_backend_load_all();

    ggml_backend_dev_t cuda_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0) {
            cuda_dev = dev;
            break;
        }
    }

    if (cuda_dev == nullptr) {
        std::puts("CUDA backend not available, skipping TQ3 KV cache CUDA support test");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_dev_init(cuda_dev, nullptr);
    GGML_ASSERT(backend);

    bool ok = true;
    // nb=3 catches the SM86 fallback that can otherwise select MMA/temp-buffer FA.
    ok = check_flash_case(backend, 0, GGML_TYPE_Q8_0,     GGML_TYPE_TQ3_0, 1) && ok;
    ok = check_flash_case(backend, 0, GGML_TYPE_Q8_0,     GGML_TYPE_TQ3_0, 3) && ok;
    ok = check_flash_case(backend, 0, GGML_TYPE_Q4_0,     GGML_TYPE_TQ3_0, 1) && ok;
    ok = check_flash_case(backend, 0, GGML_TYPE_Q4_0,     GGML_TYPE_TQ3_0, 3) && ok;
    ok = check_flash_case(backend, 0, GGML_TYPE_TURBO4_0, GGML_TYPE_TQ3_0, 1) && ok;

    ok = check_set_rows_case(backend, GGML_TYPE_TQ3_0)    && ok;
    ok = check_set_rows_case(backend, GGML_TYPE_TURBO3_0) && ok;
    ok = check_set_rows_case(backend, GGML_TYPE_TURBO4_0) && ok;

    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
