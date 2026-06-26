#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);
int  ggml_cuda_flash_attn_ext_kernel_id(int device, const ggml_tensor * dst);
