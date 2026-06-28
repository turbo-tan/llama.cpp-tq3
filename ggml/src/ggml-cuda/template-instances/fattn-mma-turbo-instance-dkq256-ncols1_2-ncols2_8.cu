// Hand-created turbo MMA decode instance. Do NOT run generate_cu_files.py over it.

#include "../fattn-mma-f16.cuh"
#include "../fattn-mma-turbo.cuh"

DECL_FATTN_MMA_TURBO_CASE(256, 256, 2, 8, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
DECL_FATTN_MMA_TURBO_CASE(256, 256, 2, 8, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_MMA_TURBO_CASE(256, 256, 2, 8, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
