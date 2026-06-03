#include "ggml.h"
#include "llama.h"
#include "sampling.h"

int main() {
    auto * selected_p_fn = &common_sampler_selected_p;
    GGML_ASSERT(selected_p_fn != nullptr);

    GGML_ASSERT(llama_get_sampled_token_ith_nosync(nullptr, 0) == LLAMA_TOKEN_NULL);

    return 0;
}
