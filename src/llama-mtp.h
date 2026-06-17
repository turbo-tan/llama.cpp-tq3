#pragma once

#include "llama.h"
#include "../ggml/include/ggml-backend.h"

#include <vector>

struct llama_mtp {
    llama_context * ctx_mtp = nullptr; // non-owning
    llama_batch hook_batch = {};
    ggml_backend_buffer_t hook_batch_embd_buffer = nullptr;
    std::vector<llama_token> hook_tokens;

    // Cross-ubatch carryover for the final hidden-state row.
    std::vector<float> pending_h;   // [n_embd]
    llama_pos          pending_pos = -1;
};
