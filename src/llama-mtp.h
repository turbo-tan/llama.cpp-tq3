#pragma once

#include "llama.h"
#include "../ggml/include/ggml-backend.h"

#include <unordered_map>
#include <vector>

struct llama_mtp {
    llama_context * ctx_mtp = nullptr; // non-owning
    llama_batch hook_batch = {};
    ggml_backend_buffer_t hook_batch_embd_buffer = nullptr;
    std::vector<llama_token> hook_tokens;

    // Cross-ubatch carryover for the final hidden-state row.
    //
    // This is PER SEQUENCE. It used to be a single pending_h/pending_pos pair,
    // which is correct only while one sequence is ever in flight. With -np > 1
    // the server interleaves slots, so slot B's ubatch would consume the
    // carryover slot A had left behind and MTP would decode with a hidden state
    // belonging to a different conversation at a position from a third one --
    // surfacing as "sequence 0 positions are decreasing" and the M-RoPE
    // "X < Y" assertion. See issue #78.
    struct seq_state {
        std::vector<float> pending_h;   // [n_embd]
        llama_pos          pending_pos = -1;
    };

    std::unordered_map<llama_seq_id, seq_state> seq;

    // Staging buffer for one ubatch worth of hidden-state rows.
    // Rows belonging to a single sequence are not necessarily contiguous within
    // a ubatch, so the rows are pulled to the host once and then distributed
    // per sequence, rather than fetched with a per-sequence strided copy.
    std::vector<float> host_h;

    void reset() {
        seq.clear();
        host_h.clear();
        host_h.shrink_to_fit();
    }
};
