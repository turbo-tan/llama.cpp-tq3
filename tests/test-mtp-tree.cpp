#include "speculative.h"
#include "llama-batch.h"

#include <cassert>

int main() {
    common_speculative_mtp_tree_plan plan;
    plan.nodes = {
        { 100, -1, 0, 1.0f, 1.00f, 0, 0 },
        { 101,  0, 1, 0.9f, 0.90f, 0, 1 },
        { 102,  1, 2, 0.8f, 0.72f, 0, 2 },
        { 201,  0, 1, 0.1f, 0.10f, 0, 1 },
        { 202,  3, 2, 0.7f, 0.07f, 0, 2 },
    };

    assert(common_speculative_mtp_tree_is_ancestor(plan, 2, 0));
    assert(common_speculative_mtp_tree_is_ancestor(plan, 2, 1));
    assert(common_speculative_mtp_tree_is_ancestor(plan, 2, 2));
    assert(common_speculative_mtp_tree_is_ancestor(plan, 4, 3));

    assert(!common_speculative_mtp_tree_is_ancestor(plan, 2, 3));
    assert(!common_speculative_mtp_tree_is_ancestor(plan, 4, 1));
    assert(!common_speculative_mtp_tree_is_ancestor(plan, 1, 2));
    assert(!common_speculative_mtp_tree_is_ancestor(plan, -1, 0));
    assert(!common_speculative_mtp_tree_is_ancestor(plan, 2, 99));

    llama_batch batch = llama_batch_init(4, 0, 1);
    batch.n_tokens = 4;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = 100 + i;
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
        batch.tree_node[i] = i;
        batch.tree_parent[i] = i == 0 ? -1 : i - 1;
        batch.tree_aux[i] = i > 1;
    }

    llama_batch_allocr balloc(1);
    assert(balloc.init(batch, llama_vocab{}, nullptr, 0, 1, true));
    llama_ubatch ubatch = balloc.split_simple(4);
    assert(ubatch.n_tokens == 4);
    assert(ubatch.tree_node != nullptr);
    assert(ubatch.tree_parent != nullptr);
    assert(ubatch.tree_aux != nullptr);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        assert(ubatch.tree_node[i] == (int32_t) i);
        assert(ubatch.tree_parent[i] == (i == 0 ? -1 : (int32_t) i - 1));
        assert(ubatch.tree_aux[i] == (i > 1));
    }

    llama_batch_free(batch);

    return 0;
}
