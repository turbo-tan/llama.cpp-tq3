#include "speculative.h"

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

    return 0;
}
