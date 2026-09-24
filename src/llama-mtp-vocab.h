#pragma once

#include <cstdint>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

struct llama_mtp_vocab_map {
    int64_t n_vocab = 0;
    std::vector<int32_t> ids;
};

inline llama_mtp_vocab_map llama_mtp_vocab_read(std::istream & in, int64_t max_size = 65536) {
    llama_mtp_vocab_map result;
    std::string magic;
    int64_t count = 0;

    if (!(in >> magic >> result.n_vocab >> count) || magic != "llama-mtp-vocab-v1" ||
            result.n_vocab <= 0 || result.n_vocab > (1 << 24) || count <= 0 ||
            count > max_size || count > result.n_vocab) {
        throw std::runtime_error("invalid MTP vocabulary header or shortlist size");
    }

    std::vector<bool> seen(result.n_vocab, false);
    result.ids.reserve((size_t) count);
    for (int64_t i = 0; i < count; ++i) {
        int64_t id = -1;
        if (!(in >> id) || id < 0 || id >= result.n_vocab || seen[id]) {
            throw std::runtime_error("invalid or duplicate MTP vocabulary token ID");
        }
        seen[id] = true;
        result.ids.push_back((int32_t) id);
    }

    std::string extra;
    if (in >> extra) {
        throw std::runtime_error("unexpected trailing MTP vocabulary data");
    }

    return result;
}
