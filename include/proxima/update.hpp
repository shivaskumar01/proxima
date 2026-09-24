#pragma once

#include "proxima/types.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace proxima::detail {

// Shared validation for update(labels, data, n). Every index resolves the
// whole batch BEFORE mutating anything, so a bad label fails the batch
// atomically instead of leaving it half-applied.

// label -> row in the update batch. Two new vectors for one label have no
// well-defined result, so duplicates are rejected.
inline std::unordered_map<label_t, std::size_t>
index_update_batch(const label_t* labels, std::size_t n) {
    std::unordered_map<label_t, std::size_t> want;
    want.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!want.emplace(labels[i], i).second) {
            throw std::invalid_argument("update: duplicate label " +
                                        std::to_string(labels[i]));
        }
    }
    return want;
}

// out_of_range so the bindings can surface it as KeyError.
[[noreturn]] inline void throw_missing_label(label_t l) {
    throw std::out_of_range("update: label " + std::to_string(l) +
                            " is not in the index");
}

}  // namespace proxima::detail
