#pragma once

#include <cstddef>

namespace vectordb {

// Squared L2 distance: sum_i (a_i - b_i)^2
// Squared form avoids a sqrt that doesn't change the ordering of nearest
// neighbors. We hand-unroll for NEON on Apple Silicon, scalar elsewhere.
float l2sq(const float* a, const float* b, std::size_t d) noexcept;

// Inner product: sum_i a_i * b_i
float dot(const float* a, const float* b, std::size_t d) noexcept;

// Negated inner product, so "smaller is better" matches L2's ordering.
// Lets the same heap code work for both metrics.
inline float neg_dot(const float* a, const float* b, std::size_t d) noexcept {
    return -dot(a, b, d);
}

}  // namespace vectordb
