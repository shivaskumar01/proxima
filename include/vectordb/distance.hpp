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

// ---- batch kernels --------------------------------------------------------
//
// One query against four candidates in a single pass (FAISS calls this
// fvec_L2sqr_batch_4). The win is twofold: the query vector is loaded once
// per 16-element block instead of four times, and the sixteen independent
// FMA chains (4 candidates x 4 accumulators) give the OoO engine enough ILP
// to hide both FMA latency and the candidate loads.
//
// Accumulation order per candidate is IDENTICAL to the single-pair kernels
// (same 16-wide chains, same pairwise horizontal sum, same scalar tail), so
// out[i] is bit-for-bit equal to l2sq(q, p_i, d) / dot(q, p_i, d). Callers
// can mix single and batched calls without changing results.

void l2sq_x4(const float* q,
             const float* p0, const float* p1,
             const float* p2, const float* p3,
             std::size_t d, float* out) noexcept;

void dot_x4(const float* q,
            const float* p0, const float* p1,
            const float* p2, const float* p3,
            std::size_t d, float* out) noexcept;

// One query against ny contiguous rows (base[i*d .. i*d+d)). Convenience
// wrapper over the x4 kernel with a single-pair tail; used by the flat scan,
// k-means assignment, and the IVF coarse-quantizer scan.
void l2sq_ny(float* out, const float* q, const float* base,
             std::size_t ny, std::size_t d) noexcept;

void dot_ny(float* out, const float* q, const float* base,
            std::size_t ny, std::size_t d) noexcept;

// Register-tiled 4-queries x 4-rows L2sq: out[i*4 + j] = ||q_i - c_j||^2.
// Each loaded q/c block is reused four times, cutting memory traffic ~4x
// versus four l2sq_ny passes, this is the GEMM-style blocking that makes
// a query-batched IVF coarse scan competitive at high dim, where the scan
// is bandwidth-bound. One accumulator chain per pair (16 live), so results
// are NOT bit-matched to l2sq()/l2sq_x4(); callers must not mix kernels
// inside a comparison whose termination depends on consistency (sorts and
// bounded heaps are fine, see the greedy-descent war story).
void l2sq_4x4(const float* q, std::size_t q_stride,
              const float* c, std::size_t c_stride,
              std::size_t d, float* out) noexcept;

}  // namespace vectordb
