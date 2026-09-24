#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace proxima {

// In-place Lloyd's algorithm with k-means++ initialization.
//
// data:       n * dim row-major float buffer
// k:          number of centroids
// niter:      Lloyd iterations
// seed:       RNG seed (deterministic across runs)
// out_centroids: resized to k * dim, filled with final centroids
// out_assign: optional, resized to n, filled with cluster id per point
// nthreads:   0 = all cores. The parallel parts (assignment, seeding-distance
//             updates) are per-point independent, so results are identical
//             for any thread count, determinism depends only on `seed`.
//             Pass 1 when calling from an already-parallel context (e.g. the
//             per-subspace PQ training loop) to avoid oversubscription.
void kmeans(const float* data, std::size_t n, std::size_t dim,
            std::size_t k, std::size_t niter, uint64_t seed,
            std::vector<float>& out_centroids,
            std::vector<int32_t>* out_assign = nullptr,
            unsigned nthreads = 0);

// Single-query nearest-centroid lookup over contiguous centroid rows.
// Internally batched 4 rows at a time (l2sq_x4); used by IVF/PQ encode and
// the k-means assignment step.
int32_t nearest_centroid(const float* x, const float* centroids,
                         std::size_t k, std::size_t dim) noexcept;

// Argmax-inner-product variant: the assignment rule for inner-product IVF
// indexes (list membership and probe selection must agree, and both go by
// max <x, centroid> when the metric is IP). Batched 4 rows at a time.
int32_t nearest_centroid_ip(const float* x, const float* centroids,
                            std::size_t k, std::size_t dim) noexcept;

}  // namespace proxima
