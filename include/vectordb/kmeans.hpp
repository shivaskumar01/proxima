#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectordb {

// In-place Lloyd's algorithm with k-means++ initialization.
//
// data:       n * dim row-major float buffer
// k:          number of centroids
// niter:      Lloyd iterations
// seed:       RNG seed (deterministic across runs)
// out_centroids: resized to k * dim, filled with final centroids
// out_assign: optional, resized to n, filled with cluster id per point
void kmeans(const float* data, std::size_t n, std::size_t dim,
            std::size_t k, std::size_t niter, uint64_t seed,
            std::vector<float>& out_centroids,
            std::vector<int32_t>* out_assign = nullptr);

// Single-query nearest-centroid lookup; small helper used by IVF/PQ encode.
int32_t nearest_centroid(const float* x, const float* centroids,
                         std::size_t k, std::size_t dim) noexcept;

}  // namespace vectordb
