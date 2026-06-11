#pragma once

#include "vectordb/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vectordb {

// Brute-force linear scan. Used as the recall ground truth for HNSW/IVF-PQ
// and as a baseline for the SIMD distance kernels.
class FlatIndex {
public:
    FlatIndex(std::size_t dim, Metric metric);

    // Append n vectors of dimension dim. Vectors are copied into a contiguous
    // row-major buffer so search loops stride sequentially through memory.
    void add(const float* data, std::size_t n);

    // Pre-size internal storage for a total of n vectors. Callers streaming
    // the dataset in chunks should reserve once up front: without it the
    // backing vector's geometric growth peaks at ~2x the final footprint
    // during the last realloc (old + new buffer live simultaneously) —
    // enough to push a 1M x 960 build into swap on a 16 GB machine.
    void reserve(std::size_t n) { data_.reserve(n * dim_); }

    // Top-k nearest neighbors for each of the nq queries.
    // out_distances and out_labels must each have nq * k slots.
    void search(const float* queries, std::size_t nq, std::size_t k,
                float* out_distances, label_t* out_labels) const;

    std::size_t size() const noexcept { return ntotal_; }
    std::size_t dim()  const noexcept { return dim_; }
    Metric      metric() const noexcept { return metric_; }

    // Raw access used by index implementations that delegate storage to Flat.
    const float* vector(id_t i) const noexcept { return data_.data() + i * dim_; }

    // Persistence. Format: magic "VFL1", u32 version, u64 dim, u8 metric,
    // u64 ntotal, ntotal*dim float32 vectors. Round-trip preserves search
    // results bit-for-bit.
    void save(const std::string& path) const;
    static FlatIndex load(const std::string& path);

private:
    std::size_t dim_;
    Metric      metric_;
    std::size_t ntotal_ = 0;
    std::vector<float> data_;
};

}  // namespace vectordb
