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
