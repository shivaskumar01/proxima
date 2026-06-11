#include "vectordb/flat.hpp"
#include "vectordb/distance.hpp"
#include "vectordb/io.hpp"
#include "vectordb/parallel.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <utility>

namespace vectordb {

FlatIndex::FlatIndex(std::size_t dim, Metric metric)
    : dim_(dim), metric_(metric) {
    if (dim == 0) throw std::invalid_argument("FlatIndex: dim must be > 0");
}

void FlatIndex::add(const float* data, std::size_t n) {
    data_.insert(data_.end(), data, data + n * dim_);
    ntotal_ += n;
}

namespace {
// Max-heap of (distance, label). For L2 we minimize distance; for IP we
// negate so smaller-is-better still applies — keeps one heap implementation.
struct HeapEntry {
    float distance;
    label_t label;
    bool operator<(const HeapEntry& other) const {
        return distance < other.distance;  // max-heap on distance
    }
};
}  // namespace

void FlatIndex::search(const float* queries, std::size_t nq, std::size_t k,
                       float* out_distances, label_t* out_labels) const {
    if (k == 0) return;  // caller asked for nothing; no work, no UB on heap.top()
    const bool is_l2 = (metric_ == Metric::L2);

    // Brute force is embarrassingly parallel and the workload is balanced —
    // every query scans every vector — so a stride schedule is fine.
    parallel_for(nq, [&](std::size_t qi) {
        const float* q = queries + qi * dim_;

        std::priority_queue<HeapEntry> heap;

        auto consider = [&](float d, std::size_t i) {
            if (heap.size() < k) {
                heap.push({d, static_cast<label_t>(i)});
            } else if (d < heap.top().distance) {
                heap.pop();
                heap.push({d, static_cast<label_t>(i)});
            }
        };

        // Rows are contiguous, so scan 4 at a time with the batch kernel:
        // the query block is loaded once per 16 floats instead of 4x, and
        // 16 independent FMA chains keep the pipes full. Heap updates stay
        // in row order, so results are identical to the one-by-one scan.
        std::size_t i = 0;
        float d4[4];
        for (; i + 4 <= ntotal_; i += 4) {
            const float* v = data_.data() + i * dim_;
            if (is_l2) {
                l2sq_x4(q, v, v + dim_, v + 2 * dim_, v + 3 * dim_, dim_, d4);
            } else {
                dot_x4(q, v, v + dim_, v + 2 * dim_, v + 3 * dim_, dim_, d4);
                for (int j = 0; j < 4; ++j) d4[j] = -d4[j];
            }
            for (int j = 0; j < 4; ++j) consider(d4[j], i + j);
        }
        for (; i < ntotal_; ++i) {
            const float* v = data_.data() + i * dim_;
            consider(is_l2 ? l2sq(q, v, dim_) : -dot(q, v, dim_), i);
        }

        std::size_t out_n = std::min(k, heap.size());
        for (std::size_t j = 0; j < out_n; ++j) {
            std::size_t pos = out_n - 1 - j;
            const HeapEntry& e = heap.top();
            out_distances[qi * k + pos] = is_l2 ? e.distance : -e.distance;
            out_labels[qi * k + pos]    = e.label;
            heap.pop();
        }
        for (std::size_t j = out_n; j < k; ++j) {
            out_distances[qi * k + j] = -1.0f;
            out_labels[qi * k + j]    = -1;
        }
    });
}

// ---- persistence ---------------------------------------------------------

void FlatIndex::save(const std::string& path) const {
    io::Writer w(path);
    w.write_magic("VFL1");
    w.write_pod<uint32_t>(1);                              // version
    w.write_pod<uint64_t>(dim_);
    w.write_pod<uint8_t>(metric_ == Metric::L2 ? 0 : 1);
    w.write_pod<uint64_t>(ntotal_);
    if (ntotal_ > 0) {
        w.write_raw(data_.data(), ntotal_ * dim_ * sizeof(float));
    }
}

FlatIndex FlatIndex::load(const std::string& path) {
    io::Reader r(path);
    r.check_magic("VFL1");
    uint32_t version = r.read_pod<uint32_t>();
    if (version != 1) throw std::runtime_error("FlatIndex: unsupported file version");

    uint64_t dim    = r.read_pod<uint64_t>();
    uint8_t  mb     = r.read_pod<uint8_t>();
    uint64_t ntotal = r.read_pod<uint64_t>();

    Metric metric = (mb == 0) ? Metric::L2 : Metric::InnerProduct;
    FlatIndex idx(static_cast<std::size_t>(dim), metric);
    if (ntotal > 0) {
        std::vector<float> buf(static_cast<std::size_t>(ntotal) * dim);
        r.read_raw(buf.data(), buf.size() * sizeof(float));
        idx.add(buf.data(), static_cast<std::size_t>(ntotal));
    }
    return idx;
}

}  // namespace vectordb
