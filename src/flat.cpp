#include "proxima/flat.hpp"
#include "proxima/distance.hpp"
#include "proxima/io.hpp"
#include "proxima/parallel.hpp"
#include "proxima/update.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace proxima {

FlatIndex::FlatIndex(std::size_t dim, Metric metric)
    : dim_(dim), metric_(metric) {
    if (dim == 0) throw std::invalid_argument("FlatIndex: dim must be > 0");
}

void FlatIndex::add(const float* data, std::size_t n) {
    data_.insert(data_.end(), data, data + n * dim_);
    labels_.reserve(labels_.size() + n);
    for (std::size_t i = 0; i < n; ++i) labels_.push_back(next_label_++);
    ntotal_ += n;
}

std::size_t FlatIndex::remove_ids(const label_t* labels, std::size_t n) {
    std::unordered_set<label_t> kill(labels, labels + n);
    std::size_t w = 0;
    for (std::size_t r = 0; r < ntotal_; ++r) {
        if (kill.count(labels_[r])) continue;
        if (w != r) {
            std::copy(data_.begin() + r * dim_, data_.begin() + (r + 1) * dim_,
                      data_.begin() + w * dim_);
            labels_[w] = labels_[r];
        }
        ++w;
    }
    const std::size_t removed = ntotal_ - w;
    ntotal_ = w;
    data_.resize(w * dim_);
    labels_.resize(w);
    return removed;
}

void FlatIndex::update(const label_t* labels, const float* data, std::size_t n) {
    if (n == 0) return;
    const auto want = detail::index_update_batch(labels, n);

    // Resolve every label to its row first; nothing is written until the
    // whole batch is known to be valid.
    constexpr std::size_t kMissing = static_cast<std::size_t>(-1);
    std::vector<std::size_t> row(n, kMissing);
    std::size_t found = 0;
    for (std::size_t r = 0; r < ntotal_ && found < n; ++r) {
        auto it = want.find(labels_[r]);
        if (it != want.end()) { row[it->second] = r; ++found; }
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (row[i] == kMissing) detail::throw_missing_label(labels[i]);
    }

    for (std::size_t i = 0; i < n; ++i) {
        std::copy(data + i * dim_, data + (i + 1) * dim_,
                  data_.begin() + row[i] * dim_);
    }
}

namespace {
// Max-heap of (distance, label). For L2 we minimize distance; for IP we
// negate so smaller-is-better still applies, keeps one heap implementation.
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

    // Brute force is embarrassingly parallel and the workload is balanced, 
    // every query scans every vector, so a stride schedule is fine.
    parallel_for(nq, [&](std::size_t qi) {
        const float* q = queries + qi * dim_;

        std::priority_queue<HeapEntry> heap;

        auto consider = [&](float d, std::size_t i) {
            if (heap.size() < k) {
                heap.push({d, labels_[i]});
            } else if (d < heap.top().distance) {
                heap.pop();
                heap.push({d, labels_[i]});
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
    w.write_pod<uint32_t>(2);                  // v2 adds explicit labels
    w.write_pod<uint64_t>(dim_);
    w.write_pod<uint8_t>(metric_ == Metric::L2 ? 0 : 1);
    w.write_pod<uint64_t>(ntotal_);
    if (ntotal_ > 0) {
        w.write_raw(data_.data(), ntotal_ * dim_ * sizeof(float));
        w.write_raw(labels_.data(), ntotal_ * sizeof(label_t));
    }
}

FlatIndex FlatIndex::load(const std::string& path) {
    io::Reader r(path);
    r.check_magic("VFL1");
    uint32_t version = r.read_pod<uint32_t>();
    if (version != 1 && version != 2) {
        throw std::runtime_error("FlatIndex: unsupported file version");
    }

    uint64_t dim    = r.read_pod<uint64_t>();
    uint8_t  mb     = r.read_pod<uint8_t>();
    uint64_t ntotal = r.read_pod<uint64_t>();

    Metric metric = (mb == 0) ? Metric::L2 : Metric::InnerProduct;
    FlatIndex idx(static_cast<std::size_t>(dim), metric);
    if (ntotal > 0) {
        std::vector<float> buf(static_cast<std::size_t>(ntotal) * dim);
        r.read_raw(buf.data(), buf.size() * sizeof(float));
        idx.add(buf.data(), static_cast<std::size_t>(ntotal));  // labels 0..n-1
        if (version >= 2) {
            r.read_raw(idx.labels_.data(), ntotal * sizeof(label_t));
            label_t mx = -1;
            for (label_t l : idx.labels_) mx = std::max(mx, l);
            idx.next_label_ = mx + 1;
        }
    }
    return idx;
}

}  // namespace proxima
