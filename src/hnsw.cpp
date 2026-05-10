#include "vectordb/hnsw.hpp"
#include "vectordb/distance.hpp"
#include "vectordb/io.hpp"
#include "vectordb/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace vectordb {

HnswIndex::HnswIndex(std::size_t dim, Metric metric,
                     std::size_t M, std::size_t ef_construction,
                     uint64_t seed)
    : dim_(dim),
      metric_(metric),
      M_(M),
      M_max_(M),
      M_max0_(M * 2),
      ef_construction_(ef_construction),
      level_mult_(M <= 1 ? 0.0 : 1.0 / std::log(static_cast<double>(M))),
      dist_fn_((metric == Metric::L2) ? &l2sq : &neg_dot),
      rng_(seed) {
    // M=0 makes the geometric level distribution undefined; M=1 makes
    // log(M)=0 and level_mult divide-by-zero. The algorithm assumes M >= 2.
    if (dim == 0) throw std::invalid_argument("HNSW: dim must be > 0");
    if (M < 2)    throw std::invalid_argument("HNSW: M must be >= 2");
    if (ef_construction == 0) {
        throw std::invalid_argument("HNSW: ef_construction must be > 0");
    }
}

float HnswIndex::distance(const float* a, const float* b) const noexcept {
    return dist_fn_(a, b, dim_);
}

int HnswIndex::random_level() {
    // Level ~ floor(-ln(U) / ln(M)). Geometric distribution; p(level >= l) = M^-l.
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng_);
    if (r < 1e-12) r = 1e-12;
    return static_cast<int>(std::floor(-std::log(r) * level_mult_));
}

std::vector<HnswIndex::PairF>
HnswIndex::search_layer(const float* q, id_t entry,
                        std::size_t ef, int layer,
                        VisitedSet& vs) const {
    // Generation-counter visited tracking, owned by the caller so search() can
    // run with one VisitedSet per OpenMP thread (no shared mutable state).
    vs.resize_to(ntotal_);
    const uint32_t mark = vs.bump();
    auto& visited = vs.marks;

    std::priority_queue<PairF, std::vector<PairF>, std::greater<PairF>> candidates;  // min-heap
    std::priority_queue<PairF> top;  // max-heap (default less<>)

    float d0 = distance(q, vec(entry));
    candidates.emplace(d0, entry);
    top.emplace(d0, entry);
    visited[entry] = mark;

    while (!candidates.empty()) {
        PairF cur = candidates.top();
        if (cur.first > top.top().first) break;
        candidates.pop();

        const auto& nbrs = links_[cur.second][layer];

        // Software prefetch: pull the first neighbor's vector into L1 ahead
        // of the distance call. Each neighbor lookup is a random ~dim*4 byte
        // load; prefetching one ahead hides the L1/L2 miss latency.
        if (!nbrs.empty()) __builtin_prefetch(vec(nbrs[0]), 0, 1);

        for (std::size_t i = 0; i < nbrs.size(); ++i) {
            id_t e = nbrs[i];
            if (i + 1 < nbrs.size()) {
                __builtin_prefetch(vec(nbrs[i + 1]), 0, 1);
            }
            if (visited[e] == mark) continue;
            visited[e] = mark;

            float ed = distance(q, vec(e));
            if (top.size() < ef || ed < top.top().first) {
                candidates.emplace(ed, e);
                top.emplace(ed, e);
                if (top.size() > ef) top.pop();
            }
        }
    }

    std::vector<PairF> out;
    out.reserve(top.size());
    while (!top.empty()) { out.push_back(top.top()); top.pop(); }
    return out;
}

std::vector<HnswIndex::PairF>
HnswIndex::select_neighbors_heuristic(const float* q,
                                      const std::vector<PairF>& candidates,
                                      std::size_t M) const {
    // Algorithm 4 from the HNSW paper, simplified (no extension/keep-pruned).
    // Iterate candidates by ascending distance to q. Accept e iff it is closer
    // to q than to every already-accepted neighbor — this enforces angular
    // diversity and avoids redundant edges in the same direction.
    std::vector<PairF> W = candidates;
    std::sort(W.begin(), W.end(),
              [](const PairF& a, const PairF& b) { return a.first < b.first; });

    std::vector<PairF> R;
    R.reserve(M);
    for (const PairF& cand : W) {
        if (R.size() >= M) break;
        float d_qe = cand.first;
        id_t  e    = cand.second;
        bool good = true;
        for (const PairF& acc : R) {
            float d_er = distance(vec(e), vec(acc.second));
            if (d_er < d_qe) { good = false; break; }
        }
        if (good) R.push_back(cand);
    }
    return R;
}

void HnswIndex::connect(id_t new_id, const std::vector<PairF>& neighbors, int layer) {
    auto& my_links = links_[new_id][layer];
    my_links.clear();
    my_links.reserve(neighbors.size());
    for (const PairF& p : neighbors) my_links.push_back(p.second);

    const std::size_t cap = (layer == 0) ? M_max0_ : M_max_;
    for (const PairF& p : neighbors) {
        id_t n = p.second;
        auto& their_links = links_[n][layer];
        their_links.push_back(new_id);
        if (their_links.size() > cap) {
            // Re-prune via heuristic so the existing node keeps its best edges.
            std::vector<PairF> cand;
            cand.reserve(their_links.size());
            for (id_t x : their_links) {
                cand.emplace_back(distance(vec(n), vec(x)), x);
            }
            auto pruned = select_neighbors_heuristic(vec(n), cand, cap);
            their_links.clear();
            their_links.reserve(pruned.size());
            for (const PairF& pp : pruned) their_links.push_back(pp.second);
        }
    }
}

void HnswIndex::add(const float* data_in, std::size_t n) {
    // Build is single-threaded; one VisitedSet for the whole batch.
    VisitedSet vs;
    for (std::size_t k = 0; k < n; ++k) {
        const float* v = data_in + k * dim_;
        id_t id = static_cast<id_t>(ntotal_);
        ntotal_++;

        data_.insert(data_.end(), v, v + dim_);
        int new_level = random_level();
        level_.push_back(new_level);
        links_.emplace_back();
        links_[id].resize(new_level + 1);

        if (id == 0) {
            entry_point_ = 0;
            max_level_   = new_level;
            continue;
        }

        id_t curr = entry_point_;

        // Greedy descent through layers strictly above new_level.
        for (int lc = max_level_; lc > new_level; --lc) {
            bool changed = true;
            while (changed) {
                changed = false;
                const auto& nbrs = links_[curr][lc];
                float best = distance(v, vec(curr));
                for (id_t e : nbrs) {
                    float d = distance(v, vec(e));
                    if (d < best) { best = d; curr = e; changed = true; }
                }
            }
        }

        // Beam search + connect for layers min(max_level, new_level) down to 0.
        for (int lc = std::min(max_level_, new_level); lc >= 0; --lc) {
            auto W = search_layer(v, curr, ef_construction_, lc, vs);
            auto neighbors = select_neighbors_heuristic(v, W, M_);
            connect(id, neighbors, lc);

            // Hand off to next layer's greedy entry: nearest of the candidates.
            if (!W.empty()) {
                auto it = std::min_element(W.begin(), W.end(),
                    [](const PairF& a, const PairF& b) { return a.first < b.first; });
                curr = it->second;
            }
        }

        if (new_level > max_level_) {
            max_level_   = new_level;
            entry_point_ = id;
        }
    }
}

void HnswIndex::search(const float* queries, std::size_t nq, std::size_t k,
                       std::size_t ef,
                       float* out_distances, label_t* out_labels) const {
    if (k == 0) return;  // (nq, 0) output; no per-query work needed.
    if (ntotal_ == 0) {
        for (std::size_t i = 0; i < nq * k; ++i) {
            out_distances[i] = -1.0f;
            out_labels[i]    = -1;
        }
        return;
    }
    if (ef < k) ef = k;

    const bool is_l2 = (metric_ == Metric::L2);

    // Each query is independent. Per-thread state is just a VisitedSet; the
    // HnswIndex itself is read-only here. parallel_for builds one ctx per
    // worker thread before dispatching this thread's slice of queries.
    parallel_for<VisitedSet>(nq,
        []() { return VisitedSet{}; },
        [&](std::size_t qi, VisitedSet& vs) {
            const float* q = queries + qi * dim_;
            id_t curr = entry_point_;

            for (int lc = max_level_; lc >= 1; --lc) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    const auto& nbrs = links_[curr][lc];
                    float best = distance(q, vec(curr));
                    for (id_t e : nbrs) {
                        float d = distance(q, vec(e));
                        if (d < best) { best = d; curr = e; changed = true; }
                    }
                }
            }

            auto W = search_layer(q, curr, ef, 0, vs);
            std::sort(W.begin(), W.end(),
                      [](const PairF& a, const PairF& b) { return a.first < b.first; });

            std::size_t out_n = std::min(k, W.size());
            for (std::size_t j = 0; j < out_n; ++j) {
                float d = W[j].first;
                out_distances[qi * k + j] = is_l2 ? d : -d;
                out_labels[qi * k + j]    = static_cast<label_t>(W[j].second);
            }
            for (std::size_t j = out_n; j < k; ++j) {
                out_distances[qi * k + j] = -1.0f;
                out_labels[qi * k + j]    = -1;
            }
        });
}

// ---- persistence ---------------------------------------------------------

void HnswIndex::save(const std::string& path) const {
    io::Writer w(path);
    w.write_magic("VHN1");
    w.write_pod<uint32_t>(1);
    w.write_pod<uint64_t>(dim_);
    w.write_pod<uint8_t>(metric_ == Metric::L2 ? 0 : 1);
    w.write_pod<uint64_t>(M_);
    w.write_pod<uint64_t>(M_max_);
    w.write_pod<uint64_t>(M_max0_);
    w.write_pod<uint64_t>(ef_construction_);
    w.write_pod<double>(level_mult_);
    w.write_pod<uint64_t>(ntotal_);
    w.write_pod<int32_t>(max_level_);
    w.write_pod<uint32_t>(entry_point_);

    if (ntotal_ > 0) {
        w.write_raw(level_.data(), ntotal_ * sizeof(int));
        w.write_raw(data_.data(),  ntotal_ * dim_ * sizeof(float));
    }

    // Per-node, per-layer neighbor lists. Variable-width by design.
    for (std::size_t id = 0; id < ntotal_; ++id) {
        int lv = level_[id];
        for (int layer = 0; layer <= lv; ++layer) {
            const auto& nbrs = links_[id][layer];
            uint32_t count = static_cast<uint32_t>(nbrs.size());
            w.write_pod<uint32_t>(count);
            if (count > 0) {
                w.write_raw(nbrs.data(), count * sizeof(id_t));
            }
        }
    }
}

HnswIndex HnswIndex::load(const std::string& path) {
    io::Reader r(path);
    r.check_magic("VHN1");
    uint32_t version = r.read_pod<uint32_t>();
    if (version != 1) throw std::runtime_error("HnswIndex: unsupported file version");

    uint64_t dim             = r.read_pod<uint64_t>();
    uint8_t  mb              = r.read_pod<uint8_t>();
    uint64_t M               = r.read_pod<uint64_t>();
    uint64_t M_max           = r.read_pod<uint64_t>();
    uint64_t M_max0          = r.read_pod<uint64_t>();
    uint64_t ef_construction = r.read_pod<uint64_t>();
    double   level_mult      = r.read_pod<double>();
    uint64_t ntotal          = r.read_pod<uint64_t>();
    int32_t  max_level       = r.read_pod<int32_t>();
    uint32_t entry_point     = r.read_pod<uint32_t>();

    Metric metric = (mb == 0) ? Metric::L2 : Metric::InnerProduct;
    HnswIndex idx(static_cast<std::size_t>(dim), metric,
                  static_cast<std::size_t>(M),
                  static_cast<std::size_t>(ef_construction),
                  /*seed=*/0);
    // Constructor only sets a subset; restore the rest directly.
    idx.M_max_      = static_cast<std::size_t>(M_max);
    idx.M_max0_     = static_cast<std::size_t>(M_max0);
    idx.level_mult_ = level_mult;
    idx.ntotal_     = static_cast<std::size_t>(ntotal);
    idx.max_level_  = max_level;
    idx.entry_point_= entry_point;

    idx.level_.resize(ntotal);
    idx.data_.resize(static_cast<std::size_t>(ntotal) * dim);
    if (ntotal > 0) {
        r.read_raw(idx.level_.data(), ntotal * sizeof(int));
        r.read_raw(idx.data_.data(),  ntotal * dim * sizeof(float));
    }

    idx.links_.assign(ntotal, {});
    for (std::size_t id = 0; id < ntotal; ++id) {
        int lv = idx.level_[id];
        idx.links_[id].resize(lv + 1);
        for (int layer = 0; layer <= lv; ++layer) {
            uint32_t count = r.read_pod<uint32_t>();
            idx.links_[id][layer].resize(count);
            if (count > 0) {
                r.read_raw(idx.links_[id][layer].data(), count * sizeof(id_t));
            }
        }
    }
    return idx;
}

std::vector<std::size_t> HnswIndex::level_histogram() const {
    std::vector<std::size_t> h;
    if (max_level_ < 0) return h;
    h.assign(max_level_ + 1, 0);
    for (int lv : level_) {
        if (lv >= 0 && lv < static_cast<int>(h.size())) h[lv]++;
    }
    return h;
}

}  // namespace vectordb
