#include "proxima/hnsw.hpp"
#include "proxima/distance.hpp"
#include "proxima/io.hpp"
#include "proxima/parallel.hpp"
#include "proxima/update.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>

namespace proxima {

namespace {
// Below this batch size a parallel build doesn't pay: thread spawn plus the
// O(ntotal) mutex array would dominate. Keeps streaming one-at-a-time add()
// calls on the cheap serial path.
constexpr std::size_t kParallelBuildThreshold = 256;
}  // namespace

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

void HnswIndex::distance_x4_ids(const float* q, const id_t* ids,
                                float* out) const noexcept {
    if (metric_ == Metric::L2) {
        l2sq_x4(q, vec(ids[0]), vec(ids[1]), vec(ids[2]), vec(ids[3]),
                dim_, out);
    } else {
        dot_x4(q, vec(ids[0]), vec(ids[1]), vec(ids[2]), vec(ids[3]),
               dim_, out);
        out[0] = -out[0]; out[1] = -out[1];
        out[2] = -out[2]; out[3] = -out[3];
    }
}

int HnswIndex::random_level() {
    // Level ~ floor(-ln(U) / ln(M)). Geometric distribution; p(level >= l) = M^-l.
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng_);
    if (r < 1e-12) r = 1e-12;
    return static_cast<int>(std::floor(-std::log(r) * level_mult_));
}

template <bool kLocked>
id_t HnswIndex::greedy_descend(const float* q, id_t curr, int from, int to,
                               SearchCtx& ctx, std::mutex* locks) const {
    if (from <= to) return curr;

    // `best` is computed ONCE and only ever lowered by strict comparison.
    // Termination therefore does not depend on distance evaluations being
    // reproducible: each accepted hop strictly decreases a single scalar.
    //
    // The earlier version recomputed best = distance(q, vec(curr)) on every
    // pass, with the single-pair kernel, while neighbors were evaluated
    // with the batched kernel. Under -ffast-math those two kernels rounded
    // differently (~5% of pairs, few-ulp deltas), and on near-tie pairs the
    // inconsistent comparisons let greedy descent ping-pong A->B->A forever.
    // GIST1M, with ~1% exact-duplicate descriptors, hit this within one
    // 100k batch; SIFT1M never did in 1M inserts. Fixed both ways: this
    // carried-best loop terminates under ANY kernel discrepancy, and
    // -ffast-math is gone so the kernels are bit-identical again.
    float best = distance(q, vec(curr));
    float d4[4];
    for (int lc = from; lc > to; --lc) {
        bool changed = true;
        while (changed) {
            changed = false;
            const std::vector<id_t>* nb;
            if constexpr (kLocked) {
                // Copy under the node's lock, then compute lock-free.
                {
                    std::lock_guard<std::mutex> g(locks[curr]);
                    ctx.snapshot = links_[curr][lc];
                }
                nb = &ctx.snapshot;
            } else {
                nb = &links_[curr][lc];
            }
            const auto& nbrs = *nb;
            std::size_t i = 0;
            for (; i + 4 <= nbrs.size(); i += 4) {
                distance_x4_ids(q, nbrs.data() + i, d4);
                for (int j = 0; j < 4; ++j) {
                    if (d4[j] < best) {
                        best = d4[j];
                        curr = nbrs[i + j];
                        changed = true;
                    }
                }
            }
            for (; i < nbrs.size(); ++i) {
                float d = distance(q, vec(nbrs[i]));
                if (d < best) { best = d; curr = nbrs[i]; changed = true; }
            }
        }
    }
    return curr;
}

template <bool kLocked>
std::vector<HnswIndex::PairF>
HnswIndex::search_layer(const float* q, id_t entry,
                        std::size_t ef, int layer,
                        SearchCtx& ctx, std::mutex* locks,
                        const uint8_t* deleted) const {
    // Generation-counter visited tracking, owned by the caller so search()
    // can run with one SearchCtx per std::thread worker (no shared state).
    VisitedSet& vs = ctx.visited;
    vs.resize_to(ntotal_);
    const uint32_t mark = vs.bump();
    auto& visited = vs.marks;

    std::priority_queue<PairF, std::vector<PairF>, std::greater<PairF>> candidates;  // min-heap
    std::priority_queue<PairF> top;  // max-heap (default less<>)

    float d0 = distance(q, vec(entry));
    candidates.emplace(d0, entry);
    if (!deleted || !deleted[entry]) top.emplace(d0, entry);
    visited[entry] = mark;

    // Tombstoned nodes are traversed (they remain routing waypoints) but
    // never enter `top`, so they cannot occupy result slots. With deletions
    // `top` may briefly be empty; the termination bound only applies once
    // ef live results exist.
    auto consider = [&](id_t e, float ed) {
        if (top.size() < ef || ed < top.top().first) {
            candidates.emplace(ed, e);
            if (!deleted || !deleted[e]) {
                top.emplace(ed, e);
                if (top.size() > ef) top.pop();
            }
        }
    };

    float d4[4];
    while (!candidates.empty()) {
        PairF cur = candidates.top();
        if (top.size() >= ef && cur.first > top.top().first) break;
        candidates.pop();

        const std::vector<id_t>* nb;
        if constexpr (kLocked) {
            {
                std::lock_guard<std::mutex> g(locks[cur.second]);
                ctx.snapshot = links_[cur.second][layer];
            }
            nb = &ctx.snapshot;
        } else {
            nb = &links_[cur.second][layer];
        }

        // Gather the unvisited neighbors (marking as we go) and prefetch
        // their vectors, each is a random ~dim*4-byte load, then compute
        // distances four at a time. Same neighbor order as a one-by-one
        // scan, so heap contents are identical.
        auto& batch = ctx.gather;
        batch.clear();
        for (id_t e : *nb) {
            if (visited[e] == mark) continue;
            visited[e] = mark;
            __builtin_prefetch(vec(e), 0, 1);
            batch.push_back(e);
        }

        std::size_t i = 0;
        for (; i + 4 <= batch.size(); i += 4) {
            distance_x4_ids(q, batch.data() + i, d4);
            consider(batch[i],     d4[0]);
            consider(batch[i + 1], d4[1]);
            consider(batch[i + 2], d4[2]);
            consider(batch[i + 3], d4[3]);
        }
        for (; i < batch.size(); ++i) {
            consider(batch[i], distance(q, vec(batch[i])));
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
    // to q than to every already-accepted neighbor, this enforces angular
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

template <bool kLocked>
void HnswIndex::connect(id_t new_id, const std::vector<PairF>& neighbors,
                        int layer, std::mutex* locks) {
    {
        std::unique_lock<std::mutex> g;
        if constexpr (kLocked) g = std::unique_lock<std::mutex>(locks[new_id]);
        // Merge rather than overwrite: during a parallel build another
        // thread may have already pushed a backlink into new_id's list at
        // this layer; clobbering it would leave a one-directional edge.
        auto& my_links = links_[new_id][layer];
        for (const PairF& p : neighbors) {
            id_t nid = p.second;
            if (std::find(my_links.begin(), my_links.end(), nid) ==
                my_links.end()) {
                my_links.push_back(nid);
            }
        }
    }

    const std::size_t cap = (layer == 0) ? M_max0_ : M_max_;
    for (const PairF& p : neighbors) {
        id_t n = p.second;
        std::unique_lock<std::mutex> g;
        if constexpr (kLocked) g = std::unique_lock<std::mutex>(locks[n]);
        auto& their_links = links_[n][layer];
        // Dedupe: under a parallel build, n (itself mid-insert) may already
        // have merged new_id into its list via its own connect().
        if (std::find(their_links.begin(), their_links.end(), new_id) !=
            their_links.end()) {
            continue;
        }
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

template <bool kLocked>
void HnswIndex::insert_one(id_t id, SearchCtx& ctx,
                           std::mutex* locks, std::mutex* entry_mtx) {
    const float* v = vec(id);
    const int new_level = level_[id];

    // Snapshot the entry point. A node whose level exceeds the current max
    // holds the entry lock for its whole insertion (it will become the new
    // entry point, and two max-raising inserts must serialize). This happens
    // ~log_M(n) times per build, so the serialization is negligible.
    std::unique_lock<std::mutex> entry_lock;
    if constexpr (kLocked) {
        entry_lock = std::unique_lock<std::mutex>(*entry_mtx);
    }
    id_t curr     = entry_point_;
    int  snap_max = max_level_;
    [[maybe_unused]] const bool hold = kLocked && new_level > snap_max;
    if constexpr (kLocked) {
        if (!hold) entry_lock.unlock();
    }

    // Greedy descent through layers strictly above new_level.
    curr = greedy_descend<kLocked>(v, curr, snap_max, new_level, ctx, locks);

    // Beam search + connect for layers min(max_level, new_level) down to 0.
    for (int lc = std::min(snap_max, new_level); lc >= 0; --lc) {
        auto W = search_layer<kLocked>(v, curr, ef_construction_, lc, ctx, locks);
        auto neighbors = select_neighbors_heuristic(v, W, M_);
        connect<kLocked>(id, neighbors, lc, locks);

        // Hand off to next layer's greedy entry: nearest of the candidates.
        if (!W.empty()) {
            auto it = std::min_element(W.begin(), W.end(),
                [](const PairF& a, const PairF& b) { return a.first < b.first; });
            curr = it->second;
        }
    }

    if (new_level > snap_max) {
        if constexpr (kLocked) {
            // `hold` inserts still own the lock; re-check under it either way
            // (another thread may have raised max_level_ past us meanwhile).
            if (!hold) entry_lock.lock();
            if (new_level > max_level_) {
                max_level_   = new_level;
                entry_point_ = id;
            }
        } else {
            max_level_   = new_level;
            entry_point_ = id;
        }
    }
}

void HnswIndex::add(const float* data_in, std::size_t n) {
    if (n == 0) return;
    const std::size_t start = ntotal_;

    // ---- Serial pre-phase: reserve every slot the parallel phase touches.
    // Vectors and levels are appended up front so data_ never reallocates
    // (and vec() stays valid) while worker threads run. Levels are drawn
    // serially so the rng_ sequence, and the level distribution, is
    // deterministic regardless of thread count.
    data_.insert(data_.end(), data_in, data_in + n * dim_);
    level_.reserve(start + n);
    links_.reserve(start + n);
    for (std::size_t k = 0; k < n; ++k) {
        int lv = random_level();
        level_.push_back(lv);
        links_.emplace_back();
        links_[start + k].resize(lv + 1);
    }
    ntotal_ = start + n;
    nlive_ += n;
    deleted_.resize(ntotal_, 0);

    std::size_t first = 0;
    if (start == 0) {
        // Bootstrap: node 0 is the initial entry point, trivially inserted.
        entry_point_ = 0;
        max_level_   = level_[0];
        first = 1;
    }
    const std::size_t m = n - first;
    if (m == 0) return;

    if (m < kParallelBuildThreshold || default_num_threads() <= 1) {
        SearchCtx ctx;
        for (std::size_t k = 0; k < m; ++k) {
            insert_one<false>(static_cast<id_t>(start + first + k),
                              ctx, nullptr, nullptr);
        }
        return;
    }

    // ---- Parallel phase: hnswlib-style fine-grained locking. One mutex per
    // node guards that node's link lists; entry_mtx guards entry_point_ and
    // max_level_. data_/level_ are read-only here. Lock order is one node
    // lock at a time, with entry_mtx never acquired while holding a node
    // lock, no cycles, no deadlock.
    std::vector<std::mutex> locks(ntotal_);
    std::mutex entry_mtx;
    parallel_for<SearchCtx>(m,
        []() { return SearchCtx{}; },
        [&](std::size_t k, SearchCtx& ctx) {
            insert_one<true>(static_cast<id_t>(start + first + k),
                             ctx, locks.data(), &entry_mtx);
        });
}

std::size_t HnswIndex::remove_ids(const label_t* labels, std::size_t n) {
    std::size_t removed = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const label_t l = labels[i];
        if (l < 0 || static_cast<std::size_t>(l) >= ntotal_) continue;
        if (!deleted_[static_cast<std::size_t>(l)]) {
            deleted_[static_cast<std::size_t>(l)] = 1;
            --nlive_;
            ++removed;
        }
    }
    return removed;
}

void HnswIndex::update(const label_t* labels, const float* data, std::size_t n) {
    if (n == 0) return;
    detail::index_update_batch(labels, n);   // rejects duplicates
    for (std::size_t i = 0; i < n; ++i) {
        const label_t l = labels[i];
        if (l < 0 || static_cast<std::size_t>(l) >= ntotal_ ||
            deleted_[static_cast<std::size_t>(l)]) {
            detail::throw_missing_label(l);
        }
    }
    SearchCtx ctx;
    for (std::size_t i = 0; i < n; ++i) {
        update_one(static_cast<id_t>(labels[i]), data + i * dim_, ctx);
    }
}

void HnswIndex::update_one(id_t id, const float* v, SearchCtx& ctx) {
    std::copy(v, v + dim_, data_.begin() + static_cast<std::size_t>(id) * dim_);
    if (ntotal_ == 1) return;   // lone node: no edges to repair
    const int lv = level_[id];

    // Step 1 (hnswlib updatePoint): every out-neighbor of `id` may now hold
    // a stale edge to it. Re-select each one's list from the two-hop
    // neighborhood, which contains its current neighbors plus `id` at its
    // new position.
    std::vector<id_t> cand;
    std::vector<PairF> pool;
    for (int layer = 0; layer <= lv; ++layer) {
        const std::vector<id_t> one_hop = links_[id][layer];
        if (one_hop.empty()) continue;
        cand.assign(1, id);
        for (id_t n1 : one_hop) {
            cand.push_back(n1);
            const auto& two_hop = links_[n1][layer];
            cand.insert(cand.end(), two_hop.begin(), two_hop.end());
        }
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

        const std::size_t cap = (layer == 0) ? M_max0_ : M_max_;
        for (id_t n1 : one_hop) {
            pool.clear();
            for (id_t c : cand) {
                if (c != n1) pool.emplace_back(distance(vec(n1), vec(c)), c);
            }
            if (pool.size() > ef_construction_) {
                std::nth_element(pool.begin(), pool.begin() + ef_construction_,
                                 pool.end());
                pool.resize(ef_construction_);
            }
            auto kept = select_neighbors_heuristic(vec(n1), pool, cap);
            auto& links = links_[n1][layer];
            links.clear();
            for (const PairF& p : kept) links.push_back(p.second);
        }
    }

    // Step 2 (repairConnectionsForUpdate): re-link `id` itself the way
    // insert_one would at its existing level. If `id` is the entry point
    // its level is max_level_ and the descent is a no-op.
    id_t curr = greedy_descend<false>(v, entry_point_, max_level_, lv,
                                      ctx, nullptr);
    for (int lc = lv; lc >= 0; --lc) {
        auto W = search_layer<false>(v, curr, ef_construction_, lc, ctx, nullptr);
        W.erase(std::remove_if(W.begin(), W.end(),
                               [id](const PairF& p) { return p.second == id; }),
                W.end());
        if (W.empty()) continue;   // alone on this layer: keep its links
        auto neighbors = select_neighbors_heuristic(v, W, M_);
        links_[id][lc].clear();
        connect<false>(id, neighbors, lc, nullptr);
        curr = std::min_element(W.begin(), W.end(),
            [](const PairF& a, const PairF& b) { return a.first < b.first; })->second;
    }
}

void HnswIndex::search(const float* queries, std::size_t nq, std::size_t k,
                       std::size_t ef,
                       float* out_distances, label_t* out_labels) const {
    if (k == 0) return;  // (nq, 0) output; no per-query work needed.
    if (nlive_ == 0) {
        for (std::size_t i = 0; i < nq * k; ++i) {
            out_distances[i] = -1.0f;
            out_labels[i]    = -1;
        }
        return;
    }
    if (ef < k) ef = k;

    const bool is_l2 = (metric_ == Metric::L2);

    // Each query is independent and the index is read-only here, so the
    // traversal runs lock-free. parallel_for builds one SearchCtx per worker
    // thread before dispatching this thread's slice of queries.
    parallel_for<SearchCtx>(nq,
        []() { return SearchCtx{}; },
        [&](std::size_t qi, SearchCtx& ctx) {
            const float* q = queries + qi * dim_;

            id_t curr = greedy_descend<false>(q, entry_point_, max_level_, 0,
                                              ctx, nullptr);

            auto W = search_layer<false>(q, curr, ef, 0, ctx, nullptr,
                                         nlive_ == ntotal_ ? nullptr
                                                           : deleted_.data());
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
    w.write_pod<uint32_t>(2);                  // v2 adds the tombstone bitmap
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
        w.write_raw(deleted_.data(), ntotal_ * sizeof(uint8_t));
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
    if (version != 1 && version != 2) {
        throw std::runtime_error("HnswIndex: unsupported file version");
    }

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
    idx.deleted_.assign(ntotal, 0);
    if (ntotal > 0) {
        r.read_raw(idx.level_.data(), ntotal * sizeof(int));
        r.read_raw(idx.data_.data(),  ntotal * dim * sizeof(float));
        if (version >= 2) {
            r.read_raw(idx.deleted_.data(), ntotal * sizeof(uint8_t));
        }
    }
    idx.nlive_ = idx.ntotal_;
    for (uint8_t dflag : idx.deleted_) idx.nlive_ -= dflag;

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

}  // namespace proxima
