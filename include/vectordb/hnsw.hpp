#pragma once

#include "vectordb/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace vectordb {

// Per-thread visited tracker. The generation counter lets us "reset" the
// whole buffer in O(1) by bumping the counter; only when it wraps do we
// memset (~once per 4B searches per thread, never in practice).
struct VisitedSet {
    std::vector<uint32_t> marks;
    uint32_t gen = 0;

    void resize_to(std::size_t n) {
        if (marks.size() < n) marks.resize(n, 0);
    }
    uint32_t bump() {
        if (gen == std::numeric_limits<uint32_t>::max()) {
            std::fill(marks.begin(), marks.end(), 0);
            gen = 0;
        }
        return ++gen;
    }
};

// HNSW (Hierarchical Navigable Small World) index following Malkov & Yashunin
// 2016 (arXiv:1603.09320).
//
// Concurrency model:
//   - add() parallelizes insertion across the batch with hnswlib-style
//     fine-grained locking: one mutex per node guarding its link lists, plus
//     a global entry-point mutex. Levels are drawn serially from the seeded
//     RNG before the parallel phase, so the level distribution is
//     deterministic; the link structure depends on insertion interleaving.
//   - search() on a quiescent index is lock-free and thread-safe (the
//     existing guarantee). Concurrent add() + search() is NOT supported.
//
// Tunables:
//   M               base connectivity per layer above 0 (typ. 8..32)
//   M_max0          max connectivity at layer 0 (default 2*M)
//   ef_construction beam width during insertion (typ. 100..400)
//   ef              beam width during search (typ. k..256)
class HnswIndex {
public:
    HnswIndex(std::size_t dim, Metric metric,
              std::size_t M = 16,
              std::size_t ef_construction = 200,
              uint64_t seed = 42);

    void add(const float* data, std::size_t n);

    // Pre-size storage for a total of n vectors. Essential when streaming
    // chunked add() calls on large datasets: vector growth doubles capacity,
    // so without an up-front reserve the data buffer transiently needs ~2x
    // the final footprint during the last realloc.
    void reserve(std::size_t n) {
        data_.reserve(n * dim_);
        level_.reserve(n);
        links_.reserve(n);
    }

    void search(const float* queries, std::size_t nq, std::size_t k,
                std::size_t ef,
                float* out_distances, label_t* out_labels) const;

    // Tombstone deletion (hnswlib-style mark_deleted): nodes stay in the
    // graph as routing waypoints — removing edges would shred connectivity —
    // but never appear in results. Memory is not reclaimed; labels (= node
    // ids) stay stable. Returns the number newly marked.
    std::size_t remove_ids(const label_t* labels, std::size_t n);

    std::size_t size() const noexcept { return nlive_; }   // live nodes
    std::size_t dim()  const noexcept { return dim_; }

    // Diagnostic: histogram of node levels (size = max_level+1).
    std::vector<std::size_t> level_histogram() const;

    // Persistence. Format: magic "VHN1", u32 version, hyperparams,
    // ntotal/level/data, then per-node nested neighbor lists.
    void save(const std::string& path) const;
    static HnswIndex load(const std::string& path);

private:
    using DistFn = float (*)(const float*, const float*, std::size_t);
    using PairF  = std::pair<float, id_t>;   // (distance, node id)

    // Per-thread traversal scratch: visited marks plus reusable buffers for
    // (a) the unvisited-neighbor batch handed to the 4-wide distance kernel
    // and (b) locked snapshots of a node's neighbor list during parallel
    // build. Owned by the caller so a const HnswIndex can be searched from
    // many threads at once.
    struct SearchCtx {
        VisitedSet visited;
        std::vector<id_t> gather;
        std::vector<id_t> snapshot;
    };

    int    random_level();
    float  distance(const float* a, const float* b) const noexcept;
    const float* vec(id_t id) const noexcept { return data_.data() + id * dim_; }

    // One query against four node ids via the batched NEON kernel
    // (bit-identical to four distance() calls).
    void distance_x4_ids(const float* q, const id_t* ids,
                         float* out) const noexcept;

    // The kLocked variants of the traversal primitives take each node's
    // mutex while reading/writing its link list; the lock-free variants
    // (kLocked = false) are used by query-time search on a quiescent index
    // and by small serial builds. `locks` may be null when !kLocked.

    // Greedy steepest-descent through layers (from, to]: at each layer hop
    // to the closest neighbor until a local minimum, then drop a layer.
    template <bool kLocked>
    id_t greedy_descend(const float* q, id_t curr, int from, int to,
                        SearchCtx& ctx, std::mutex* locks) const;

    // Beam search within one layer; returns up to ef closest nodes
    // (unsorted max-heap snapshot, caller may sort).
    // `deleted` (may be null) filters nodes out of the RESULT set while
    // still traversing them.
    template <bool kLocked>
    std::vector<PairF> search_layer(const float* q, id_t entry,
                                    std::size_t ef, int layer,
                                    SearchCtx& ctx, std::mutex* locks,
                                    const uint8_t* deleted = nullptr) const;

    // Heuristic neighbor selection (Algorithm 4 in the paper).
    // candidates is a list of (distance, id) pairs to consider; M is target.
    std::vector<PairF> select_neighbors_heuristic(
        const float* q,
        const std::vector<PairF>& candidates,
        std::size_t M) const;

    template <bool kLocked>
    void connect(id_t new_id, const std::vector<PairF>& neighbors, int layer,
                 std::mutex* locks);

    // Insert one pre-allocated node (data_/level_/links_ slots must already
    // exist). entry_mtx guards entry_point_/max_level_ when kLocked.
    template <bool kLocked>
    void insert_one(id_t id, SearchCtx& ctx,
                    std::mutex* locks, std::mutex* entry_mtx);

    std::size_t dim_;
    Metric      metric_;
    std::size_t M_;
    std::size_t M_max_;
    std::size_t M_max0_;
    std::size_t ef_construction_;
    double      level_mult_;       // 1 / ln(M)
    DistFn      dist_fn_;

    std::size_t ntotal_ = 0;              // storage slots (incl. tombstones)
    std::size_t nlive_  = 0;              // ntotal_ minus tombstones
    std::vector<uint8_t> deleted_;        // per-node tombstone marks
    std::vector<float> data_;             // ntotal * dim, row-major
    std::vector<int>   level_;            // per-node max layer
    std::vector<std::vector<std::vector<id_t>>> links_;  // [id][layer] -> neighbor ids

    int    max_level_   = -1;
    id_t   entry_point_ = 0;

    mutable std::mt19937_64 rng_;
};

}  // namespace vectordb
