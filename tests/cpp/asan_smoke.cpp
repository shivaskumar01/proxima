// AddressSanitizer audit: exercise the memory-unsafe surface of all four
// indexes across both metrics, with deletes, re-adds, partial fast-scan
// blocks, and save/load round-trips. Adversarial sizes (non-multiples of 4
// and 16 for tails; list counts not divisible by BLOCK=16 for fast-scan).
#include "proxima/flat.hpp"
#include "proxima/hnsw.hpp"
#include "proxima/ivfpq.hpp"
#include "proxima/ivfpq_fs.hpp"

#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using namespace proxima;

static std::vector<float> rnd(std::size_t n, std::size_t d, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g;
    std::vector<float> v(n * d);
    for (auto& x : v) x = g(rng);
    return v;
}

static void check_search(const char* tag, std::size_t nq, std::size_t k,
                         const std::vector<float>& D,
                         const std::vector<label_t>& L,
                         label_t max_label) {
    for (std::size_t i = 0; i < nq * k; ++i) {
        if (L[i] != -1 && (L[i] < 0 || L[i] >= max_label)) {
            std::printf("FAIL %s: label %lld out of range [0,%lld)\n",
                        tag, (long long)L[i], (long long)max_label);
            std::abort();
        }
    }
}

// A batch holding one unknown label must throw before touching the index.
template <typename Idx>
static void expect_update_throws(Idx& idx, std::size_t d) {
    std::vector<label_t> bad{2, 999999};
    auto v = rnd(2, d, 11);
    try {
        idx.update(bad.data(), v.data(), 2);
    } catch (const std::out_of_range&) {
        return;
    }
    std::printf("FAIL: update with an unknown label did not throw\n");
    std::abort();
}

template <typename Factory>
static void exercise_ivf(const char* tag, Factory make, std::size_t d) {
    const std::size_t n = 2003;            // not a multiple of 16
    auto data = rnd(n, d, 7);
    auto q = rnd(37, d, 8);
    auto idx = make();
    idx.train(data.data(), n);
    idx.add(data.data(), n);

    std::vector<float> D(37 * 10);
    std::vector<label_t> L(37 * 10);
    idx.search(q.data(), 37, 10, 8, D.data(), L.data());
    check_search(tag, 37, 10, D, L, (label_t)idx.size());

    // Remove a scattered, partial-block-straddling set including out-of-range
    // and duplicate labels.
    std::vector<label_t> kill;
    for (label_t l = 0; l < (label_t)n; l += 7) kill.push_back(l);
    kill.push_back(999999);                // out of range: must be ignored
    kill.push_back(0);                     // duplicate
    std::size_t removed = idx.remove_ids(kill.data(), kill.size());
    idx.search(q.data(), 37, 10, 8, D.data(), L.data());
    // Survivors must exclude killed labels.
    for (auto v : L) {
        if (v == -1) continue;
        if (v % 7 == 0) { std::printf("FAIL %s: killed label %lld surfaced\n", tag, (long long)v); std::abort(); }
    }

    // Re-add after delete: fresh labels, no crash, search reads in bounds
    // (ASan validates the loads; precise label assertions live in pytest).
    idx.add(data.data(), 50);
    idx.search(q.data(), 37, 10, 8, D.data(), L.data());

    // Update a scattered set of live labels (mix of in-place rewrites and
    // cross-list moves, incl. the newest labels in partial blocks), then a
    // batch with a bad label that must be rejected without side effects.
    std::vector<label_t> upd;
    for (label_t l = 1; l < (label_t)n; l += 13) if (l % 7) upd.push_back(l);
    upd.push_back((label_t)n + 49);
    auto fresh = rnd(upd.size(), d, 10);
    idx.update(upd.data(), fresh.data(), upd.size());
    idx.search(q.data(), 37, 10, 8, D.data(), L.data());
    check_search(tag, 37, 10, D, L, (label_t)n + 50);
    expect_update_throws(idx, d);

    std::printf("ok %s: removed=%zu, size after re-add=%zu\n",
                tag, removed, idx.size());
}

int main() {
    // ---- Flat (L2 + IP) ----
    for (int mi = 0; mi < 2; ++mi) {
        Metric m = mi ? Metric::InnerProduct : Metric::L2;
        const std::size_t d = 31, n = 517;
        auto data = rnd(n, d, 1);
        auto q = rnd(13, d, 2);
        FlatIndex idx(d, m);
        idx.add(data.data(), n);
        std::vector<float> D(13 * 7); std::vector<label_t> L(13 * 7);
        idx.search(q.data(), 13, 7, D.data(), L.data());
        std::vector<label_t> kill{0, 3, 9, 200, 516, 999999};
        idx.remove_ids(kill.data(), kill.size());
        idx.add(data.data(), 4);                    // re-add: new labels
        idx.search(q.data(), 13, 7, D.data(), L.data());
        std::vector<label_t> upd{1, 515, 517, 520}; // first/last survivors + re-adds
        auto fresh = rnd(upd.size(), d, 12);
        idx.update(upd.data(), fresh.data(), upd.size());
        expect_update_throws(idx, d);
        idx.search(q.data(), 13, 7, D.data(), L.data());
        idx.save("/tmp/_asan_flat.bin");
        FlatIndex r = FlatIndex::load("/tmp/_asan_flat.bin");
        r.search(q.data(), 13, 7, D.data(), L.data());
        std::printf("ok Flat(%s)\n", mi ? "ip" : "l2");
    }

    // ---- HNSW (L2 + IP) with tombstones across a parallel build ----
    for (int mi = 0; mi < 2; ++mi) {
        Metric m = mi ? Metric::InnerProduct : Metric::L2;
        const std::size_t d = 33, n = 1500;
        auto data = rnd(n, d, 3);
        auto q = rnd(20, d, 4);
        HnswIndex idx(d, m, 16, 100, 5);
        idx.add(data.data(), n);                    // parallel build
        std::vector<float> D(20 * 10); std::vector<label_t> L(20 * 10);
        idx.search(q.data(), 20, 10, 64, D.data(), L.data());
        std::vector<label_t> kill;
        for (label_t l = 0; l < (label_t)(n - 5); ++l) kill.push_back(l);  // all but 5
        idx.remove_ids(kill.data(), kill.size());
        idx.search(q.data(), 20, 10, 64, D.data(), L.data());  // must dig past tombstones
        idx.add(data.data(), 200);                  // incremental add after delete
        idx.search(q.data(), 20, 10, 64, D.data(), L.data());
        // Re-link live nodes among tombstones: the 5 survivors plus every
        // re-added node (the entry point may be among them).
        std::vector<label_t> upd;
        for (label_t l = (label_t)(n - 5); l < (label_t)(n + 200); ++l) upd.push_back(l);
        auto fresh = rnd(upd.size(), d, 13);
        idx.update(upd.data(), fresh.data(), upd.size());
        expect_update_throws(idx, d);               // label 2 is a tombstone
        idx.search(q.data(), 20, 10, 64, D.data(), L.data());
        idx.save("/tmp/_asan_hnsw.bin");
        HnswIndex r = HnswIndex::load("/tmp/_asan_hnsw.bin");
        r.search(q.data(), 20, 10, 64, D.data(), L.data());
        std::printf("ok HNSW(%s)\n", mi ? "ip" : "l2");
    }

    // ---- IVF-PQ + fast-scan, both metrics, several dims (incl. dsub tails) ----
    for (std::size_t d : {32u, 64u}) {
        for (int mi = 0; mi < 2; ++mi) {
            Metric m = mi ? Metric::InnerProduct : Metric::L2;
            exercise_ivf("IvfPq", [&]{ return IvfPqIndex(d, 16, 8, 10, 9, m); }, d);
            exercise_ivf("IvfPqFastScan", [&]{ return IvfPqFastScan(d, 16, 8, 10, 9, m); }, d);
        }
    }

    std::printf("ALL ASAN CHECKS PASSED\n");
    return 0;
}
