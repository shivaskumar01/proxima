#pragma once

// Tiny std::thread-based parallel_for. We don't use OpenMP because faiss-cpu
// bundles its own libomp and macOS dyld refuses to load a second one, using
// std::thread keeps us from being dependency-incompatible with FAISS in the
// same Python process.

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace vectordb {

inline unsigned default_num_threads() noexcept {
    unsigned t = std::thread::hardware_concurrency();
    return t == 0 ? 1u : t;
}

// Stride-scheduled parallel_for. Each worker also gets its own ThreadCtx
// (built by mk_ctx()), so callers can hold per-thread scratch (visited sets,
// LUT buffers) without touching a mutex.
//
// body signature: void(std::size_t i, ThreadCtx& ctx)
template <typename ThreadCtx, typename CtxFactory, typename Body>
void parallel_for(std::size_t n, CtxFactory mk_ctx, Body body,
                  unsigned nthreads = 0) {
    if (n == 0) return;
    if (nthreads == 0) nthreads = default_num_threads();
    if (nthreads > n) nthreads = static_cast<unsigned>(n);

    if (nthreads <= 1) {
        ThreadCtx ctx = mk_ctx();
        for (std::size_t i = 0; i < n; ++i) body(i, ctx);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) {
        workers.emplace_back([=, &body, &mk_ctx]() {
            ThreadCtx ctx = mk_ctx();
            // Stride: thread t handles indices t, t+T, t+2T, ...
            for (std::size_t i = t; i < n; i += nthreads) {
                body(i, ctx);
            }
        });
    }
    for (auto& w : workers) w.join();
}

// Overload for the common case where workers don't need per-thread context.
template <typename Body>
void parallel_for(std::size_t n, Body body, unsigned nthreads = 0) {
    parallel_for<int>(n,
        []() { return 0; },
        [&](std::size_t i, int&) { body(i); },
        nthreads);
}

}  // namespace vectordb
