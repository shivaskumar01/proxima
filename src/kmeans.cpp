#include "vectordb/kmeans.hpp"
#include "vectordb/distance.hpp"
#include "vectordb/parallel.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>

namespace vectordb {

int32_t nearest_centroid(const float* x, const float* centroids,
                         std::size_t k, std::size_t dim) noexcept {
    int32_t best   = 0;
    float   best_d = std::numeric_limits<float>::max();
    float d4[4];
    std::size_t c = 0;
    // Batched scan: 4 centroid rows per pass through x. Strict `<` keeps the
    // lowest index on ties, matching the original sequential scan.
    for (; c + 4 <= k; c += 4) {
        const float* row = centroids + c * dim;
        l2sq_x4(x, row, row + dim, row + 2 * dim, row + 3 * dim, dim, d4);
        for (int j = 0; j < 4; ++j) {
            if (d4[j] < best_d) {
                best_d = d4[j];
                best   = static_cast<int32_t>(c + j);
            }
        }
    }
    for (; c < k; ++c) {
        float d = l2sq(x, centroids + c * dim, dim);
        if (d < best_d) { best_d = d; best = static_cast<int32_t>(c); }
    }
    return best;
}

int32_t nearest_centroid_ip(const float* x, const float* centroids,
                            std::size_t k, std::size_t dim) noexcept {
    int32_t best   = 0;
    float   best_d = std::numeric_limits<float>::lowest();
    float d4[4];
    std::size_t c = 0;
    for (; c + 4 <= k; c += 4) {
        const float* row = centroids + c * dim;
        dot_x4(x, row, row + dim, row + 2 * dim, row + 3 * dim, dim, d4);
        for (int j = 0; j < 4; ++j) {
            if (d4[j] > best_d) {
                best_d = d4[j];
                best   = static_cast<int32_t>(c + j);
            }
        }
    }
    for (; c < k; ++c) {
        float d = dot(x, centroids + c * dim, dim);
        if (d > best_d) { best_d = d; best = static_cast<int32_t>(c); }
    }
    return best;
}

namespace {

// k-means++ seeding: each new centroid is sampled with probability
// proportional to its squared distance to the closest already-chosen
// centroid. Reduces the variance of Lloyd's final WCSS dramatically.
//
// Parallelism note: only the min_d2 refresh is parallel (per-point
// independent, so bitwise-deterministic). The cumulative-sum sampling stays
// serial — a parallel reduction would change float summation order and with
// it the sampled index, breaking seed-determinism across thread counts.
void kmeanspp_init(const float* data, std::size_t n, std::size_t dim,
                   std::size_t k, std::mt19937_64& rng,
                   std::vector<float>& centroids, unsigned nthreads) {
    centroids.resize(k * dim);
    std::uniform_int_distribution<std::size_t> uniform_n(0, n - 1);
    std::size_t first = uniform_n(rng);
    std::memcpy(centroids.data(), data + first * dim, dim * sizeof(float));

    std::vector<float> min_d2(n);
    parallel_for(n, [&](std::size_t i) {
        min_d2[i] = l2sq(data + i * dim, centroids.data(), dim);
    }, nthreads);

    for (std::size_t c = 1; c < k; ++c) {
        // Sample idx with probability min_d2[i] / sum(min_d2)
        double total = 0.0;
        for (float v : min_d2) total += v;
        if (total <= 0.0) {
            // Degenerate (all duplicates of chosen centroids); pick any.
            std::size_t idx = uniform_n(rng);
            std::memcpy(centroids.data() + c * dim, data + idx * dim,
                        dim * sizeof(float));
        } else {
            std::uniform_real_distribution<double> u(0.0, total);
            double r = u(rng);
            std::size_t idx = 0;
            double cum = 0.0;
            for (; idx < n; ++idx) {
                cum += min_d2[idx];
                if (cum >= r) break;
            }
            if (idx >= n) idx = n - 1;
            std::memcpy(centroids.data() + c * dim, data + idx * dim,
                        dim * sizeof(float));
        }
        // Update min_d2 with the new centroid (per-point independent).
        const float* new_c = centroids.data() + c * dim;
        parallel_for(n, [&](std::size_t i) {
            float d = l2sq(data + i * dim, new_c, dim);
            if (d < min_d2[i]) min_d2[i] = d;
        }, nthreads);
    }
}

}  // namespace

void kmeans(const float* data, std::size_t n, std::size_t dim,
            std::size_t k, std::size_t niter, uint64_t seed,
            std::vector<float>& out_centroids,
            std::vector<int32_t>* out_assign,
            unsigned nthreads) {
    if (k > n) k = n;  // can't have more clusters than points
    std::mt19937_64 rng(seed);

    kmeanspp_init(data, n, dim, k, rng, out_centroids, nthreads);

    std::vector<int32_t> assign(n, 0);
    std::vector<double> sums(k * dim, 0.0);   // double accumulators reduce drift
    std::vector<int64_t> counts(k, 0);

    for (std::size_t iter = 0; iter < niter; ++iter) {
        // Assign step: each point -> nearest centroid. This is the O(n*k*dim)
        // hot loop of Lloyd's; per-point independent, so the parallel result
        // is identical to the serial one for any thread count.
        parallel_for(n, [&](std::size_t i) {
            assign[i] = nearest_centroid(data + i * dim, out_centroids.data(),
                                         k, dim);
        }, nthreads);

        // Update step: centroid = mean of assigned points. O(n*dim) and
        // memory-bound; kept serial so the double-precision sums accumulate
        // in a fixed order (deterministic centroids).
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (std::size_t i = 0; i < n; ++i) {
            int32_t c = assign[i];
            const float* x = data + i * dim;
            double* s = sums.data() + c * dim;
            for (std::size_t j = 0; j < dim; ++j) s[j] += x[j];
            counts[c]++;
        }
        for (std::size_t c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                // Empty cluster: re-seed from a random point. Common k-means
                // hygiene; without it whole clusters can stay dead forever.
                std::uniform_int_distribution<std::size_t> u(0, n - 1);
                std::size_t pick = u(rng);
                std::memcpy(out_centroids.data() + c * dim,
                            data + pick * dim, dim * sizeof(float));
            } else {
                float* cptr = out_centroids.data() + c * dim;
                const double* s = sums.data() + c * dim;
                double inv = 1.0 / static_cast<double>(counts[c]);
                for (std::size_t j = 0; j < dim; ++j) {
                    cptr[j] = static_cast<float>(s[j] * inv);
                }
            }
        }
    }

    if (out_assign) *out_assign = std::move(assign);
}

}  // namespace vectordb
