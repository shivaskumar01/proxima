#include "proxima/ivfpq.hpp"
#include "proxima/coarse.hpp"
#include "proxima/distance.hpp"
#include "proxima/io.hpp"
#include "proxima/kmeans.hpp"
#include "proxima/parallel.hpp"
#include "proxima/update.hpp"

#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#include "proxima/simd.hpp"

namespace proxima {

IvfPqIndex::IvfPqIndex(std::size_t dim, std::size_t nlist, std::size_t M,
                       std::size_t kmeans_iters, uint64_t seed, Metric metric)
    : dim_(dim), nlist_(nlist), M_(M),
      dsub_(M == 0 ? 0 : dim / M),
      kmeans_iters_(kmeans_iters),
      seed_(seed),
      metric_(metric) {
    if (dim == 0)   throw std::invalid_argument("IVF-PQ: dim must be > 0");
    if (nlist == 0) throw std::invalid_argument("IVF-PQ: nlist must be > 0");
    if (M == 0)     throw std::invalid_argument("IVF-PQ: M must be > 0");
    if (dim % M != 0) {
        throw std::invalid_argument("IVF-PQ: dim must be divisible by M");
    }
    // Pre-allocate inverted-list slots so list_sizes() and other introspection
    // is safe before train(). They stay empty until train() (which re-assigns).
    inv_labels_.assign(nlist_, {});
    inv_codes_.assign(nlist_, {});
}

void IvfPqIndex::train(const float* data, std::size_t n) {
    if (n < nlist_) {
        throw std::invalid_argument("IVF-PQ: training set smaller than nlist");
    }
    // PQ codebooks need n >= KSUB per subspace (each subspace runs kmeans
    // with k=KSUB). Without this, kmeans clamps k to n and returns a smaller
    // codebook than the M*KSUB*dsub_ memcpy below assumes, silent OOB read.
    if (n < KSUB) {
        throw std::invalid_argument(
            "IVF-PQ: training set smaller than KSUB (256); PQ codebooks would be undertrained");
    }

    // 1. Train coarse quantizer on raw vectors. kmeans parallelizes its
    //    assignment step internally (deterministic for fixed seed).
    std::vector<int32_t> coarse_assign;
    kmeans(data, n, dim_, nlist_, kmeans_iters_, seed_,
           coarse_centroids_, &coarse_assign, /*nthreads=*/0);

    // 2. PQ training input: residuals for L2 (r_i = x_i - centroid), the raw
    //    vectors for IP (no residual encoding, see header).
    const float* pq_train = data;
    std::vector<float> residuals;
    if (metric_ == Metric::L2) {
        residuals.resize(static_cast<std::size_t>(n) * dim_);
        parallel_for(n, [&](std::size_t i) {
            const float* x = data + i * dim_;
            const float* c = coarse_centroids_.data() + coarse_assign[i] * dim_;
            float* r = residuals.data() + i * dim_;
            for (std::size_t j = 0; j < dim_; ++j) r[j] = x[j] - c[j];
        });
        pq_train = residuals.data();
    }

    // 3. Train one PQ codebook per subspace on the residual slice.
    //    Subspaces are independent → train them in parallel. Each thread keeps
    //    its own gather buffer; results are written to disjoint slices of
    //    pq_codebooks_, so no synchronization is needed.
    pq_codebooks_.assign(M_ * KSUB * dsub_, 0.0f);

    struct PqCtx {
        std::vector<float> sub_buf;
        std::vector<float> cb;
    };
    parallel_for<PqCtx>(M_,
        [n, dsub = dsub_]() {
            return PqCtx{std::vector<float>(static_cast<std::size_t>(n) * dsub),
                         std::vector<float>{}};
        },
        [&](std::size_t m, PqCtx& ctx) {
            for (std::size_t i = 0; i < n; ++i) {
                std::memcpy(ctx.sub_buf.data() + i * dsub_,
                            pq_train + i * dim_ + m * dsub_,
                            dsub_ * sizeof(float));
            }
            // nthreads=1: this loop is already one thread per subspace;
            // letting the inner kmeans spawn its own workers would
            // oversubscribe M * hardware_concurrency threads.
            kmeans(ctx.sub_buf.data(), n, dsub_, KSUB, kmeans_iters_,
                   seed_ + 1 + m, ctx.cb, /*out_assign=*/nullptr,
                   /*nthreads=*/1);
            std::memcpy(pq_codebooks_.data() + m * KSUB * dsub_,
                        ctx.cb.data(), KSUB * dsub_ * sizeof(float));
        });

    // Full state reset: re-training invalidates any previously encoded codes
    // (they reference the old PQ codebooks). Wiping inv_labels_/inv_codes_ but
    // not ntotal_ would silently drift label assignments, new add() calls
    // would start labels at the stale offset, leaving holes the user can't see.
    inv_labels_.assign(nlist_, {});
    inv_codes_.assign(nlist_, {});
    ntotal_ = 0;
    next_label_ = 0;
    rebuild_codebooks_T();
    if (metric_ == Metric::L2) {
        rebuild_precomputed_table();   // IP has no per-probe centroid term
    } else {
        precomp_.clear();
    }
    trained_ = true;
}

void IvfPqIndex::rebuild_precomputed_table() {
    // cb_norm[m*KSUB + k] = ||r_mk||^2, shared across all coarse centroids.
    std::vector<float> cb_norm(M_ * KSUB);
    for (std::size_t m = 0; m < M_; ++m) {
        const float* cb_m = pq_codebooks_.data() + m * KSUB * dsub_;
        for (std::size_t k = 0; k < KSUB; ++k) {
            const float* r = cb_m + k * dsub_;
            cb_norm[m * KSUB + k] = dot(r, r, dsub_);
        }
    }
    precomp_.assign(nlist_ * M_ * KSUB, 0.0f);
    parallel_for(nlist_, [&](std::size_t c) {
        const float* cc  = coarse_centroids_.data() + c * dim_;
        float*       dst = precomp_.data() + c * M_ * KSUB;
        for (std::size_t m = 0; m < M_; ++m) {
            const float* c_m  = cc + m * dsub_;
            const float* cb_m = pq_codebooks_.data() + m * KSUB * dsub_;
            for (std::size_t k = 0; k < KSUB; ++k) {
                dst[m * KSUB + k] = cb_norm[m * KSUB + k] +
                    2.0f * dot(c_m, cb_m + k * dsub_, dsub_);
            }
        }
    });
}

void IvfPqIndex::rebuild_codebooks_T() {
    // pq_codebooks_   layout: [m][k][j]      (KSUB rows of dsub each)
    // pq_codebooks_T_ layout: [m][j][k]      (dsub rows of KSUB each)
    // The transpose lets the LUT build inner loop stride across k contiguously.
    pq_codebooks_T_.assign(M_ * dsub_ * KSUB, 0.0f);
    for (std::size_t m = 0; m < M_; ++m) {
        const float* src = pq_codebooks_.data()  + m * KSUB * dsub_;
        float*       dst = pq_codebooks_T_.data() + m * dsub_ * KSUB;
        for (std::size_t k = 0; k < KSUB; ++k) {
            for (std::size_t j = 0; j < dsub_; ++j) {
                dst[j * KSUB + k] = src[k * dsub_ + j];
            }
        }
    }
}

void IvfPqIndex::encode_vector(const float* x, int32_t coarse_id,
                               uint8_t* out_code,
                               float* r_sub_scratch) const {
    const float* c = coarse_centroids_.data() + coarse_id * dim_;
    for (std::size_t m = 0; m < M_; ++m) {
        const float* x_m = x + m * dsub_;
        const float* src = x_m;
        if (metric_ == Metric::L2) {
            const float* c_m = c + m * dsub_;
            for (std::size_t j = 0; j < dsub_; ++j) {
                r_sub_scratch[j] = x_m[j] - c_m[j];
            }
            src = r_sub_scratch;
        }
        // Codebook assignment is always nearest-by-L2: PQ minimizes
        // reconstruction error of the encoded vector; the metric only
        // changes how reconstructed vectors are SCORED at search time.
        const float* cb = pq_codebooks_.data() + m * KSUB * dsub_;
        out_code[m] = static_cast<uint8_t>(
            nearest_centroid(src, cb, KSUB, dsub_));
    }
}

void IvfPqIndex::assign_and_encode(const float* data, std::size_t n,
                                   int32_t* coarse, uint8_t* codes) const {
    // Dominated by the nearest_centroid scan over nlist coarse centroids per
    // vector; each vector is independent and writes to disjoint slots, so
    // the result is identical to the serial encode.
    parallel_for<std::vector<float>>(n,
        [dsub = dsub_]() { return std::vector<float>(dsub); },  // residual scratch
        [&](std::size_t i, std::vector<float>& r_sub) {
            const float* x = data + i * dim_;
            coarse[i] = (metric_ == Metric::L2)
                ? nearest_centroid(x, coarse_centroids_.data(), nlist_, dim_)
                : nearest_centroid_ip(x, coarse_centroids_.data(), nlist_, dim_);
            encode_vector(x, coarse[i], codes + i * M_, r_sub.data());
        });
}

void IvfPqIndex::append(int32_t c, label_t label, const uint8_t* code) {
    inv_labels_[c].push_back(label);
    inv_codes_[c].insert(inv_codes_[c].end(), code, code + M_);
}

void IvfPqIndex::add(const float* data, std::size_t n) {
    if (!trained_) throw std::logic_error("IVF-PQ: add() before train()");
    if (n == 0) return;

    // Phase 1 (parallel): coarse-assign + PQ-encode every vector.
    std::vector<int32_t> coarse(n);
    std::vector<uint8_t> codes(n * M_);
    assign_and_encode(data, n, coarse.data(), codes.data());

    // Phase 2 (serial): append to the inverted lists in input order so labels
    // remain sequential and list contents deterministic.
    for (std::size_t i = 0; i < n; ++i) {
        append(coarse[i], next_label_++, codes.data() + i * M_);
    }
    ntotal_ += n;
}

void IvfPqIndex::update(const label_t* labels, const float* data, std::size_t n) {
    if (!trained_) throw std::logic_error("IVF-PQ: update() before train()");
    if (n == 0) return;
    const auto want = detail::index_update_batch(labels, n);

    // Locate every label (list, position) before touching anything.
    std::vector<std::pair<int32_t, std::size_t>> loc(n, {-1, 0});
    std::size_t found = 0;
    for (std::size_t c = 0; c < nlist_ && found < n; ++c) {
        const auto& ls = inv_labels_[c];
        for (std::size_t r = 0; r < ls.size(); ++r) {
            auto it = want.find(ls[r]);
            if (it != want.end()) {
                loc[it->second] = {static_cast<int32_t>(c), r};
                ++found;
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (loc[i].first < 0) detail::throw_missing_label(labels[i]);
    }

    std::vector<int32_t> coarse(n);
    std::vector<uint8_t> codes(n * M_);
    assign_and_encode(data, n, coarse.data(), codes.data());

    // Same list: overwrite the code in place. Otherwise move it. In-place
    // writes go first, while the recorded positions are still valid;
    // remove_ids then compacts the source lists.
    std::vector<label_t> moved;
    for (std::size_t i = 0; i < n; ++i) {
        const uint8_t* code = codes.data() + i * M_;
        if (coarse[i] == loc[i].first) {
            std::copy(code, code + M_,
                      inv_codes_[loc[i].first].begin() + loc[i].second * M_);
        } else {
            moved.push_back(labels[i]);
        }
    }
    if (moved.empty()) return;
    remove_ids(moved.data(), moved.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (coarse[i] != loc[i].first) {
            append(coarse[i], labels[i], codes.data() + i * M_);
        }
    }
    ntotal_ += moved.size();
}

std::size_t IvfPqIndex::remove_ids(const label_t* labels, std::size_t n) {
    std::unordered_set<label_t> kill(labels, labels + n);
    std::size_t removed = 0;
    for (std::size_t c = 0; c < nlist_; ++c) {
        auto& ls = inv_labels_[c];
        auto& cs = inv_codes_[c];
        std::size_t w = 0;
        for (std::size_t r = 0; r < ls.size(); ++r) {
            if (kill.count(ls[r])) { ++removed; continue; }
            if (w != r) {
                ls[w] = ls[r];
                std::copy(cs.begin() + r * M_, cs.begin() + (r + 1) * M_,
                          cs.begin() + w * M_);
            }
            ++w;
        }
        ls.resize(w);
        cs.resize(w * M_);
    }
    ntotal_ -= removed;
    return removed;
}

namespace {
struct HeapEntry {
    float distance;
    label_t label;
    bool operator<(const HeapEntry& o) const { return distance < o.distance; }
};
}  // namespace

void IvfPqIndex::search(const float* queries, std::size_t nq, std::size_t k,
                        std::size_t nprobe,
                        float* out_distances, label_t* out_labels) const {
    if (!trained_) throw std::logic_error("IVF-PQ: search() before train()");
    if (k == 0) return;  // bounded-heap path below assumes k >= 1.
    if (nprobe == 0) nprobe = 1;
    if (nprobe > nlist_) nprobe = nlist_;

    // Scratch buffers allocated once per thread, reused across that thread's
    // queries. ADC lookup table is M * KSUB floats, for M=8 that's 8KB,
    // fits in L1d.
    struct SearchCtx {
        std::vector<float>   coarse_d;   // per-query path only
        std::vector<int32_t> coarse_idx;
        std::vector<float>   qdot;   // -2 * <q_m, r_mk>, built once per query
        std::vector<float>   lut;
    };

    // Phase 1 of each slab: the coarse scan for a block of queries at once,
    // 4 queries x 4 centroids register-tiled (see coarse.hpp, at high dim
    // this scan is bandwidth-bound and dominates low-nprobe searches).
    // The phase-split batched scan trades an extra coarse-matrix round-trip
    // and a phase barrier for 4x less centroid traffic. Measured: +20-44%
    // at GIST (960-dim, 3.8 MB of centroid reads per query) and a 5-15%
    // LOSS at SIFT (128-dim, 0.5 MB) where the round-trip dominates the
    // saving. Gate on per-query coarse traffic.
    const bool batched_coarse = metric_ == Metric::L2 &&
        nlist_ * dim_ * sizeof(float) >= (std::size_t{1} << 20);
    std::vector<float> coarse_all(
        batched_coarse ? std::min(nq, kCoarseSlab) * nlist_ : 0);

    for (std::size_t s0 = 0; s0 < nq; s0 += kCoarseSlab) {
        const std::size_t sn = std::min(kCoarseSlab, nq - s0);
        if (batched_coarse) {
            coarse_scan_block(coarse_all.data(), queries + s0 * dim_, sn,
                              coarse_centroids_.data(), nlist_, dim_);
        }

    parallel_for<SearchCtx>(sn,
        [nlist = nlist_, M = M_]() {
            SearchCtx c;
            c.coarse_d.resize(nlist);
            c.coarse_idx.resize(nlist);
            c.qdot.resize(M * KSUB);
            c.lut.resize(M * KSUB);
            return c;
        },
        [&](std::size_t si, SearchCtx& ctx) {
            const std::size_t qi = s0 + si;
            auto& coarse_idx = ctx.coarse_idx;
            auto& qdot       = ctx.qdot;
            auto& lut        = ctx.lut;
        const float* q = queries + qi * dim_;

        // 1. Coarse distances: this query's row of the slab matrix when
        //    batched, else computed inline (the 128-dim regime).
        const float* coarse_d;
        if (batched_coarse) {
            coarse_d = coarse_all.data() + si * nlist_;
        } else if (metric_ == Metric::L2) {
            l2sq_ny(ctx.coarse_d.data(), q, coarse_centroids_.data(),
                    nlist_, dim_);
            coarse_d = ctx.coarse_d.data();
        } else {
            // IP: probe lists by max dot; store negated so smaller-is-better
            // machinery (sort, heap) is shared with L2.
            dot_ny(ctx.coarse_d.data(), q, coarse_centroids_.data(),
                   nlist_, dim_);
            for (std::size_t c = 0; c < nlist_; ++c) {
                ctx.coarse_d[c] = -ctx.coarse_d[c];
            }
            coarse_d = ctx.coarse_d.data();
        }
        for (std::size_t c = 0; c < nlist_; ++c) {
            coarse_idx[c] = static_cast<int32_t>(c);
        }
        if (nprobe < nlist_) {
            std::partial_sort(
                coarse_idx.begin(), coarse_idx.begin() + nprobe,
                coarse_idx.end(),
                [&](int32_t a, int32_t b) { return coarse_d[a] < coarse_d[b]; });
        } else {
            std::sort(coarse_idx.begin(), coarse_idx.end(),
                [&](int32_t a, int32_t b) { return coarse_d[a] < coarse_d[b]; });
        }

        // 2. Build the query dot-table ONCE: qdot[m*KSUB+k] = s * <q_m, r_mk>
        //    with s = -2 for the L2 expansion and s = -1 for IP (where the
        //    negated dot table IS the whole ADC table, no per-probe work).
        //    Same NEON tile (transposed codebooks, 16 entries in 4 registers
        //    across the dsub reduction), amortized across all probes.
        const float qdot_scale = (metric_ == Metric::L2) ? -2.0f : -1.0f;
        for (std::size_t m = 0; m < M_; ++m) {
            const float* q_m    = q + m * dsub_;
            const float* cb_T_m = pq_codebooks_T_.data() + m * dsub_ * KSUB;
            float*       qd_m   = qdot.data() + m * KSUB;

#if PROXIMA_USE_NEON
            for (std::size_t k_start = 0; k_start < KSUB; k_start += 16) {
                float32x4_t a0 = vdupq_n_f32(0.0f);
                float32x4_t a1 = vdupq_n_f32(0.0f);
                float32x4_t a2 = vdupq_n_f32(0.0f);
                float32x4_t a3 = vdupq_n_f32(0.0f);
                for (std::size_t j = 0; j < dsub_; ++j) {
                    float32x4_t q_v = vdupq_n_f32(q_m[j]);
                    const float* cb_row = cb_T_m + j * KSUB + k_start;
                    a0 = vfmaq_f32(a0, q_v, vld1q_f32(cb_row));
                    a1 = vfmaq_f32(a1, q_v, vld1q_f32(cb_row + 4));
                    a2 = vfmaq_f32(a2, q_v, vld1q_f32(cb_row + 8));
                    a3 = vfmaq_f32(a3, q_v, vld1q_f32(cb_row + 12));
                }
                const float32x4_t m2 = vdupq_n_f32(qdot_scale);
                vst1q_f32(qd_m + k_start,      vmulq_f32(a0, m2));
                vst1q_f32(qd_m + k_start +  4, vmulq_f32(a1, m2));
                vst1q_f32(qd_m + k_start +  8, vmulq_f32(a2, m2));
                vst1q_f32(qd_m + k_start + 12, vmulq_f32(a3, m2));
            }
#else
            std::memset(qd_m, 0, KSUB * sizeof(float));
            for (std::size_t j = 0; j < dsub_; ++j) {
                float q_j = q_m[j];
                const float* cb_row = cb_T_m + j * KSUB;
                for (std::size_t k = 0; k < KSUB; ++k) {
                    qd_m[k] += q_j * cb_row[k];
                }
            }
            for (std::size_t k = 0; k < KSUB; ++k) qd_m[k] *= qdot_scale;
#endif
        }

        std::priority_queue<HeapEntry> top;

        for (std::size_t p = 0; p < nprobe; ++p) {
            int32_t c = coarse_idx[p];

            // Empty list: skip BEFORE paying for the LUT merge.
            const auto& labels_c = inv_labels_[c];
            const auto& codes_c  = inv_codes_[c];
            const std::size_t n_c = labels_c.size();
            if (n_c == 0) continue;

            // 3. Per-probe ADC table.
            //    L2: streaming MERGE of two tables plus the coarse bias, 
            //      lut[m][k] = precomp[c][m][k] + qdot[m][k], with
            //      bias = ||q - c||^2 folded into subspace 0.
            //    IP: the (negated) query dot-table IS the table, no
            //      per-probe work at all; just point at it.
            const float* lut_p = qdot.data();   // IP: dot table IS the table
            if (metric_ == Metric::L2) {
            const float* pc = precomp_.data() +
                static_cast<std::size_t>(c) * M_ * KSUB;
            const float  bias = coarse_d[c];
            const std::size_t total = M_ * KSUB;
#if PROXIMA_USE_NEON
            for (std::size_t i = 0; i < total; i += 16) {
                vst1q_f32(lut.data() + i,
                          vaddq_f32(vld1q_f32(pc + i), vld1q_f32(qdot.data() + i)));
                vst1q_f32(lut.data() + i + 4,
                          vaddq_f32(vld1q_f32(pc + i + 4), vld1q_f32(qdot.data() + i + 4)));
                vst1q_f32(lut.data() + i + 8,
                          vaddq_f32(vld1q_f32(pc + i + 8), vld1q_f32(qdot.data() + i + 8)));
                vst1q_f32(lut.data() + i + 12,
                          vaddq_f32(vld1q_f32(pc + i + 12), vld1q_f32(qdot.data() + i + 12)));
            }
            const float32x4_t bias_v = vdupq_n_f32(bias);
            for (std::size_t k = 0; k < KSUB; k += 4) {
                vst1q_f32(lut.data() + k,
                          vaddq_f32(vld1q_f32(lut.data() + k), bias_v));
            }
#else
            for (std::size_t i = 0; i < total; ++i) lut[i] = pc[i] + qdot[i];
            for (std::size_t k = 0; k < KSUB; ++k) lut[k] += bias;
#endif
            lut_p = lut.data();
            }

            // 4. Scan the inverted list, summing the LUT for each code.
            //    Unrolled 4 codes wide: a single code's sum is a serial
            //    dependency chain (each += waits on an L1 load); four
            //    independent chains let the OoO core overlap the lookups.
            //    This is the same ILP trick FAISS uses in its 8-bit PQ
            //    scanner. Heap updates stay in list order -> results are
            //    identical to the one-by-one scan.
            auto consider = [&](float dist, std::size_t i) {
                if (top.size() < k) {
                    top.push({dist, labels_c[i]});
                } else if (dist < top.top().distance) {
                    top.pop();
                    top.push({dist, labels_c[i]});
                }
            };

            const uint8_t* codes_p = codes_c.data();
            std::size_t i = 0;
            for (; i + 4 <= n_c; i += 4) {
                const uint8_t* c0 = codes_p + i * M_;
                const uint8_t* c1 = c0 + M_;
                const uint8_t* c2 = c1 + M_;
                const uint8_t* c3 = c2 + M_;
                __builtin_prefetch(c3 + 4 * M_, 0, 0);
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (std::size_t m = 0; m < M_; ++m) {
                    const float* lut_m = lut_p + m * KSUB;
                    s0 += lut_m[c0[m]];
                    s1 += lut_m[c1[m]];
                    s2 += lut_m[c2[m]];
                    s3 += lut_m[c3[m]];
                }
                consider(s0, i);
                consider(s1, i + 1);
                consider(s2, i + 2);
                consider(s3, i + 3);
            }
            for (; i < n_c; ++i) {
                const uint8_t* code = codes_p + i * M_;
                float dist = 0.0f;
                for (std::size_t m = 0; m < M_; ++m) {
                    dist += lut_p[m * KSUB + code[m]];
                }
                consider(dist, i);
            }
        }

        const bool is_l2 = (metric_ == Metric::L2);
        std::size_t out_n = std::min(k, top.size());
        for (std::size_t j = 0; j < out_n; ++j) {
            std::size_t pos = out_n - 1 - j;
            const HeapEntry& e = top.top();
            out_distances[qi * k + pos] = is_l2 ? e.distance : -e.distance;
            out_labels[qi * k + pos]    = e.label;
            top.pop();
        }
        for (std::size_t j = out_n; j < k; ++j) {
            out_distances[qi * k + j] = -1.0f;
            out_labels[qi * k + j]    = -1;
        }
    });
    }  // slab loop
}

// ---- persistence ---------------------------------------------------------

void IvfPqIndex::save(const std::string& path) const {
    io::Writer w(path);
    w.write_magic("VIP1");
    w.write_pod<uint32_t>(2);                  // v2 adds the metric byte
    w.write_pod<uint8_t>(metric_ == Metric::L2 ? 0 : 1);
    w.write_pod<uint64_t>(dim_);
    w.write_pod<uint64_t>(nlist_);
    w.write_pod<uint64_t>(M_);
    w.write_pod<uint64_t>(dsub_);
    w.write_pod<uint8_t>(trained_ ? 1 : 0);
    w.write_pod<uint64_t>(ntotal_);

    if (trained_) {
        w.write_raw(coarse_centroids_.data(),
                    coarse_centroids_.size() * sizeof(float));
        w.write_raw(pq_codebooks_.data(),
                    pq_codebooks_.size() * sizeof(float));
        for (std::size_t i = 0; i < nlist_; ++i) {
            uint64_t n_i = inv_labels_[i].size();
            w.write_pod<uint64_t>(n_i);
            if (n_i > 0) {
                w.write_raw(inv_labels_[i].data(), n_i * sizeof(label_t));
                w.write_raw(inv_codes_[i].data(),  n_i * M_);
            }
        }
    }
}

IvfPqIndex IvfPqIndex::load(const std::string& path) {
    io::Reader r(path);
    r.check_magic("VIP1");
    uint32_t version = r.read_pod<uint32_t>();
    if (version != 1 && version != 2) {
        throw std::runtime_error("IvfPqIndex: unsupported file version");
    }
    // v1 files predate metric support and are always L2.
    Metric metric = Metric::L2;
    if (version == 2 && r.read_pod<uint8_t>() != 0) {
        metric = Metric::InnerProduct;
    }

    uint64_t dim    = r.read_pod<uint64_t>();
    uint64_t nlist  = r.read_pod<uint64_t>();
    uint64_t M      = r.read_pod<uint64_t>();
    uint64_t dsub   = r.read_pod<uint64_t>();
    uint8_t  tr     = r.read_pod<uint8_t>();
    uint64_t ntotal = r.read_pod<uint64_t>();

    IvfPqIndex idx(static_cast<std::size_t>(dim),
                   static_cast<std::size_t>(nlist),
                   static_cast<std::size_t>(M),
                   /*kmeans_iters=*/20, /*seed=*/0, metric);
    if (idx.dsub_ != dsub) {
        throw std::runtime_error("IvfPqIndex: dsub mismatch");
    }
    idx.ntotal_  = static_cast<std::size_t>(ntotal);
    idx.trained_ = (tr != 0);

    if (idx.trained_) {
        idx.coarse_centroids_.assign(
            static_cast<std::size_t>(nlist) * dim, 0.0f);
        r.read_raw(idx.coarse_centroids_.data(),
                   idx.coarse_centroids_.size() * sizeof(float));

        idx.pq_codebooks_.assign(M * KSUB * dsub, 0.0f);
        r.read_raw(idx.pq_codebooks_.data(),
                   idx.pq_codebooks_.size() * sizeof(float));
        idx.rebuild_codebooks_T();
        if (metric == Metric::L2) idx.rebuild_precomputed_table();

        idx.inv_labels_.assign(nlist, {});
        idx.inv_codes_.assign(nlist, {});
        for (std::size_t i = 0; i < nlist; ++i) {
            uint64_t n_i = r.read_pod<uint64_t>();
            idx.inv_labels_[i].resize(n_i);
            idx.inv_codes_[i].resize(n_i * M);
            if (n_i > 0) {
                r.read_raw(idx.inv_labels_[i].data(), n_i * sizeof(label_t));
                r.read_raw(idx.inv_codes_[i].data(),  n_i * M);
            }
        }
        // Labels are monotonic but holes may exist after removals; the next
        // fresh label is one past the maximum ever assigned.
        label_t mx = -1;
        for (const auto& ls : idx.inv_labels_)
            for (label_t l : ls) mx = std::max(mx, l);
        idx.next_label_ = mx + 1;
    }
    return idx;
}

std::vector<std::size_t> IvfPqIndex::list_sizes() const {
    std::vector<std::size_t> sizes(nlist_, 0);
    for (std::size_t c = 0; c < nlist_; ++c) sizes[c] = inv_labels_[c].size();
    return sizes;
}

}  // namespace proxima
