#include "vectordb/ivfpq.hpp"
#include "vectordb/distance.hpp"
#include "vectordb/io.hpp"
#include "vectordb/kmeans.hpp"
#include "vectordb/parallel.hpp"

#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace vectordb {

IvfPqIndex::IvfPqIndex(std::size_t dim, std::size_t nlist, std::size_t M,
                       std::size_t kmeans_iters, uint64_t seed)
    : dim_(dim), nlist_(nlist), M_(M),
      dsub_(M == 0 ? 0 : dim / M),
      kmeans_iters_(kmeans_iters),
      seed_(seed) {
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
    // codebook than the M*KSUB*dsub_ memcpy below assumes — silent OOB read.
    if (n < KSUB) {
        throw std::invalid_argument(
            "IVF-PQ: training set smaller than KSUB (256); PQ codebooks would be undertrained");
    }

    // 1. Train coarse quantizer on raw vectors.
    std::vector<int32_t> coarse_assign;
    kmeans(data, n, dim_, nlist_, kmeans_iters_, seed_,
           coarse_centroids_, &coarse_assign);

    // 2. Compute residuals: r_i = x_i - coarse_centroid[assign_i].
    //    Stored densely as n * dim, then sliced per subspace for PQ training.
    std::vector<float> residuals(static_cast<std::size_t>(n) * dim_);
    for (std::size_t i = 0; i < n; ++i) {
        const float* x = data + i * dim_;
        const float* c = coarse_centroids_.data() + coarse_assign[i] * dim_;
        float* r = residuals.data() + i * dim_;
        for (std::size_t j = 0; j < dim_; ++j) r[j] = x[j] - c[j];
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
                            residuals.data() + i * dim_ + m * dsub_,
                            dsub_ * sizeof(float));
            }
            kmeans(ctx.sub_buf.data(), n, dsub_, KSUB, kmeans_iters_,
                   seed_ + 1 + m, ctx.cb, /*out_assign=*/nullptr);
            std::memcpy(pq_codebooks_.data() + m * KSUB * dsub_,
                        ctx.cb.data(), KSUB * dsub_ * sizeof(float));
        });

    inv_labels_.assign(nlist_, {});
    inv_codes_.assign(nlist_, {});
    rebuild_codebooks_T();
    trained_ = true;
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
        const float* c_m = c + m * dsub_;
        for (std::size_t j = 0; j < dsub_; ++j) {
            r_sub_scratch[j] = x_m[j] - c_m[j];
        }
        const float* cb = pq_codebooks_.data() + m * KSUB * dsub_;
        out_code[m] = static_cast<uint8_t>(
            nearest_centroid(r_sub_scratch, cb, KSUB, dsub_));
    }
}

void IvfPqIndex::add(const float* data, std::size_t n) {
    if (!trained_) throw std::logic_error("IVF-PQ: add() before train()");

    std::vector<uint8_t> code(M_);
    std::vector<float>   r_sub(dsub_);   // heap-allocated, reused per add() call
    for (std::size_t i = 0; i < n; ++i) {
        const float* x = data + i * dim_;
        int32_t coarse = nearest_centroid(x, coarse_centroids_.data(),
                                          nlist_, dim_);
        encode_vector(x, coarse, code.data(), r_sub.data());

        label_t lbl = static_cast<label_t>(ntotal_ + i);
        inv_labels_[coarse].push_back(lbl);
        auto& codes = inv_codes_[coarse];
        codes.insert(codes.end(), code.begin(), code.end());
    }
    ntotal_ += n;
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
    // queries. ADC lookup table is M * KSUB floats — for M=8 that's 8KB,
    // fits in L1d.
    struct SearchCtx {
        std::vector<float>   coarse_d;
        std::vector<int32_t> coarse_idx;
        std::vector<float>   residual;
        std::vector<float>   lut;
    };
    parallel_for<SearchCtx>(nq,
        [nlist = nlist_, dim = dim_, M = M_]() {
            SearchCtx c;
            c.coarse_d.resize(nlist);
            c.coarse_idx.resize(nlist);
            c.residual.resize(dim);
            c.lut.resize(M * KSUB);
            return c;
        },
        [&](std::size_t qi, SearchCtx& ctx) {
            auto& coarse_d   = ctx.coarse_d;
            auto& coarse_idx = ctx.coarse_idx;
            auto& residual   = ctx.residual;
            auto& lut        = ctx.lut;
        const float* q = queries + qi * dim_;

        // 1. Distance from q to every coarse centroid; take top nprobe.
        for (std::size_t c = 0; c < nlist_; ++c) {
            coarse_d[c] = l2sq(q, coarse_centroids_.data() + c * dim_, dim_);
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

        std::priority_queue<HeapEntry> top;

        for (std::size_t p = 0; p < nprobe; ++p) {
            int32_t c = coarse_idx[p];
            const float* cv = coarse_centroids_.data() + c * dim_;

            // 2. Residual = q - centroid.
            for (std::size_t j = 0; j < dim_; ++j) residual[j] = q[j] - cv[j];

            // 3. Build ADC lookup table for this list.
            //    LUT[m*KSUB + k] = ||residual_m - codebook[m][k]||^2.
            //    The transposed codebook lets us stream 16 LUT entries at a
            //    time in 4 NEON registers, accumulating across all dsub
            //    dimensions before storing — no LUT load/store in the inner
            //    loop. KSUB=256 divides evenly by 16, no tail handling.
            for (std::size_t m = 0; m < M_; ++m) {
                const float* r_m    = residual.data() + m * dsub_;
                const float* cb_T_m = pq_codebooks_T_.data() + m * dsub_ * KSUB;
                float*       lut_m  = lut.data() + m * KSUB;

#if defined(__ARM_NEON)
                for (std::size_t k_start = 0; k_start < KSUB; k_start += 16) {
                    float32x4_t lut0 = vdupq_n_f32(0.0f);
                    float32x4_t lut1 = vdupq_n_f32(0.0f);
                    float32x4_t lut2 = vdupq_n_f32(0.0f);
                    float32x4_t lut3 = vdupq_n_f32(0.0f);
                    for (std::size_t j = 0; j < dsub_; ++j) {
                        float32x4_t r_v = vdupq_n_f32(r_m[j]);
                        const float* cb_row = cb_T_m + j * KSUB + k_start;
                        float32x4_t cb0 = vld1q_f32(cb_row);
                        float32x4_t cb1 = vld1q_f32(cb_row + 4);
                        float32x4_t cb2 = vld1q_f32(cb_row + 8);
                        float32x4_t cb3 = vld1q_f32(cb_row + 12);
                        float32x4_t d0 = vsubq_f32(cb0, r_v);
                        float32x4_t d1 = vsubq_f32(cb1, r_v);
                        float32x4_t d2 = vsubq_f32(cb2, r_v);
                        float32x4_t d3 = vsubq_f32(cb3, r_v);
                        lut0 = vfmaq_f32(lut0, d0, d0);
                        lut1 = vfmaq_f32(lut1, d1, d1);
                        lut2 = vfmaq_f32(lut2, d2, d2);
                        lut3 = vfmaq_f32(lut3, d3, d3);
                    }
                    vst1q_f32(lut_m + k_start,      lut0);
                    vst1q_f32(lut_m + k_start +  4, lut1);
                    vst1q_f32(lut_m + k_start +  8, lut2);
                    vst1q_f32(lut_m + k_start + 12, lut3);
                }
#else
                std::memset(lut_m, 0, KSUB * sizeof(float));
                for (std::size_t j = 0; j < dsub_; ++j) {
                    float r_j = r_m[j];
                    const float* cb_row = cb_T_m + j * KSUB;
                    for (std::size_t k = 0; k < KSUB; ++k) {
                        float diff = cb_row[k] - r_j;
                        lut_m[k] += diff * diff;
                    }
                }
#endif
            }

            // 4. Scan the inverted list, summing the LUT for each code.
            const auto& labels_c = inv_labels_[c];
            const auto& codes_c  = inv_codes_[c];
            std::size_t n_c = labels_c.size();
            for (std::size_t i = 0; i < n_c; ++i) {
                const uint8_t* code = codes_c.data() + i * M_;
                float dist = 0.0f;
                for (std::size_t m = 0; m < M_; ++m) {
                    dist += lut[m * KSUB + code[m]];
                }
                if (top.size() < k) {
                    top.push({dist, labels_c[i]});
                } else if (dist < top.top().distance) {
                    top.pop();
                    top.push({dist, labels_c[i]});
                }
            }
        }

        std::size_t out_n = std::min(k, top.size());
        for (std::size_t j = 0; j < out_n; ++j) {
            std::size_t pos = out_n - 1 - j;
            const HeapEntry& e = top.top();
            out_distances[qi * k + pos] = e.distance;
            out_labels[qi * k + pos]    = e.label;
            top.pop();
        }
        for (std::size_t j = out_n; j < k; ++j) {
            out_distances[qi * k + j] = -1.0f;
            out_labels[qi * k + j]    = -1;
        }
    });
}

// ---- persistence ---------------------------------------------------------

void IvfPqIndex::save(const std::string& path) const {
    io::Writer w(path);
    w.write_magic("VIP1");
    w.write_pod<uint32_t>(1);
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
    if (version != 1) throw std::runtime_error("IvfPqIndex: unsupported file version");

    uint64_t dim    = r.read_pod<uint64_t>();
    uint64_t nlist  = r.read_pod<uint64_t>();
    uint64_t M      = r.read_pod<uint64_t>();
    uint64_t dsub   = r.read_pod<uint64_t>();
    uint8_t  tr     = r.read_pod<uint8_t>();
    uint64_t ntotal = r.read_pod<uint64_t>();

    IvfPqIndex idx(static_cast<std::size_t>(dim),
                   static_cast<std::size_t>(nlist),
                   static_cast<std::size_t>(M),
                   /*kmeans_iters=*/20, /*seed=*/0);
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
    }
    return idx;
}

std::vector<std::size_t> IvfPqIndex::list_sizes() const {
    std::vector<std::size_t> sizes(nlist_, 0);
    for (std::size_t c = 0; c < nlist_; ++c) sizes[c] = inv_labels_[c].size();
    return sizes;
}

}  // namespace vectordb
