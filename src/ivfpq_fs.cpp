#include "vectordb/ivfpq_fs.hpp"
#include "vectordb/coarse.hpp"
#include "vectordb/distance.hpp"
#include "vectordb/io.hpp"
#include "vectordb/kmeans.hpp"
#include "vectordb/parallel.hpp"

#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>

#include "vectordb/simd.hpp"

namespace vectordb {

IvfPqFastScan::IvfPqFastScan(std::size_t dim, std::size_t nlist, std::size_t M,
                             std::size_t kmeans_iters, uint64_t seed,
                             Metric metric)
    : dim_(dim), nlist_(nlist), M_(M),
      dsub_(M == 0 ? 0 : dim / M),
      kmeans_iters_(kmeans_iters),
      seed_(seed),
      metric_(metric) {
    if (dim == 0)   throw std::invalid_argument("IVF-PQ-FS: dim must be > 0");
    if (nlist == 0) throw std::invalid_argument("IVF-PQ-FS: nlist must be > 0");
    if (M == 0)     throw std::invalid_argument("IVF-PQ-FS: M must be > 0");
    if (M % 2 != 0) {
        throw std::invalid_argument("IVF-PQ-FS: M must be even (nibble packing)");
    }
    if (dim % M != 0) {
        throw std::invalid_argument("IVF-PQ-FS: dim must be divisible by M");
    }
    inv_labels_.assign(nlist_, {});
    inv_packed_.assign(nlist_, {});
}

void IvfPqFastScan::train(const float* data, std::size_t n) {
    if (n < nlist_) {
        throw std::invalid_argument("IVF-PQ-FS: training set smaller than nlist");
    }
    if (n < KSUB) {
        throw std::invalid_argument(
            "IVF-PQ-FS: training set smaller than KSUB (16)");
    }

    // 1. Coarse quantizer (kmeans parallelizes its assignment internally).
    std::vector<int32_t> coarse_assign;
    kmeans(data, n, dim_, nlist_, kmeans_iters_, seed_,
           coarse_centroids_, &coarse_assign, /*nthreads=*/0);

    // 2. PQ training input: residuals for L2, raw vectors for IP.
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

    // 3. One 16-entry codebook per subspace, trained in parallel across M.
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
            kmeans(ctx.sub_buf.data(), n, dsub_, KSUB, kmeans_iters_,
                   seed_ + 1 + m, ctx.cb, /*out_assign=*/nullptr,
                   /*nthreads=*/1);
            std::memcpy(pq_codebooks_.data() + m * KSUB * dsub_,
                        ctx.cb.data(), KSUB * dsub_ * sizeof(float));
        });

    // Re-training invalidates all previously encoded codes.
    inv_labels_.assign(nlist_, {});
    inv_packed_.assign(nlist_, {});
    ntotal_ = 0;
    rebuild_codebooks_T();
    if (metric_ == Metric::L2) {
        rebuild_precomputed_table();
    } else {
        precomp_.clear();
    }
    trained_ = true;
}

void IvfPqFastScan::rebuild_codebooks_T() {
    pq_codebooks_T_.assign(M_ * dsub_ * KSUB, 0.0f);
    for (std::size_t m = 0; m < M_; ++m) {
        const float* src = pq_codebooks_.data()   + m * KSUB * dsub_;
        float*       dst = pq_codebooks_T_.data() + m * dsub_ * KSUB;
        for (std::size_t k = 0; k < KSUB; ++k) {
            for (std::size_t j = 0; j < dsub_; ++j) {
                dst[j * KSUB + k] = src[k * dsub_ + j];
            }
        }
    }
}

void IvfPqFastScan::rebuild_precomputed_table() {
    // Same ADC expansion as IvfPqIndex: precomp[c][m][k] = ||r||^2 + 2<c_m,r>.
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

void IvfPqFastScan::encode_vector(const float* x, int32_t coarse_id,
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
        const float* cb = pq_codebooks_.data() + m * KSUB * dsub_;
        out_code[m] = static_cast<uint8_t>(
            nearest_centroid(src, cb, KSUB, dsub_));
    }
}

void IvfPqFastScan::add(const float* data, std::size_t n) {
    if (!trained_) throw std::logic_error("IVF-PQ-FS: add() before train()");
    if (n == 0) return;

    // Phase 1 (parallel): coarse-assign + encode into per-vector slots.
    std::vector<int32_t> coarse(n);
    std::vector<uint8_t> codes(n * M_);
    parallel_for<std::vector<float>>(n,
        [dsub = dsub_]() { return std::vector<float>(dsub); },
        [&](std::size_t i, std::vector<float>& r_sub) {
            const float* x = data + i * dim_;
            coarse[i] = (metric_ == Metric::L2)
                ? nearest_centroid(x, coarse_centroids_.data(), nlist_, dim_)
                : nearest_centroid_ip(x, coarse_centroids_.data(), nlist_, dim_);
            encode_vector(x, coarse[i], codes.data() + i * M_, r_sub.data());
        });

    // Phase 2 (serial): nibble-pack into the block layout in input order.
    const std::size_t bb = block_bytes();
    for (std::size_t i = 0; i < n; ++i) {
        const int32_t c = coarse[i];
        auto& labels = inv_labels_[c];
        auto& packed = inv_packed_[c];
        const std::size_t lane = labels.size() % BLOCK;
        if (lane == 0) packed.resize(packed.size() + bb, 0);
        const std::size_t base = (labels.size() / BLOCK) * bb;
        const uint8_t* code = codes.data() + i * M_;
        for (std::size_t p = 0; p < M_ / 2; ++p) {
            packed[base + p * BLOCK + lane] =
                static_cast<uint8_t>(code[2 * p] | (code[2 * p + 1] << 4));
        }
        labels.push_back(static_cast<label_t>(ntotal_ + i));
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

void IvfPqFastScan::search(const float* queries, std::size_t nq, std::size_t k,
                           std::size_t nprobe,
                           float* out_distances, label_t* out_labels) const {
    if (!trained_) throw std::logic_error("IVF-PQ-FS: search() before train()");
    if (k == 0) return;
    if (nprobe == 0) nprobe = 1;
    if (nprobe > nlist_) nprobe = nlist_;

    struct SearchCtx {
        std::vector<float>   coarse_d;   // per-query path only
        std::vector<int32_t> coarse_idx;
        std::vector<float>   qdot;    // -2<q_m, r_mk>, M*16
        std::vector<float>   lut_f;   // exact float LUT for current probe
        std::vector<uint8_t> lut8;    // quantized LUT (underestimates)
        std::vector<float>   bmin;    // per-subspace quantization bias
    };

    // Slab-batched coarse scan, 4 queries x 4 centroids register-tiled.
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
            c.lut_f.resize(M * KSUB);
            c.lut8.resize(M * KSUB);
            c.bmin.resize(M);
            return c;
        },
        [&](std::size_t si, SearchCtx& ctx) {
        const std::size_t qi = s0 + si;
        const float* q = queries + qi * dim_;
        auto& coarse_idx = ctx.coarse_idx;

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

        // 2. Query dot-table, once per query: qdot[m*16+k] = -2<q_m, r_mk>.
        //    KSUB=16 fits exactly one 4x4-register tile per subspace.
        const float qdot_scale = (metric_ == Metric::L2) ? -2.0f : -1.0f;
        for (std::size_t m = 0; m < M_; ++m) {
            const float* q_m    = q + m * dsub_;
            const float* cb_T_m = pq_codebooks_T_.data() + m * dsub_ * KSUB;
            float*       qd_m   = ctx.qdot.data() + m * KSUB;
#if VECTORDB_USE_NEON
            float32x4_t a0 = vdupq_n_f32(0.0f);
            float32x4_t a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f);
            float32x4_t a3 = vdupq_n_f32(0.0f);
            for (std::size_t j = 0; j < dsub_; ++j) {
                float32x4_t q_v = vdupq_n_f32(q_m[j]);
                const float* row = cb_T_m + j * KSUB;
                a0 = vfmaq_f32(a0, q_v, vld1q_f32(row));
                a1 = vfmaq_f32(a1, q_v, vld1q_f32(row + 4));
                a2 = vfmaq_f32(a2, q_v, vld1q_f32(row + 8));
                a3 = vfmaq_f32(a3, q_v, vld1q_f32(row + 12));
            }
            const float32x4_t m2 = vdupq_n_f32(qdot_scale);
            vst1q_f32(qd_m,      vmulq_f32(a0, m2));
            vst1q_f32(qd_m +  4, vmulq_f32(a1, m2));
            vst1q_f32(qd_m +  8, vmulq_f32(a2, m2));
            vst1q_f32(qd_m + 12, vmulq_f32(a3, m2));
#else
            std::memset(qd_m, 0, KSUB * sizeof(float));
            for (std::size_t j = 0; j < dsub_; ++j) {
                float q_j = q_m[j];
                const float* row = cb_T_m + j * KSUB;
                for (std::size_t kk = 0; kk < KSUB; ++kk) qd_m[kk] += q_j * row[kk];
            }
            for (std::size_t kk = 0; kk < KSUB; ++kk) qd_m[kk] *= qdot_scale;
#endif
        }

        std::priority_queue<HeapEntry> top;

        // ADC table + uint8 quantization (per-subspace bias = min, shared
        // scale = max subspace range / 255, floor rounding → quantized sums
        // UNDERESTIMATE exact distances, so the SIMD pass is a lossless
        // filter). For L2 both happen per probe (the table depends on the
        // probed centroid); for IP the negated dot table IS the table and
        // is probe-independent — built and quantized ONCE per query.
        const float* lut_f = ctx.qdot.data();
        uint8_t* lut8 = ctx.lut8.data();
        float qdelta = 1.0f, inv_delta = 1.0f, qbase = 0.0f;
        auto quantize = [&](const float* lf) {
            float* bmin = ctx.bmin.data();
            float maxrange = 0.0f;
            for (std::size_t m = 0; m < M_; ++m) {
                const float* lm = lf + m * KSUB;
                float lo = lm[0], hi = lm[0];
                for (std::size_t kk = 1; kk < KSUB; ++kk) {
                    lo = std::min(lo, lm[kk]);
                    hi = std::max(hi, lm[kk]);
                }
                bmin[m] = lo;
                maxrange = std::max(maxrange, hi - lo);
            }
            qdelta = maxrange > 0.0f ? maxrange / 255.0f : 1.0f;
            inv_delta = 1.0f / qdelta;
            qbase = 0.0f;
            for (std::size_t m = 0; m < M_; ++m) qbase += bmin[m];
            for (std::size_t m = 0; m < M_; ++m) {
                const float* lm = lf + m * KSUB;
                uint8_t* l8 = lut8 + m * KSUB;
                for (std::size_t kk = 0; kk < KSUB; ++kk) {
                    int v = static_cast<int>((lm[kk] - bmin[m]) * inv_delta);
                    l8[kk] = static_cast<uint8_t>(v > 255 ? 255 : (v < 0 ? 0 : v));
                }
            }
        };
        if (metric_ != Metric::L2) quantize(lut_f);

        for (std::size_t p = 0; p < nprobe; ++p) {
            const int32_t c = coarse_idx[p];
            const auto& labels_c = inv_labels_[c];
            const auto& packed_c = inv_packed_[c];
            const std::size_t n_c = labels_c.size();
            if (n_c == 0) continue;

            if (metric_ == Metric::L2) {
                // Exact float LUT: precomp + qdot, coarse bias folded into
                // subspace 0; then re-quantize for this probe.
                const float* pc = precomp_.data() +
                    static_cast<std::size_t>(c) * M_ * KSUB;
                const float bias = coarse_d[c];
                float* lf = ctx.lut_f.data();
                for (std::size_t i = 0; i < M_ * KSUB; ++i) {
                    lf[i] = pc[i] + ctx.qdot[i];
                }
                for (std::size_t kk = 0; kk < KSUB; ++kk) lf[kk] += bias;
                lut_f = lf;
                quantize(lut_f);
            }

            auto exact_rescore = [&](const uint8_t* bp, std::size_t lane) {
                float dist = 0.0f;
                for (std::size_t pp = 0; pp < M_ / 2; ++pp) {
                    const uint8_t byte = bp[pp * BLOCK + lane];
                    dist += lut_f[(2 * pp)     * KSUB + (byte & 0x0F)];
                    dist += lut_f[(2 * pp + 1) * KSUB + (byte >> 4)];
                }
                return dist;
            };
            auto qthresh = [&]() -> uint32_t {
                if (top.size() < k) return 0xFFFFFFFFu;
                float t = (top.top().distance - qbase) * inv_delta;
                if (!(t >= 0.0f)) return 0;
                if (t >= 65535.0f) return 0xFFFFFFFFu;
                return static_cast<uint32_t>(t);
            };

#if VECTORDB_USE_NEON
            uint32_t thr = qthresh();   // scalar path scans exactly, no filter
#endif
            const std::size_t bb = block_bytes();
            const std::size_t nblocks = (n_c + BLOCK - 1) / BLOCK;
            for (std::size_t b = 0; b < nblocks; ++b) {
                const uint8_t* bp = packed_c.data() + b * bb;
                const std::size_t lanes = std::min(BLOCK, n_c - b * BLOCK);
#if VECTORDB_USE_NEON
                // 5. The fast scan: per subspace pair, one load + two tbl
                //    lookups score 16 candidates; u16 lanes accumulate.
                __builtin_prefetch(bp + bb, 0, 0);   // next block's codes
                uint16x8_t acc_lo = vdupq_n_u16(0);
                uint16x8_t acc_hi = vdupq_n_u16(0);
                const uint8x16_t nib_mask = vdupq_n_u8(0x0F);
                for (std::size_t pp = 0; pp < M_ / 2; ++pp) {
                    uint8x16_t bytes = vld1q_u8(bp + pp * BLOCK);
                    uint8x16_t lo_codes = vandq_u8(bytes, nib_mask);
                    uint8x16_t hi_codes = vshrq_n_u8(bytes, 4);
                    uint8x16_t t0 = vqtbl1q_u8(vld1q_u8(lut8 + (2 * pp) * KSUB),
                                               lo_codes);
                    uint8x16_t t1 = vqtbl1q_u8(vld1q_u8(lut8 + (2 * pp + 1) * KSUB),
                                               hi_codes);
                    acc_lo = vaddw_u8(acc_lo, vget_low_u8(t0));
                    acc_hi = vaddw_u8(acc_hi, vget_high_u8(t0));
                    acc_lo = vaddw_u8(acc_lo, vget_low_u8(t1));
                    acc_hi = vaddw_u8(acc_hi, vget_high_u8(t1));
                }

                // 6. SIMD survivor mask: compare all 16 lanes against the
                //    quantized threshold at once. The common steady state —
                //    a tight top-k and no survivors in the block — is one
                //    compare + one reduction + one branch, with no per-lane
                //    work at all. Survivor lanes (rare) are walked via the
                //    narrowed bitmask instead of a 16-iteration loop.
                const uint16x8_t thr_v =
                    vdupq_n_u16(thr > 0xFFFFu ? 0xFFFFu
                                              : static_cast<uint16_t>(thr));
                const uint16x8_t le_lo = vcleq_u16(acc_lo, thr_v);
                const uint16x8_t le_hi = vcleq_u16(acc_hi, thr_v);
                if (vmaxvq_u16(vorrq_u16(le_lo, le_hi)) == 0) continue;

                auto rescore_lane = [&](std::size_t i) {
                    float dist = exact_rescore(bp, i);
                    if (top.size() < k) {
                        top.push({dist, labels_c[b * BLOCK + i]});
                        thr = qthresh();
                    } else if (dist < top.top().distance) {
                        top.pop();
                        top.push({dist, labels_c[b * BLOCK + i]});
                        thr = qthresh();
                    }
                };

                if (lanes == BLOCK) {
                    // vshrn narrows each 0xFFFF/0x0000 lane to a 0xFF/0x00
                    // byte; reinterpret as u64 and walk set bytes.
                    uint64_t m_lo = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(le_lo, 4)), 0);
                    uint64_t m_hi = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(le_hi, 4)), 0);
                    while (m_lo) {
                        std::size_t i = static_cast<std::size_t>(
                            __builtin_ctzll(m_lo)) >> 3;
                        m_lo &= ~(0xFFull << (i * 8));
                        rescore_lane(i);
                    }
                    while (m_hi) {
                        std::size_t i = static_cast<std::size_t>(
                            __builtin_ctzll(m_hi)) >> 3;
                        m_hi &= ~(0xFFull << (i * 8));
                        rescore_lane(8 + i);
                    }
                } else {
                    // Partial tail block (at most one per list): dead lanes
                    // hold zero-codes, so bound the walk by the live count.
                    uint16_t vals[BLOCK];
                    vst1q_u16(vals, acc_lo);
                    vst1q_u16(vals + 8, acc_hi);
                    for (std::size_t i = 0; i < lanes; ++i) {
                        if (vals[i] <= thr) rescore_lane(i);
                    }
                }
#else
                // Scalar fallback: exact float scan, no quantized filter.
                for (std::size_t i = 0; i < lanes; ++i) {
                    float dist = exact_rescore(bp, i);
                    if (top.size() < k) {
                        top.push({dist, labels_c[b * BLOCK + i]});
                    } else if (dist < top.top().distance) {
                        top.pop();
                        top.push({dist, labels_c[b * BLOCK + i]});
                    }
                }
#endif
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

void IvfPqFastScan::save(const std::string& path) const {
    io::Writer w(path);
    w.write_magic("VFS1");
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
                w.write_raw(inv_packed_[i].data(), inv_packed_[i].size());
            }
        }
    }
}

IvfPqFastScan IvfPqFastScan::load(const std::string& path) {
    io::Reader r(path);
    r.check_magic("VFS1");
    uint32_t version = r.read_pod<uint32_t>();
    if (version != 1 && version != 2) {
        throw std::runtime_error("IvfPqFastScan: unsupported file version");
    }
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

    IvfPqFastScan idx(static_cast<std::size_t>(dim),
                      static_cast<std::size_t>(nlist),
                      static_cast<std::size_t>(M),
                      /*kmeans_iters=*/20, /*seed=*/0, metric);
    if (idx.dsub_ != dsub) {
        throw std::runtime_error("IvfPqFastScan: dsub mismatch");
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

        const std::size_t bb = idx.block_bytes();
        idx.inv_labels_.assign(nlist, {});
        idx.inv_packed_.assign(nlist, {});
        for (std::size_t i = 0; i < nlist; ++i) {
            uint64_t n_i = r.read_pod<uint64_t>();
            idx.inv_labels_[i].resize(n_i);
            idx.inv_packed_[i].resize(((n_i + BLOCK - 1) / BLOCK) * bb);
            if (n_i > 0) {
                r.read_raw(idx.inv_labels_[i].data(), n_i * sizeof(label_t));
                r.read_raw(idx.inv_packed_[i].data(), idx.inv_packed_[i].size());
            }
        }
    }
    return idx;
}

std::vector<std::size_t> IvfPqFastScan::list_sizes() const {
    std::vector<std::size_t> sizes(nlist_, 0);
    for (std::size_t c = 0; c < nlist_; ++c) sizes[c] = inv_labels_[c].size();
    return sizes;
}

}  // namespace vectordb
