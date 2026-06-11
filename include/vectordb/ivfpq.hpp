#pragma once

#include "vectordb/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vectordb {

// IVF-PQ: coarse Inverted-File partitioning + Product Quantization on the
// residuals. Memory ~ M bytes per vector; search is dominated by an L1-resident
// lookup-table sum over M subspaces per probed list entry.
//
// Constraints:
//   - dim must be divisible by M (so each subspace has dsub = dim / M dims)
//   - ksub fixed at 256 (one byte per subspace code)
//   - L2 metric only in this version (residual-IP requires norms bookkeeping)
class IvfPqIndex {
public:
    IvfPqIndex(std::size_t dim,
               std::size_t nlist,        // # coarse centroids (e.g. 1024)
               std::size_t M,            // # PQ subspaces (e.g. 8 or 16)
               std::size_t kmeans_iters = 20,
               uint64_t    seed = 42);

    // Train both the coarse quantizer and the PQ codebooks.
    void train(const float* data, std::size_t n);

    // Add vectors after training. Each vector is encoded into M bytes.
    void add(const float* data, std::size_t n);

    // Search top-k. nprobe controls the recall/QPS knob: lists scanned per query.
    void search(const float* queries, std::size_t nq, std::size_t k,
                std::size_t nprobe,
                float* out_distances, label_t* out_labels) const;

    bool        is_trained() const noexcept { return trained_; }
    std::size_t size() const noexcept { return ntotal_; }
    std::size_t dim()  const noexcept { return dim_; }
    std::size_t nlist() const noexcept { return nlist_; }
    std::size_t M()     const noexcept { return M_; }

    // Diagnostic: per-list vector count.
    std::vector<std::size_t> list_sizes() const;

    // Persistence. Format: magic "VIP1", u32 version, hyperparams, coarse
    // centroids, PQ codebooks, then per-list (labels, codes).
    void save(const std::string& path) const;
    static IvfPqIndex load(const std::string& path);

private:
    static constexpr std::size_t KSUB = 256;

    // Caller passes a scratch buffer of size >= dsub_ for residual computation;
    // letting the caller own it lets add() reuse one buffer across many vectors
    // and avoids a fixed-size stack array (which used to silently overflow at
    // dsub > 64 — bug found on GIST1M, M=8, dsub=120).
    void encode_vector(const float* x, int32_t coarse_id,
                       uint8_t* out_code, float* r_sub_scratch) const;

    std::size_t dim_;
    std::size_t nlist_;
    std::size_t M_;
    std::size_t dsub_;
    std::size_t kmeans_iters_;
    uint64_t    seed_;

    bool        trained_ = false;
    std::size_t ntotal_  = 0;

    // Coarse: nlist * dim, row-major.
    std::vector<float> coarse_centroids_;

    // PQ codebooks: M * KSUB * dsub. Indexed as [m*KSUB*dsub + k*dsub + j].
    // Used by encode_vector (one centroid at a time -> standard layout best).
    std::vector<float> pq_codebooks_;

    // Transposed PQ codebooks: M * dsub * KSUB. Indexed as
    // [m*dsub*KSUB + j*KSUB + k]. Used by the per-query dot-table build so
    // the inner loop strides contiguously across all 256 codes for one
    // (m, j) pair — enables a 4-wide NEON FMA tile.
    std::vector<float> pq_codebooks_T_;
    void rebuild_codebooks_T();

    // Precomputed ADC term (FAISS's use_precomputed_table): expanding
    //   ||(q - c) - r||^2 = ||q - c||^2 + (||r||^2 + 2<c_m, r>) - 2<q_m, r>
    // the middle term depends only on (coarse centroid, codebook entry).
    // precomp_[c*M*KSUB + m*KSUB + k] = ||r_mk||^2 + 2<c_m, r_mk>.
    // ||q - c||^2 is the coarse distance (already computed by the coarse
    // scan) and folds in as a per-probe scalar; -2<q_m, r_mk> is built once
    // per QUERY. The per-probe LUT build drops from O(M*KSUB*dsub) FLOPs to
    // an O(M*KSUB) table merge — the dsub factor (120 on GIST1M with M=8)
    // is why FAISS used to win GIST IVF-PQ at every nprobe.
    // Size nlist*M*KSUB floats (8-16 MB typical); rebuilt by train()/load(),
    // never serialized.
    std::vector<float> precomp_;
    void rebuild_precomputed_table();

    // Inverted lists. Parallel arrays so labels and codes each stream
    // contiguously during scan; interleaving (label,code,label,code) would
    // pollute the cache with the unused label bytes during PQ summation.
    std::vector<std::vector<label_t>> inv_labels_;
    std::vector<std::vector<uint8_t>> inv_codes_;   // each: n_list * M bytes
};

}  // namespace vectordb
