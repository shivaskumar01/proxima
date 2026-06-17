#pragma once

#include "vectordb/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vectordb {

// IVF-PQ "fast scan": 4-bit product quantization scanned with the NEON
// `tbl` instruction, the design FAISS ships as IndexIVFPQFastScan.
//
// Versus the 8-bit IvfPqIndex:
//   - KSUB = 16 codebook entries per subspace (codes are 4 bits), so at the
//     same byte budget you run 2x the subspaces: M=32 x 4-bit == 16 bytes,
//     comparable recall to M=16 x 8-bit.
//   - Codes are nibble-packed into blocks of 16 candidates laid out so one
//     vqtbl1q_u8 performs 16 LUT lookups in a single instruction (the 8-bit
//     scan does 16 scalar loads for the same work).
//   - The per-probe float LUT is quantized to uint8 with floor() against a
//     per-subspace bias, which makes every quantized score an UNDERESTIMATE
//     of the exact ADC distance. The SIMD scan therefore acts as a lossless
//     filter: only lanes whose underestimate beats the current top-k get
//     re-scored exactly against the float LUT, so results are exact ADC
//     top-k, quantization never costs recall.
//
// Shares the precomputed-table ADC expansion with IvfPqIndex for L2; for
// InnerProduct (raw-vector encoding, FAISS-style) the dot table is
// probe-independent, so the LUT is built AND quantized once per query.
// dim % M == 0; M even. Cosine = normalize vectors + queries, use "ip".
class IvfPqFastScan {
public:
    IvfPqFastScan(std::size_t dim,
                  std::size_t nlist,
                  std::size_t M,            // # 4-bit subquantizers (even)
                  std::size_t kmeans_iters = 20,
                  uint64_t    seed = 42,
                  Metric      metric = Metric::L2);

    void train(const float* data, std::size_t n);
    void add(const float* data, std::size_t n);

    void search(const float* queries, std::size_t nq, std::size_t k,
                std::size_t nprobe,
                float* out_distances, label_t* out_labels) const;

    // Physically remove vectors by label (lists are filtered and repacked).
    // Surviving labels unchanged; labels never reused. Returns count removed.
    std::size_t remove_ids(const label_t* labels, std::size_t n);

    bool        is_trained() const noexcept { return trained_; }
    std::size_t size() const noexcept { return ntotal_; }
    std::size_t dim()  const noexcept { return dim_; }
    std::size_t nlist() const noexcept { return nlist_; }
    std::size_t M()     const noexcept { return M_; }
    Metric      metric() const noexcept { return metric_; }

    std::vector<std::size_t> list_sizes() const;

    // Persistence. Magic "VFS1". Precomputed/transposed tables are rebuilt
    // on load, never serialized.
    void save(const std::string& path) const;
    static IvfPqFastScan load(const std::string& path);

private:
    static constexpr std::size_t KSUB = 16;   // 4-bit codes
    static constexpr std::size_t BLOCK = 16;  // candidates per tbl block

    // Bytes per block: 16 candidates x (M/2) packed bytes.
    std::size_t block_bytes() const noexcept { return BLOCK * (M_ / 2); }

    void encode_vector(const float* x, int32_t coarse_id,
                       uint8_t* out_code, float* r_sub_scratch) const;
    void rebuild_codebooks_T();
    void rebuild_precomputed_table();

    std::size_t dim_;
    std::size_t nlist_;
    std::size_t M_;
    std::size_t dsub_;
    std::size_t kmeans_iters_;
    uint64_t    seed_;
    Metric      metric_;

    bool        trained_ = false;
    std::size_t ntotal_  = 0;          // LIVE count
    label_t     next_label_ = 0;       // monotonic; derived from max on load

    std::vector<float> coarse_centroids_;     // nlist * dim
    std::vector<float> pq_codebooks_;         // M * KSUB * dsub
    std::vector<float> pq_codebooks_T_;       // M * dsub * KSUB
    std::vector<float> precomp_;              // nlist * M * KSUB

    // Inverted lists. labels in list order; codes nibble-packed by block:
    // block b spans bytes [b*block_bytes, ...); within it, subspace pair
    // p in [0, M/2) owns 16 bytes at p*16, and byte i holds
    // code[2p] | code[2p+1] << 4 for candidate b*16+i. Dead lanes in the
    // final partial block are zero-filled and never read (lane loops are
    // bounded by the live count).
    std::vector<std::vector<label_t>> inv_labels_;
    std::vector<std::vector<uint8_t>> inv_packed_;
};

}  // namespace vectordb
