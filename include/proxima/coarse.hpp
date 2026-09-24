#pragma once

// Query-batched coarse-quantizer scan shared by the IVF indexes.
//
// The per-query scan (l2sq_ny inside each query's worker) reloads every
// centroid row once PER QUERY: nq * nlist * dim * 4 bytes of traffic. At
// GIST scale (960-dim, nlist=1000) that is ~3.8 GB per 1000-query batch and
// the scan is flatly bandwidth-bound, it is why FAISS (which batches the
// coarse scan as a matrix multiply) kept winning IVF-PQ at nprobe=1 even
// after the LUT work was fixed. Tiling 4 queries x 4 centroids in registers
// (l2sq_4x4) reuses each loaded block four times.
//
// Callers process queries in slabs and pass a (slab_n x nlist) output
// buffer, bounding memory instead of materializing an nq x nlist matrix.

#include "proxima/distance.hpp"
#include "proxima/parallel.hpp"

#include <cstddef>

namespace proxima {

// Queries processed in blocks of 4 per worker; remainder queries fall back
// to the row-wise kernel. out[(qi)*nlist + c] = ||q_qi - cent_c||^2.
inline void coarse_scan_block(float* out,
                              const float* queries, std::size_t nq,
                              const float* cents, std::size_t nlist,
                              std::size_t d,
                              unsigned nthreads = 0) {
    const std::size_t nblocks = (nq + 3) / 4;
    parallel_for(nblocks, [&](std::size_t blk) {
        const std::size_t q0 = blk * 4;
        if (q0 + 4 <= nq) {
            float tile[16];
            std::size_t c = 0;
            for (; c + 4 <= nlist; c += 4) {
                l2sq_4x4(queries + q0 * d, d, cents + c * d, d, d, tile);
                for (int i = 0; i < 4; ++i) {
                    float* row = out + (q0 + i) * nlist + c;
                    row[0] = tile[i * 4 + 0];
                    row[1] = tile[i * 4 + 1];
                    row[2] = tile[i * 4 + 2];
                    row[3] = tile[i * 4 + 3];
                }
            }
            for (; c < nlist; ++c) {
                for (int i = 0; i < 4; ++i) {
                    out[(q0 + i) * nlist + c] =
                        l2sq(queries + (q0 + i) * d, cents + c * d, d);
                }
            }
        } else {
            for (std::size_t qi = q0; qi < nq; ++qi) {
                l2sq_ny(out + qi * nlist, queries + qi * d, cents, nlist, d);
            }
        }
    }, nthreads);
}

// Slab size for the coarse-distance matrix: 4096 queries x nlist floats
// (16 MB at nlist=1024) keeps the buffer L2/L3-friendly and bounded.
constexpr std::size_t kCoarseSlab = 4096;

}  // namespace proxima
