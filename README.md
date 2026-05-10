# vectordb

Hand-rolled **HNSW** and **IVF-PQ** vector indexes in C++17 with NumPy-friendly
Python bindings. Built to understand how production ANN libraries (FAISS,
hnswlib, ScaNN) actually work — not as a drop-in replacement for them.

## What's in here

| Index       | Algorithm                                     | Build  | Memory          |
| ----------- | --------------------------------------------- | ------ | --------------- |
| `FlatIndex` | brute-force linear scan (recall ground truth) | O(1)   | `n·d·4` bytes   |
| `HnswIndex` | hierarchical NSW (Malkov & Yashunin 2016)     | O(n log n) | ~`n·d·4 + n·M·8` bytes |
| `IvfPqIndex`| inverted file + product quantization          | O(n·k·iter) train | ~`n·M` bytes (codes) |

Distance kernels: `l2sq` and `dot` with hand-unrolled NEON intrinsics on Apple
Silicon, scalar fallback elsewhere.

## Build

```bash
cd ~/vectordb
python3 -m venv .venv
.venv/bin/pip install numpy faiss-cpu pybind11 pytest matplotlib
cmake -S . -B build -DPython3_EXECUTABLE=$(pwd)/.venv/bin/python
cmake --build build -j
.venv/bin/python -m pytest tests/ -v
```

The compiled extension lands in `python/vectordb/_vectordb.*.so`. The Python
shim (`python/vectordb/__init__.py`) re-exports the three index classes, so
benchmarks and tests just `from vectordb import HnswIndex`.

## Use

```python
import numpy as np
from vectordb import HnswIndex, IvfPqIndex

xb = np.random.randn(100_000, 128).astype("float32")
xq = np.random.randn(100, 128).astype("float32")

# HNSW
idx = HnswIndex(dim=128, metric="l2", M=16, ef_construction=200)
idx.add(xb)
D, I = idx.search(xq, k=10, ef=64)

# IVF-PQ
ivf = IvfPqIndex(dim=128, nlist=1024, M=8)
ivf.train(xb[:50_000])
ivf.add(xb)
D, I = ivf.search(xq, k=10, nprobe=8)

# Persistence — round-trip preserves search results bit-for-bit.
idx.save("hnsw.bin")
loaded = HnswIndex.load("hnsw.bin")
```

## Benchmarks

```bash
.venv/bin/python benchmarks/bench.py --dataset synthetic --n 100000 --d 128
.venv/bin/python benchmarks/sift_loader.py --download   # 161 MB
.venv/bin/python benchmarks/bench.py --dataset sift1m
.venv/bin/python benchmarks/plot_results.py --dataset sift1m
.venv/bin/python benchmarks/gist_loader.py --download   # ~2.6 GB, harder dataset
.venv/bin/python benchmarks/bench.py --dataset gist1m
.venv/bin/python benchmarks/plot_results.py --dataset gist1m
```

### SIFT1M (1M × 128, Apple M2 Pro, std::thread parallel search)

![SIFT1M recall vs QPS](benchmarks/results_sift1m.png)

**Honest read of the chart:**
- **HNSW (blue):** ours sits 19-27% behind FAISS across the recall axis (QPS ratio 0.73-0.81 from the SIFT1M benchmark run); same curve shape, just shifted right. The remaining gap is the bulk-distance kernel.
- **IVF-PQ at low nprobe:** **ours beats FAISS** at nprobe=1 for both M=8 (416k vs 382k qps, +9%) and M=16 (349k vs 299k qps, +17%) — the vectorized LUT build pays off most when LUT cost dominates list-scan cost.
- **IVF-PQ at high nprobe:** ours is now within ~2× of FAISS on SIFT1M (was 2-3×) after vectorizing the LUT build. The remaining gap is the inverted-list code-scan loop — FAISS's `IndexIVFPQ` uses a SIMD-accelerated PQ scan path that we don't yet match. That's the v3 work below.


HNSW (M=16, efConstruction=200):

| Index             | efSearch | recall@10 | QPS     | build (s) |
| ----------------- | -------- | --------- | ------- | --------- |
| ours-HNSW         | 16       | 0.802     | 143.4k  | 342       |
| faiss-HNSW        | 16       | 0.815     | 176.8k  |  44       |
| ours-HNSW         | 64       | 0.964     |  50.1k  | 356       |
| faiss-HNSW        | 64       | 0.970     |  59.4k  |  51       |
| **ours-HNSW**     | **256**  | **0.997** | **15.3k** | 343     |
| **faiss-HNSW**    | **256**  | **0.998** | **16.8k** |  45     |

IVF-PQ (nlist=1000):

| Index               | M  | bytes | nprobe | recall@10 | QPS     |
| ------------------- | -- | ----- | ------ | --------- | ------- |
| **ours-IVFPQ**      | 8  | 8     |  1     | 0.229     | **416.2k** |
| faiss-IVFPQ         | 8  | 8     |  1     | 0.227     | 382.4k  |
| ours-IVFPQ          | 8  | 8     |  8     | 0.347     | 118.8k  |
| faiss-IVFPQ         | 8  | 8     |  8     | 0.347     | 203.5k  |
| ours-IVFPQ          | 8  | 8     | 32     | 0.360     |  36.5k  |
| faiss-IVFPQ         | 8  | 8     | 32     | 0.359     |  68.9k  |
| **ours-IVFPQ**      | 16 | 16    |  1     | 0.296     | **349.1k** |
| faiss-IVFPQ         | 16 | 16    |  1     | 0.294     | 299.4k  |
| ours-IVFPQ          | 16 | 16    |  8     | 0.514     |  80.6k  |
| faiss-IVFPQ         | 16 | 16    |  8     | 0.512     | 116.6k  |
| **ours-IVFPQ**      | 16 | 16    | 32     | **0.546** |  23.8k  |
| **faiss-IVFPQ**     | 16 | 16    | 32     | **0.545** |  40.0k  |

**Recall matches FAISS within 0.002 in every cell.** Algorithm and
quantization are correct end-to-end on a real dataset.

**HNSW QPS sits 19-27% behind FAISS** across the efSearch sweep (ratios
0.73-0.81 from the SIFT1M benchmark run). The gap narrows at higher recall —
at efS=256 we're at 81% of FAISS QPS. The remaining gap is SIMD distance
kernels (FAISS uses bulk batched-distance routines that vectorize across
many candidates at once).

**IVF-PQ at nprobe=1 beats FAISS** (M=8: +9%, M=16: +17%) — when only one
list is scanned, the LUT build dominates per-query work, and our NEON FMA
tile kernel is tighter than FAISS's general-purpose path. As `nprobe`
grows, the inverted-list code scan dominates instead and FAISS's
`tbl`-instruction scan opens a ~2× gap. Closing the scan loop is v3.

M=16 doubles the code budget and lifts recall from 0.36 → 0.55 at the same
nprobe, validating the recall-vs-memory tradeoff.

### GIST1M (1M × 960, harder dataset)

SIFT1M is the easy benchmark. GIST1M is the harder one — 7.5× higher
dimensionality, less cluster structure, recall ceilings drop sharply for both
implementations. Running here is a credibility check: tricks that work on
SIFT1M don't always survive at 960-dim.

![GIST1M recall vs QPS](benchmarks/results_gist1m.png)

HNSW (M=16, efConstruction=200):

| efSearch | recall@10 ours / FAISS | QPS ours / FAISS |
| -------- | ---------------------- | ---------------- |
| 16  | 0.485 / 0.515 | 22.9k / 23.5k |
| 32  | 0.627 / 0.648 | 14.2k / 16.4k |
| 64  | 0.758 / 0.773 |  8.6k / 10.7k |
| 128 | 0.860 / 0.868 |  5.1k /  6.1k |
| **256** | **0.927 / 0.927** | **2.9k / 3.5k** |

**At efS=256 our HNSW recall matches FAISS exactly (0.927 vs 0.927)**, and
QPS is 83% of FAISS (17% behind). Across the full sweep we sit 3-20%
behind on QPS and within ~0.03 recall — same shape as SIFT1M, just at a
lower absolute recall ceiling because nearest-neighbor structure at
960-dim is fundamentally harder.

IVF-PQ (nlist=1000):

| config | recall@10 ours / FAISS | QPS ours / FAISS |
| ------ | ---------------------- | ---------------- |
| M=8,  nprobe=1  | 0.074 / 0.075 |  66.0k / 192.3k |
| M=8,  nprobe=8  | 0.093 / 0.098 |  27.8k /  92.4k |
| M=8,  nprobe=64 | 0.093 / 0.099 |   5.0k /  19.9k |
| M=16, nprobe=1  | 0.111 / 0.112 |  59.3k / 121.9k |
| M=16, nprobe=8  | 0.157 / 0.165 |  19.4k /  53.9k |
| **M=16, nprobe=64** | **0.161 / 0.169** | **3.6k / 11.9k** |

**IVF-PQ recall matches FAISS within 0.01** in every cell. The QPS gap is
2-4× — wider than on SIFT1M.

**Why the LUT vectorization didn't help much on GIST1M:** with `dim=960`
and `M=8`, each subspace has `dsub=120` dims, so the original per-centroid
`l2sq` call already did 30 NEON FMAs of useful work — the call overhead
was already amortized. The tile pattern wins on SIFT (`dsub=16`, only 4
NEON FMAs per call so call overhead dominates) and is roughly a wash on
GIST. The remaining GIST gap is the inverted-list code scan, same as on
SIFT — that's the v3 SIMD PQ scan work.

The recall ceiling at ~0.16 (M=16, full scan) is what 240× compression buys
on 960-dim GIST descriptors: PQ throws away too much to recover much
beyond that.
This is the dataset, not a bug — FAISS plateaus at the same place.

> **Bug found while benching GIST1M**: my IVF-PQ `encode_vector` had a
> fixed `float r_sub[64]` stack array, which silently overflowed at
> dsub > 64. SIFT1M (dsub ≤ 32) never tripped it; GIST1M with M=8
> (dsub=120) did. Caught by macOS stack canary → SIGABRT. Fixed by passing
> a heap-allocated scratch buffer from `add()`. Regression test now exercises
> dsub=120 explicitly.

## Architecture

### HNSW (`src/hnsw.cpp`)

- **Storage**: `data_[id*dim..(id+1)*dim]` for vectors, `links_[id][layer]`
  for neighbor IDs. Entry point and max level updated only when a node draws a
  level above the current max.
- **Insertion**: greedy descent from entry point through layers above the new
  node's level, then beam search (`ef_construction`) at each lower layer.
  Bidirectional links added; receiving nodes get re-pruned to `M_max` if they
  overflow.
- **Neighbor selection**: heuristic from Algorithm 4 in the paper (a candidate
  is accepted only if it's closer to the query than to every already-accepted
  neighbor — kills near-duplicate edges).
- **Search**: greedy descent through upper layers, then `search_layer` at
  level 0 with `ef`. Returns top-k from a max-heap.
- **Visited set**: single buffer + monotonic generation counter. Bumping the
  counter resets all marks in O(1) (vs O(n) memset per query).

### IVF-PQ (`src/ivfpq.cpp`)

- **Train**: k-means on raw vectors → `nlist` coarse centroids. Compute
  residuals `r = x - centroid(x)`. For each of `M` subspaces, run k-means with
  `KSUB=256` on the `dsub`-dim slice → `M` codebooks of 256 vectors each.
- **Add**: nearest coarse centroid → encode residual into `M` bytes (one per
  subspace) → push (label, code) into the inverted list.
- **Search (ADC)**: for each of `nprobe` nearest coarse lists, build a
  `M × 256` lookup table where `LUT[m][k]` is the squared distance from the
  query's residual subspace to PQ centroid `k`. Each list entry's distance is
  a sum of `M` lookups.

### k-means (`src/kmeans.cpp`)

Lloyd's algorithm with k-means++ seeding. Empty-cluster re-seeding from a
random data point each iteration. Double-precision sum accumulators
(centroid drift becomes visible at ~10K updates with float32).

## SIMD / cache tuning notes

Apple M2 Pro: 4-wide NEON (128-bit), 64KB L1d per P-core, 16MB shared L2.

**1. Distance kernels (`src/distance.cpp`)**: 4-way unrolled with four
independent FMA accumulators. Single-accumulator code stalls on the FMA
latency (~3 cycles); four parallel chains saturate the issue width and let
the OoO engine overlap loads with arithmetic.

```cpp
acc0 = vfmaq_f32(acc0, d0, d0);   // four independent dependency chains
acc1 = vfmaq_f32(acc1, d1, d1);
acc2 = vfmaq_f32(acc2, d2, d2);
acc3 = vfmaq_f32(acc3, d3, d3);
```

For d=128 (16 NEON regs × 16 floats per iter × 8 iters), the inner loop
processes the whole vector in 8 unrolled iterations with no scalar tail.

**2. HNSW neighbor prefetch (`src/hnsw.cpp`)**: each neighbor expansion is a
random ~512-byte load (d=128 floats). One-ahead `__builtin_prefetch` hides
the L2 miss; the distance call on the current neighbor proceeds while the
next one is fetched into L1.

```cpp
for (size_t i = 0; i < nbrs.size(); ++i) {
    if (i + 1 < nbrs.size()) __builtin_prefetch(vec(nbrs[i+1]), 0, 1);
    // ... compute distance to nbrs[i]
}
```

**3. IVF-PQ LUT layout + vectorized build**: `M × 256` floats = 8KB for M=8,
fits comfortably in L1d. Inner loop is `M` constant-stride loads with no
pointer chasing.

The LUT *build* (compute distance from each subspace's residual to all 256
codebook entries) used to dominate IVF-PQ search at high `nprobe` — naively
it's `M × 256` calls to `l2sq`, ~70% of per-query work at nprobe=32. We
vectorize it with a NEON FMA tile pattern:

- A second copy of the codebook is kept transposed: instead of `cb[k][j]`
  (per-centroid, then per-dim) it's `cb_T[j][k]` (per-dim, then per-centroid)
  so all 256 distances stride contiguously.
- The build processes 16 LUT entries at a time, holding 4 NEON accumulators
  in registers across the entire `dsub` reduction — no LUT loads/stores in
  the inner loop.

```cpp
for (k_start = 0; k_start < 256; k_start += 16) {
    float32x4_t lut0..lut3 = zero;          // four parallel chains
    for (j = 0; j < dsub; ++j) {
        float32x4_t r = vdupq_n_f32(r_m[j]);
        // four 4-wide loads from cb_T_m[j*256 + k_start..]
        // four (cb - r) and four FMA into lut0..lut3
    }
    store lut0..lut3 -> LUT[m][k_start..]
}
```

This kicks the LUT build from ~70% of search time down to ~30% on SIFT1M
and roughly doubles IVF-PQ QPS at high nprobe.

**4. PQ inverted-list layout**: parallel arrays for labels and codes. Codes
stream contiguously during scan (`n_list × M` byte buffer); labels are
touched only on heap insert. Interleaving (label, code, label, code) would
pollute L1d with `int64` labels during the dominant code-summation loop.

**5. Visited-set generation counter**: mentioned above. The naive approach
of `std::vector<bool> visited(n)` would memset N bytes per query call;
incrementing a `uint32_t` instead is O(1) per query, with a single full
reset only when the counter wraps (~4B searches in).

**6. Parallel batch search via std::thread (`include/vectordb/parallel.hpp`)**:
each query is independent; a stride-scheduled `parallel_for` fans out across
`hardware_concurrency()` workers. Per-thread context (HNSW VisitedSet,
IVF-PQ ADC LUT scratch) is built once per worker and reused across that
worker's slice of queries.

We use `std::thread` rather than OpenMP specifically because faiss-cpu
bundles its own libomp; on macOS the dyld refuses to load a second copy and
errors with `OMP: Error #15`. `std::thread` has no such problem and lets us
coexist with FAISS in the same Python process for honest side-by-side
benchmarks.

This single change closed the HNSW QPS gap from ~8× to ~1.2-1.4× (19-27%
behind). The IVF-PQ gap closed from 8-20× to roughly 1-2× on SIFT1M and
~2-4× on GIST1M (the high-dim, high-nprobe regime where the SIMD code-scan
matters most).

### Remaining gap vs FAISS

After parallelism + vectorized LUT build, the residual gap is now entirely
in two specific places:

- **SIMD PQ code scan**: the inverted-list scan loop. FAISS's `IndexIVFPQ`
  (what we benchmark against) uses a SIMD-accelerated PQ scan path that we
  don't yet match; ours sums M scalar lookups per code. Visible as the gap
  that *opens* as `nprobe` grows in the IVF-PQ table — at nprobe=1 we beat
  FAISS, at nprobe=32 we're ~1.7× behind on SIFT1M (~3-4× on GIST1M). The
  natural reference design for closing this is the `tbl`-instruction trick
  used in FAISS's separate `IndexPQFastScan` class (4-bit codes, 16-byte
  LUT, one `tbl` per 16 lookups) — that's a different index type and a
  weekend of work to bring over.
- **Bulk distance kernels** (`fvec_L2sqr_ny`): one query against many
  candidates in a vectorized pass. Visible as the modest residual HNSW gap
  at high efSearch where each query touches thousands of candidates.

Each is a focused weekend of NEON work — clean v3 scope.

## Layout

```
vectordb/
├── CMakeLists.txt
├── include/vectordb/   # public headers
├── src/                # C++ implementations + pybind11 bindings
├── python/vectordb/    # Python package; .so lands here after build
├── tests/              # pytest correctness tests vs Flat
└── benchmarks/         # SIFT1M loader + side-by-side FAISS bench
```

## Clean state for distribution

`.gitignore` excludes `build/`, `data/`, `.venv/`, compiled `*.so`,
`__pycache__/`, and `benchmarks/results_*.json` — so a `git clone` is
clean. If you instead `zip` the local working tree, those artifacts will
be included. To wipe them:

```bash
git clean -fdX     # remove every gitignored file (safe; ignores tracked)
```


## References

- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs*, TPAMI 2018.
  arXiv:1603.09320
- Jégou, Douze & Schmid, *Product Quantization for Nearest Neighbor Search*,
  TPAMI 2011.
- TexMex SIFT1M corpus: <http://corpus-texmex.irisa.fr/>
