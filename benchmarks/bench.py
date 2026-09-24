"""Benchmark harness.

Compares this project's HNSW and IVF-PQ to FAISS on either SIFT1M or a
synthetic random dataset (so the script stays useful before the user
downloads SIFT1M).

Reports build time, recall@k, and QPS for each (index, parameter) cell.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from proxima import FlatIndex, HnswIndex, IvfPqIndex  # noqa: E402

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False


# ---- dataset loading ---------------------------------------------------

def load_dataset(name: str, n: int, d: int, seed: int = 0):
    """Returns (xb, xq, gt). gt is computed only when not loading SIFT1M."""
    if name == "sift1m":
        # When run as `python benchmarks/bench.py`, Python puts benchmarks/
        # on sys.path[0], so sift_loader resolves as a sibling module.
        from sift_loader import load_sift1m
        xb, xq, gt = load_sift1m()
        return xb, xq, gt
    if name == "gist1m":
        from gist_loader import load_gist1m
        xb, xq, gt = load_gist1m()
        return xb, xq, gt
    # synthetic
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((min(1000, n // 10), d), dtype=np.float32)
    print(f"computing brute-force ground truth for synthetic ({n}×{d})...")
    flat = FlatIndex(dim=d, metric="l2")
    flat.add(xb)
    _, gt = flat.search(xq, k=100)
    return xb, xq, gt


# ---- metric helpers ----------------------------------------------------

def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    hits = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k].tolist()) & set(t[:k].tolist()))
    return hits / (len(pred) * k)


@dataclass
class Result:
    name: str
    build_s: float
    qps: float
    recall_at_10: float
    extra: str = ""

    def fmt(self) -> str:
        return (
            f"{self.name:<32}  build={self.build_s:7.2f}s  "
            f"recall@10={self.recall_at_10:6.3f}  "
            f"qps={self.qps:9.1f}  {self.extra}"
        )


# ---- benchmark cells ---------------------------------------------------

def time_search(fn, queries, repeats: int = 3) -> tuple[float, np.ndarray]:
    """Returns (qps, last_labels). Picks the fastest of `repeats` to denoise."""
    best = float("inf")
    labels = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, labels = fn(queries)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return len(queries) / best, labels


def bench_ours_hnsw(xb, xq, gt, *, M, ef_construction, ef_search) -> Result:
    d = xb.shape[1]
    idx = HnswIndex(dim=d, metric="l2", M=M, ef_construction=ef_construction, seed=42)
    t0 = time.perf_counter()
    idx.add(xb)
    build = time.perf_counter() - t0

    qps, labels = time_search(lambda q: idx.search(q, k=10, ef=ef_search), xq)
    r = recall_at_k(labels, gt, k=10)
    return Result(
        name=f"ours-HNSW(M={M},efC={ef_construction},efS={ef_search})",
        build_s=build, qps=qps, recall_at_10=r,
    )


def bench_ours_ivfpq(xb, xq, gt, *, nlist, M, nprobe) -> Result:
    d = xb.shape[1]
    idx = IvfPqIndex(dim=d, nlist=nlist, M=M, kmeans_iters=15, seed=42)
    t0 = time.perf_counter()
    # Train on a subset for speed; standard practice is ~30*nlist samples.
    n_train = min(len(xb), max(30 * nlist, 10_000))
    idx.train(xb[:n_train])
    idx.add(xb)
    build = time.perf_counter() - t0

    qps, labels = time_search(lambda q: idx.search(q, k=10, nprobe=nprobe), xq)
    r = recall_at_k(labels, gt, k=10)
    return Result(
        name=f"ours-IVFPQ(nlist={nlist},M={M},nprobe={nprobe})",
        build_s=build, qps=qps, recall_at_10=r,
    )


def bench_faiss_hnsw(xb, xq, gt, *, M, ef_construction, ef_search) -> Result:
    if not HAVE_FAISS:
        return Result(name="faiss-HNSW (skipped)", build_s=0, qps=0, recall_at_10=0)
    d = xb.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    t0 = time.perf_counter()
    index.add(xb)
    build = time.perf_counter() - t0
    index.hnsw.efSearch = ef_search

    qps, _ = time_search(lambda q: index.search(q, 10), xq)
    _, labels = index.search(xq, 10)
    r = recall_at_k(labels, gt, k=10)
    return Result(
        name=f"faiss-HNSW(M={M},efC={ef_construction},efS={ef_search})",
        build_s=build, qps=qps, recall_at_10=r,
    )


def bench_faiss_ivfpq(xb, xq, gt, *, nlist, M, nprobe) -> Result:
    if not HAVE_FAISS:
        return Result(name="faiss-IVFPQ (skipped)", build_s=0, qps=0, recall_at_10=0)
    d = xb.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8)  # 8 bits per code
    n_train = min(len(xb), max(30 * nlist, 10_000))
    t0 = time.perf_counter()
    index.train(xb[:n_train])
    index.add(xb)
    build = time.perf_counter() - t0
    index.nprobe = nprobe

    qps, _ = time_search(lambda q: index.search(q, 10), xq)
    _, labels = index.search(xq, 10)
    r = recall_at_k(labels, gt, k=10)
    return Result(
        name=f"faiss-IVFPQ(nlist={nlist},M={M},nprobe={nprobe})",
        build_s=build, qps=qps, recall_at_10=r,
    )


# ---- driver -----------------------------------------------------------

def run(dataset: str, n: int, d: int) -> None:
    xb, xq, gt = load_dataset(dataset, n, d)
    print(f"\nDataset: base={xb.shape}  queries={xq.shape}  truth={gt.shape}")
    print(f"FAISS available: {HAVE_FAISS}\n")

    results: list[Result] = []

    print("=== HNSW sweep ===")
    for ef_search in (16, 64, 256):
        results.append(bench_ours_hnsw(xb, xq, gt, M=16, ef_construction=200,
                                       ef_search=ef_search))
        print("  " + results[-1].fmt())
        if HAVE_FAISS:
            results.append(bench_faiss_hnsw(xb, xq, gt, M=16, ef_construction=200,
                                            ef_search=ef_search))
            print("  " + results[-1].fmt())

    # IVF-PQ requires dim divisible by M. Pick M=8 (compact) and M=16 (more
    # accurate) when both are valid for this dim.
    pq_Ms = []
    if d % 8 == 0:  pq_Ms.append(8)
    if d % 16 == 0: pq_Ms.append(16)
    if not pq_Ms:   pq_Ms = [4 if d % 4 == 0 else 2]

    nlist = max(64, min(1024, int(np.sqrt(len(xb)))))
    for pq_M in pq_Ms:
        print(f"\n=== IVF-PQ sweep (nlist={nlist}, M={pq_M}, "
              f"{pq_M}-byte codes / {pq_M*8} bits) ===")
        for nprobe in (1, 8, 32):
            results.append(bench_ours_ivfpq(xb, xq, gt, nlist=nlist, M=pq_M, nprobe=nprobe))
            print("  " + results[-1].fmt())
            if HAVE_FAISS:
                results.append(bench_faiss_ivfpq(xb, xq, gt, nlist=nlist, M=pq_M, nprobe=nprobe))
                print("  " + results[-1].fmt())

    print("\n=== summary ===")
    for r in results:
        print("  " + r.fmt())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("sift1m", "gist1m", "synthetic"),
                   default="synthetic")
    p.add_argument("--n", type=int, default=50_000, help="synthetic only")
    p.add_argument("--d", type=int, default=128, help="synthetic only")
    args = p.parse_args()
    run(args.dataset, args.n, args.d)


if __name__ == "__main__":
    main()
