"""Cosine on IVF-PQ: metric="l2" on normalized vectors vs metric="ip".

On unit vectors L2 and cosine rank identically, but IVF-PQ's L2 path
encodes residuals from the coarse centroid while its IP path encodes raw
vectors. This measures what that costs on SIFT1M (recall@10 only, against
the cached exact cosine ground truth that
`plot_results.py --dataset sift1m --metric cosine` writes), and checks
whether more k-means iterations change the IP result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "benchmarks")]

import faiss                                                        # noqa: E402
from proxima import IvfPqIndex                                      # noqa: E402
from plot_results import (dataset_paths, ivfpq_nlist, ivfpq_ntrain,  # noqa: E402
                          load_truth, prep, recall_at_k)
from sift_loader import fvecs_read                                  # noqa: E402

NPROBES = (8, 64)


def faiss_recall(metric: int, M: int, nlist: int, xt, xb, xq, gt) -> list[float]:
    q = faiss.IndexFlatIP(xb.shape[1]) if metric == faiss.METRIC_INNER_PRODUCT \
        else faiss.IndexFlatL2(xb.shape[1])
    idx = faiss.IndexIVFPQ(q, xb.shape[1], nlist, M, 8, metric)
    idx.train(xt)
    idx.add(xb)
    out = []
    for p in NPROBES:
        idx.nprobe = p
        out.append(recall_at_k(idx.search(xq, 10)[1], gt, 10))
    return out


def main() -> None:
    paths = dataset_paths("sift1m")
    xb = prep(fvecs_read(paths["base"]), "cosine")
    xq = prep(fvecs_read(paths["query"]), "cosine")
    gt = load_truth("sift1m", "cosine", xq)
    nlist = ivfpq_nlist(len(xb))
    xt = xb[:ivfpq_ntrain(len(xb), nlist)]
    d = xb.shape[1]

    for M in (8, 16):
        rows: list[tuple[str, list[float]]] = []
        for metric, iters in (("ip", 15), ("ip", 25), ("l2", 15)):
            idx = IvfPqIndex(dim=d, nlist=nlist, M=M, kmeans_iters=iters,
                             seed=42, metric=metric)
            idx.train(xt)
            idx.add(xb)
            rows.append((f"ours {metric} iters={iters}",
                         [recall_at_k(idx.search(xq, k=10, nprobe=p)[1], gt, 10)
                          for p in NPROBES]))
        rows.append(("faiss ip (default)",
                     faiss_recall(faiss.METRIC_INNER_PRODUCT, M, nlist, xt, xb, xq, gt)))
        rows.append(("faiss l2", faiss_recall(faiss.METRIC_L2, M, nlist, xt, xb, xq, gt)))

        print(f"\nM={M}  recall@10 vs cosine truth   " +
              "   ".join(f"nprobe={p}" for p in NPROBES))
        for label, rs in rows:
            print(f"  {label:<22} " + "      ".join(f"{r:.3f}" for r in rs))


if __name__ == "__main__":
    main()
