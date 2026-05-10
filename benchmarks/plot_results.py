"""Generate the recall@10 vs QPS plot for SIFT1M.

This is the standard ANN-benchmark visualization: each (algorithm, parameter)
cell becomes a point; sweeping the speed/quality knob (efSearch for HNSW,
nprobe for IVF-PQ) traces a curve. Up-and-right is better.

Output: benchmarks/results_sift1m.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from vectordb import HnswIndex, IvfPqIndex                # noqa: E402
from sift_loader import load_sift1m                       # noqa: E402
from gist_loader import load_gist1m                       # noqa: E402

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    hits = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k].tolist()) & set(t[:k].tolist()))
    return hits / (len(pred) * k)


def time_qps(fn, queries, repeats: int = 3) -> tuple[float, np.ndarray]:
    best = float("inf")
    labels = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        _, labels = fn(queries)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return len(queries) / best, labels


def sweep_ours_hnsw(xb, xq, gt, ef_search_grid: Iterable[int]):
    d = xb.shape[1]
    print("building ours-HNSW...", flush=True)
    idx = HnswIndex(dim=d, metric="l2", M=16, ef_construction=200, seed=42)
    idx.add(xb)
    pts = []
    for ef in ef_search_grid:
        qps, lbl = time_qps(lambda q: idx.search(q, k=10, ef=ef), xq)
        r = recall_at_k(lbl, gt, k=10)
        print(f"  ours-HNSW efS={ef:<3}  recall={r:.3f}  qps={qps:.0f}", flush=True)
        pts.append((r, qps))
    return pts


def sweep_faiss_hnsw(xb, xq, gt, ef_search_grid: Iterable[int]):
    if not HAVE_FAISS: return []
    d = xb.shape[1]
    print("building faiss-HNSW...", flush=True)
    idx = faiss.IndexHNSWFlat(d, 16)
    idx.hnsw.efConstruction = 200
    idx.add(xb)
    pts = []
    for ef in ef_search_grid:
        idx.hnsw.efSearch = ef
        qps, _ = time_qps(lambda q: idx.search(q, 10), xq)
        _, lbl = idx.search(xq, 10)
        r = recall_at_k(lbl, gt, k=10)
        print(f"  faiss-HNSW efS={ef:<3}  recall={r:.3f}  qps={qps:.0f}", flush=True)
        pts.append((r, qps))
    return pts


def sweep_ours_ivfpq(xb, xq, gt, M: int, nprobe_grid: Iterable[int]):
    d = xb.shape[1]
    nlist = max(64, min(1024, int(np.sqrt(len(xb)))))
    print(f"building ours-IVFPQ(M={M})...", flush=True)
    idx = IvfPqIndex(dim=d, nlist=nlist, M=M, kmeans_iters=15, seed=42)
    n_train = min(len(xb), max(30 * nlist, 10_000))
    idx.train(xb[:n_train])
    idx.add(xb)
    pts = []
    for nprobe in nprobe_grid:
        qps, lbl = time_qps(lambda q: idx.search(q, k=10, nprobe=nprobe), xq)
        r = recall_at_k(lbl, gt, k=10)
        print(f"  ours-IVFPQ M={M} nprobe={nprobe:<3}  recall={r:.3f}  qps={qps:.0f}", flush=True)
        pts.append((r, qps))
    return pts


def sweep_faiss_ivfpq(xb, xq, gt, M: int, nprobe_grid: Iterable[int]):
    if not HAVE_FAISS: return []
    d = xb.shape[1]
    nlist = max(64, min(1024, int(np.sqrt(len(xb)))))
    print(f"building faiss-IVFPQ(M={M})...", flush=True)
    quantizer = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8)
    n_train = min(len(xb), max(30 * nlist, 10_000))
    idx.train(xb[:n_train])
    idx.add(xb)
    pts = []
    for nprobe in nprobe_grid:
        idx.nprobe = nprobe
        qps, _ = time_qps(lambda q: idx.search(q, 10), xq)
        _, lbl = idx.search(xq, 10)
        r = recall_at_k(lbl, gt, k=10)
        print(f"  faiss-IVFPQ M={M} nprobe={nprobe:<3}  recall={r:.3f}  qps={qps:.0f}", flush=True)
        pts.append((r, qps))
    return pts


def main() -> None:
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("sift1m", "gist1m"), default="sift1m")
    args = p.parse_args()

    if args.dataset == "sift1m":
        xb, xq, gt = load_sift1m()
        title_dataset = "SIFT1M (1M × 128)"
    else:
        xb, xq, gt = load_gist1m()
        title_dataset = "GIST1M (1M × 960)"
    print(f"{args.dataset}: base={xb.shape}  queries={xq.shape}", flush=True)

    ef_grid = [16, 32, 64, 128, 256]
    nprobe_grid = [1, 4, 8, 16, 32, 64]

    series = {
        "ours-HNSW":         sweep_ours_hnsw(xb, xq, gt, ef_grid),
        "faiss-HNSW":        sweep_faiss_hnsw(xb, xq, gt, ef_grid),
        "ours-IVFPQ(M=8)":   sweep_ours_ivfpq(xb, xq, gt, 8,  nprobe_grid),
        "faiss-IVFPQ(M=8)":  sweep_faiss_ivfpq(xb, xq, gt, 8,  nprobe_grid),
        "ours-IVFPQ(M=16)":  sweep_ours_ivfpq(xb, xq, gt, 16, nprobe_grid),
        "faiss-IVFPQ(M=16)": sweep_faiss_ivfpq(xb, xq, gt, 16, nprobe_grid),
    }

    out_path = ROOT / "benchmarks" / f"results_{args.dataset}.png"
    out_json = ROOT / "benchmarks" / f"results_{args.dataset}.json"
    out_json.write_text(json.dumps(series, indent=2))

    fig, ax = plt.subplots(figsize=(8, 6))
    style = {
        "ours-HNSW":         dict(marker="o", linestyle="-",  color="C0"),
        "faiss-HNSW":        dict(marker="o", linestyle="--", color="C0", alpha=0.6),
        "ours-IVFPQ(M=8)":   dict(marker="s", linestyle="-",  color="C1"),
        "faiss-IVFPQ(M=8)":  dict(marker="s", linestyle="--", color="C1", alpha=0.6),
        "ours-IVFPQ(M=16)":  dict(marker="^", linestyle="-",  color="C2"),
        "faiss-IVFPQ(M=16)": dict(marker="^", linestyle="--", color="C2", alpha=0.6),
    }
    for name, pts in series.items():
        if not pts: continue
        recalls = [p[0] for p in pts]
        qps     = [p[1] for p in pts]
        ax.plot(recalls, qps, label=name, **style.get(name, {}))

    ax.set_xlabel("recall@10")
    ax.set_ylabel("QPS (queries per second)")
    ax.set_yscale("log")
    ax.set_title(f"{title_dataset} — recall@10 vs QPS (Apple M2 Pro, std::thread)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
