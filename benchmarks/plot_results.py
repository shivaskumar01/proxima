"""Generate the recall@10 vs QPS plot for SIFT1M / GIST1M.

This is the standard ANN-benchmark visualization: each (algorithm, parameter)
cell becomes a point; sweeping the speed/quality knob (efSearch for HNSW,
nprobe for IVF-PQ) traces a curve. Up-and-right is better.

Memory model: the default invocation re-execs itself once per series
(`--series NAME`), so each sweep runs in its own process that loads the
data, builds ONE index, measures, merges its points into the results JSON,
and exits. Combined with the chunk-streamed loaders this keeps peak RSS
around base+index for the current series only (~5 GB for GIST1M) instead of
accumulating every index and numpy temporary in one long-lived process, 
the all-in-one version drove a 16 GB machine deep into swap on GIST1M,
where it burned an hour inside the memory compressor without finishing a
single build.

Metrics: --metric l2 (default) scores against the dataset's own ground
truth. --metric cosine L2-normalizes base and queries and searches by inner
product; --metric ip is raw maximum-inner-product search on the vectors as
stored. For both, exact top-100 ground truth is computed once with
faiss.IndexFlatIP (cross-checked against our FlatIndex) and cached next to
the dataset.

Output: benchmarks/results_{dataset}.{json,png} (l2) or
benchmarks/results_{dataset}_{metric}.{json,png}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from proxima import HnswIndex, IvfPqIndex, IvfPqFastScan          # noqa: E402
from sift_loader import fvecs_read, fvecs_chunks, fvecs_shape, ivecs_read  # noqa: E402
import sift_loader                                                  # noqa: E402
import gist_loader                                                  # noqa: E402

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False


DATASETS = {
    "sift1m": (sift_loader.DEFAULT_DIR, "sift", "SIFT1M (1M × 128)"),
    "gist1m": (gist_loader.DEFAULT_DIR, "gist", "GIST1M (1M × 960)"),
}

SERIES = [
    "ours-HNSW",
    "faiss-HNSW",
    "ours-IVFPQ(M=8)",
    "faiss-IVFPQ(M=8)",
    "ours-IVFPQ(M=16)",
    "faiss-IVFPQ(M=16)",
    "ours-IVFPQfs(M=32)",
    "faiss-IVFPQfs(M=32)",
]

# Inner-product runs add FAISS IVFPQ with by_residual=False: FAISS's IP
# default encodes residuals, ours encodes raw vectors, so this variant is
# the same design as ours and isolates implementation from encoding choice.
SERIES_IP = SERIES + [
    "faiss-IVFPQraw(M=8)",
    "faiss-IVFPQraw(M=16)",
]

METRICS = ("l2", "cosine", "ip")

EF_GRID = [16, 32, 64, 128, 256]
NPROBE_GRID = [1, 4, 8, 16, 32, 64]
CHUNK_ROWS = 100_000


def dataset_paths(dataset: str) -> dict[str, Path]:
    root, prefix, _ = DATASETS[dataset]
    return {
        "base":  root / f"{prefix}_base.fvecs",
        "query": root / f"{prefix}_query.fvecs",
        "gt":    root / f"{prefix}_groundtruth.ivecs",
    }


def results_stem(dataset: str, metric: str) -> str:
    return f"results_{dataset}" if metric == "l2" else f"results_{dataset}_{metric}"


def prep(x: np.ndarray, metric: str) -> np.ndarray:
    """Cosine runs normalize every vector; l2 and raw ip use them as stored."""
    if metric != "cosine":
        return x
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def load_truth(dataset: str, metric: str, xq: np.ndarray) -> np.ndarray:
    paths = dataset_paths(dataset)
    if metric == "l2":
        return ivecs_read(paths["gt"])
    cache = paths["gt"].with_name(f"{paths['gt'].stem}_{metric}.npy")
    if cache.exists():
        return np.load(cache)
    if not HAVE_FAISS:
        raise SystemExit("computing inner-product ground truth needs faiss")
    print(f"computing exact {metric} top-100 ground truth -> {cache}", flush=True)
    flat = faiss.IndexFlatIP(xq.shape[1])
    for c in fvecs_chunks(paths["base"], CHUNK_ROWS):
        flat.add(prep(c, metric))
    _, gt = flat.search(xq, 100)
    # Independent check with our own brute-force scan on a query slice.
    from proxima import FlatIndex
    ours = FlatIndex(dim=xq.shape[1], metric="ip")
    for c in fvecs_chunks(paths["base"], CHUNK_ROWS):
        ours.add(prep(c, metric))
    _, gt_ours = ours.search(xq[:200], k=10)
    agree = recall_at_k(gt_ours, gt[:200], k=10)
    print(f"  ground-truth cross-check vs FlatIndex (200 q): {agree:.4f}", flush=True)
    if agree < 0.999:
        raise SystemExit(f"ground truth disagreement: {agree}")
    del flat, ours
    np.save(cache, gt.astype(np.int32))
    return gt


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


# Index builds saturate every core (both ours and FAISS), which heats the
# package right before the first search sweep and depresses its QPS by up to
# ~30%. A short settle keeps the sweep cells comparable to each other.
SETTLE_S = 15.0


def settle() -> None:
    print(f"  (settling {SETTLE_S:.0f}s after build...)", flush=True)
    time.sleep(SETTLE_S)


def ivfpq_nlist(n_base: int) -> int:
    return max(64, min(1024, int(np.sqrt(n_base))))


def ivfpq_ntrain(n_base: int, nlist: int) -> int:
    return min(n_base, max(30 * nlist, 10_000))


# ---- one sweep per process ------------------------------------------------

def run_series(dataset: str, name: str, metric: str = "l2") -> list[tuple[float, float]]:
    paths = dataset_paths(dataset)
    n_base, d = fvecs_shape(paths["base"])
    xq = prep(fvecs_read(paths["query"]), metric)
    gt = load_truth(dataset, metric, xq)
    chunks = lambda: (prep(c, metric)                            # noqa: E731
                      for c in fvecs_chunks(paths["base"], CHUNK_ROWS))
    ours_metric = "l2" if metric == "l2" else "ip"
    fmetric = faiss.METRIC_L2 if (not HAVE_FAISS or metric == "l2") \
        else faiss.METRIC_INNER_PRODUCT
    flat_q = (lambda dd: faiss.IndexFlatL2(dd)) if metric == "l2" \
        else (lambda dd: faiss.IndexFlatIP(dd))                  # noqa: E731

    print(f"[{name}] metric={metric}  base=({n_base}, {d})  queries={xq.shape}", flush=True)
    t0 = time.perf_counter()
    pts: list[tuple[float, float]] = []

    if name == "ours-HNSW":
        idx = HnswIndex(dim=d, metric=ours_metric, M=16, ef_construction=200, seed=42)
        idx.reserve(n_base)   # chunked adds must not pay 2x realloc peaks
        for c in chunks():
            idx.add(c)
        print(f"  build: {time.perf_counter() - t0:.1f}s", flush=True)
        settle()
        for ef in EF_GRID:
            qps, lbl = time_qps(lambda q: idx.search(q, k=10, ef=ef), xq)
            r = recall_at_k(lbl, gt, k=10)
            print(f"  ours-HNSW efS={ef:<3}  recall={r:.3f}  qps={qps:.0f}", flush=True)
            pts.append((r, qps))

    elif name == "faiss-HNSW":
        idx = faiss.IndexHNSWFlat(d, 16, fmetric)
        idx.hnsw.efConstruction = 200
        # Chunked add. faiss has no reserve(), so each add realloc-copies its
        # storage vector (brief old+new transients), but that beats holding
        # a full 3.8 GB numpy copy NEXT TO faiss's internal copy for the
        # whole multi-minute build: the full-load variant sat at ~7.7 GB
        # sustained and swap-thrashed for an hour+ on GIST1M on a 16 GB
        # machine, exactly like the all-in-one harness used to.
        for c in chunks():
            idx.add(c)
        print(f"  build: {time.perf_counter() - t0:.1f}s", flush=True)
        settle()
        for ef in EF_GRID:
            idx.hnsw.efSearch = ef
            qps, _ = time_qps(lambda q: idx.search(q, 10), xq)
            _, lbl = idx.search(xq, 10)
            r = recall_at_k(lbl, gt, k=10)
            print(f"  faiss-HNSW efS={ef:<3}  recall={r:.3f}  qps={qps:.0f}", flush=True)
            pts.append((r, qps))

    elif "IVFPQfs" in name:
        # 4-bit fast-scan: M=32 x 4 bits = 16-byte codes (byte parity with
        # the M=16 8-bit series).
        M = 32
        nlist = ivfpq_nlist(n_base)
        n_train = ivfpq_ntrain(n_base, nlist)
        xt = prep(fvecs_read(paths["base"], 0, n_train), metric)
        if name.startswith("ours"):
            idx = IvfPqFastScan(dim=d, nlist=nlist, M=M, kmeans_iters=15, seed=42,
                                metric=ours_metric)
        else:
            quantizer = flat_q(d)
            idx = faiss.IndexIVFPQFastScan(quantizer, d, nlist, M, 4, fmetric)
        idx.train(xt)
        del xt
        for c in chunks():
            idx.add(c)
        print(f"  train+add: {time.perf_counter() - t0:.1f}s", flush=True)
        settle()
        for nprobe in NPROBE_GRID:
            if name.startswith("ours"):
                qps, lbl = time_qps(lambda q: idx.search(q, k=10, nprobe=nprobe), xq)
            else:
                idx.nprobe = nprobe
                qps, _ = time_qps(lambda q: idx.search(q, 10), xq)
                _, lbl = idx.search(xq, 10)
            r = recall_at_k(lbl, gt, k=10)
            print(f"  {name} nprobe={nprobe:<3}  recall={r:.3f}  qps={qps:.0f}", flush=True)
            pts.append((r, qps))

    elif name.startswith("ours-IVFPQ") or name.startswith("faiss-IVFPQ"):
        M = 8 if "M=8" in name else 16
        nlist = ivfpq_nlist(n_base)
        n_train = ivfpq_ntrain(n_base, nlist)
        xt = prep(fvecs_read(paths["base"], 0, n_train), metric)  # contiguous slice
        if name.startswith("ours"):
            idx = IvfPqIndex(dim=d, nlist=nlist, M=M, kmeans_iters=15, seed=42,
                             metric=ours_metric)
            idx.train(xt)
            del xt
            for c in chunks():
                idx.add(c)
            print(f"  train+add: {time.perf_counter() - t0:.1f}s", flush=True)
            settle()
            for nprobe in NPROBE_GRID:
                qps, lbl = time_qps(lambda q: idx.search(q, k=10, nprobe=nprobe), xq)
                r = recall_at_k(lbl, gt, k=10)
                print(f"  {name} nprobe={nprobe:<3}  recall={r:.3f}  qps={qps:.0f}", flush=True)
                pts.append((r, qps))
        else:
            quantizer = flat_q(d)
            idx = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8, fmetric)
            if "IVFPQraw" in name:
                idx.by_residual = False     # raw-vector encoding, like ours
            idx.train(xt)
            del xt
            for c in chunks():
                idx.add(c)
            print(f"  train+add: {time.perf_counter() - t0:.1f}s", flush=True)
            settle()
            for nprobe in NPROBE_GRID:
                idx.nprobe = nprobe
                qps, _ = time_qps(lambda q: idx.search(q, 10), xq)
                _, lbl = idx.search(xq, 10)
                r = recall_at_k(lbl, gt, k=10)
                print(f"  {name} nprobe={nprobe:<3}  recall={r:.3f}  qps={qps:.0f}", flush=True)
                pts.append((r, qps))
    else:
        raise SystemExit(f"unknown series: {name}")

    return pts


def merge_into_json(dataset: str, name: str, pts: list[tuple[float, float]],
                    metric: str = "l2") -> None:
    out_json = ROOT / "benchmarks" / f"{results_stem(dataset, metric)}.json"
    series = json.loads(out_json.read_text()) if out_json.exists() else {}
    series[name] = pts
    out_json.write_text(json.dumps(series, indent=2))
    print(f"merged {name} -> {out_json}", flush=True)


# ---- plotting ---------------------------------------------------------------

def plot(dataset: str, metric: str = "l2") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, _, title_dataset = DATASETS[dataset]
    stem = results_stem(dataset, metric)
    out_json = ROOT / "benchmarks" / f"{stem}.json"
    out_path = ROOT / "benchmarks" / f"{stem}.png"
    series = json.loads(out_json.read_text())

    fig, ax = plt.subplots(figsize=(8, 6))
    style = {
        "ours-HNSW":         dict(marker="o", linestyle="-",  color="C0"),
        "faiss-HNSW":        dict(marker="o", linestyle="--", color="C0", alpha=0.6),
        "ours-IVFPQ(M=8)":   dict(marker="s", linestyle="-",  color="C1"),
        "faiss-IVFPQ(M=8)":  dict(marker="s", linestyle="--", color="C1", alpha=0.6),
        "ours-IVFPQ(M=16)":  dict(marker="^", linestyle="-",  color="C2"),
        "faiss-IVFPQ(M=16)": dict(marker="^", linestyle="--", color="C2", alpha=0.6),
        "ours-IVFPQfs(M=32)":  dict(marker="D", linestyle="-",  color="C3"),
        "faiss-IVFPQfs(M=32)": dict(marker="D", linestyle="--", color="C3", alpha=0.6),
        "faiss-IVFPQraw(M=8)":  dict(marker="s", linestyle=":", color="C1", alpha=0.6),
        "faiss-IVFPQraw(M=16)": dict(marker="^", linestyle=":", color="C2", alpha=0.6),
    }
    for name in (SERIES if metric == "l2" else SERIES_IP):
        pts = series.get(name)
        if not pts:
            continue
        recalls = [p[0] for p in pts]
        qps     = [p[1] for p in pts]
        ax.plot(recalls, qps, label=name, **style.get(name, {}))

    ax.set_xlabel("recall@10")
    ax.set_ylabel("QPS (queries per second)")
    ax.set_yscale("log")
    label = {"l2": "L2", "cosine": "cosine (normalized, IP)", "ip": "raw inner product"}[metric]
    ax.set_title(f"{title_dataset}, {label}: recall@10 vs QPS (Apple M2 Pro)")
    ax.grid(True, which="both", alpha=0.3)
    # IVF curves hug the left edge and HNSW the right; the middle is empty.
    ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=tuple(DATASETS), default="sift1m")
    p.add_argument("--metric", choices=METRICS, default="l2")
    p.add_argument("--series", choices=SERIES_IP, default=None,
                   help="run ONE sweep in this process and merge into the JSON")
    p.add_argument("--plot-only", action="store_true",
                   help="render the PNG from the existing JSON")
    args = p.parse_args()

    if args.plot_only:
        plot(args.dataset, args.metric)
        return

    if args.series:
        pts = run_series(args.dataset, args.series, args.metric)
        merge_into_json(args.dataset, args.series, pts, args.metric)
        return

    # Orchestrator: one subprocess per series so memory is returned to the
    # OS between sweeps, then render.
    for name in (SERIES if args.metric == "l2" else SERIES_IP):
        if name.startswith("faiss") and not HAVE_FAISS:
            print(f"skipping {name} (faiss not installed)", flush=True)
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--dataset", args.dataset, "--metric", args.metric, "--series", name],
            check=True,
        )
    plot(args.dataset, args.metric)


if __name__ == "__main__":
    main()
