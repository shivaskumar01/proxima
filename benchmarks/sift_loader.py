"""SIFT1M loader.

The TexMex / corpus-texmex SIFT files use the .fvecs and .ivecs format:
each record is a 4-byte little-endian dim followed by `dim` floats (or ints).
Vectors are all the same dim, so we read the first one to learn it then
reshape the whole file in one pass.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

SIFT1M_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"  # 161 MB
DEFAULT_DIR = Path.home() / "vectordb" / "data" / "sift"


def _fvecs_mm(path: str | Path) -> np.ndarray:
    """Memmap an .fvecs file as an (n, d+1) int32 view (col 0 = dim)."""
    mm = np.memmap(str(path), dtype=np.int32, mode="r")
    if mm.size == 0:
        return mm.reshape(0, 1)
    d = int(mm[0])
    return mm.reshape(-1, d + 1)


def fvecs_shape(path: str | Path) -> tuple[int, int]:
    mm = _fvecs_mm(path)
    return mm.shape[0], mm.shape[1] - 1


def fvecs_read(path: str | Path, lo: int = 0, hi: int | None = None) -> np.ndarray:
    """Read rows [lo, hi) of an .fvecs file as float32.

    Memmap + chunked column-strip: the old np.fromfile + .copy() version
    held BOTH the raw int32 array and the stripped copy at peak, 7.7 GB
    transient for GIST1M, which alone could push a 16 GB machine into swap.
    This version peaks at the output array + one ~128 MB chunk.
    """
    mm = _fvecs_mm(path)
    n, d = mm.shape[0], mm.shape[1] - 1
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if hi is None:
        hi = n
    out = np.empty((hi - lo, d), dtype=np.float32)
    step = max(1, (1 << 25) // max(d + 1, 1))   # ~128 MB of int32 per chunk
    for c in range(lo, hi, step):
        e = min(hi, c + step)
        # int32 payload copy -> reinterpret the bits as float32.
        out[c - lo:e - lo] = mm[c:e, 1:].copy().view(np.float32)
    return out


def fvecs_chunks(path: str | Path, chunk_rows: int = 100_000):
    """Yield contiguous float32 row-chunks of an .fvecs file.

    Lets index builds stream the base vectors instead of materializing the
    whole matrix next to the index's own internal copy."""
    mm = _fvecs_mm(path)
    n = mm.shape[0]
    for lo in range(0, n, chunk_rows):
        hi = min(n, lo + chunk_rows)
        yield mm[lo:hi, 1:].copy().view(np.float32)


def ivecs_read(path: str | Path) -> np.ndarray:
    a = np.fromfile(str(path), dtype="int32")
    if a.size == 0:
        return np.zeros((0, 0), dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy()


def load_sift1m(root: str | Path = DEFAULT_DIR):
    """Returns (xb, xq, gt), base 1M*128, queries 10K*128, ground-truth 10K*100."""
    root = Path(root)
    xb_path = root / "sift_base.fvecs"
    xq_path = root / "sift_query.fvecs"
    gt_path = root / "sift_groundtruth.ivecs"
    for p in (xb_path, xq_path, gt_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}. Download SIFT1M with:\n"
                f"  python {Path(__file__).name} --download"
            )
    return fvecs_read(xb_path), fvecs_read(xq_path), ivecs_read(gt_path)


def download_sift1m(dest: str | Path = DEFAULT_DIR) -> None:
    """Fetches and extracts the SIFT1M tarball. ~161 MB compressed."""
    import tarfile, tempfile, shutil

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "sift_base.fvecs").exists():
        print(f"already present at {dest}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "sift.tar.gz"
        print(f"downloading {SIFT1M_URL} → {tar_path}")
        urllib.request.urlretrieve(SIFT1M_URL, str(tar_path))
        print("extracting...")
        with tarfile.open(tar_path, "r:gz") as tf:
            # filter='data' rejects absolute paths and special files (the
            # default in Python 3.14). SIFT tarball is data-only so this is safe.
            tf.extractall(tmp, filter="data")
        sift_dir = Path(tmp) / "sift"
        for f in sift_dir.iterdir():
            shutil.move(str(f), str(dest / f.name))
    print(f"done. files in {dest}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true",
                   help="download SIFT1M into ~/vectordb/data/sift/")
    p.add_argument("--describe", action="store_true",
                   help="describe the dataset on disk")
    args = p.parse_args()

    if args.download:
        download_sift1m()
        return
    if args.describe:
        xb, xq, gt = load_sift1m()
        print(f"base    {xb.shape} dtype={xb.dtype}")
        print(f"queries {xq.shape} dtype={xq.dtype}")
        print(f"truth   {gt.shape} dtype={gt.dtype}")
        return
    p.print_help()


if __name__ == "__main__":
    main()
