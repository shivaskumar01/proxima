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


def fvecs_read(path: str | Path) -> np.ndarray:
    a = np.fromfile(str(path), dtype="int32")
    if a.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    d = int(a[0])
    # Each record is (1 + d) int32s; reinterpret payload as float32.
    a = a.reshape(-1, d + 1)[:, 1:]
    return a.copy().view(np.float32)


def ivecs_read(path: str | Path) -> np.ndarray:
    a = np.fromfile(str(path), dtype="int32")
    if a.size == 0:
        return np.zeros((0, 0), dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy()


def load_sift1m(root: str | Path = DEFAULT_DIR):
    """Returns (xb, xq, gt) — base 1M*128, queries 10K*128, ground-truth 10K*100."""
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
