"""GIST1M loader.

Same TexMex .fvecs/.ivecs format as SIFT1M, just larger and harder:
1M base vectors at 960 dimensions (~3.6 GB on disk), 1000 queries.
The high dimensionality is what makes this a useful credibility check, 
ANN tricks that work on 128-dim SIFT can fall apart at 960-dim.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

# Reuse the .fvecs/.ivecs readers from sift_loader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sift_loader import fvecs_read, ivecs_read  # noqa: E402

GIST1M_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"  # ~2.6 GB
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "gist"   # <repo>/data/gist


def load_gist1m(root: str | Path = DEFAULT_DIR):
    """Returns (xb, xq, gt), base 1M*960, queries 1000*960, ground-truth 1000*100."""
    root = Path(root)
    xb_path = root / "gist_base.fvecs"
    xq_path = root / "gist_query.fvecs"
    gt_path = root / "gist_groundtruth.ivecs"
    for p in (xb_path, xq_path, gt_path):
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}. Download GIST1M with:\n"
                f"  python {Path(__file__).name} --download"
            )
    return fvecs_read(xb_path), fvecs_read(xq_path), ivecs_read(gt_path)


def download_gist1m(dest: str | Path = DEFAULT_DIR) -> None:
    """Fetches and extracts the GIST1M tarball. ~2.6 GB compressed; expect minutes."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "gist_base.fvecs").exists():
        print(f"already present at {dest}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "gist.tar.gz"
        print(f"downloading {GIST1M_URL} → {tar_path} (~2.6 GB)")
        urllib.request.urlretrieve(GIST1M_URL, str(tar_path))
        print("extracting...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp, filter="data")
        gist_dir = Path(tmp) / "gist"
        for f in gist_dir.iterdir():
            shutil.move(str(f), str(dest / f.name))
    print(f"done. files in {dest}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true",
                   help="download GIST1M into <repo>/data/gist/")
    p.add_argument("--describe", action="store_true",
                   help="describe the dataset on disk")
    args = p.parse_args()

    if args.download:
        download_gist1m()
        return
    if args.describe:
        xb, xq, gt = load_gist1m()
        print(f"base    {xb.shape} dtype={xb.dtype}")
        print(f"queries {xq.shape} dtype={xq.dtype}")
        print(f"truth   {gt.shape} dtype={gt.dtype}")
        return
    p.print_help()


if __name__ == "__main__":
    main()
