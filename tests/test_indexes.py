"""Correctness tests on synthetic data.

Each test uses Flat as the recall ground truth. HNSW and IVF-PQ should both
match Flat exactly on small data (HNSW) or hit a target recall (IVF-PQ, since
PQ is lossy).
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import numpy as np
import pytest

from vectordb import FlatIndex, HnswIndex, IvfPqIndex


RNG = np.random.default_rng(0)


def gen_data(n: int, d: int) -> np.ndarray:
    return RNG.standard_normal((n, d), dtype=np.float32)


def recall_at_k(pred_labels: np.ndarray, truth_labels: np.ndarray, k: int) -> float:
    hits = 0
    for p, t in zip(pred_labels, truth_labels):
        hits += len(set(p[:k].tolist()) & set(t[:k].tolist()))
    return hits / (len(pred_labels) * k)


# ---- FlatIndex ----------------------------------------------------------

def test_flat_basic_l2():
    d = 16
    data = gen_data(500, d)
    queries = gen_data(20, d)
    idx = FlatIndex(dim=d, metric="l2")
    idx.add(data)
    assert idx.size == 500

    D, L = idx.search(queries, k=5)
    assert D.shape == (20, 5)
    assert L.shape == (20, 5)
    # Distances should be non-decreasing along k.
    assert np.all(np.diff(D, axis=1) >= -1e-5)


def test_flat_returns_self_at_distance_zero():
    d = 8
    data = gen_data(100, d)
    idx = FlatIndex(dim=d, metric="l2")
    idx.add(data)
    D, L = idx.search(data, k=1)
    assert np.array_equal(L[:, 0], np.arange(100))
    assert np.allclose(D[:, 0], 0.0, atol=1e-4)


# ---- HnswIndex ----------------------------------------------------------

def test_hnsw_high_recall_vs_flat():
    """HNSW with reasonable ef should reach near-perfect recall@10 on small data."""
    d = 32
    data = gen_data(2000, d)
    queries = gen_data(100, d)

    flat = FlatIndex(dim=d, metric="l2")
    flat.add(data)
    _, truth = flat.search(queries, k=10)

    hnsw = HnswIndex(dim=d, metric="l2", M=16, ef_construction=200, seed=1)
    hnsw.add(data)
    _, pred = hnsw.search(queries, k=10, ef=64)

    r = recall_at_k(pred, truth, k=10)
    assert r >= 0.95, f"recall@10 too low: {r}"


def test_hnsw_self_query():
    d = 16
    data = gen_data(500, d)
    hnsw = HnswIndex(dim=d, metric="l2", M=16, ef_construction=100)
    hnsw.add(data)
    _, L = hnsw.search(data, k=1, ef=32)
    # Each vector should find itself.
    self_hits = (L[:, 0] == np.arange(500)).sum()
    assert self_hits >= 495, f"only {self_hits}/500 found themselves"


def test_hnsw_level_distribution():
    """Levels should be geometrically distributed (most nodes at level 0)."""
    d = 16
    hnsw = HnswIndex(dim=d, metric="l2", M=16, seed=42)
    hnsw.add(gen_data(5000, d))
    hist = hnsw.level_histogram()
    assert hist[0] >= 4500  # vast majority at level 0
    assert sum(hist) == 5000


# ---- IvfPqIndex ---------------------------------------------------------

def test_ivfpq_train_and_search():
    d = 32         # divisible by M
    M = 8
    nlist = 32
    n = 5000

    data = gen_data(n, d)
    queries = gen_data(50, d)

    flat = FlatIndex(dim=d, metric="l2")
    flat.add(data)
    _, truth = flat.search(queries, k=10)

    idx = IvfPqIndex(dim=d, nlist=nlist, M=M, kmeans_iters=15, seed=7)
    idx.train(data)
    idx.add(data)
    assert idx.is_trained
    assert idx.size == n

    # PQ is lossy; with small d/M we expect recall@10 around 0.4-0.7 at high nprobe.
    _, pred = idx.search(queries, k=10, nprobe=nlist)  # exhaustive scan
    r = recall_at_k(pred, truth, k=10)
    assert r >= 0.30, f"recall@10 with full nprobe too low: {r}"


def test_ivfpq_nprobe_monotonic_recall():
    """Recall should grow (weakly) as nprobe increases."""
    d = 32
    M = 8
    nlist = 16
    n = 2000

    data = gen_data(n, d)
    queries = gen_data(40, d)

    flat = FlatIndex(dim=d, metric="l2")
    flat.add(data)
    _, truth = flat.search(queries, k=10)

    idx = IvfPqIndex(dim=d, nlist=nlist, M=M, kmeans_iters=15, seed=11)
    idx.train(data)
    idx.add(data)

    recalls = []
    for nprobe in (1, 4, 16):
        _, pred = idx.search(queries, k=10, nprobe=nprobe)
        recalls.append(recall_at_k(pred, truth, k=10))
    # Allow ties; nprobe=16 (full scan here) should be best.
    assert recalls[-1] >= recalls[0]


def test_ivfpq_rejects_bad_dim():
    with pytest.raises(ValueError):
        IvfPqIndex(dim=33, nlist=16, M=8)  # 33 not divisible by 8


def test_ivfpq_large_dsub():
    """Regression: GIST1M (dim=960, M=8) hits dsub=120. A previous version
    used a 64-float stack array in encode_vector which overflowed silently.
    This test would crash with stack-buffer-overflow before the fix."""
    d, M = 240, 8           # dsub = 30 -> within old limit
    n = 500
    data = gen_data(n, d)
    idx = IvfPqIndex(dim=d, nlist=16, M=M, kmeans_iters=10, seed=0)
    idx.train(data); idx.add(data)
    _, _ = idx.search(data[:5], k=3, nprobe=4)

    # Now exercise dsub > 64 — would have crashed previously.
    d, M = 960, 8           # dsub = 120
    data = gen_data(n, d)
    idx = IvfPqIndex(dim=d, nlist=16, M=M, kmeans_iters=10, seed=0)
    idx.train(data); idx.add(data)
    D, L = idx.search(data[:5], k=3, nprobe=4)
    assert L.shape == (5, 3)
    assert (L >= 0).all()


# ---- distance kernel correctness ----------------------------------------

@pytest.mark.parametrize("d", [1, 4, 7, 16, 17, 31, 64, 96, 128, 129, 200])
def test_l2_distance_matches_numpy(d):
    """NEON l2sq must match numpy across dims that hit every loop tail.
    The tested dims include non-multiples of 4 and 16 to exercise the
    scalar-tail and 4-wide-tail paths in the unrolled NEON kernel."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, d), dtype=np.float32)
    queries = rng.standard_normal((10, d), dtype=np.float32)
    idx = FlatIndex(dim=d, metric="l2"); idx.add(data)
    D, L = idx.search(queries, k=5)
    for qi in range(10):
        expected = np.sum((data - queries[qi]) ** 2, axis=1)
        for j, lbl in enumerate(L[qi]):
            # Float32 + ffast-math; allow ~1e-3 absolute tolerance.
            assert abs(D[qi, j] - expected[lbl]) < 1e-3, \
                f"d={d}: distance mismatch at (qi={qi}, j={j})"


@pytest.mark.parametrize("d", [4, 17, 64, 129])
def test_ip_distance_matches_numpy(d):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((80, d), dtype=np.float32)
    queries = rng.standard_normal((8, d), dtype=np.float32)
    idx = FlatIndex(dim=d, metric="ip"); idx.add(data)
    D, L = idx.search(queries, k=3)
    for qi in range(8):
        expected = data @ queries[qi]
        for j, lbl in enumerate(L[qi]):
            assert abs(D[qi, j] - expected[lbl]) < 1e-3


# ---- inner-product metric -----------------------------------------------

def test_flat_inner_product_returns_top_dots():
    """IP search returns top-k vectors by descending dot product."""
    d = 16
    data = gen_data(200, d)
    queries = gen_data(20, d)
    idx = FlatIndex(dim=d, metric="ip")
    idx.add(data)
    D, L = idx.search(queries, k=5)

    # Top-1 distance should match np.max of dot products per query.
    full_dots = data @ queries.T   # (200, 20)
    np.testing.assert_allclose(D[:, 0], full_dots.max(axis=0), atol=1e-3)
    # Distances should be descending along k (largest-dot first).
    assert np.all(np.diff(D, axis=1) <= 1e-5)


def test_hnsw_inner_product_metric():
    d = 16
    data = gen_data(800, d)
    queries = gen_data(50, d)

    flat = FlatIndex(dim=d, metric="ip")
    flat.add(data)
    _, truth = flat.search(queries, k=10)

    idx = HnswIndex(dim=d, metric="ip", M=16, ef_construction=100)
    idx.add(data)
    _, pred = idx.search(queries, k=10, ef=64)
    assert recall_at_k(pred, truth, k=10) >= 0.90


# ---- edge cases ---------------------------------------------------------

def test_search_empty_index_returns_sentinels():
    d = 8
    idx = HnswIndex(dim=d, metric="l2")
    queries = gen_data(3, d)
    D, L = idx.search(queries, k=5, ef=10)
    assert D.shape == (3, 5)
    assert (L == -1).all()
    assert (D == -1.0).all()


def test_search_k_larger_than_size():
    d = 8
    data = gen_data(7, d)
    idx = FlatIndex(dim=d, metric="l2")
    idx.add(data)
    queries = gen_data(2, d)
    D, L = idx.search(queries, k=20)
    assert D.shape == (2, 20)
    # First 7 slots are real results; rest are -1 sentinels.
    assert (L[:, 7:] == -1).all()
    assert (L[:, :7] >= 0).all()


def test_hnsw_dimension_mismatch_raises():
    d = 16
    idx = HnswIndex(dim=d, metric="l2")
    bad = gen_data(10, d + 1)
    with pytest.raises(ValueError):
        idx.add(bad)


def test_ivfpq_train_too_few_samples_raises():
    idx = IvfPqIndex(dim=32, nlist=128, M=8)
    with pytest.raises(ValueError):
        idx.train(gen_data(50, 32))   # 50 < nlist


# ---- persistence (round-trip preserves search results) ------------------

def test_flat_save_load(tmp_path):
    d = 16
    data = gen_data(300, d)
    queries = gen_data(20, d)
    idx = FlatIndex(dim=d, metric="l2")
    idx.add(data)
    D1, L1 = idx.search(queries, k=10)

    p = str(tmp_path / "flat.bin")
    idx.save(p)
    loaded = FlatIndex.load(p)
    assert loaded.size == idx.size and loaded.dim == idx.dim

    D2, L2 = loaded.search(queries, k=10)
    assert np.array_equal(L1, L2)
    assert np.allclose(D1, D2)


def test_hnsw_save_load(tmp_path):
    d = 32
    data = gen_data(2000, d)
    queries = gen_data(50, d)
    idx = HnswIndex(dim=d, metric="l2", M=16, ef_construction=200, seed=7)
    idx.add(data)
    D1, L1 = idx.search(queries, k=10, ef=64)

    p = str(tmp_path / "hnsw.bin")
    idx.save(p)
    loaded = HnswIndex.load(p)
    assert loaded.size == idx.size and loaded.dim == idx.dim

    D2, L2 = loaded.search(queries, k=10, ef=64)
    # Bit-for-bit: same graph, same query order, same distance kernel.
    assert np.array_equal(L1, L2)
    assert np.allclose(D1, D2)


def test_ivfpq_save_load(tmp_path):
    d = 32
    data = gen_data(3000, d)
    queries = gen_data(40, d)
    idx = IvfPqIndex(dim=d, nlist=32, M=8, kmeans_iters=15, seed=3)
    idx.train(data)
    idx.add(data)
    D1, L1 = idx.search(queries, k=10, nprobe=4)

    p = str(tmp_path / "ivfpq.bin")
    idx.save(p)
    loaded = IvfPqIndex.load(p)
    assert loaded.is_trained
    assert loaded.size == idx.size
    assert loaded.nlist == idx.nlist
    assert loaded.M == idx.M

    D2, L2 = loaded.search(queries, k=10, nprobe=4)
    assert np.array_equal(L1, L2)
    assert np.allclose(D1, D2)


def test_load_rejects_bad_magic(tmp_path):
    p = tmp_path / "junk.bin"
    p.write_bytes(b"GARB" + b"\x00" * 100)
    with pytest.raises(RuntimeError):
        FlatIndex.load(str(p))


# ---- recall sweep -------------------------------------------------------

@pytest.mark.parametrize("ef_search,min_recall", [(16, 0.55), (64, 0.90), (256, 0.99)])
def test_hnsw_recall_grows_with_ef(ef_search, min_recall):
    d = 32
    data = gen_data(3000, d)
    queries = gen_data(80, d)

    flat = FlatIndex(dim=d, metric="l2"); flat.add(data)
    _, truth = flat.search(queries, k=10)

    idx = HnswIndex(dim=d, metric="l2", M=16, ef_construction=200, seed=2)
    idx.add(data)
    _, pred = idx.search(queries, k=10, ef=ef_search)
    r = recall_at_k(pred, truth, k=10)
    assert r >= min_recall, f"ef={ef_search} recall {r:.3f} < {min_recall}"


# ---- robustness regressions --------------------------------------------

def test_search_k_zero_returns_empty():
    """k=0 must not crash. Heap-based top-k loops would deref empty heap.top()."""
    d = 8
    data = gen_data(50, d)
    queries = gen_data(5, d)

    flat = FlatIndex(dim=d, metric="l2"); flat.add(data)
    D, L = flat.search(queries, k=0)
    assert D.shape == (5, 0) and L.shape == (5, 0)

    hnsw = HnswIndex(dim=d, metric="l2", M=8); hnsw.add(data)
    D, L = hnsw.search(queries, k=0, ef=16)
    assert D.shape == (5, 0) and L.shape == (5, 0)

    ivf = IvfPqIndex(dim=d, nlist=4, M=4)
    ivf.train(gen_data(300, d))   # >= KSUB
    ivf.add(data)
    D, L = ivf.search(queries, k=0, nprobe=2)
    assert D.shape == (5, 0) and L.shape == (5, 0)


def test_ivfpq_list_sizes_before_train():
    """list_sizes() previously indexed inv_labels_[0..nlist) before train()
    populated them, which is OOB and crashed."""
    idx = IvfPqIndex(dim=16, nlist=8, M=4)
    sizes = idx.list_sizes()
    assert sizes == [0] * 8


def test_ivfpq_retrain_resets_ntotal():
    """train() invalidates previously encoded vectors, so it must reset
    ntotal_ too. Otherwise new add() calls start labels at the stale offset
    and the inverted lists end up with phantom holes."""
    d = 16
    train_data = gen_data(400, d)            # >= KSUB
    add_data   = gen_data(20, d)
    idx = IvfPqIndex(dim=d, nlist=4, M=4)
    idx.train(train_data)
    idx.add(add_data)
    assert idx.size == 20

    # Re-training should fully reset.
    idx.train(train_data)
    assert idx.size == 0
    assert sum(idx.list_sizes()) == 0

    # New labels start at 0, not 20.
    idx.add(add_data)
    assert idx.size == 20
    _, L = idx.search(add_data, k=1, nprobe=4)
    assert L.min() >= 0 and L.max() < 20


def test_ivfpq_train_too_few_for_pq_codebooks():
    """KSUB=256 codes per subspace require n >= 256 training points; otherwise
    kmeans clamps internally and the codebook memcpy reads past the end."""
    idx = IvfPqIndex(dim=16, nlist=4, M=4)
    with pytest.raises(ValueError):
        idx.train(gen_data(100, 16))   # 100 < 256


@pytest.mark.parametrize("ctor", [
    lambda: FlatIndex(dim=0),
    lambda: HnswIndex(dim=0),
    lambda: HnswIndex(dim=16, M=0),
    lambda: HnswIndex(dim=16, M=1),
    lambda: HnswIndex(dim=16, M=8, ef_construction=0),
    lambda: IvfPqIndex(dim=0,  nlist=4, M=4),
    lambda: IvfPqIndex(dim=16, nlist=0, M=4),
    lambda: IvfPqIndex(dim=16, nlist=4, M=0),
])
def test_constructors_reject_bad_params(ctor):
    # Strict ValueError-only: previously also accepted ZeroDivisionError
    # to paper over HNSW M<=1 (which divided by log(M)=0). The constructors
    # now validate explicitly; M=0/M=1 must be a clean ValueError.
    with pytest.raises(ValueError):
        ctor()


# ---- parallel / incremental build ----------------------------------------

def test_hnsw_incremental_add_recall():
    """Two add() batches (both above the parallel-build threshold) must yield
    one coherent graph: recall over the combined set stays high."""
    d = 32
    a = gen_data(1500, d)
    b = gen_data(1500, d)
    both = np.vstack([a, b])
    queries = gen_data(80, d)

    flat = FlatIndex(dim=d, metric="l2")
    flat.add(both)
    _, truth = flat.search(queries, k=10)

    idx = HnswIndex(dim=d, metric="l2", M=16, ef_construction=200, seed=3)
    idx.add(a)
    idx.add(b)
    assert idx.size == 3000
    _, pred = idx.search(queries, k=10, ef=64)
    assert recall_at_k(pred, truth, k=10) >= 0.93


def test_hnsw_streaming_small_adds():
    """Many tiny add() calls take the serial insert path (below the parallel
    threshold) and must keep entry-point/max-level bookkeeping intact."""
    d = 16
    data = gen_data(600, d)
    queries = gen_data(40, d)

    flat = FlatIndex(dim=d, metric="l2")
    flat.add(data)
    _, truth = flat.search(queries, k=5)

    idx = HnswIndex(dim=d, metric="l2", M=16, ef_construction=100, seed=9)
    for i in range(0, 600, 30):          # 20 batches of 30, all serial
        idx.add(data[i:i + 30])
    assert idx.size == 600
    _, pred = idx.search(queries, k=5, ef=64)
    assert recall_at_k(pred, truth, k=5) >= 0.93


def test_flat_distances_exact_on_batch_tail():
    """n=7 rows: one 4-wide batch + 3 single-pair tail rows. Distances from
    both paths must agree with numpy — guards the batched-kernel tail."""
    d = 24
    data = gen_data(7, d)
    queries = gen_data(3, d)
    idx = FlatIndex(dim=d, metric="l2")
    idx.add(data)
    D, L = idx.search(queries, k=7)
    for qi in range(3):
        expected = np.sum((data - queries[qi]) ** 2, axis=1)
        for j in range(7):
            assert abs(D[qi, j] - expected[L[qi, j]]) < 1e-3


def test_ivfpq_add_after_load_continues_labels(tmp_path):
    """Labels are assigned from ntotal_; a loaded index must continue the
    sequence instead of restarting at 0."""
    d = 16
    train = gen_data(400, d)
    first = gen_data(30, d)
    second = gen_data(20, d)

    idx = IvfPqIndex(dim=d, nlist=4, M=4)
    idx.train(train)
    idx.add(first)

    p = str(tmp_path / "ivf.bin")
    idx.save(p)
    loaded = IvfPqIndex.load(p)
    loaded.add(second)
    assert loaded.size == 50

    _, L = loaded.search(second, k=1, nprobe=4)
    # New vectors should mostly resolve to the new label range [30, 50).
    assert L.max() < 50
    assert (L >= 30).sum() >= 15


def _build_duplicate_heavy(q):
    """Child-process worker for test_hnsw_duplicate_heavy_build_terminates."""
    import numpy as np
    from vectordb import HnswIndex

    rng = np.random.default_rng(0)
    base = rng.standard_normal((2000, 64), dtype=np.float32)
    # ~30% exact duplicates + a block of all-zero rows: lots of distance
    # ties, the profile that triggered the GIST1M hang.
    dup_idx = rng.integers(0, len(base), size=900)
    data = np.vstack([base, base[dup_idx], np.zeros((100, 64), np.float32)])
    rng.shuffle(data)

    idx = HnswIndex(dim=64, M=16, ef_construction=200, seed=5)
    idx.add(data)          # parallel build (above threshold)
    _, L = idx.search(data[:20], k=3, ef=32)
    q.put(int((L >= 0).all()))


def test_hnsw_duplicate_heavy_build_terminates():
    """Regression: greedy descent must terminate on tie/duplicate-heavy data.

    Under -ffast-math the single-pair and batched distance kernels rounded
    differently (few-ulp deltas on ~5% of pairs); greedy descent compared a
    recomputed single-kernel `best` against batched-kernel neighbors and
    could ping-pong forever between near-tied nodes. GIST1M (~1% duplicate
    descriptors) hung inside its first 100k batch. Fixed by carrying `best`
    (strictly decreasing scalar -> unconditional termination) and dropping
    -ffast-math (kernels bit-identical again). Run in a child process with a
    hard deadline so a regression fails the suite instead of hanging it.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    qout = ctx.Queue()
    p = ctx.Process(target=_build_duplicate_heavy, args=(qout,))
    p.start()
    p.join(timeout=120)
    if p.is_alive():
        p.terminate()
        p.join()
        pytest.fail("duplicate-heavy parallel build did not terminate in 120s")
    assert p.exitcode == 0
    assert qout.get() == 1


# ---- multithread safety -------------------------------------------------

def test_concurrent_search_threadsafe():
    """Multiple Python threads search a shared index — no races, identical results."""
    import threading

    d = 16
    data = gen_data(2000, d)
    queries = gen_data(200, d)
    idx = HnswIndex(dim=d, metric="l2", M=16, ef_construction=100, seed=11)
    idx.add(data)
    expected_D, expected_L = idx.search(queries, k=10, ef=32)

    results = [None] * 4
    def worker(i):
        results[i] = idx.search(queries, k=10, ef=32)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    for D, L in results:
        assert np.array_equal(L, expected_L)
        assert np.allclose(D, expected_D)
