# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals, too-many-statements, too-many-arguments]
"""Scalability & K-truncation error.

Backs the paper's truncation scalability claims. Quantifies the
Appendix-H K-nearest-anchor approximation: accuracy
(|alpha_K - alpha_full|, membership-flip rate), query latency and
memory vs. N, and exact-vs-approximate nearest-neighbor determinism.
The exact affinity is the production ``Samples.affinity_dual``
(JAX-tiled); the K-truncated affinity reuses the SAME per-anchor
diagonal covariance, restricted to each query's K nearest anchors.

Run::

    uv run python -m experiments.benchmark.run_kernel_truncation --quick
"""

import time
import tracemalloc
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import typer
from scipy.spatial import KDTree

from experiments.benchmark.common import (
    build_odd,
    normalize_fit_apply,
    score_autosafe,
    write_config,
    write_dat,
)
from experiments.benchmark.synthetic_odds import get_odd

NPArray = npt.NDArray[np.float64]
DEFAULT_OUTDIR = Path("experiments/benchmark/results/kernel_truncation")


def truncated_affinity(
    anchors: NPArray,
    inv_diag: NPArray,
    x: NPArray,
    k: int,
    *,
    tree: KDTree | None = None,
    chunk: int = 20_000,
) -> tuple[NPArray, float, float]:
    """K-nearest-anchor approximation of the global affinity.

    Uses the SAME per-anchor diagonal covariance as the exact ODD,
    summing the log-space noisy-OR over only the ``k`` Euclidean-nearest
    anchors of each query (the Appendix-H heuristic).

    Args:
        anchors (NPArray): (N, D) anchor coordinates
            (``odd._anchors_np``).
        inv_diag (NPArray): (N, D) diagonal of each kernel's sigma_inv
            (``odd._inv_diag_np``).
        x (NPArray): (M, D) query points.
        k (int): nearest anchors to keep (clipped to N).
        tree (KDTree | None): prebuilt tree; built here if None.
        chunk (int): query-batch size capping the (chunk, k, D)
            temporary.

    Returns:
        tuple[NPArray, float, float]: (alpha_k (M,), query_seconds,
            tree_build_seconds). query_seconds EXCLUDES the tree build.
    """
    anchors = np.ascontiguousarray(anchors, dtype=float)
    inv_diag = np.ascontiguousarray(inv_diag, dtype=float)
    x = np.ascontiguousarray(x, dtype=float)
    k = int(min(k, anchors.shape[0]))
    t_build = 0.0
    if tree is None:
        tb = time.perf_counter()
        tree = KDTree(anchors)
        t_build = time.perf_counter() - tb
    out = np.empty(x.shape[0], dtype=float)
    t0 = time.perf_counter()
    for lo in range(0, x.shape[0], chunk):
        xb = x[lo : lo + chunk]
        _, idx = tree.query(xb, k=k, workers=1)  # deterministic
        idx = idx.reshape(xb.shape[0], k)
        diff = xb[:, None, :] - anchors[idx]  # (m, k, D)
        mahal = np.einsum("mkd,mkd->mk", diff * diff, inv_diag[idx])
        alpha_i = np.exp(-0.5 * np.maximum(mahal, 0.0))
        log_surv = np.sum(np.log1p(-np.clip(alpha_i, 0.0, 1.0 - 1e-16)), axis=1)
        out[lo : lo + chunk] = -np.expm1(log_surv)
    return out, time.perf_counter() - t0, t_build


def _ivf_flip_rate(
    anchors: NPArray, inv_diag: NPArray, x: NPArray, k: int, *, zeta: float
) -> dict[str, float] | None:
    """Membership-flip rate of FAISS IVF (approximate) vs. exact K-NN.

    Demonstrates why certification mode must use exact search: an
    approximate index can miss high-affinity anchors.

    Returns:
        dict[str, float] | None: Flip rate keyed by ``nprobe<P>``, or
            None if FAISS is unavailable.
    """
    try:
        import faiss  # ruff: ignore[import-outside-top-level]
    except ImportError:
        return None
    n, d = anchors.shape
    k = int(min(k, n))
    a32 = np.ascontiguousarray(anchors, dtype=np.float32)
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    exact, _, _ = truncated_affinity(anchors, inv_diag, x, k)
    exact_in = exact >= zeta
    out: dict[str, float] = {}
    for nprobe in (1, 8):
        nlist = max(1, int(np.sqrt(n)))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(a32)
        index.add(a32)
        index.nprobe = min(nprobe, nlist)
        _, idx = index.search(x32, k)
        idx = np.clip(idx, 0, n - 1).reshape(x.shape[0], k)
        diff = x[:, None, :] - anchors[idx]
        mahal = np.einsum("mkd,mkd->mk", diff * diff, inv_diag[idx])
        alpha_i = np.exp(-0.5 * np.maximum(mahal, 0.0))
        log_surv = np.sum(np.log1p(-np.clip(alpha_i, 0.0, 1.0 - 1e-16)), axis=1)
        approx_in = (-np.expm1(log_surv)) >= zeta
        out[f"nprobe{nprobe}"] = float(np.mean(approx_in != exact_in))
    return out


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
    huge: bool = False,
) -> None:
    """Run the sweep; write error/latency dats, determinism report.

    Args:
        quick (bool): Run the small ``--quick`` smoke configuration
            instead of the full-size sweep.
        seed (int): Random seed.
        outdir (Path): Directory the results are written to.
        huge (bool): Also run N = 600000 (VCAS scale) in the full-size
            sweep; ignored with ``quick``.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    zeta = 0.5
    if quick:
        n_list = [200, 1000]
        k_list = [4, 16]
        m_query = 500
    else:
        n_list = [1000, 10000, 100000]
        k_list = [8, 32, 128, 512]
        m_query = 10000
        if huge:
            n_list.append(600000)  # Johann-only; ~VCAS scale

    odd_obj = get_odd("poly5d")
    rows: list[dict] = []
    determinism = {"exact_deterministic": True, "ivf_flip_rate": None}

    for n in n_list:
        id_raw = odd_obj.sample_id(n, rng)
        val_raw, _ = odd_obj.sample_validation(m_query, rng)
        _norm, (id_n, val_n) = normalize_fit_apply(id_raw, val_raw)
        odd = build_odd(id_n, mode="calibrated", gamma=1.0, s=3.0)

        # Warm up the JAX tile compilation so the timed call excludes
        # it. Must use the SAME query shape as the timed call: a
        # 64-point warmup traces a different point-tile shape, so the
        # first N in the sweep still paid the XLA compile (~0.4 s) and
        # its q_secs_full was not comparable.
        _ = score_autosafe(odd, val_n)
        tracemalloc.start()
        t0 = time.perf_counter()
        alpha_full = score_autosafe(odd, val_n)
        full_secs = time.perf_counter() - t0
        full_py_peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()

        anchors = np.asarray(odd._anchors_np, dtype=float)  # ruff: ignore[private-member-access]
        inv_diag = np.asarray(odd._inv_diag_np, dtype=float)  # ruff: ignore[private-member-access]
        model_mem_mb = (anchors.nbytes + inv_diag.nbytes) / 1e6
        tree = KDTree(anchors)  # shared across K for fair per-K query timing

        for k in k_list:
            tracemalloc.start()
            alpha_k, q_secs, _ = truncated_affinity(
                anchors, inv_diag, val_n, k, tree=tree
            )
            trunc_py_peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
            tb = time.perf_counter()
            _ = KDTree(anchors)
            tree_build_secs = time.perf_counter() - tb

            err = np.abs(alpha_k - alpha_full)
            flip = float(np.mean((alpha_k >= zeta) != (alpha_full >= zeta)))
            rows.append({
                "n": n,
                "k": k,
                "err_max": float(np.max(err)),
                "err_mean": float(np.mean(err)),
                "err_p99": float(np.percentile(err, 99)),
                "flip_zeta05": flip,
                "q_secs_full": full_secs,
                "q_secs_trunc": q_secs,
                "tree_build_secs": tree_build_secs,
                "model_mem_mb": model_mem_mb,
                "full_py_peak_mb": full_py_peak_mb,
                "trunc_py_peak_mb": trunc_py_peak_mb,
            })

        # Determinism: exact-NN truncation is bit-identical across two
        # runs.
        a1, _, _ = truncated_affinity(anchors, inv_diag, val_n, k_list[-1])
        a2, _, _ = truncated_affinity(anchors, inv_diag, val_n, k_list[-1])
        if not np.array_equal(a1, a2):
            determinism["exact_deterministic"] = False
        # Approximate-NN flip rate (largest N only, to bound runtime).
        if n == n_list[-1]:
            determinism["ivf_flip_rate"] = _ivf_flip_rate(
                anchors, inv_diag, val_n, k_list[-1], zeta=zeta
            )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(outdir / "results.csv")

    write_dat(
        outdir / "truncation_error.dat",
        {
            "n": df["n"].to_list(),
            "k": df["k"].to_list(),
            "err_max": df["err_max"].to_list(),
            "err_mean": df["err_mean"].to_list(),
            "err_p99": df["err_p99"].to_list(),
            "flip_zeta05": df["flip_zeta05"].to_list(),
        },
    )
    write_dat(
        outdir / "truncation_latency.dat",
        {
            "n": df["n"].to_list(),
            "k": df["k"].to_list(),
            "q_secs_full": df["q_secs_full"].to_list(),
            "q_secs_trunc": df["q_secs_trunc"].to_list(),
            "tree_build_secs": df["tree_build_secs"].to_list(),
            "model_mem_mb": df["model_mem_mb"].to_list(),
            "full_py_peak_mb": df["full_py_peak_mb"].to_list(),
            "trunc_py_peak_mb": df["trunc_py_peak_mb"].to_list(),
        },
    )
    import json  # ruff: ignore[import-outside-top-level]

    (outdir / "truncation_determinism.json").write_text(
        json.dumps(determinism, indent=2)
    )
    write_config(
        outdir,
        {
            "experiment": "kernel_truncation",
            "quick": quick,
            "huge": huge,
            "seed": seed,
            "dataset": "poly5d",
            "n_list": n_list,
            "k_list": k_list,
            "m_query": m_query,
            "zeta": zeta,
            "note": (
                "Euclidean-NN truncation is a heuristic (kernels are "
                "anisotropic-diagonal); full_py_peak_mb is the Python-tracemalloc "
                "peak and EXCLUDES XLA/JAX device memory. Model memory is the "
                "shared anchors+inv_diag footprint; per-query working sets are "
                "O(M*K*D) truncated vs O(M*N) touched (XLA-tiled) full."
            ),
        },
        start_time=start,
    )
    typer.echo(f"kernel truncation done: {len(rows)} rows -> {outdir}")


if __name__ == "__main__":
    typer.run(main)
