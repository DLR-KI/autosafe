# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[magic-value-comparison, docstring-missing-exception, boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals, complex-structure, too-many-branches, too-many-return-statements]
r"""Baseline comparison against one-class classifiers.

Backs the paper's baseline-comparison claim: compares the autoSAFE
affinity against KDE, One-Class SVM, k-NN distance, Isolation Forest, a
GMM, and the convex hull on the synthetic suite, with an
order-independence check. Continuous-score wrappers live here, not in
``src/autosafe`` (the repo baselines stay binary).

Run::

    uv run python -m experiments.benchmark.run_baseline_comparison \
        --quick
"""

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import typer
from scipy.spatial import Delaunay, KDTree
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.svm import OneClassSVM

from experiments.benchmark.common import (
    boundary_error,
    build_odd,
    classification_metrics,
    confusion_at,
    normalize_fit_apply,
    score_autosafe,
    write_config,
    write_dat,
)
from experiments.benchmark.synthetic_odds import get_odd

NPArray = npt.NDArray[np.float64]
DEFAULT_OUTDIR = Path("experiments/benchmark/results/baseline_comparison")


def _score(
    method: str, id_pts: NPArray, val: NPArray, rng: np.random.Generator
) -> NPArray:
    """Return continuous membership scores (higher = more inside)."""
    if method == "autosafe":
        return score_autosafe(build_odd(id_pts, mode="calibrated"), val)
    if method == "kde":
        best: NPArray | None = None
        best_ll = -np.inf
        for bw in ("scott", "silverman", 0.05, 0.1, 0.2, 0.5):
            kde = KernelDensity(bandwidth=bw).fit(id_pts)
            ll = kde.score(id_pts)
            if ll > best_ll:
                best_ll, best = ll, kde.score_samples(val)
        return np.asarray(best, dtype=float)
    if method == "ocsvm":
        best = None
        best_margin = -np.inf
        for nu in (0.01, 0.05, 0.1):
            for gamma in ("scale", 1.0, 10.0):
                svm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(id_pts)
                margin = float(np.mean(svm.decision_function(id_pts)))
                if margin > best_margin:
                    best_margin = margin
                    best = svm.decision_function(val)
        return np.asarray(best, dtype=float)
    if method == "knn_dist":
        tree = KDTree(id_pts)
        k = min(5, len(id_pts))
        d, _ = tree.query(val, k=k)
        return -np.asarray(d[:, -1] if d.ndim == 2 else d, dtype=float)
    if method == "iforest":
        iso = IsolationForest(n_estimators=200, random_state=int(rng.integers(1 << 30)))
        iso.fit(id_pts)
        return np.asarray(iso.score_samples(val), dtype=float)
    if method == "gmm":
        best = None
        best_bic = np.inf
        for nc in (1, 2, 4, 8):
            if nc > len(id_pts):
                break
            gmm = GaussianMixture(n_components=nc, random_state=0).fit(id_pts)
            bic = gmm.bic(id_pts)
            if bic < best_bic:
                best_bic = bic
                best = gmm.score_samples(val)
        return np.asarray(best, dtype=float)
    if method == "convex_hull":
        try:
            tri = Delaunay(id_pts)
            return (tri.find_simplex(val) >= 0).astype(float)
        except Exception:  # ruff: ignore[blind-except]  -- Qhull failure -> all-outside fallback
            return np.zeros(len(val), dtype=float)
    raise ValueError(f"unknown method {method!r}")


def _order_independence(id_pts: NPArray, val: NPArray, seed: int) -> float:
    """Max abs deviation of autoSAFE scores across 5 ID permutations.

    Returns:
        float: Largest absolute score difference to the unpermuted ODD.
    """
    rng = np.random.default_rng(seed)
    base = score_autosafe(build_odd(id_pts, mode="calibrated"), val)
    worst = 0.0
    for _ in range(5):
        perm = rng.permutation(len(id_pts))
        s = score_autosafe(build_odd(id_pts[perm], mode="calibrated"), val)
        worst = max(worst, float(np.max(np.abs(s - base))))
    return worst


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Run the baseline comparison; write results, dats and config.

    Args:
        quick (bool): Run the small ``--quick`` smoke configuration
            instead of the full-size sweep.
        seed (int): Random seed.
        outdir (Path): Directory the results are written to.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    if quick:
        datasets = ["linear2d", "annulus2d"]
        n_anchor = [50]
        methods = ["autosafe", "kde", "knn_dist", "convex_hull"]
        n_val = 1000
    else:
        datasets = ["linear2d", "annulus2d", "twoblobs2d", "banana2d", "poly5d"]
        n_anchor = [10, 100, 1000]
        methods = [
            "autosafe",
            "kde",
            "ocsvm",
            "knn_dist",
            "iforest",
            "gmm",
            "convex_hull",
        ]
        n_val = 100_000

    zeta = 0.5
    rows: list[dict] = []
    fixed_rows: list[dict] = []
    for ds in datasets:
        odd = get_odd(ds)
        val_raw, labels = odd.sample_validation(n_val, rng)
        for n in n_anchor:
            id_raw = odd.sample_id(n, rng)
            _, (id_n, val_n) = normalize_fit_apply(id_raw, val_raw)
            for method in methods:
                scores = _score(method, id_n, val_n, rng)
                m = classification_metrics(scores, labels)
                order_dev = (
                    _order_independence(id_n, val_n, seed)
                    if method == "autosafe"
                    else float("nan")
                )
                rows.append({
                    "dataset": ds,
                    "n_anchor": n,
                    "method": method,
                    "order_dev": order_dev,
                    **m,
                })
                # Fixed-threshold operating point + boundary error
                # (backs the paper's fixed-threshold claim): autoSAFE at
                # zeta=0.5 vs the hull's binary membership. Boundary
                # distances are measured in the RAW ground-truth frame.
                if method in {"autosafe", "convex_hull"}:
                    pred = scores >= zeta
                    fixed_rows.append({
                        "dataset": ds,
                        "n_anchor": n,
                        "method": method,
                        **{
                            k: v
                            for k, v in confusion_at(scores, labels, zeta).items()
                            if k not in {"tp", "fp", "tn", "fn"}
                        },
                        **boundary_error(val_raw, labels, pred),
                    })

    outdir = Path(outdir)
    df = pl.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.write_csv(outdir / "results.csv")

    for ds in datasets:
        sub = df.filter(pl.col("dataset") == ds).sort(["method", "n_anchor"])
        write_dat(
            outdir / f"aupr_vs_n_{ds}.dat",
            {
                "n": sub["n_anchor"].to_list(),
                "method": sub["method"].to_list(),
                "aupr": sub["aupr"].to_list(),
            },
        )

    fdf = pl.DataFrame(fixed_rows)
    write_dat(
        outdir / "fixed_zeta.dat",
        {c: fdf[c].to_list() for c in fdf.columns},
    )

    write_config(
        outdir,
        {
            "experiment": "baseline_comparison",
            "quick": quick,
            "seed": seed,
            "datasets": datasets,
            "n_anchor": n_anchor,
            "methods": methods,
            "n_val": n_val,
        },
        start_time=start,
    )
    typer.echo(f"baseline comparison done: {len(rows)} rows -> {outdir}")


if __name__ == "__main__":
    typer.run(main)
