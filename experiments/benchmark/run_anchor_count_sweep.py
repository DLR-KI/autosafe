# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals]
r"""Per-N precision/recall curves replacing the averaged Figure 2.

Backs the paper's Monte Carlo per-N curve figure. Reproduces the
camera-ready Monte Carlo Figure-2 configuration: linear2d, FIXED
kappa=eta=1, lam_rel=e^-10, on the RAW [-5,5]^2 coordinates (no
normalization, no calibration), validation uniform on the raw [-10,10]^2
box. The curve-R^2 is between the ODD-referenced curve and the
convex-hull-referenced curve as zeta sweeps (main.tex:609), exactly as
the paper computes it. Thresholding is evaluated through the equivalent
stable score S=-log(1-alpha), so the zeta=1 endpoint and large-N
rankings are not corrupted by linear-space affinity saturation.

Run::

    uv run python -m experiments.benchmark.run_anchor_count_sweep \
        --quick
"""

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import typer
from scipy.spatial import Delaunay

from experiments.benchmark.common import (
    build_odd,
    classification_metrics,
    curve_r2,
    default_thresholds,
    pr_curves,
    write_config,
    write_dat,
)
from experiments.benchmark.synthetic_odds import get_odd

NPArray = npt.NDArray[np.float64]
DEFAULT_OUTDIR = Path("experiments/benchmark/results/anchor_count_sweep")


def _hull_labels(anchors: NPArray, pts: NPArray) -> npt.NDArray[np.bool_]:
    """Convex-hull membership of ``pts`` w.r.t. the anchor hull.

    Returns:
        npt.NDArray[np.bool_]: (M,) membership mask; all False if the
            hull is degenerate.
    """
    try:
        return Delaunay(anchors).find_simplex(pts) >= 0
    except Exception:  # ruff: ignore[blind-except]
        return np.zeros(len(pts), dtype=bool)


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Run the sweep; write per-N curve, AUPR/R2 dats, results.csv.

    Args:
        quick (bool): Run the small ``--quick`` smoke configuration
            instead of the full-size sweep.
        seed (int): Random seed.
        outdir (Path): Directory the results are written to.
    """
    start = time.perf_counter()
    odd_gt = get_odd("linear2d")
    thresholds = default_thresholds(101 if quick else 256)
    with np.errstate(divide="ignore"):
        log_thresholds = -np.log1p(-thresholds)
    if quick:
        n_list = [10, 30, 100]
        n_seeds = 2
        n_val = 2000
    else:
        # {5, 25, 50} match the paper's requested per-N grid verbatim;
        # safe to extend because the RNG is re-seeded per (n, seed)
        # cell, so the pre-existing N cells reproduce bit-identically.
        n_list = [3, 5, 10, 25, 30, 50, 100, 300, 1000, 3000, 10000]
        n_seeds = 5
        n_val = 100_000

    rows: list[dict] = []
    per_n_curves: dict[int, dict[str, NPArray]] = {}
    for n in n_list:
        prec_odd_runs, rec_odd_runs = [], []
        aupr_runs, r2p_runs, r2r_runs = [], [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(seed + s)
            # RAW coordinates, fixed parameters -- the paper's MCM
            # config.
            anchors = odd_gt.sample_id(n, rng)
            val, labels_odd = odd_gt.sample_validation(n_val, rng)
            labels_hull = _hull_labels(anchors, val)
            odd = build_odd(anchors, mode="fixed")
            _, survival = odd.affinity_dual(val)
            scores = -np.asarray(survival, dtype=float)

            prec_odd, rec_odd = pr_curves(scores, labels_odd, log_thresholds)
            prec_hull, rec_hull = pr_curves(scores, labels_hull, log_thresholds)
            prec_odd_runs.append(prec_odd)
            rec_odd_runs.append(rec_odd)
            aupr_runs.append(classification_metrics(scores, labels_odd)["aupr"])
            r2p_runs.append(curve_r2(prec_odd, prec_hull))
            r2r_runs.append(curve_r2(rec_odd, rec_hull))

        prec_mean = np.mean(prec_odd_runs, axis=0)
        prec_std = np.std(prec_odd_runs, axis=0)
        rec_mean = np.mean(rec_odd_runs, axis=0)
        rec_std = np.std(rec_odd_runs, axis=0)
        per_n_curves[n] = {
            "zeta": thresholds,
            "log_threshold": log_thresholds,
            "precision_mean": prec_mean,
            "precision_std": prec_std,
            "recall_mean": rec_mean,
            "recall_std": rec_std,
        }
        rows.append({
            "n": n,
            "aupr_mean": float(np.mean(aupr_runs)),
            "aupr_std": float(np.std(aupr_runs)),
            "r2_precision": float(np.mean(r2p_runs)),
            "r2_recall": float(np.mean(r2r_runs)),
        })

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(outdir / "results.csv")
    for n, cur in per_n_curves.items():
        write_dat(outdir / f"pr_n{n}.dat", cur)
    write_dat(
        outdir / "aupr_vs_n.dat",
        {
            "n": df["n"].to_list(),
            "aupr_mean": df["aupr_mean"].to_list(),
            "aupr_std": df["aupr_std"].to_list(),
        },
    )
    write_dat(
        outdir / "curve_r2.dat",
        {
            "n": df["n"].to_list(),
            "r2_precision": df["r2_precision"].to_list(),
            "r2_recall": df["r2_recall"].to_list(),
        },
    )
    write_config(
        outdir,
        {
            "experiment": "anchor_count_sweep",
            "quick": quick,
            "seed": seed,
            "config_note": (
                "fixed kappa=eta=1, raw coords (paper Figure 2); "
                "curves and AUPR evaluated with stable -log(1-alpha) scores"
            ),
            "n_list": n_list,
            "n_seeds": n_seeds,
        },
        start_time=start,
    )
    typer.echo(f"anchor count sweep done: {len(rows)} N values -> {outdir}")


if __name__ == "__main__":
    typer.run(main)
