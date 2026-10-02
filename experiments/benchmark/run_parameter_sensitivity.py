# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals, complex-structure, too-many-statements]
r"""Hyperparameter ablation.

Backs the paper's parameter-sensitivity claim. Primary grid: the
dimensionless calibrated constants gamma x s (matching the revised
Limitations text), plus a validation-noise multiple m sweep. Secondary
grid: the fixed-parameter kappa x eta on linear2d (for the MCM section).
Metric: AUPR (threshold-free, so the zeta sweep does not confound).

Run::

    uv run python -m experiments.benchmark.run_parameter_sensitivity \
        --quick
"""

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import typer
from scipy.spatial import KDTree

from autosafe.samples import Samples
from experiments.benchmark.common import (
    build_odd,
    classification_metrics,
    normalize_fit_apply,
    score_autosafe,
    write_config,
    write_dat,
)
from experiments.benchmark.synthetic_odds import SyntheticODD, get_odd

NPArray = npt.NDArray[np.float64]
DEFAULT_OUTDIR = Path("experiments/benchmark/results/parameter_sensitivity")


def _aupr(odd: Samples, val_n: NPArray, labels: npt.NDArray[np.bool_]) -> float:
    """Return the AUPR of the ODD's affinity on labelled points."""
    return classification_metrics(score_autosafe(odd, val_n), labels)["aupr"]


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Run the sweep; write heatmap/m/fixed dats, results, config.

    Args:
        quick (bool): Run the small ``--quick`` smoke configuration
            instead of the full-size sweep.
        seed (int): Random seed.
        outdir (Path): Directory the results are written to.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    if quick:
        datasets = ["linear2d"]
        n_list = [100]
        gamma_grid = [1.0]
        s_grid = [1.0, 3.0]
        m_grid = [3.0]
        kappa_grid = [0.5, 1.0]
        eta_grid = [0.5, 1.0]
        n_val = 1000
    else:
        datasets = ["linear2d", "annulus2d", "poly5d"]
        n_list = [100, 1000]
        gamma_grid = [0.5, 1.0, 2.0]
        s_grid = [1.0, 2.0, 3.0, 5.0]
        m_grid = [1.0, 3.0, 5.0]
        kappa_grid = [0.1, 0.5, 1.0, 2.0, 5.0]
        eta_grid = [0.1, 0.5, 1.0, 2.0, 5.0]
        n_val = 50_000

    rows: list[dict] = []
    for ds in datasets:
        odd_gt = get_odd(ds)
        for n in n_list:
            id_raw = odd_gt.sample_id(n, rng)
            val_raw, labels = odd_gt.sample_validation(n_val, rng)
            _, (id_n, val_n) = normalize_fit_apply(id_raw, val_raw)
            for gamma in gamma_grid:
                for s in s_grid:
                    odd = build_odd(id_n, mode="calibrated", gamma=gamma, s=s)
                    rows.append({
                        "dataset": ds,
                        "n": n,
                        "kind": "calibrated",
                        "gamma": gamma,
                        "s": s,
                        "m": float("nan"),
                        "kappa": float("nan"),
                        "eta": float("nan"),
                        "aupr": _aupr(odd, val_n, labels),
                    })
            # m sweep at the default calibration: perturb anchors by
            # m*d_tilde.
            d_tilde = float(np.median(_nn_dists(id_n)))
            odd_def = build_odd(id_n, mode="calibrated", gamma=1.0, s=3.0)
            for m in m_grid:
                val_probe, lab_probe = _perturbed_validation(
                    odd_gt, id_raw, m * d_tilde, n_val, rng
                )
                rows.append({
                    "dataset": ds,
                    "n": n,
                    "kind": "m_sweep",
                    "gamma": 1.0,
                    "s": 3.0,
                    "m": m,
                    "kappa": float("nan"),
                    "eta": float("nan"),
                    "aupr": _aupr(odd_def, val_probe, lab_probe),
                })

    # Secondary fixed-parameter grid on linear2d at the largest N.
    odd_gt = get_odd("linear2d")
    n_fixed = n_list[-1]
    id_raw = odd_gt.sample_id(n_fixed, rng)
    val_raw, labels = odd_gt.sample_validation(n_val, rng)
    _, (id_n, val_n) = normalize_fit_apply(id_raw, val_raw)
    for kappa in kappa_grid:
        for eta in eta_grid:
            odd = build_odd(id_n, mode="manual", kappa=kappa, eta=eta)
            rows.append({
                "dataset": "linear2d",
                "n": n_fixed,
                "kind": "fixed",
                "gamma": float("nan"),
                "s": float("nan"),
                "m": float("nan"),
                "kappa": kappa,
                "eta": eta,
                "aupr": _aupr(odd, val_n, labels),
            })

    outdir = Path(outdir)
    df = pl.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.write_csv(outdir / "results.csv")

    for ds in datasets:
        for n in n_list:
            sub = df.filter(
                (pl.col("dataset") == ds)
                & (pl.col("n") == n)
                & (pl.col("kind") == "calibrated")
            )
            if len(sub):
                write_dat(
                    outdir / f"sensitivity_{ds}_n{n}.dat",
                    {
                        "gamma": sub["gamma"].to_list(),
                        "s": sub["s"].to_list(),
                        "aupr": sub["aupr"].to_list(),
                    },
                )
    fixed = df.filter(pl.col("kind") == "fixed")
    write_dat(
        outdir / "sensitivity_fixed_linear2d.dat",
        {
            "kappa": fixed["kappa"].to_list(),
            "eta": fixed["eta"].to_list(),
            "aupr": fixed["aupr"].to_list(),
        },
    )
    cal = df.filter(pl.col("kind") == "calibrated")
    aupr_vals = np.asarray(cal["aupr"].to_list(), dtype=float)
    spread = (
        float(np.nanmax(aupr_vals) - np.nanmin(aupr_vals)) if len(aupr_vals) else 0.0
    )
    write_config(
        outdir,
        {
            "experiment": "parameter_sensitivity",
            "quick": quick,
            "seed": seed,
            "calibrated_aupr_spread": spread,
            "datasets": datasets,
        },
        start_time=start,
    )
    typer.echo(
        f"parameter sensitivity done: {len(rows)} rows, "
        f"calibrated AUPR spread {spread:.4f} -> {outdir}"
    )


def _nn_dists(pts: NPArray) -> NPArray:
    """Full-space L2 distance from each point to its nearest other one.

    Returns:
        NPArray: (N,) nearest-neighbor distances.
    """
    return KDTree(pts).query(pts, k=2)[0][:, 1]


def _perturbed_validation(
    odd_gt: SyntheticODD,
    id_raw: NPArray,
    sigma_raw: float,
    n: int,
    rng: np.random.Generator,
) -> tuple[NPArray, npt.NDArray[np.bool_]]:
    """Half uniform-on-2x-box, half anchor-perturbed validation points.

    Returns:
        tuple[NPArray, npt.NDArray[np.bool_]]: Points in the same
            normalized frame as the ID anchors, and their ground-truth
            labels.
    """
    half = n // 2
    lo, hi = odd_gt.enlarged_bbox(2.0)
    uni_raw = rng.uniform(lo, hi, size=(n - half, odd_gt.dim))
    idx = rng.integers(0, id_raw.shape[0], size=half)
    pert_raw = id_raw[idx] + rng.normal(0.0, sigma_raw, size=(half, odd_gt.dim))
    val_raw = np.vstack([uni_raw, pert_raw])
    labels = odd_gt.contains(val_raw)
    # Reuse the ID min/max so the frame matches id_n exactly.
    _, (_, val_n) = normalize_fit_apply(id_raw, val_raw)
    return val_n, labels


if __name__ == "__main__":
    typer.run(main)
