# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals, assert]
"""First empirical validation of the OOD consistency adjustment.

Exercises ``Samples.enforce_ood_consistency`` (paper Def. 4.5 /
Algorithm 1): records iteration counts, asserts the post-adjustment
constraint ``max alpha(OOD) <= xi`` (empirical Prop. B.1), and measures
the collateral damage on in-distribution coverage.

Run::

    uv run python -m experiments.benchmark.run_ood_adjustment --quick
"""

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import typer

from experiments.benchmark.common import (
    build_odd,
    normalize_fit_apply,
    score_autosafe,
    write_config,
    write_dat,
)
from experiments.benchmark.synthetic_odds import SyntheticODD, get_odd

NPArray = npt.NDArray[np.float64]
DEFAULT_OUTDIR = Path("experiments/benchmark/results/ood_adjustment")


def _uniform_ood(odd: SyntheticODD, m: int, rng: np.random.Generator) -> NPArray:
    """Uniform points on the 2x box, rejected if inside the true ODD.

    Returns:
        NPArray: (m, D) RAW out-of-distribution points.
    """
    lo, hi = odd.enlarged_bbox(2.0)
    out: list[NPArray] = []
    while sum(len(a) for a in out) < m:
        batch = rng.uniform(lo, hi, size=(4 * m + 64, odd.dim))
        out.append(batch[~odd.contains(batch)])
    return np.vstack(out)[:m]


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Run the study; write iteration/collateral dats and results."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    zeta = 0.6
    if quick:
        datasets = ["linear2d"]
        n_id = 100
        m_grid = [20]
        xi_grid = [0.3]
        c_grid = [0.9]
        n_val = 1000
    else:
        datasets = ["linear2d", "annulus2d"]
        n_id = 1000
        m_grid = [10, 100, 1000]
        xi_grid = [0.1, 0.3, 0.5]
        c_grid = [0.5, 0.8, 0.9, 0.99]
        n_val = 100_000

    rows: list[dict] = []
    for ds in datasets:
        odd_gt = get_odd(ds)
        id_raw = odd_gt.sample_id(n_id, rng)
        val_raw, labels = odd_gt.sample_validation(n_val, rng)
        id_eval_raw = odd_gt.sample_id(min(n_id, 500), rng)
        norm, (id_n, val_n, id_eval_n) = normalize_fit_apply(
            id_raw, val_raw, id_eval_raw
        )
        for m in m_grid:
            ood_raw = _uniform_ood(odd_gt, m, rng)
            ood_n = np.asarray(norm.transform(ood_raw), dtype=float)
            for xi in xi_grid:
                for c in c_grid:
                    odd = build_odd(id_n, mode="calibrated")
                    pred_pre = score_autosafe(odd, val_n) >= zeta
                    id_alpha_pre = score_autosafe(odd, id_eval_n)
                    summary = odd.enforce_ood_consistency(ood_n, xi=xi, shrink_factor=c)
                    post_max = float(np.max(score_autosafe(odd, ood_n)))
                    assert post_max <= xi + 1e-9, (
                        f"Prop B.1 violated: max OOD alpha {post_max} > xi {xi}"
                    )
                    pred_post = score_autosafe(odd, val_n) >= zeta
                    id_alpha_post = score_autosafe(odd, id_eval_n)
                    iterations = summary["iterations"]
                    adj = summary["adjusted_kernels"]
                    assert isinstance(iterations, int)
                    assert isinstance(adj, dict)
                    rows.append({
                        "dataset": ds,
                        "m": m,
                        "xi": xi,
                        "c": c,
                        "iterations": iterations,
                        "max_ood_affinity_post": post_max,
                        "precision_pre": _precision(pred_pre, labels),
                        "precision_post": _precision(pred_post, labels),
                        "recall_pre": _recall(pred_pre, labels),
                        "recall_post": _recall(pred_post, labels),
                        "id_drop_frac": float(
                            np.mean((id_alpha_pre >= zeta) & (id_alpha_post < zeta))
                        ),
                        "kernels_touched": len(adj),
                        "max_kernel_scalings": max(adj.values(), default=0),
                    })

    outdir = Path(outdir)
    df = pl.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.write_csv(outdir / "results.csv")
    write_dat(
        outdir / "ood_iterations.dat",
        {
            "dataset": df["dataset"].to_list(),
            "m": df["m"].to_list(),
            "xi": df["xi"].to_list(),
            "c": df["c"].to_list(),
            "iterations": df["iterations"].to_list(),
        },
    )
    write_dat(
        outdir / "ood_collateral.dat",
        {
            "dataset": df["dataset"].to_list(),
            "m": df["m"].to_list(),
            "xi": df["xi"].to_list(),
            "c": df["c"].to_list(),
            "recall_pre": df["recall_pre"].to_list(),
            "recall_post": df["recall_post"].to_list(),
            "precision_pre": df["precision_pre"].to_list(),
            "precision_post": df["precision_post"].to_list(),
            "id_drop_frac": df["id_drop_frac"].to_list(),
        },
    )
    write_config(
        outdir,
        {
            "experiment": "ood_adjustment",
            "quick": quick,
            "seed": seed,
            "zeta": zeta,
            "datasets": datasets,
            "n_id": n_id,
        },
        start_time=start,
    )
    typer.echo(f"OOD adjustment done: {len(rows)} configs -> {outdir}")


def _precision(pred: npt.NDArray[np.bool_], labels: npt.NDArray[np.bool_]) -> float:
    pp = int(np.count_nonzero(pred))
    return float(np.count_nonzero(pred & labels) / pp) if pp else 1.0


def _recall(pred: npt.NDArray[np.bool_], labels: npt.NDArray[np.bool_]) -> float:
    pos = int(np.count_nonzero(labels))
    return float(np.count_nonzero(pred & labels) / pos) if pos else 0.0


if __name__ == "__main__":
    typer.run(main)
