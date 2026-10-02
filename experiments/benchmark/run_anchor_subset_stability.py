# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals]
r"""Anchor-subsampling stability from a fixed pool.

Backs the paper's subset-stability claim: how sensitive the recovered
ODD and the P/R curves are to anchor subsampling, "varying size and
selection". The anchor-count-sweep experiment already reports cross-seed
variance over independent ID draws; this experiment answers the sharper
"fixed pool, varying subsets" framing directly: from ONE fixed master
pool of ID points, draw K independent subsamples at each size N, rebuild
the calibrated ODD per draw (including its own normalizer — the deployed
pipeline), and measure

-   AUPR and level-set IoU vs. ground truth (mean +- std over
    draws), and
-   the mean/min/max PAIRWISE IoU between the K level sets at the same
    size (selection stability at fixed size).

Expectation: spread shrinks and pairwise IoU rises toward 1 as N grows.

Run::

    uv run python -m \
        experiments.benchmark.run_anchor_subset_stability --quick
"""

import time
from itertools import combinations, starmap
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import typer

from experiments.benchmark.common import (
    build_odd,
    classification_metrics,
    iou,
    normalize_fit_apply,
    score_autosafe,
    write_config,
    write_dat,
)
from experiments.benchmark.synthetic_odds import get_odd

NPArray = npt.NDArray[np.float64]
DEFAULT_OUTDIR = Path("experiments/benchmark/results/anchor_subset_stability")


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Run the anchor-subset stability sweep; write the table."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    zeta = 0.5
    if quick:
        datasets = ["annulus2d"]
        n_pool = 1000
        n_list = [50, 200]
        n_draws = 3
        n_grid = 2000
    else:
        datasets = ["linear2d", "annulus2d"]
        n_pool = 10_000
        n_list = [100, 300, 1000, 3000]
        n_draws = 10
        n_grid = 50_000

    rows: list[dict] = []
    agg_rows: list[dict] = []
    for ds in datasets:
        odd_obj = get_odd(ds)
        pool = odd_obj.sample_id(n_pool, rng)
        grid_raw, grid_lab = odd_obj.sample_validation(n_grid, rng)
        for n in n_list:
            masks: list[npt.NDArray[np.bool_]] = []
            auprs: list[float] = []
            ious: list[float] = []
            for draw in range(n_draws):
                idx = rng.choice(n_pool, size=n, replace=False)
                # Deployed pipeline per draw: normalizer fit on the
                # subsample.
                _, (id_n, grid_n) = normalize_fit_apply(pool[idx], grid_raw)
                odd = build_odd(id_n, mode="calibrated")
                scores = score_autosafe(odd, grid_n)
                mask = scores >= zeta
                masks.append(mask)
                aupr = classification_metrics(scores, grid_lab)["aupr"]
                iou_true = iou(mask, grid_lab)
                auprs.append(aupr)
                ious.append(iou_true)
                rows.append({
                    "dataset": ds,
                    "n": n,
                    "draw": draw,
                    "aupr": aupr,
                    "iou_true": iou_true,
                })
            pairwise = list(starmap(iou, combinations(masks, 2)))
            agg_rows.append({
                "dataset": ds,
                "n": n,
                "aupr_mean": float(np.mean(auprs)),
                "aupr_std": float(np.std(auprs)),
                "iou_true_mean": float(np.mean(ious)),
                "iou_true_std": float(np.std(ious)),
                "iou_pairwise_mean": float(np.mean(pairwise)),
                "iou_pairwise_min": float(np.min(pairwise)),
                "iou_pairwise_max": float(np.max(pairwise)),
            })

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(outdir / "results.csv")
    adf = pl.DataFrame(agg_rows)
    write_dat(
        outdir / "subset_stability.dat",
        {c: adf[c].to_list() for c in adf.columns},
    )
    write_config(
        outdir,
        {
            "experiment": "anchor_subset_stability",
            "quick": quick,
            "seed": seed,
            "datasets": datasets,
            "n_pool": n_pool,
            "n_list": n_list,
            "n_draws": n_draws,
            "n_grid": n_grid,
            "zeta": zeta,
            "note": (
                "Fixed-pool anchor-subsampling stability: K independent "
                "subsets per size from ONE master ID pool; per-draw calibrated ODD "
                "with its own normalizer (deployed pipeline). iou_pairwise_* compares "
                "the zeta=0.5 level sets of different draws at the same size on a "
                "shared raw-frame grid."
            ),
        },
        start_time=start,
    )
    typer.echo(f"anchor subset stability done: {len(rows)} draws -> {outdir}")


if __name__ == "__main__":
    typer.run(main)
