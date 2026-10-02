# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals, too-many-statements]
r"""Sampling-density & duplicate-anchor robustness.

Backs the paper's duplicate-sensitivity claim. The noisy-OR makes a
repeated anchor's contribution grow a -> 1-(1-a)^m, so the ODD depends
on sampling *rate* at fixed support, which motivates normalization /
weighting / deduplication. This experiment measures:
    (A) sampling-density convergence of the calibrated level set (IoU vs
        truth);
    (B) the level-set change under exact/near duplicates, fixed vs
        calibrated, plus a controlled check of the 1-(1-a)^m noisy-OR
        formula;
    (C) that de-duplication before building restores the
        sampling-rate-invariant ODD (IoU vs the no-duplicate baseline ->
        1).

Run::

    uv run python -m experiments.benchmark.run_duplicate_sensitivity \
        --quick
"""

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import typer

from experiments.benchmark.common import (
    build_odd,
    iou,
    level_set_mask,
    normalize_fit_apply,
    score_autosafe,
    write_config,
    write_dat,
)
from experiments.benchmark.synthetic_odds import get_odd

NPArray = npt.NDArray[np.float64]
DEFAULT_OUTDIR = Path("experiments/benchmark/results/duplicate_sensitivity")


def inject_duplicates(
    base: NPArray, frac: float, repeats: int, jitter: float, rng: np.random.Generator
) -> NPArray:
    """Append duplicates of a random fraction of anchors.

    Args:
        base (NPArray): (N, D) unique anchor points.
        frac (float): fraction of base anchors to duplicate (0..1).
        repeats (int): extra copies per selected anchor (>=1).
        jitter (float): std of isotropic Gaussian noise per copy (0 =>
            exact).
        rng (np.random.Generator): seeded RNG.

    Returns:
        NPArray: (N + n_sel*repeats, D) base followed by the added
            copies.
    """
    n = base.shape[0]
    n_sel = round(frac * n)
    if n_sel == 0 or repeats <= 0:
        return base.copy()
    sel = rng.choice(n, size=n_sel, replace=False)
    copies = np.repeat(base[sel], repeats, axis=0)
    if jitter > 0.0:
        copies += rng.normal(0.0, jitter, size=copies.shape)
    return np.vstack([base, copies])


def dedup(points: NPArray, tol: float) -> NPArray:
    """Collapse points within L2 distance ``tol`` to one representative.

    Deterministic: snaps to a tol-grid and keeps the first occurrence.

    Args:
        points (NPArray): (N, D) points.
        tol (float): dedup radius (0 => unchanged).

    Returns:
        NPArray: (K, D) deduplicated points, order-stable.
    """
    if tol <= 0.0:
        return points.copy()
    keys = np.round(points / tol).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def _frame(arm: str, id_raw: NPArray, grid_raw: NPArray) -> tuple[NPArray, NPArray]:
    """Return (id, grid) in the arm's coordinate frame.

    The calibrated arm normalizes to [-1, 1] (deployed path); the fixed
    arm stays on raw coordinates (kappa=eta=1 saturates on normalized
    data, so the raw frame is where the noisy-OR inflation is visible —
    cf. the halo-vs-anchor-count and anchor-count-sweep experiments).
    """
    if arm == "calibrated":
        _, (id_n, grid_n) = normalize_fit_apply(id_raw, grid_raw)
        return id_n, grid_n
    return np.asarray(id_raw, dtype=float), np.asarray(grid_raw, dtype=float)


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Run the study: A density, B duplicates + formula, C dedup.

    Args:
        quick (bool): Run the small ``--quick`` smoke configuration
            instead of the full-size sweep.
        seed (int): Random seed.
        outdir (Path): Directory the results are written to.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    zeta = 0.5
    if quick:
        datasets = ["annulus2d"]
        density_n = [50, 200]
        n_grid = 2000
        n_base = 100
        dup_grid = [(3, 0.5, 0.0)]
        dedup_tols = [0.0, 0.05]
        arms = ["calibrated", "fixed"]
    else:
        datasets = ["annulus2d", "linear2d"]
        density_n = [100, 300, 1000, 3000]
        n_grid = 100000
        n_base = 1000
        dup_grid = [
            (r, f, j) for r in (1, 3, 10) for f in (0.1, 0.5) for j in (0.0, 0.02)
        ]
        dedup_tols = [0.0, 0.01, 0.05]
        arms = ["calibrated", "fixed"]

    rows: list[dict] = []
    density_rows: list[dict] = []
    dup_rows: list[dict] = []
    dedup_rows: list[dict] = []

    for ds in datasets:
        odd_obj = get_odd(ds)
        grid_raw, grid_lab = odd_obj.sample_validation(n_grid, rng)

        prev_mask: npt.NDArray[np.bool_] | None = None
        for n in density_n:
            id_raw = odd_obj.sample_id(n, rng)
            id_f, grid_f = _frame("calibrated", id_raw, grid_raw)
            odd = build_odd(id_f, mode="calibrated")
            mask = level_set_mask(odd, grid_f, zeta)
            iou_true = iou(mask, grid_lab)
            iou_self = iou(mask, prev_mask) if prev_mask is not None else float("nan")
            prev_mask = mask
            density_rows.append({
                "dataset": ds,
                "n": n,
                "iou_true": iou_true,
                "iou_selfstab": iou_self,
            })
            rows.append({
                "study": "A",
                "dataset": ds,
                "arm": "calibrated",
                "n": n,
                "iou_true": iou_true,
                "iou_selfstab": iou_self,
            })

        base_raw = odd_obj.sample_id(n_base, rng)
        for arm in arms:
            base_f, grid_f = _frame(arm, base_raw, grid_raw)
            odd_base = build_odd(base_f, mode=arm)
            base_scores = score_autosafe(odd_base, grid_f)
            base_mask = base_scores >= zeta
            for repeats, frac, jitter in dup_grid:
                dup_f = inject_duplicates(base_f, frac, repeats, jitter, rng)
                odd_dup = build_odd(dup_f, mode=arm)
                dup_scores = score_autosafe(odd_dup, grid_f)
                dalpha = np.abs(dup_scores - base_scores)
                iou_vs_base = iou(dup_scores >= zeta, base_mask)
                dup_rows.append({
                    "dataset": ds,
                    "arm": arm,
                    "repeats": repeats,
                    "frac": frac,
                    "jitter": jitter,
                    "dalpha_max": float(np.max(dalpha)),
                    "dalpha_mean": float(np.mean(dalpha)),
                    "iou_vs_base": iou_vs_base,
                })
                rows.append({
                    "study": "B",
                    "dataset": ds,
                    "arm": arm,
                    "repeats": repeats,
                    "frac": frac,
                    "jitter": jitter,
                    "dalpha_max": float(np.max(dalpha)),
                    "iou_vs_base": iou_vs_base,
                })

        for arm in arms:
            base_f, grid_f = _frame(arm, base_raw, grid_raw)
            odd_base = build_odd(base_f, mode=arm)
            base_mask = score_autosafe(odd_base, grid_f) >= zeta
            worst = inject_duplicates(base_f, 0.5, 10, 0.02, rng)
            for tol in dedup_tols:
                dd = dedup(worst, tol)
                odd_dd = build_odd(dd, mode=arm)
                iou_vs_base = iou(score_autosafe(odd_dd, grid_f) >= zeta, base_mask)
                dedup_rows.append({
                    "dataset": ds,
                    "arm": arm,
                    "tol": tol,
                    "n_after": int(dd.shape[0]),
                    "iou_vs_base": iou_vs_base,
                })
                rows.append({
                    "study": "C",
                    "dataset": ds,
                    "arm": arm,
                    "tol": tol,
                    "iou_vs_base": iou_vs_base,
                })

    # Controlled check of the 1-(1-a)^m noisy-OR formula
    # Must hold the kernel WIDTH fixed across duplication to isolate the
    # pure noisy-OR effect: with the sigma-law's eta>0, duplicating an
    # anchor drives its nearest-neighbor distance to 0 and *changes*
    # sigma (a separate, genuine autoSAFE reaction to duplicates
    # measured in Study B). Using mode="manual" with eta=0 gives
    # sigma=kappa=1 independent of duplication, so exp(-r^2/2)=0.5 at
    # r=sqrt(2 ln2) and the copies each contribute a=0.5.
    formula_rows: list[dict] = []
    single = np.array([[0.0, 0.0], [10.0, 0.0]])  # 2nd anchor far away (inert)
    odd1 = build_odd(single, mode="manual", kappa=1.0, eta=0.0)
    r = float(np.sqrt(2.0 * np.log(2.0)))
    probe = np.array([[r, 0.0]])
    a_single = float(score_autosafe(odd1, probe)[0])
    for m in (1, 2, 3, 5, 10):
        dup = np.vstack([np.zeros((m + 1, 2)), np.array([[10.0, 0.0]])])
        odd_m = build_odd(dup, mode="manual", kappa=1.0, eta=0.0)
        alpha_after = float(score_autosafe(odd_m, probe)[0])
        predicted = 1.0 - (1.0 - a_single) ** (m + 1)
        formula_rows.append({
            "m": m,
            "a_single": a_single,
            "alpha_after": alpha_after,
            "predicted_noisy_or": predicted,
        })

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(outdir / "results.csv")

    d = pl.DataFrame(density_rows)
    write_dat(
        outdir / "density_stability.dat",
        {
            "dataset": d["dataset"].to_list(),
            "n": d["n"].to_list(),
            "iou_true": d["iou_true"].to_list(),
            "iou_selfstab": d["iou_selfstab"].to_list(),
        },
    )
    b = pl.DataFrame(dup_rows)
    write_dat(
        outdir / "duplicate.dat",
        {
            "dataset": b["dataset"].to_list(),
            "arm": b["arm"].to_list(),
            "repeats": b["repeats"].to_list(),
            "frac": b["frac"].to_list(),
            "jitter": b["jitter"].to_list(),
            "dalpha_max": b["dalpha_max"].to_list(),
            "dalpha_mean": b["dalpha_mean"].to_list(),
            "iou_vs_base": b["iou_vs_base"].to_list(),
        },
    )
    c = pl.DataFrame(dedup_rows)
    write_dat(
        outdir / "dedup.dat",
        {
            "dataset": c["dataset"].to_list(),
            "arm": c["arm"].to_list(),
            "tol": c["tol"].to_list(),
            "n_after": c["n_after"].to_list(),
            "iou_vs_base": c["iou_vs_base"].to_list(),
        },
    )
    f = pl.DataFrame(formula_rows)
    write_dat(
        outdir / "formula_check.dat",
        {
            "m": f["m"].to_list(),
            "a_single": f["a_single"].to_list(),
            "alpha_after": f["alpha_after"].to_list(),
            "predicted_noisy_or": f["predicted_noisy_or"].to_list(),
        },
    )

    write_config(
        outdir,
        {
            "experiment": "duplicate_sensitivity",
            "quick": quick,
            "seed": seed,
            "datasets": datasets,
            "density_n": density_n,
            "n_base": n_base,
            "n_grid": n_grid,
            "zeta": zeta,
            "note": (
                "Arms use distinct frames: calibrated normalizes to [-1,1]; fixed "
                "(kappa=eta=1) stays raw (normalized fixed saturates). IoU/dalpha are "
                "within-arm (dup vs no-dup base). formula_check.dat demonstrates the "
                "1-(1-a)^(m+1) noisy-OR inflation on a controlled single anchor."
            ),
        },
        start_time=start,
    )
    typer.echo(f"duplicate sensitivity done: {len(rows)} rows -> {outdir}")


if __name__ == "__main__":
    typer.run(main)
