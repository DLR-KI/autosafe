# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals, too-many-arguments, too-many-positional-arguments, too-many-statements]
r"""Non-convex "real-world-style" hole.

Backs the paper's non-convex-hole claim: demonstrates the central
advantage over a convex hull: a carved-out region a convex hull cannot
represent. Two sources:

-   ``vcas``: the real VCAS anchor set with a carved sub-box
    (h in [-300, 300] AND tau in [0, 8], an excluded near-CPA regime);
    removed anchors become the OOD set. Constant columns (hdot_int) are
    dropped, matching the paper's n = 4 effective dimensions.
-   ``synthetic``: poly5d with a carved hyperbox (hermetic fallback,
    used by ``--quick`` and the smoke tests).

Reports the false-positive mass *inside the hole* for autoSAFE (with and
without the hole points as OOD samples) and for the convex hull.

Run::

    uv run python -m experiments.benchmark.run_held_out_vcas_hole \
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
DEFAULT_OUTDIR = Path("experiments/benchmark/results/held_out_vcas_hole")
VCAS_CSV = Path("data/vcas_state_variables.csv")
#: Hole-split layout: (id_pts, hole_pts, val_pts, val_labels, probe).
HoleSplit = tuple[NPArray, NPArray, NPArray, npt.NDArray[np.bool_], NPArray]
# Near-CPA exclusion carve (raw VCAS units): h in [-300, 300], tau in
# [0, 8].
VCAS_HOLE = {"h": (-300.0, 300.0), "tau": (0.0, 8.0)}


def _poly5d_with_hole(
    n_id: int, n_val: int, m_ood: int, rng: np.random.Generator
) -> HoleSplit:
    """poly5d region minus a carved hyperbox (well inside the region).

    Returns:
        (id_pts, hole_pts, val_pts, val_labels, probe) where
        ``val_labels`` is true ground-truth membership (region minus
        hole), ``hole_pts`` are in-hole points used as the OOD set, and
        ``probe`` is a DISJOINT held-out in-hole set for the FP metric.
    """
    odd = get_odd("poly5d")
    dim = odd.dim
    # Hole = hyperbox centered at origin, half-width 1.5. The whole box
    # lies inside the poly5d region (corner check: 5*1.5^2 <= 60 and
    # x2 >= x1 - 3 at every corner), so uniform-in-box sampling is
    # valid.
    hole_lo = np.full(dim, -1.5)
    hole_hi = np.full(dim, 1.5)

    def in_hole(x: NPArray) -> npt.NDArray[np.bool_]:
        return np.all((x >= hole_lo) & (x <= hole_hi), axis=1)

    region = odd.sample_id(n_id * 3, rng)
    id_pts = region[~in_hole(region)][:n_id]
    n_probe = 2000
    in_hole_pts = rng.uniform(hole_lo, hole_hi, size=(m_ood + n_probe, dim))
    hole_pts = in_hole_pts[:m_ood]
    probe = in_hole_pts[m_ood:]

    lo, hi = odd.enlarged_bbox(2.0)
    val = rng.uniform(lo, hi, size=(n_val, dim))
    val_labels = odd.contains(val) & ~in_hole(val)
    return id_pts, hole_pts, val, val_labels, probe


def _vcas_with_hole(
    n_id: int, n_val: int, m_ood: int, rng: np.random.Generator
) -> HoleSplit:
    """VCAS anchors minus the near-CPA sub-box (non-convex hole).

    Ground truth = the anchor bounding box (the VCAS ODD is a
    hyperrectangle, r = 0) minus the carved sub-box. Constant columns
    are dropped (n = 4 effective dimensions, cf. paper Appendix G.2).

    Args:
        n_id (int): Anchor subsample size (kept out of the hole).
        n_val (int): Validation points, uniform on the 2x enlarged box.
        m_ood (int): Subsample of removed (in-hole) anchors used as OOD.
        rng (np.random.Generator): Seeded RNG.

    Returns:
        Same tuple layout as :func:`_poly5d_with_hole`.
    """
    df = pl.read_csv(VCAS_CSV)
    cols = [c for c, dt in zip(df.columns, df.dtypes, strict=False) if dt.is_numeric()]
    data = df.select(cols).to_numpy().astype(float)
    keep = [i for i in range(data.shape[1]) if np.ptp(data[:, i]) > 0.0]
    kept_names = [cols[i] for i in keep]
    data = data[:, keep]

    def in_hole(x: NPArray) -> npt.NDArray[np.bool_]:
        mask = np.ones(len(x), dtype=bool)
        for name, (lo, hi) in VCAS_HOLE.items():
            j = kept_names.index(name)
            mask &= (x[:, j] >= lo) & (x[:, j] <= hi)
        return mask

    hole_mask = in_hole(data)
    outside = data[~hole_mask]
    inside = data[hole_mask]
    id_pts = outside[
        rng.choice(len(outside), size=min(n_id, len(outside)), replace=False)
    ]
    # Split the removed in-hole anchors: an OOD set for the adjustment
    # and a DISJOINT held-out probe set for the FP-in-hole metric.
    # Probing with real in-hole data points (not uniform corner points)
    # is essential: the VCAS mass is concentrated, so uniform in-hole
    # points mostly fall outside the hull's span in the other dimensions
    # and would understate the hull's leakage.
    perm = rng.permutation(len(inside))
    hole_pts = inside[perm[: min(m_ood, len(inside))]]
    probe = inside[perm[min(m_ood, len(inside)) : min(m_ood + 2000, len(inside))]]

    # Ground-truth box from the FULL anchor set (r = 0 hyperrectangle).
    lower, upper = data.min(axis=0), data.max(axis=0)
    center, half = 0.5 * (lower + upper), 0.5 * (upper - lower) * 2.0
    val = rng.uniform(center - half, center + half, size=(n_val, data.shape[1]))
    val_box = np.all((val >= lower) & (val <= upper), axis=1)
    return id_pts, hole_pts, val, val_box & ~in_hole(val), probe


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
    source: str = "auto",
    n_id: int = 0,
    m_ood: int = 500,
) -> None:
    """Run the held-out VCAS hole; write hole-FP .dat, results, config.

    ``source``: ``auto`` (vcas when the CSV exists, else synthetic),
    ``vcas``, or ``synthetic``; ``--quick`` always forces synthetic so
    the smoke tests stay hermetic. ``n_id`` = 0 selects the per-source
    default (2000 synthetic, 20000 vcas).
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    zeta = 0.5
    n_val = 2000 if quick else 100_000

    if quick:
        source = "synthetic"
    elif source == "auto":
        source = "vcas" if VCAS_CSV.exists() else "synthetic"

    if source == "vcas":
        n_id = n_id or 20_000
        id_raw, hole_raw, val_raw, labels, probe_raw = _vcas_with_hole(
            n_id, n_val, m_ood, rng
        )
        source = "vcas_hole"
    else:
        n_id = n_id or (200 if quick else 2000)
        id_raw, hole_raw, val_raw, labels, probe_raw = _poly5d_with_hole(
            n_id, n_val, m_ood, rng
        )
        source = "poly5d_hole"

    _norm, (id_n, hole_n, val_n, probe_n) = normalize_fit_apply(
        id_raw, hole_raw, val_raw, probe_raw
    )

    rows: list[dict] = []
    fixed_rows: list[dict] = []

    def _fixed(method: str, val_scores: NPArray) -> None:
        """Fixed-zeta operating point and raw-frame boundary error."""
        pred = val_scores >= zeta
        fixed_rows.append({
            "method": method,
            **{
                k: v
                for k, v in confusion_at(val_scores, labels, zeta).items()
                if k not in {"tp", "fp", "tn", "fn"}
            },
            **boundary_error(val_raw, labels, pred),
        })

    # (i) autoSAFE without OOD.
    odd_plain = build_odd(id_n, mode="calibrated")
    plain_val_scores = score_autosafe(odd_plain, val_n)
    rows.append({
        "method": "autosafe_no_ood",
        "fp_rate_in_hole": float(np.mean(score_autosafe(odd_plain, probe_n) >= zeta)),
        "aupr": classification_metrics(plain_val_scores, labels)["aupr"],
    })
    _fixed("autosafe_no_ood", plain_val_scores)

    # (ii) autoSAFE with the hole points as OOD samples.
    odd_ood = build_odd(id_n, mode="calibrated")
    if len(hole_n) >= 1:
        odd_ood.enforce_ood_consistency(hole_n, xi=0.1, shrink_factor=0.9)
    ood_val_scores = score_autosafe(odd_ood, val_n)
    rows.append({
        "method": "autosafe_with_ood",
        "fp_rate_in_hole": float(np.mean(score_autosafe(odd_ood, probe_n) >= zeta)),
        "aupr": classification_metrics(ood_val_scores, labels)["aupr"],
    })
    _fixed("autosafe_with_ood", ood_val_scores)

    # (iii) convex hull.
    try:
        hull = Delaunay(id_n)
        pred_hull_probe = hull.find_simplex(probe_n) >= 0
        pred_hull_val = hull.find_simplex(val_n) >= 0
    except Exception:  # ruff: ignore[blind-except]
        pred_hull_probe = np.zeros(len(probe_n), dtype=bool)
        pred_hull_val = np.zeros(len(val_n), dtype=bool)
    rows.append({
        "method": "convex_hull",
        "fp_rate_in_hole": float(np.mean(pred_hull_probe)),
        "aupr": classification_metrics(pred_hull_val.astype(float), labels)["aupr"],
    })
    _fixed("convex_hull", pred_hull_val.astype(float))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(outdir / "results.csv")
    write_dat(
        outdir / "hole_fp.dat",
        {
            "method": df["method"].to_list(),
            "fp_rate_in_hole": df["fp_rate_in_hole"].to_list(),
            "aupr": df["aupr"].to_list(),
        },
    )
    fdf = pl.DataFrame(fixed_rows)
    write_dat(
        outdir / "hole_fixed_zeta.dat",
        {c: fdf[c].to_list() for c in fdf.columns},
    )
    write_config(
        outdir,
        {
            "experiment": "held_out_vcas_hole",
            "quick": quick,
            "seed": seed,
            "source": source,
            "zeta": zeta,
            "n_id": n_id,
            "m_ood": m_ood,
            "vcas_available": VCAS_CSV.exists(),
            "vcas_hole": VCAS_HOLE,
        },
        start_time=start,
    )
    typer.echo(f"held-out VCAS hole done ({source}): {len(rows)} methods -> {outdir}")


if __name__ == "__main__":
    typer.run(main)
