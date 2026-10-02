# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[magic-value-comparison, boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals]
r"""Boundary halo vs. N — the paper's headline experiment.

Compares three bandwidth arms on linear2d: ``fixed`` (kappa=eta=1, the
paper's MCM parameters, expected halo GROWING with N), ``calibrated``
(gamma=1, s=3, expected halo shrinking), and ``calibrated_sN``
(s=sqrt(ln N), validating thm:AsympODDSimCalibrated). Halo width is
measured on a near-boundary band (closed-form signed distance to the
line x2 = x1 - 3), not a full grid, to stay within budget at large N.

Run::

    uv run python -m experiments.benchmark.run_halo_vs_anchor_count \
        --quick
"""

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import typer
from scipy.spatial import KDTree

from experiments.benchmark.common import (
    build_odd,
    score_autosafe,
    write_config,
    write_dat,
)
from experiments.benchmark.synthetic_odds import get_odd

NPArray = npt.NDArray[np.float64]
DEFAULT_OUTDIR = Path("experiments/benchmark/results/halo_vs_anchor_count")

# linear2d region: x2 >= x1 - 3, i.e. g(x) = x2 - x1 + 3 >= 0. Signed
# distance to the boundary line is g(x) / sqrt(2); outside => g < 0.
_NORMAL = np.array([-1.0, 1.0]) / np.sqrt(2.0)


def _signed_distance_raw(pts: NPArray) -> NPArray:
    """Signed distance to the linear2d boundary in RAW coords.

    Returns:
        NPArray: (M,) signed distances, positive inside.
    """
    return (pts[:, 1] - pts[:, 0] + 3.0) / np.sqrt(2.0)


def _halo_band_raw(n_band: int, max_dist: float, rng: np.random.Generator) -> NPArray:
    """Sample RAW points in an exterior band just outside the boundary.

    Points lie within signed distance ``[-max_dist, 0)`` of the line and
    inside the 2x box, by offsetting boundary points along the inward
    normal (negated for the exterior).

    Returns:
        NPArray: (n_band, 2) RAW exterior band points.
    """
    lo, hi = get_odd("linear2d").enlarged_bbox(2.0)
    out: list[NPArray] = []
    while sum(len(a) for a in out) < n_band:
        batch = rng.uniform(lo, hi, size=(4 * n_band + 64, 2))
        sd = _signed_distance_raw(batch)
        keep = batch[(sd < 0.0) & (sd >= -max_dist)]
        out.append(keep)
    return np.vstack(out)[:n_band]


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Run the sweep; write halo/level-set dats, results.csv, config."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    zeta = 0.5
    if quick:
        n_list = [30, 100, 300]
        n_band = 2000
        n_val = 1000
    else:
        n_list = [30, 100, 300, 1000, 3000, 10000, 30000]
        n_band = 50_000
        n_val = 100_000

    # Runs on RAW [-5,5]^2 coordinates (not normalized): fixed
    # kappa=eta=1 is the paper's MCM regime there and shows the
    # sqrt(ln N) halo growth, whereas on normalized data it is the
    # degenerate saturating case. The calibrated arms are
    # scale-invariant, so raw vs. normalized is identical for them; halo
    # widths are then all in comparable raw units.
    odd_gt = get_odd("linear2d")
    max_band = 6.0  # raw units; wide enough to capture the fixed-param halo
    rows: list[dict] = []
    for n in n_list:
        id_raw = odd_gt.sample_id(n, rng)
        val_raw, labels = odd_gt.sample_validation(n_val, rng)
        band_raw = _halo_band_raw(n_band, max_band, rng)
        d_tilde = float(np.median(KDTree(id_raw).query(id_raw, k=2)[0][:, 1]))
        band_sd = -_signed_distance_raw(band_raw)  # exterior distance >= 0 (raw)
        for arm in ("fixed", "calibrated", "calibrated_sN"):
            if arm == "fixed":
                odd = build_odd(id_raw, mode="fixed")
            elif arm == "calibrated":
                odd = build_odd(id_raw, mode="calibrated", gamma=1.0, s=3.0)
            else:
                odd = build_odd(
                    id_raw, mode="calibrated", gamma=1.0, s=float(np.sqrt(np.log(n)))
                )
            band_alpha = score_autosafe(odd, band_raw)
            inside_band = band_sd[band_alpha >= zeta]
            halo_max = float(np.max(inside_band)) if inside_band.size else 0.0
            halo_p95 = (
                float(np.percentile(inside_band, 95)) if inside_band.size else 0.0
            )
            pred = score_autosafe(odd, val_raw) >= zeta
            tp = int(np.count_nonzero(pred & labels))
            pp = int(np.count_nonzero(pred))
            pos = int(np.count_nonzero(labels))
            rows.append({
                "arm": arm,
                "n": n,
                "halo_max": halo_max,
                "halo_p95": halo_p95,
                "precision": tp / pp if pp else 1.0,
                "recall": tp / pos if pos else 0.0,
                "d_tilde": d_tilde,
            })

    outdir = Path(outdir)
    df = pl.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.write_csv(outdir / "results.csv")
    write_dat(
        outdir / "halo.dat",
        {
            "arm": df["arm"].to_list(),
            "n": df["n"].to_list(),
            "halo_max": df["halo_max"].to_list(),
            "halo_p95": df["halo_p95"].to_list(),
            "precision": df["precision"].to_list(),
            "recall": df["recall"].to_list(),
            "d_tilde": df["d_tilde"].to_list(),
        },
    )

    # Acceptance check: fixed halo should grow, calibrated should not.
    def _trend(arm: str) -> float:
        sub = df.filter(pl.col("arm") == arm).sort("n")
        h = np.asarray(sub["halo_max"].to_list(), dtype=float)
        return float(h[-1] - h[0]) if len(h) >= 2 else 0.0

    write_config(
        outdir,
        {
            "experiment": "halo_vs_anchor_count",
            "quick": quick,
            "seed": seed,
            "zeta": zeta,
            "n_list": n_list,
            "halo_trend_fixed": _trend("fixed"),
            "halo_trend_calibrated": _trend("calibrated"),
            "halo_trend_calibrated_sN": _trend("calibrated_sN"),
            "note": (
                "Expect halo_trend_fixed > 0 (grows) and calibrated trends <= 0 "
                "(shrink). If calibrated does not shrink, STOP and report."
            ),
        },
        start_time=start,
    )
    typer.echo(
        f"halo vs. anchor count done: fixed trend {_trend('fixed'):+.3f}, "
        f"calibrated trend {_trend('calibrated'):+.3f} -> {outdir}"
    )


if __name__ == "__main__":
    typer.run(main)
