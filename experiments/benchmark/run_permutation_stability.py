# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[magic-value-comparison, boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals]
r"""Determinism / order-independence harness.

Backs the paper's determinism claim. Runs the full pipeline (build + OOD
adjustment) on annulus2d under many simultaneous permutations of the ID
and OOD data, then asserts that the per-anchor sigma diagonals (keyed by
anchor COORDINATE, never by index) and the affinity on a fixed grid
match the unpermuted run to <= 1e-12.

Run::

    uv run python -m experiments.benchmark.run_permutation_stability \
        --quick
"""

import json
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import typer

from autosafe.samples import Samples
from experiments.benchmark.common import (
    build_odd,
    kernel_sigma,
    normalize_fit_apply,
    score_autosafe,
)
from experiments.benchmark.run_ood_adjustment import _uniform_ood
from experiments.benchmark.synthetic_odds import get_odd

NPArray = npt.NDArray[np.float64]
DEFAULT_OUTDIR = Path("experiments/benchmark/results/permutation_stability")


def _sigma_by_coord(odd: Samples) -> dict[tuple, NPArray]:
    """Map each rounded anchor coordinate to its sigma diagonal.

    Returns:
        dict[tuple, NPArray]: Sigma diagonal keyed by the anchor
            coordinate rounded to 9 decimals.
    """
    out: dict[tuple, NPArray] = {}
    for s in odd.samples:
        key = tuple(np.round(np.asarray(s.x, dtype=float), 9))
        out[key] = np.diag(kernel_sigma(s))
    return out


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Run the permutation harness; write permutation_report.json.

    Args:
        quick (bool): Run the small ``--quick`` smoke configuration
            instead of the full-size sweep.
        seed (int): Random seed.
        outdir (Path): Directory the results are written to.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    xi, c = 0.3, 0.9
    if quick:
        n_id, n_perm = 100, 4
    else:
        n_id, n_perm = 1000, 20

    odd_gt = get_odd("annulus2d")
    id_raw = odd_gt.sample_id(n_id, rng)
    ood_raw = _uniform_ood(odd_gt, 200 if not quick else 40, rng)
    _norm, (id_n, ood_n) = normalize_fit_apply(id_raw, ood_raw)
    grid = rng.uniform(-1.0, 1.0, size=(10_000 if not quick else 500, 2))

    base_odd = build_odd(id_n, mode="calibrated")
    base_odd.enforce_ood_consistency(ood_n, xi=xi, shrink_factor=c)
    base_sigma = _sigma_by_coord(base_odd)
    base_alpha = score_autosafe(base_odd, grid)

    results = []
    max_sigma_dev = 0.0
    max_alpha_dev = 0.0
    for p in range(n_perm):
        pid = rng.permutation(n_id)
        pod = rng.permutation(len(ood_n))
        odd = build_odd(id_n[pid], mode="calibrated")
        odd.enforce_ood_consistency(ood_n[pod], xi=xi, shrink_factor=c)
        sig = _sigma_by_coord(odd)
        # Compare sigma diagonals matched by coordinate key.
        sdev = 0.0
        for key, val in base_sigma.items():
            sdev = max(sdev, float(np.max(np.abs(val - sig[key]))))
        adev = float(np.max(np.abs(base_alpha - score_autosafe(odd, grid))))
        max_sigma_dev = max(max_sigma_dev, sdev)
        max_alpha_dev = max(max_alpha_dev, adev)
        results.append({
            "permutation": p,
            "max_sigma_dev": sdev,
            "max_alpha_dev": adev,
            "pass": bool(sdev <= 1e-12 and adev <= 1e-12),
        })

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "permutation_stability",
        "quick": quick,
        "seed": seed,
        "n_permutations": n_perm,
        "max_sigma_dev": max_sigma_dev,
        "max_alpha_dev": max_alpha_dev,
        "all_pass": all(r["pass"] for r in results),
        "per_permutation": results,
        "note": (
            "Comparison is coordinate-keyed (index-invariant). If a value "
            "deviation > 1e-12 is caused only by JAX summation order, add the "
            "opt-in canonical_order sort and re-run."
        ),
        "wall_time_seconds": time.perf_counter() - start,
    }
    (outdir / "permutation_report.json").write_text(json.dumps(report, indent=2))
    typer.echo(
        f"permutation stability done: all_pass={report['all_pass']} "
        f"max_sigma_dev={max_sigma_dev:.2e} max_alpha_dev={max_alpha_dev:.2e} "
        f"-> {outdir}"
    )


if __name__ == "__main__":
    typer.run(main)
