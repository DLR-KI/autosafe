# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals]
r"""Split-conformal membership thresholds for the VCAS and HCAS data.

Backs the conformal thresholds marked in the aviation precision/recall
figure. The recorded states of each use case are split at random into a
calibration set, a test set, and the anchors; the ODD is built on the
anchors exactly as in the calibrated dataset mode (min-max normalization
to [-1, 1] on the ground-truth YAML bounds, calibrated RBF kernels with
the width schedule s_N = sqrt(ln N)), and the threshold is the
split-conformal quantile of the calibration states' log-survival scores
(see ``run_conformal_threshold``). The test set measures the realized
false-exclusion rate.

Consecutive simulator states are close to each other, so a random split
places most calibration states next to an anchor; the thresholds are
therefore optimistic relative to a split by trajectory.

Run::

    uv run python -m experiments.benchmark.run_aviation_conformal \
        --quick
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import typer
import yaml

from autosafe.kernels.rbf import resolve_width_multiple
from autosafe.preprocessing import RangeNormalizer, create_robust_normalization_pipeline
from experiments.benchmark.common import build_odd, write_config
from experiments.benchmark.run_conformal_threshold import (
    alpha_equivalent,
    conformal_score,
    conformal_threshold,
)

DEFAULT_OUTDIR = Path("experiments/benchmark/results/aviation_conformal")
DATASETS = {
    "vcas": Path("data/vcas_state_variables.csv"),
    "hcas": Path("data/hcas_state_variables.csv"),
}


def yaml_bounds_normalizer(limits_yaml: Path) -> RangeNormalizer:
    """Fit the dataset mode's min-max normalizer on the YAML bounds.

    Args:
        limits_yaml (Path): Ground-truth limits YAML of the data set.

    Returns:
        RangeNormalizer: Normalizer mapping each variable's range in the
            YAML to [-1, 1].
    """
    limits = yaml.safe_load(limits_yaml.read_text(encoding="utf-8"))["limits"]
    lower = [float(min(entry["values"])) for entry in limits.values()]
    upper = [float(max(entry["values"])) for entry in limits.values()]
    normalizer = create_robust_normalization_pipeline(
        target_range=(-1.0, 1.0), method="minmax"
    )
    normalizer.fit(np.array([lower, upper], dtype=float))
    return normalizer


def split_states(
    states: np.ndarray, n_cal: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the recorded states into calibration, test and anchor sets.

    Args:
        states (np.ndarray): (N, D) normalized recorded states.
        n_cal (int): Number of calibration states.
        n_test (int): Number of test states.
        seed (int): Random seed.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Calibration, test and
            anchor states.
    """
    perm = np.random.default_rng(seed).permutation(states.shape[0])
    return (
        states[perm[:n_cal]],
        states[perm[n_cal : n_cal + n_test]],
        states[perm[n_cal + n_test :]],
    )


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Compute the conformal thresholds; write ``results.csv``.

    Args:
        quick (bool): Run a small VCAS subsample instead of the full
            data sets.
        seed (int): Random seed of the split.
        outdir (Path): Directory the results are written to.
    """
    start = time.perf_counter()
    if quick:
        datasets, n_states, n_cal, n_test = ["vcas"], 3000, 500, 500
    else:
        datasets, n_states, n_cal, n_test = ["vcas", "hcas"], None, 10_000, 10_000
    eps_list = [0.01, 0.05, 0.10]

    rows: list[dict] = []
    for name in datasets:
        path = DATASETS[name]
        normalizer = yaml_bounds_normalizer(path.with_suffix(".yml"))
        raw = pl.read_csv(path).to_numpy().astype(float)
        if n_states is not None:
            rng = np.random.default_rng(seed)
            raw = raw[np.sort(rng.choice(raw.shape[0], n_states, replace=False))]
        states = np.asarray(normalizer.transform(raw), dtype=float)
        cal, test, anchors = split_states(states, n_cal, n_test, seed)
        odd = build_odd(anchors, mode="calibrated", gamma=1.0)
        cal_scores = conformal_score(odd, cal)
        test_scores = conformal_score(odd, test)
        for eps in eps_list:
            thr, certifiable = conformal_threshold(cal_scores, eps)
            rows.append({
                "dataset": name,
                "n_states": states.shape[0],
                "n_anchors": anchors.shape[0],
                "n_cal": n_cal,
                "n_test": n_test,
                "width_multiple": resolve_width_multiple(
                    "sqrt_log_n", anchors.shape[0]
                ),
                "eps": eps,
                "log_threshold": thr,
                "zeta_hat": alpha_equivalent(thr),
                "one_minus_zeta_hat": (
                    float(np.exp(-thr)) if np.isfinite(thr) else None
                ),
                "test_false_exclusion": (
                    float(np.mean(test_scores < thr)) if certifiable else 0.0
                ),
            })
        typer.echo(f"{name}: done ({time.perf_counter() - start:.0f} s)")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(outdir / "results.csv")
    write_config(
        outdir,
        {
            "experiment": "aviation_conformal",
            "quick": quick,
            "seed": seed,
            "datasets": datasets,
            "n_cal": n_cal,
            "n_test": n_test,
            "eps": eps_list,
            "calibration": {"gamma": 1.0, "s": "sqrt_log_n"},
            "normalization": "min-max to [-1, 1] on the ground-truth YAML bounds",
        },
        start_time=start,
    )


if __name__ == "__main__":
    typer.run(main)
