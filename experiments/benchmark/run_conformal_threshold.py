# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (grid magic numbers, typer boolean flags,
# driver-length rules); scoped per file since experiment scripts are not
# library code.
# ruff: file-ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-locals]
r"""Split-conformal calibration of the membership threshold zeta.

Backs the paper's threshold-calibration claim (give zeta a
distribution-free coverage guarantee, e.g. split-conformal, bounding the
false-exclusion rate) and its zeta-selection claim (how zeta is chosen
automatically). Chooses zeta from a held-out ID calibration set so that
a fresh ID point is wrongly excluded with probability at most eps, then
verifies the empirical coverage on disjoint test points.

Run::

    uv run python -m experiments.benchmark.run_conformal_threshold \
        --quick
"""

import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import typer

from autosafe.samples import Samples
from experiments.benchmark.common import (
    build_odd,
    normalize_fit_apply,
    write_config,
    write_dat,
)
from experiments.benchmark.synthetic_odds import get_odd

NPArray = npt.NDArray[np.float64]
DEFAULT_OUTDIR = Path("experiments/benchmark/results/conformal_threshold")


def conformal_score(odd: Samples, x: NPArray) -> NPArray:
    """Saturation-free membership score for conformal calibration.

    Returns ``-S(x) = -log(1 - alpha(x)) >= 0`` (higher = more clearly
    inside). The linear affinity ``alpha`` saturates to exactly 1.0 once
    the accumulated kernel mass exceeds ~37 (cf. the paper's
    numerical-evaluation subsection), which makes the empirical quantile
    degenerate in higher dimensions -- measured on poly5d, every
    calibration score was exactly 1.0, so the conformal threshold
    collapsed to 1.0 and excluded nothing. Because ``-S`` is a strictly
    increasing function of ``alpha``, quantiles are equivariant and the
    split-conformal guarantee transfers unchanged, while the score stays
    finite and discriminative.

    Args:
        odd (Samples): The constructed ODD.
        x (NPArray): (M, D) query points.

    Returns:
        NPArray: (M,) scores ``-log(1 - alpha)``; ``+inf`` at exact
            anchors.
    """
    _, survival = odd.affinity_dual(np.asarray(x, dtype=float))
    return -np.asarray(survival, dtype=float)


def alpha_equivalent(log_score: float) -> float:
    """Convert a ``-log(1 - alpha)`` threshold back to the alpha scale.

    Args:
        log_score (float): Threshold in the ``-log(1 - alpha)`` scale.

    Returns:
        float: The equivalent affinity threshold
            ``1 - exp(-log_score)``.
    """
    if not np.isfinite(log_score):
        return 1.0 if log_score > 0 else 0.0
    return float(-np.expm1(-log_score))


def conformal_threshold(cal_scores: NPArray, eps: float) -> tuple[float, bool]:
    """Split-conformal lower threshold bounding false exclusion.

    With ``cal_scores`` the affinities of a held-out ID calibration set,
    the ODD ``{x : alpha(x) >= zeta_hat}`` excludes a fresh exchangeable
    ID point with probability at most ``eps`` (finite-sample,
    distribution-free).

    The threshold is the k-th smallest calibration score with
    ``k = floor(eps * (n + 1))`` (the standard split-conformal lower
    prediction set). ``k == 0`` means ``n`` is too small for level
    ``eps``: no exclusion can be certified, so ``zeta_hat = -inf``.

    Args:
        cal_scores (NPArray): (n,) ID calibration affinities in [0, 1].
        eps (float): target false-exclusion rate in (0, 1).

    Returns:
        tuple[float, bool]: (zeta_hat, certifiable).
    """
    s = np.sort(np.asarray(cal_scores, dtype=float))
    n = s.shape[0]
    k = int(np.floor(eps * (n + 1)))
    if k <= 0:
        return float("-inf"), False
    k = min(k, n)
    return float(s[k - 1]), True


def main(
    quick: bool = False,
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    """Run conformal calibration; write coverage, zeta-vs-eps dats.

    Args:
        quick (bool): Run the small ``--quick`` smoke configuration
            instead of the full-size sweep.
        seed (int): Random seed.
        outdir (Path): Directory the results are written to.
    """
    start = time.perf_counter()
    if quick:
        datasets = ["linear2d"]
        n_list = [400]
        eps_list = [0.1]
        n_repeats = 3
        n_test = 500
        n_out = 500
    else:
        datasets = ["linear2d", "annulus2d", "poly5d"]
        n_list = [500, 1000, 5000]
        eps_list = [0.01, 0.05, 0.10]
        n_repeats = 20
        n_test = 5000
        n_out = 5000

    rows: list[dict] = []
    for ds in datasets:
        odd_obj = get_odd(ds)
        for n in n_list:
            for rep in range(n_repeats):
                rng = np.random.default_rng(seed + 1000 * rep)
                pool = odd_obj.sample_id(n, rng)
                rng.shuffle(pool)
                half = n // 2
                train_raw, cal_raw = pool[:half], pool[half:]
                test_id_raw = odd_obj.sample_id(n_test, rng)
                val_raw, val_lab = odd_obj.sample_validation(n_out * 6, rng)
                out_raw = val_raw[~val_lab][:n_out]

                _, (train_n, cal_n, test_n, out_n) = normalize_fit_apply(
                    train_raw, cal_raw, test_id_raw, out_raw
                )
                odd = build_odd(train_n, mode="calibrated", gamma=1.0, s=3.0)
                # Log-space (survival) score: saturation-free, see
                # conformal_score.
                cal_scores = conformal_score(odd, cal_n)
                test_scores = conformal_score(odd, test_n)
                out_scores = conformal_score(odd, out_n)

                for eps in eps_list:
                    thr, certifiable = conformal_threshold(cal_scores, eps)
                    emp_fe = float(np.mean(test_scores < thr)) if certifiable else 0.0
                    spec_out = float(np.mean(out_scores < thr)) if certifiable else 0.0
                    rows.append({
                        "dataset": ds,
                        "n": n,
                        "eps": eps,
                        "repeat": rep,
                        "log_threshold": thr,
                        "zeta_hat": alpha_equivalent(thr),
                        "emp_false_excl": emp_fe,
                        "specificity_out": spec_out,
                        "certifiable": certifiable,
                    })

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows)
    df.write_csv(outdir / "results.csv")

    # Aggregate across repeats.
    agg = (
        df
        .group_by(["dataset", "n", "eps"])
        .agg(
            pl
            .col("zeta_hat")
            .filter(pl.col("zeta_hat").is_finite())
            .mean()
            .alias("zeta_hat_mean"),
            pl
            .col("log_threshold")
            .filter(pl.col("log_threshold").is_finite())
            .mean()
            .alias("log_threshold_mean"),
            pl.col("emp_false_excl").mean().alias("emp_false_excl_mean"),
            pl.col("emp_false_excl").max().alias("emp_false_excl_max"),
            pl.col("specificity_out").mean().alias("specificity_out_mean"),
            pl.col("certifiable").mean().alias("frac_certifiable"),
        )
        .sort(["dataset", "n", "eps"])
    )
    write_dat(
        outdir / "conformal_coverage.dat",
        {
            "dataset": agg["dataset"].to_list(),
            "n": agg["n"].to_list(),
            "eps": agg["eps"].to_list(),
            "zeta_hat_mean": [
                v if v is not None else float("nan")
                for v in agg["zeta_hat_mean"].to_list()
            ],
            "log_threshold_mean": [
                v if v is not None else float("nan")
                for v in agg["log_threshold_mean"].to_list()
            ],
            "emp_false_excl_mean": agg["emp_false_excl_mean"].to_list(),
            "emp_false_excl_max": agg["emp_false_excl_max"].to_list(),
            "specificity_out_mean": agg["specificity_out_mean"].to_list(),
        },
    )
    for ds in datasets:
        for n in n_list:
            sub = agg.filter((pl.col("dataset") == ds) & (pl.col("n") == n)).sort("eps")
            if sub.height == 0:
                continue
            write_dat(
                outdir / f"zeta_vs_eps_{ds}_n{n}.dat",
                {
                    "eps": sub["eps"].to_list(),
                    "zeta_hat_mean": [
                        v if v is not None else float("nan")
                        for v in sub["zeta_hat_mean"].to_list()
                    ],
                    "emp_false_excl_mean": sub["emp_false_excl_mean"].to_list(),
                },
            )

    write_config(
        outdir,
        {
            "experiment": "conformal_threshold",
            "quick": quick,
            "seed": seed,
            "datasets": datasets,
            "n_list": n_list,
            "eps_list": eps_list,
            "n_repeats": n_repeats,
            "n_test": n_test,
            "n_out": n_out,
            "score": "log-space survival, -log(1-alpha) (saturation-free)",
            "note": (
                "Split-conformal on the LOG-SPACE score -log(1-alpha): threshold = "
                "k-th smallest calibration score, k=floor(eps*(n_cal+1)), which "
                "guarantees P(score(X_test) < threshold) <= eps. Because "
                "-log(1-alpha) is strictly increasing in alpha, quantiles are "
                "equivariant and the guarantee transfers to the alpha scale "
                "(zeta_hat = 1-exp(-threshold)). "
                "The linear affinity was tried first and is DEGENERATE in 5D: it "
                "saturates at exactly 1.0, so every calibration score tied at 1.0 and "
                "the threshold excluded nothing (poly5d, all eps). emp_false_excl is "
                "the measured ID exclusion on a disjoint test set; specificity_out is "
                "the fraction of true-outside points excluded."
            ),
        },
        start_time=start,
    )
    typer.echo(f"conformal threshold done: {len(rows)} rows -> {outdir}")


if __name__ == "__main__":
    typer.run(main)
