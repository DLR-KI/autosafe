# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Smoke tests for the benchmark experiment scripts (--quick configs).

Each test runs one experiment in its miniature configuration and checks
that the expected artifacts exist and that metric columns lie in valid
ranges. These must finish in seconds.
"""

import json
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from experiments.benchmark import (
    run_anchor_count_sweep,
    run_anchor_subset_stability,
    run_baseline_comparison,
    run_conformal_threshold,
    run_covariance_structure,
    run_duplicate_sensitivity,
    run_halo_vs_anchor_count,
    run_held_out_vcas_hole,
    run_kernel_truncation,
    run_ood_adjustment,
    run_parameter_sensitivity,
    run_permutation_stability,
)


def _assert_metric_range(df: pl.DataFrame, col: str) -> None:
    vals = np.asarray(df[col].to_list(), dtype=float)
    vals = vals[~np.isnan(vals)]
    assert np.all((vals >= -1e-9) & (vals <= 1.0 + 1e-9)), f"{col} out of [0,1]"


def test_baseline_comparison(tmp_path: Path) -> None:
    """Baseline comparison writes results + per-dataset AUPR dat with metrics in range."""
    out = tmp_path / "baseline_comparison"
    run_baseline_comparison.main(quick=True, seed=0, outdir=out)
    assert (out / "results.csv").exists()
    assert (out / "config.json").exists()
    assert (out / "aupr_vs_n_linear2d.dat").exists()
    _assert_metric_range(pl.read_csv(out / "results.csv"), "aupr")
    fixed = pl.read_csv(out / "fixed_zeta.dat", separator=" ", infer_schema_length=1000)
    assert set(fixed["method"].to_list()) == {"autosafe", "convex_hull"}
    _assert_metric_range(fixed, "iou")
    assert np.all(np.asarray(fixed["be_fp_mean"].to_list(), dtype=float) >= 0.0)


def test_anchor_subset_stability(tmp_path: Path) -> None:
    """Anchor subset stability writes the subsampling-stability table with IoUs in range."""
    out = tmp_path / "anchor_subset_stability"
    run_anchor_subset_stability.main(quick=True, seed=0, outdir=out)
    assert (out / "results.csv").exists()
    assert (out / "config.json").exists()
    agg = pl.read_csv(
        out / "subset_stability.dat", separator=" ", infer_schema_length=1000
    )
    for col in ("iou_true_mean", "iou_pairwise_mean", "iou_pairwise_min"):
        _assert_metric_range(agg, col)
    # Pairwise stability cannot exceed 1 and the min is <= the mean.
    mins = np.asarray(agg["iou_pairwise_min"].to_list(), dtype=float)
    means = np.asarray(agg["iou_pairwise_mean"].to_list(), dtype=float)
    assert np.all(mins <= means + 1e-12)


def test_ood_adjustment(tmp_path: Path) -> None:
    """OOD adjustment writes iteration/collateral dat; the OOD constraint holds."""
    out = tmp_path / "ood_adjustment"
    run_ood_adjustment.main(quick=True, seed=0, outdir=out)
    df = pl.read_csv(out / "results.csv")
    assert (out / "ood_iterations.dat").exists()
    assert (out / "ood_collateral.dat").exists()
    for xi_v, post_v in zip(
        df["xi"].to_list(), df["max_ood_affinity_post"].to_list(), strict=True
    ):
        assert post_v <= xi_v + 1e-9
    assert cast("int", df["iterations"].min()) >= 0


def test_parameter_sensitivity(tmp_path: Path) -> None:
    """Parameter sensitivity writes heatmap + fixed-grid dat with AUPR in range."""
    out = tmp_path / "parameter_sensitivity"
    run_parameter_sensitivity.main(quick=True, seed=0, outdir=out)
    assert (out / "sensitivity_fixed_linear2d.dat").exists()
    assert any(out.glob("sensitivity_*_n*.dat"))
    _assert_metric_range(pl.read_csv(out / "results.csv"), "aupr")


def test_halo_vs_anchor_count(tmp_path: Path) -> None:
    """Halo vs. anchor count writes the halo dat with non-negative halo widths."""
    out = tmp_path / "halo_vs_anchor_count"
    run_halo_vs_anchor_count.main(quick=True, seed=0, outdir=out)
    df = pl.read_csv(out / "results.csv")
    assert (out / "halo.dat").exists()
    assert cast("float", df["halo_max"].min()) >= -1e-9
    _assert_metric_range(df, "precision")
    _assert_metric_range(df, "recall")


def test_anchor_count_sweep(tmp_path: Path) -> None:
    """Anchor count sweep writes per-N curves and an R2 table."""
    out = tmp_path / "anchor_count_sweep"
    run_anchor_count_sweep.main(quick=True, seed=0, outdir=out)
    assert (out / "curve_r2.dat").exists()
    assert (out / "aupr_vs_n.dat").exists()
    assert any(out.glob("pr_n*.dat"))
    _assert_metric_range(pl.read_csv(out / "results.csv"), "aupr_mean")


def test_kernel_truncation(tmp_path: Path) -> None:
    """Kernel truncation writes error/latency/determinism; exact NN is bit-reproducible."""
    out = tmp_path / "kernel_truncation"
    run_kernel_truncation.main(quick=True, seed=0, outdir=out)
    assert (out / "truncation_latency.dat").exists()
    det = json.loads((out / "truncation_determinism.json").read_text())
    assert det["exact_deterministic"] is True
    df = pl.read_csv(out / "results.csv")
    _assert_metric_range(df, "err_max")
    _assert_metric_range(df, "flip_zeta05")


def test_kernel_truncation_matches_full_at_k_equals_n() -> None:
    """K-truncation with K = N reproduces the exact affinity bit-for-bit."""
    rng = np.random.default_rng(0)
    from experiments.benchmark.common import build_odd, score_autosafe

    pts = rng.uniform(-1, 1, size=(40, 3))
    odd = build_odd(pts, mode="calibrated")
    x = rng.uniform(-1, 1, size=(50, 3))
    full = score_autosafe(odd, x)
    trunc, _, _ = run_kernel_truncation.truncated_affinity(
        np.asarray(odd._anchors_np), np.asarray(odd._inv_diag_np), x, 40
    )
    assert float(np.max(np.abs(trunc - full))) < 1e-9


def test_conformal_threshold(tmp_path: Path) -> None:
    """Conformal threshold writes coverage; metrics in range."""
    out = tmp_path / "conformal_threshold"
    run_conformal_threshold.main(quick=True, seed=0, outdir=out)
    assert (out / "conformal_coverage.dat").exists()
    df = pl.read_csv(out / "results.csv")
    _assert_metric_range(df, "emp_false_excl")
    _assert_metric_range(df, "specificity_out")


def test_conformal_threshold_quantile() -> None:
    """conformal_threshold picks the eps-quantile and flags small n."""
    zeta, ok = run_conformal_threshold.conformal_threshold(
        np.linspace(0.0, 1.0, 1000), 0.1
    )
    assert ok
    assert abs(zeta - 0.1) < 0.02
    _, ok_small = run_conformal_threshold.conformal_threshold(
        np.array([0.1, 0.5, 0.9]), 0.01
    )
    assert ok_small is False


def test_duplicate_sensitivity(tmp_path: Path) -> None:
    """Duplicate sensitivity writes dup/dedup/formula dats; the noisy-OR formula holds exactly."""
    out = tmp_path / "duplicate_sensitivity"
    run_duplicate_sensitivity.main(quick=True, seed=0, outdir=out)
    for name in (
        "duplicate",
        "dedup",
        "formula_check",
        "density_stability",
    ):
        assert (out / f"{name}.dat").exists()
    f = pl.read_csv(out / "formula_check.dat", separator=" ")
    for after, pred in zip(f["alpha_after"], f["predicted_noisy_or"], strict=True):
        assert abs(after - pred) < 1e-9


def test_duplicate_sensitivity_helpers() -> None:
    """inject_duplicates grows the set; dedup collapses exact copies."""
    rng = np.random.default_rng(0)
    base = rng.uniform(-1, 1, size=(20, 2))
    dup = run_duplicate_sensitivity.inject_duplicates(
        base, frac=0.5, repeats=3, jitter=0.0, rng=rng
    )
    assert dup.shape[0] == 20 + 10 * 3
    dd = run_duplicate_sensitivity.dedup(np.vstack([base, base]), tol=1e-6)
    assert dd.shape[0] == 20


def test_covariance_structure(tmp_path: Path) -> None:
    """Covariance structure writes anisotropy + gap dats with metrics in range."""
    out = tmp_path / "covariance_structure"
    run_covariance_structure.main(quick=True, seed=0, outdir=out)
    assert (out / "gap_vs_rho.dat").exists()
    df = pl.read_csv(out / "results.csv")
    _assert_metric_range(df, "aupr")
    _assert_metric_range(df, "iou")
    assert cast("float", df["halo_max"].min()) >= -1e-9


def test_permutation_stability(tmp_path: Path) -> None:
    """Permutation stability writes a determinism report; the pipeline is bit-reproducible."""
    out = tmp_path / "permutation_stability"
    run_permutation_stability.main(quick=True, seed=0, outdir=out)
    report = json.loads((out / "permutation_report.json").read_text())
    assert report["all_pass"] is True
    assert report["max_alpha_dev"] <= 1e-12


def test_held_out_vcas_hole(tmp_path: Path) -> None:
    """Held-out VCAS hole writes hole-FP results; the convex hull leaks into the hole."""
    out = tmp_path / "held_out_vcas_hole"
    run_held_out_vcas_hole.main(quick=True, seed=0, outdir=out)
    df = pl.read_csv(out / "hole_fp.dat", separator=" ")
    methods = df["method"].to_list()
    assert {"autosafe_no_ood", "autosafe_with_ood", "convex_hull"} <= set(methods)
    hull_fp = df.filter(pl.col("method") == "convex_hull")["fp_rate_in_hole"][0]
    assert hull_fp >= 0.5  # hull cannot exclude the hole
