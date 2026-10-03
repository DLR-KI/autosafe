# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the paper-data exporter.

Two layers: focused unit tests build tiny synthetic ``.dat`` fixtures to pin
down each pivot/filter/join recipe exactly, and one integration test runs a
representative subset of the real experiment scripts in ``--quick`` mode
(the project's existing convention -- see ``test_runs.py``) to exercise the
exporter end to end against a real, if miniature, results tree.
"""

from pathlib import Path

import pytest

from experiments.benchmark import (
    export_paper_data,
    run_anchor_count_sweep,
    run_baseline_comparison,
    run_covariance_structure,
    run_halo_vs_anchor_count,
    run_held_out_vcas_hole,
    run_kernel_truncation,
    run_parameter_sensitivity,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_exports_list_matches_expected_filenames() -> None:
    """The builder table and the canonical filename list never drift apart."""
    assert [e.filename for e in export_paper_data._EXPORTS] == (
        export_paper_data.EXPECTED_FILENAMES
    )
    assert len(export_paper_data.EXPECTED_FILENAMES) == 18


def test_halo_verbatim_and_wide_pivot(tmp_path: Path) -> None:
    """``halo.dat`` is a verbatim copy; ``halo_wide.dat`` pivots arm to columns."""
    _write(
        tmp_path / "halo_vs_anchor_count" / "halo.dat",
        "arm n halo_max halo_p95 precision recall d_tilde\n"
        "fixed 30 1.0 0.9 0.8 0.7 0.5\n"
        "calibrated 30 3.0 2.9 0.4 1 0.5\n"
        "fixed 100 1.2 1.0 0.6 1 0.4\n"
        "calibrated 100 1.4 1.1 0.9 1 0.4\n",
    )
    header, rows = export_paper_data._halo(tmp_path)
    assert header == [
        "arm",
        "n",
        "halo_max",
        "halo_p95",
        "precision",
        "recall",
        "d_tilde",
    ]
    assert len(rows) == 4

    header, rows = export_paper_data._halo_wide(tmp_path)
    assert header == [
        "n",
        "fixed_halo",
        "calibrated_halo",
        "fixed_precision",
        "calibrated_precision",
    ]
    by_n = {r["n"]: r for r in rows}
    assert by_n["30"]["fixed_halo"] == "1.0"
    assert by_n["30"]["calibrated_halo"] == "3.0"
    assert by_n["100"]["fixed_precision"] == "0.6"
    assert by_n["100"]["calibrated_precision"] == "0.9"


def test_sensitivity_filters_gamma_le_2(tmp_path: Path) -> None:
    """``sensitivity_*_n1000.dat`` keeps only ``gamma <= 2``."""
    _write(
        tmp_path / "parameter_sensitivity" / "sensitivity_linear2d_n1000.dat",
        "gamma s aupr\n0.5 1 0.9\n2 1 0.8\n3 1 0.7\n5 1 0.6\n",
    )
    _, rows = export_paper_data._sensitivity(tmp_path, "sensitivity_linear_n1000.dat")
    assert [r["gamma"] for r in rows] == ["0.5", "2"]


def test_baselines_pivots_method_by_n_and_drops_poly5d(tmp_path: Path) -> None:
    """``baselines.dat`` pivots method x N, renames convex_hull, drops poly5d."""
    for ds in ("linear2d", "annulus2d", "twoblobs2d", "banana2d", "poly5d"):
        rows = "n method aupr\n"
        for n in ("10", "1000"):
            for method in ("autosafe", "kde", "gmm", "convex_hull"):
                rows += f"{n} {method} 0.5\n"
        _write(
            tmp_path / "baseline_comparison" / f"aupr_vs_n_{ds}.dat",
            "n method aupr\n" + rows.split("\n", 1)[1],
        )
    header, rows = export_paper_data._baselines(tmp_path)
    assert header == [
        "dataset",
        "autosafe10",
        "kde10",
        "gmm10",
        "hull10",
        "autosafe1000",
        "kde1000",
        "gmm1000",
        "hull1000",
    ]
    datasets = [r["dataset"] for r in rows]
    assert datasets == ["linear", "annulus", "blobs", "banana"]  # poly5d dropped
    assert rows[0]["hull10"] == "0.5"  # convex_hull -> hull


def test_mcm_anchor_count_mean_averages_over_n(tmp_path: Path) -> None:
    """``mcm_anchor_count_mean.dat`` averages precision/recall over N, by zeta."""
    _write(
        tmp_path / "anchor_count_sweep" / "aupr_vs_n.dat",
        "n aupr_mean aupr_std\n3 0.5 0.1\n5 0.6 0.1\n",
    )
    _write(
        tmp_path / "anchor_count_sweep" / "pr_n3.dat",
        "zeta log_threshold precision_mean precision_std recall_mean recall_std\n"
        "0 0 0.2 0.0 1.0 0.0\n"
        "1 0 0.8 0.0 0.4 0.0\n",
    )
    _write(
        tmp_path / "anchor_count_sweep" / "pr_n5.dat",
        "zeta log_threshold precision_mean precision_std recall_mean recall_std\n"
        "0 0 0.4 0.0 1.0 0.0\n"
        "1 0 0.6 0.0 0.6 0.0\n",
    )
    header, rows = export_paper_data._mcm_per_n(tmp_path)
    assert header == [
        "n",
        "zeta",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
    ]
    assert len(rows) == 4  # 2 n-values x 2 zeta-values, in N order

    header, rows = export_paper_data._mcm_anchor_count_mean(tmp_path)
    assert header == ["zeta", "precision_mean", "recall_mean"]
    by_zeta = {r["zeta"]: r for r in rows}
    assert float(by_zeta["0"]["precision_mean"]) == pytest.approx(0.3)  # (0.2+0.4)/2
    assert float(by_zeta["0"]["recall_mean"]) == pytest.approx(1.0)
    assert float(by_zeta["1"]["precision_mean"]) == pytest.approx(0.7)  # (0.8+0.6)/2


def test_subset_density_dedup_pivots(tmp_path: Path) -> None:
    """``subset.dat``, ``density.dat``, ``dedup.dat`` pivot dataset to columns."""
    _write(
        tmp_path / "anchor_subset_stability" / "subset_stability.dat",
        "dataset n aupr_mean aupr_std iou_true_mean iou_true_std "
        "iou_pairwise_mean iou_pairwise_min iou_pairwise_max\n"
        "linear2d 100 0.9 0.0 0.6 0.0 0.8 0.7 0.9\n"
        "annulus2d 100 0.9 0.0 0.6 0.0 0.7 0.6 0.8\n",
    )
    header, rows = export_paper_data._subset(tmp_path)
    assert header == [
        "n",
        "linear_mean",
        "linear_min",
        "linear_max",
        "annulus_mean",
        "annulus_min",
        "annulus_max",
    ]
    assert rows[0]["linear_mean"] == "0.8"
    assert rows[0]["annulus_mean"] == "0.7"

    _write(
        tmp_path / "duplicate_sensitivity" / "density_stability.dat",
        "dataset n iou_true iou_selfstab\n"
        "annulus2d 100 0.6 nan\n"
        "linear2d 100 0.62 nan\n",
    )
    header, rows = export_paper_data._density(tmp_path)
    assert header == ["n", "annulus", "linear"]
    assert rows[0] == {"n": "100", "annulus": "0.6", "linear": "0.62"}

    _write(
        tmp_path / "duplicate_sensitivity" / "dedup.dat",
        "dataset arm tol n_after iou_vs_base\n"
        "annulus2d calibrated 0 100 0.6\n"
        "annulus2d fixed 0 100 0.9\n"  # non-calibrated arm must be dropped
        "linear2d calibrated 0 100 0.7\n",
    )
    header, rows = export_paper_data._dedup(tmp_path)
    assert header == ["tol", "annulus", "linear"]
    assert len(rows) == 1
    assert rows[0] == {"tol": "0", "annulus": "0.6", "linear": "0.7"}


def test_covariance_pivots(tmp_path: Path) -> None:
    """``covariance_n100.dat`` and ``deployed_covariance.dat`` pivot correctly."""
    _write(
        tmp_path / "covariance_structure" / "anisotropy.dat",
        "arm rho n aupr iou halo_max sigma_aniso_frac sigma_aniso_ratio\n"
        "iso 0 100 0.9 0.5 0.1 1 1.0\n"
        "deployed 0 100 0.9 0.6 0.1 1 1.0\n"
        "iso 0 300 0.9 0.5 0.1 1 1.0\n"
        "deployed 0 300 0.9 0.7 0.1 1 1.0\n"
        "iso 1 100 0.9 0.45 0.1 1 1.0\n"
        "deployed 1 100 0.9 0.55 0.1 1 1.0\n"
        "deployed 1 300 0.9 0.65 0.1 1 1.0\n",
    )
    header, rows = export_paper_data._covariance_n100(tmp_path)
    assert header == ["rho", "iso", "deployed"]
    by_rho = {r["rho"]: r for r in rows}
    assert by_rho["0"]["iso"] == "0.5"
    assert by_rho["0"]["deployed"] == "0.6"
    assert by_rho["1"]["iso"] == "0.45"  # n==300 row correctly excluded (n != 100)

    header, rows = export_paper_data._deployed_covariance(tmp_path)
    assert header == ["n", "rho0", "rho1"]
    by_n = {r["n"]: r for r in rows}
    assert by_n["100"]["rho0"] == "0.6"
    assert by_n["100"]["rho1"] == "0.55"
    assert by_n["300"]["rho0"] == "0.7"


def test_truncation_latency_drops_n1000_and_needs_k128_512(tmp_path: Path) -> None:
    """``truncation_latency.dat`` drops N=1000 and needs both K=128 and K=512."""
    _write(
        tmp_path / "kernel_truncation" / "truncation_latency.dat",
        "n k q_secs_full q_secs_trunc tree_build_secs model_mem_mb "
        "full_py_peak_mb trunc_py_peak_mb\n"
        "1000 128 0.5 0.1 0.0 0.1 0.1 0.1\n"
        "1000 512 0.5 0.2 0.0 0.1 0.1 0.1\n"
        "10000 128 0.6 0.3 0.0 0.1 0.1 0.1\n"
        "10000 512 0.6 0.4 0.0 0.1 0.1 0.1\n"
        "100000 8 0.9 0.5 0.0 0.1 0.1 0.1\n",  # missing 128/512 -> excluded
    )
    header, rows = export_paper_data._truncation_latency(tmp_path)
    assert header == ["n", "exact", "k128", "k512"]
    ns = [r["n"] for r in rows]
    assert ns == ["10000"]  # 1000 dropped by recipe, 100000 lacks required K's
    assert rows[0] == {"n": "10000", "exact": "0.6", "k128": "0.3", "k512": "0.4"}


def test_hole_shortens_labels_and_drops_aupr(tmp_path: Path) -> None:
    """``hole.dat`` keeps only fp_rate, with paper label shortening."""
    _write(
        tmp_path / "held_out_vcas_hole" / "hole_fp.dat",
        "method fp_rate_in_hole aupr\n"
        "autosafe_no_ood 0.03 0.9\n"
        "autosafe_with_ood 0.01 0.9\n"
        "convex_hull 0.5 0.2\n",
    )
    header, rows = export_paper_data._hole(tmp_path)
    assert header == ["method", "fp_rate"]
    assert [r["method"] for r in rows] == ["autosafe", "autosafe_ood", "hull"]
    assert rows[0]["fp_rate"] == "0.03"


def test_conformal_n1000_filters_and_pivots(tmp_path: Path) -> None:
    """``conformal_n1000.dat`` filters n=1000 and pivots dataset by eps."""
    _write(
        tmp_path / "conformal_threshold" / "conformal_coverage.dat",
        "dataset n eps zeta_hat_mean log_threshold_mean emp_false_excl_mean "
        "emp_false_excl_max specificity_out_mean\n"
        "annulus2d 500 0.01 0.9 1.0 0.99 0.1 0.9\n"
        "annulus2d 1000 0.01 0.9 1.0 0.01 0.1 0.9\n"
        "linear2d 1000 0.01 0.9 1.0 0.02 0.1 0.9\n",
    )
    header, rows = export_paper_data._conformal_n1000(tmp_path)
    assert header == ["eps", "annulus", "linear"]
    assert len(rows) == 1  # only the n=1000 eps=0.01 row survives the filter
    assert rows[0] == {"eps": "0.01", "annulus": "0.01", "linear": "0.02"}


def test_run_against_quick_experiment_outputs(tmp_path: Path) -> None:
    """The exporter runs end to end against real ``--quick`` results.

    Most recipes are generic pivots/joins that adapt to whatever the smaller
    ``--quick`` sweep produced, so they succeed with fewer rows than the real
    run. ``baselines.dat`` is a documented exception: ``--quick``
    baseline_comparison does not sweep the ``gmm`` method the paper's table
    needs, so that one recipe is expected to report a graceful skip rather
    than raising -- proving the "results tree may be partially populated"
    contract, not a bug.
    """
    results = tmp_path / "results"
    run_halo_vs_anchor_count.main(
        quick=True, seed=0, outdir=results / "halo_vs_anchor_count"
    )
    run_kernel_truncation.main(quick=True, seed=0, outdir=results / "kernel_truncation")
    run_parameter_sensitivity.main(
        quick=True, seed=0, outdir=results / "parameter_sensitivity"
    )
    run_anchor_count_sweep.main(
        quick=True, seed=0, outdir=results / "anchor_count_sweep"
    )
    run_covariance_structure.main(
        quick=True, seed=0, outdir=results / "covariance_structure"
    )
    run_held_out_vcas_hole.main(
        quick=True, seed=0, outdir=results / "held_out_vcas_hole"
    )
    run_baseline_comparison.main(
        quick=True, seed=0, outdir=results / "baseline_comparison"
    )

    outdir = tmp_path / "export"
    status = export_paper_data.run(results_dir=results, outdir=outdir)

    assert set(status) == set(export_paper_data.EXPECTED_FILENAMES)
    assert status["baselines.dat"].startswith("skipped:")  # quick lacks gmm

    for name, message in status.items():
        if message.startswith("written"):
            written = outdir / name
            assert written.exists()
            assert written.read_text(encoding="utf-8").splitlines()[0]  # has a header

    # At least the recipes fed by scripts we actually ran should succeed.
    for name in ("halo.dat", "truncation_error.dat", "halo_wide.dat", "hole.dat"):
        assert status[name].startswith("written"), (name, status[name])
