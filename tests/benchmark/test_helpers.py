# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Unit tests for benchmark helpers not reached by the ``--quick`` runs."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from autosafe.odd.comparison.oneclass import OneClassSVMBoundary
from autosafe.pointsets import rows_in
from autosafe.sample import Sample
from autosafe.samples import Samples
from experiments.benchmark import (
    common,
    export_paper_data,
    run_anchor_count_sweep,
    run_baseline_comparison,
    run_conformal_threshold,
    run_covariance_structure,
    run_duplicate_sensitivity,
    run_held_out_vcas_hole,
    run_kernel_truncation,
    synthetic_odds,
)


def _blob_and_probes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    id_pts = rng.normal(size=(60, 2))
    inside = rng.normal(scale=0.3, size=(20, 2))
    outside = rng.normal(scale=0.3, size=(20, 2)) + 8.0
    return id_pts, inside, outside


# --- run_baseline_comparison._score -------------------------------------------


@pytest.mark.parametrize("method", ["ocsvm", "knn_dist", "iforest", "gmm"])
def test_baseline_scores_rank_inside_above_outside(method: str) -> None:
    id_pts, inside, outside = _blob_and_probes()
    scores = run_baseline_comparison._score(
        method, id_pts, np.vstack([inside, outside]), np.random.default_rng(1)
    )
    assert scores.shape == (40,)
    assert np.min(scores[:20]) > np.max(scores[20:])


def test_baseline_ocsvm_matches_library_selection() -> None:
    """The script's grid search mirrors ``OneClassSVMBoundary`` verbatim."""
    id_pts, inside, outside = _blob_and_probes()
    val = np.vstack([inside, outside])
    script = run_baseline_comparison._score(
        "ocsvm", id_pts, val, np.random.default_rng(0)
    )
    library = (
        OneClassSVMBoundary(auto_select=True).fit(id_pts.T).decision_function(val.T)
    )
    np.testing.assert_allclose(script, library)


def test_baseline_hull_falls_back_to_all_outside_on_degenerate_anchors() -> None:
    collinear = np.column_stack([np.arange(5.0), np.zeros(5)])
    scores = run_baseline_comparison._score(
        "convex_hull", collinear, np.zeros((3, 2)), np.random.default_rng(0)
    )
    np.testing.assert_array_equal(scores, np.zeros(3))
    np.testing.assert_array_equal(
        run_anchor_count_sweep._hull_labels(collinear, np.zeros((3, 2))),
        np.zeros(3, dtype=bool),
    )


def test_baseline_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        run_baseline_comparison._score(
            "bogus", np.zeros((3, 2)), np.zeros((1, 2)), np.random.default_rng(0)
        )


# --- run_held_out_vcas_hole ----------------------------------------------------


@pytest.mark.skipif(
    not run_held_out_vcas_hole.VCAS_CSV.exists(), reason="VCAS data not available"
)
def test_vcas_hole_split_is_consistent() -> None:
    id_pts, hole_pts, val, labels, probe = run_held_out_vcas_hole._vcas_with_hole(
        300, 500, 50, np.random.default_rng(0)
    )
    # Constant columns are dropped: n = 4 effective dimensions.
    assert id_pts.shape == (300, 4)
    assert hole_pts.shape == (50, 4)
    assert val.shape == (500, 4)
    assert labels.dtype == bool
    assert 0 < labels.sum() < len(labels)
    # Anchors never come from the hole; the probe set is disjoint from the OOD set.
    assert not rows_in(id_pts, hole_pts).any()
    assert not rows_in(id_pts, probe).any()
    assert not rows_in(probe, hole_pts).any()


# --- small helpers and error paths ------------------------------------------


def test_build_odd_rejects_single_anchor_and_unknown_mode() -> None:
    with pytest.raises(ValueError, match="N >= 2"):
        common.build_odd(np.zeros((1, 2)), mode="calibrated")
    with pytest.raises(ValueError, match="unknown mode"):
        common.build_odd(np.random.default_rng(0).normal(size=(5, 2)), mode="bogus")


def test_kernel_sigma_rejects_non_rbf_kernel() -> None:
    odd = Samples(
        [Sample(x=np.array(p)) for p in ([0.0, 0.0], [1.0, 0.0])],
        kernel_cls="Laplacian",
        kernel_kwargs={"alpha": 0.5},
    )
    with pytest.raises(TypeError, match="not RBFKernel"):
        common.kernel_sigma(odd.samples[0])


def test_curve_r2_with_a_constant_target() -> None:
    flat = np.ones(4)
    assert common.curve_r2(flat, flat) == pytest.approx(1.0)
    assert np.isnan(common.curve_r2(flat, flat + 1.0))


def test_git_hash_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_args: object, **_kwargs: object) -> None:
        raise OSError

    monkeypatch.setattr("subprocess.run", _fail)
    assert common._git_hash() == "unknown"


def test_alpha_equivalent_of_non_finite_scores() -> None:
    assert run_conformal_threshold.alpha_equivalent(np.inf) == pytest.approx(1.0)
    assert run_conformal_threshold.alpha_equivalent(-np.inf) == pytest.approx(0.0)


def test_local_cov_scores_on_coincident_anchors_and_unknown_kind() -> None:
    same = np.zeros((4, 2))  # zero covariance -> trace floored, still finite
    scores = run_covariance_structure.local_cov_scores(
        same, np.zeros((1, 2)), k=2, s=1.0, kind="iso"
    )
    assert np.all(np.isfinite(scores))
    with pytest.raises(ValueError, match="unknown kind"):
        run_covariance_structure.local_cov_scores(
            np.random.default_rng(0).normal(size=(5, 2)),
            np.zeros((1, 2)),
            k=2,
            s=1.0,
            kind="bogus",
        )


def test_inject_duplicates_with_nothing_to_duplicate_is_a_copy() -> None:
    base = np.arange(6.0).reshape(3, 2)
    out = run_duplicate_sensitivity.inject_duplicates(
        base, 0.0, 3, 0.0, np.random.default_rng(0)
    )
    np.testing.assert_array_equal(out, base)
    assert out is not base


def test_ivf_flip_rate_without_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faiss", None)  # makes `import faiss` fail
    assert (
        run_kernel_truncation._ivf_flip_rate(
            np.zeros((4, 2)), np.ones((4, 2)), np.zeros((2, 2)), 2, zeta=0.5
        )
        is None
    )


def test_synthetic_odd_bbox_and_unknown_name() -> None:
    odd = synthetic_odds.get_odd("linear2d")
    lower, upper = odd.bbox
    np.testing.assert_array_equal(lower, odd.lower)
    np.testing.assert_array_equal(upper, odd.upper)
    with pytest.raises(KeyError, match="unknown synthetic ODD"):
        synthetic_odds.get_odd("bogus")


# --- export_paper_data --------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_read_dat_rejects_an_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty results file"):
        export_paper_data._read_dat(_write(tmp_path / "empty.dat", "\n"))


def test_main_reports_written_and_skipped_files(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write(results / "halo_vs_anchor_count" / "halo.dat", "n halo\n10 0.5\n")
    out = tmp_path / "out"

    app = export_paper_data.typer.Typer()
    app.command()(export_paper_data.main)
    result = CliRunner().invoke(
        app, ["--outdir", str(out), "--results-dir", str(results)]
    )
    assert result.exit_code == 0
    assert "halo.dat: written (1 rows)" in result.output
    # Everything but halo.dat is missing from the partial results tree.
    assert "per_n.dat: skipped" in result.output
    assert (
        f"1/{len(export_paper_data.EXPECTED_FILENAMES)} paper data files"
        in result.output
    )
    assert (out / "halo.dat").exists()


def test_dat_written_by_main_round_trips(tmp_path: Path) -> None:
    """The exporter's own output parses back to the same table."""
    path = _write(tmp_path / "t.dat", "a b\n1 2\n3 4\n")
    header, rows = export_paper_data._read_dat(path)
    out = tmp_path / "copy.dat"
    export_paper_data._write_table(out, header, rows)
    assert pl.read_csv(out, separator=" ").equals(pl.read_csv(path, separator=" "))
