# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Integration and regression tests for dataset-mode de-duplication.

Do NOT import anything from experiments/ here: the library tests must not
depend on the benchmark scripts. The regression probe below is an
independently reconstructed synthetic scenario, not a re-run of the
paper's experiment.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.spatial

import autosafe
from autosafe.deduplication import DeduplicationPolicy, deduplicate_points
from autosafe.kernels.rbf import calibrate_rbf_scale_d_tilde
from autosafe.samples import Samples
from autosafe.tools.evaluate.dataset.anchors import _extract_anchor_points
from autosafe.tools.evaluate.workflows import evaluate_dataset_mode


def _sidecar(csv_path: Path) -> dict:
    return json.loads(csv_path.with_name(csv_path.stem + "-params.json").read_text())


def _write_dataset_with_duplicates(
    tmp_path: Path,
    *,
    name: str = "synthetic",
    n_unique: int = 20,
    n_dup_copies: int = 5,
) -> tuple[Path, int]:
    """Write a 3D dataset with a known number of exact duplicate rows.

    Args:
        tmp_path (Path): pytest temporary directory.
        name (str): Base file name (without extension).
        n_unique (int): Number of distinct rows.
        n_dup_copies (int): How many of the unique rows get one extra
            exact duplicate appended.

    Returns:
        tuple[Path, int]: (dataset CSV path, total row count).
    """
    rng = np.random.default_rng(0)
    unique_rows = rng.normal(loc=0.0, scale=0.3, size=(n_unique, 3))
    duplicated = unique_rows[:n_dup_copies]
    data = np.vstack([unique_rows, duplicated])

    ds = tmp_path / f"{name}.csv"
    rows = ["x0,x1,x2"] + [f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f}" for r in data]
    ds.write_text("\n".join(rows), encoding="utf-8")
    return ds, data.shape[0]


def test_disabled_dedup_reproduces_current_behavior_exactly(tmp_path: Path) -> None:
    """The most important test: dedup_policy=None changes nothing.

    Duplicate rows are NOT collapsed, the ODD path carries no
    ``-dedup`` tag, and repeated runs are byte-identical, exactly as
    without de-duplication support.
    """
    ds, n_total = _write_dataset_with_duplicates(tmp_path)

    _, csv_path, odd_path = evaluate_dataset_mode(
        ds,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        references=["knn"],
        n_samples=1200,
        threshold_count=5,
    )

    assert "-dedup" not in odd_path.name
    sc = _sidecar(csv_path)
    assert sc["dedup_enabled"] is False
    assert sc["dedup_policy"] is None
    assert sc["dedup_provenance_path"] is None

    odd = autosafe.from_json(odd_path)
    anchor_points = _extract_anchor_points(odd)
    # Every raw row --- including the exact duplicates --- still
    # instantiates its own kernel/anchor when de-duplication is off.
    assert anchor_points.shape[0] == n_total
    assert sc["n_anchors"] == n_total

    # Determinism: an independent second run on the identical data (same
    # seed inside _write_dataset_with_duplicates) reproduces every metric
    # column exactly; the two only differ in the embedded source-file
    # name column.
    ds2, _ = _write_dataset_with_duplicates(tmp_path, name="synthetic_repeat")
    _, csv_path2, _odd_path2 = evaluate_dataset_mode(
        ds2,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        references=["knn"],
        n_samples=1200,
        threshold_count=5,
    )
    df1 = pl.read_csv(csv_path).drop("source")
    df2 = pl.read_csv(csv_path2).drop("source")
    assert df1.equals(df2)


def test_dedup_collapses_duplicates_and_tags_cache(tmp_path: Path) -> None:
    """With dedup_policy set, exact duplicates collapse to one anchor."""
    ds, n_total = _write_dataset_with_duplicates(tmp_path)
    policy = DeduplicationPolicy(resolution=(1e-6, 1e-6, 1e-6), origin=(0.0, 0.0, 0.0))

    _, csv_path, odd_path = evaluate_dataset_mode(
        ds,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        references=["knn"],
        n_samples=1200,
        threshold_count=5,
        dedup_policy=policy,
    )

    assert "-dedup" in odd_path.name
    sc = _sidecar(csv_path)
    assert sc["dedup_enabled"] is True
    assert sc["dedup_id_n_input"] == n_total
    assert sc["dedup_id_n_output"] < n_total
    assert sc["n_anchors"] == sc["dedup_id_n_output"]
    assert sc["dedup_provenance_path"] is not None
    assert Path(sc["dedup_provenance_path"]).exists()
    assert sc["dedup_provenance_digest"]


def test_cache_identity_differs_by_policy(tmp_path: Path) -> None:
    """A different resolution/origin gives a different ODD cache path."""
    ds, _ = _write_dataset_with_duplicates(tmp_path)
    policy_a = DeduplicationPolicy(
        resolution=(1e-6, 1e-6, 1e-6), origin=(0.0, 0.0, 0.0)
    )
    policy_b = DeduplicationPolicy(resolution=(0.4, 0.4, 0.4), origin=(0.0, 0.0, 0.0))

    def _run(policy: DeduplicationPolicy) -> Path:
        _, _csv, odd_path = evaluate_dataset_mode(
            ds,
            closest_sample_mode="global",
            kernel_kwargs={"calibration": "auto"},
            references=["knn"],
            n_samples=1000,
            threshold_count=5,
            dedup_policy=policy,
        )
        return odd_path

    path_a = _run(policy_a)
    path_b = _run(policy_b)
    assert path_a != path_b


def test_cache_identity_differs_by_representative_set(tmp_path: Path) -> None:
    """Changing the underlying data changes the cache tag, same policy."""
    ds, _ = _write_dataset_with_duplicates(
        tmp_path, name="mutable", n_unique=15, n_dup_copies=3
    )
    policy = DeduplicationPolicy(resolution=(1e-6, 1e-6, 1e-6), origin=(0.0, 0.0, 0.0))

    _, _csv1, odd_path1 = evaluate_dataset_mode(
        ds,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        references=["knn"],
        n_samples=800,
        threshold_count=5,
        dedup_policy=policy,
    )

    # Overwrite the SAME path with different underlying data (different
    # representative set under the identical policy).
    rng = np.random.default_rng(99)
    new_rows = rng.normal(loc=2.0, scale=0.1, size=(15, 3))
    ds.write_text(
        "\n".join(
            ["x0,x1,x2"] + [f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f}" for r in new_rows]
        ),
        encoding="utf-8",
    )
    _, _csv2, odd_path2 = evaluate_dataset_mode(
        ds,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        references=["knn"],
        n_samples=800,
        threshold_count=5,
        dedup_policy=policy,
    )

    assert odd_path1 != odd_path2


def test_subsampling_occurs_after_deduplication(tmp_path: Path) -> None:
    """subsample_anchors draws from the de-duplicated pool, not raw rows."""
    ds, n_total = _write_dataset_with_duplicates(
        tmp_path, name="heavy_dup", n_unique=12, n_dup_copies=10
    )
    policy = DeduplicationPolicy(resolution=(1e-6, 1e-6, 1e-6), origin=(0.0, 0.0, 0.0))

    _, csv_path, _odd_path = evaluate_dataset_mode(
        ds,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        references=["knn"],
        n_samples=800,
        threshold_count=5,
        dedup_policy=policy,
        subsample_anchors=5,
        seed=1,
    )

    sc = _sidecar(csv_path)
    assert sc["dedup_id_n_input"] == n_total
    assert sc["dedup_id_n_output"] == 12  # 12 unique rows survive de-dup
    assert sc["n_anchors"] == 5  # subsampled AFTER de-duplication


def test_calibration_records_reserved_before_deduplication(tmp_path: Path) -> None:
    """Reserved calibration rows are excluded from the de-duplication input."""
    ds, n_total = _write_dataset_with_duplicates(
        tmp_path, name="calib", n_unique=20, n_dup_copies=4
    )
    policy = DeduplicationPolicy(resolution=(1e-6, 1e-6, 1e-6), origin=(0.0, 0.0, 0.0))
    n_reserved = 6

    _, csv_path, _odd_path = evaluate_dataset_mode(
        ds,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        references=["knn"],
        n_samples=800,
        threshold_count=5,
        dedup_policy=policy,
        dedup_n_calibration_reserved=n_reserved,
    )

    sc = _sidecar(csv_path)
    assert sc["dedup_calibration_reserved_count"] == n_reserved
    # The calibration rows never reach the de-duplication candidate pool.
    assert sc["dedup_id_n_input"] == n_total - n_reserved


def test_ood_still_constrained_with_dedup_enabled(tmp_path: Path) -> None:
    """Every OOD sample remains constrained when de-duplication is on."""
    rng = np.random.default_rng(42)
    blob_a = rng.normal(loc=[-0.5, -0.5, -0.5], scale=0.12, size=(30, 3))
    blob_b = rng.normal(loc=[0.5, 0.5, 0.5], scale=0.12, size=(30, 3))
    data = np.vstack([blob_a, blob_b, blob_a[:5]])  # 5 exact duplicates

    ds = tmp_path / "with_ood.csv"
    ds.write_text(
        "\n".join(["x0,x1,x2"] + [f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f}" for r in data]),
        encoding="utf-8",
    )
    ood = tmp_path / "ood.csv"
    ood.write_text(
        "x0,x1,x2\n0.5,0.5,0.5\n-0.5,-0.5,-0.5\n0.0,0.0,0.0\n", encoding="utf-8"
    )

    policy = DeduplicationPolicy(resolution=(1e-6, 1e-6, 1e-6), origin=(0.0, 0.0, 0.0))
    xi = 0.1
    _, csv_path, odd_path = evaluate_dataset_mode(
        ds,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        references=["knn"],
        n_samples=1200,
        threshold_count=5,
        ood_path=ood,
        ood_xi=xi,
        dedup_policy=policy,
    )

    sc = _sidecar(csv_path)
    assert sc["dedup_enabled"] is True
    assert sc["dedup_ood_n_input"] == 3
    assert float(sc["ood_max_affinity_final"]) <= xi + 1e-9
    assert "-dedup" in odd_path.name
    assert "-ood" in odd_path.name


class TestDuplicateSensitivityRegression:
    """Regression probe: dedup restores IoU degraded by duplicate anchors.

    The paper reports that collapsing records within shared resolution
    cells restores IoU with the duplicate-free region from 0.665 to
    0.938 (the paper's duplicate-sensitivity figure). This is an INDEPENDENTLY reconstructed synthetic probe (a 2D
    disk of anchors, global RBF auto-calibration), not a re-run of the
    paper's experiment or its result files, so the exact figures are
    used as an order-of-magnitude regression target with a generous
    tolerance, not asserted bit-for-bit.
    """

    @staticmethod
    def _median_nn_distance(points: np.ndarray) -> float:
        tree = scipy.spatial.KDTree(points)
        dist, _ = tree.query(points, k=2)
        positive = dist[:, 1][dist[:, 1] > 0]
        return float(np.median(positive)) if positive.size else 1.0

    @classmethod
    def _region_mask(
        cls, anchors: np.ndarray, test_points: np.ndarray, threshold: float
    ) -> np.ndarray:
        d_median = cls._median_nn_distance(anchors)
        kappa, eta = calibrate_rbf_scale_d_tilde(
            np.full(anchors.shape[0], d_median), gamma=1.0, s=3.0
        )
        odd = Samples(
            samples=anchors,
            closest_sample_mode="global",
            kernel_cls="RBF",
            kernel_kwargs={"kappa": kappa, "eta": eta},
        )
        affinity = np.asarray(odd(test_points))
        return affinity > threshold

    @staticmethod
    def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return float(intersection / union) if union > 0 else 1.0

    def test_duplicate_sensitivity_probe(  # ruff:ignore[too-many-locals]
        self,
    ) -> None:
        """Duplicates degrade IoU; de-duplication mostly (not fully) restores it.

        Reproduces the *regime* of the paper's duplicate-sensitivity figure
        (Study C, ``annulus2d``, ``calibrated`` arm; see
        ``experiments/benchmark/results/duplicate_sensitivity/dedup.dat``):
        6000 anchors (1000 base + ``inject_duplicates(frac=0.5, repeats=10,
        jitter=0.02)``), ``tol=0`` (no dedup) gives IoU 0.665 vs. the
        duplicate-free region; a grid-dedup at ``tol=0.05`` (2.5x the
        jitter) restores it to only 0.938, because the merge width also
        collapses genuinely distinct anchors (6000 -> 1085 points) --
        recovery plateaus below 1.0 rather than reaching it.

        This probe mirrors that structure at unit-test scale: an
        ``annulus2d``-like ring (r in [1, 4], i.e. r^2 in [1, 16] as in
        ``synthetic_odds._annulus2d``) rather than a filled disk, the
        paper's duplicate load (frac=0.5, repeats=10), and a dedup merge
        width coarser than the jitter, via the production
        ``DeduplicationPolicy``/``deduplicate_points``.

        The paper's *absolute* jitter (0.02) and tol (0.05) are calibrated
        in the deployed pipeline's min-max-normalized [-1, 1] frame on
        1000 anchors; reproducing those literal values on ~150 raw-frame
        annulus anchors makes the jittered clusters far narrower than the
        anchors' own spacing and collapses the ODD almost to nothing
        (iou_with_duplicates below 0.05 in exploratory sweeps at
        jitter=0.02, well past the paper's 0.665). Instead, jitter is
        scaled to the reference set's own median nearest-neighbor spacing
        (1x) and the dedup resolution to 1.5x that jitter -- still
        "coarser than jitter" per the paper's relationship, just a
        smaller multiple, which is what a parameter sweep over jitter
        scale and merge-width ratio found necessary to land near the
        paper's operating point at this reduced anchor count (see the
        task report for the sweep). At seed=7 this measures
        iou_with_duplicates=0.6623 and iou_after_dedup=0.9097 against the
        paper's 0.665 / 0.938 (both are exact/deterministic under the
        fixed seed; the tolerances below are not covering run-to-run
        noise, only the gap between this reconstruction and the paper's
        own setup).
        """
        rng = np.random.default_rng(7)
        n_ref = 150
        theta = rng.uniform(0, 2 * np.pi, n_ref)
        radius = np.sqrt(rng.uniform(1.0, 16.0, n_ref))  # annulus2d: r^2 in [1, 16]
        reference_anchors = np.stack(
            [radius * np.cos(theta), radius * np.sin(theta)], axis=1
        )

        grid = np.linspace(-5.0, 5.0, 70)
        xx, yy = np.meshgrid(grid, grid)
        test_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        threshold = 0.5

        reference_region = self._region_mask(reference_anchors, test_points, threshold)

        # Paper's duplicate load (Study C): 50% of anchors get 10 extra
        # jittered copies each -- mirrors inject_duplicates(frac=0.5,
        # repeats=10, jitter=...) in run_duplicate_sensitivity.py. Jitter
        # is scaled to the reference anchors' own median NN spacing
        # rather than the paper's literal 0.02 (calibrated in a
        # different, normalized coordinate frame -- see docstring).
        frac, repeats = 0.5, 10
        n_dup_anchors = round(frac * n_ref)
        dup_source_idx = rng.choice(n_ref, size=n_dup_anchors, replace=False)
        jitter = self._median_nn_distance(reference_anchors)
        jittered_copies = np.repeat(reference_anchors[dup_source_idx], repeats, axis=0)
        jittered_copies += rng.normal(0.0, jitter, size=jittered_copies.shape)
        corrupted_anchors = np.vstack([reference_anchors, jittered_copies])

        corrupted_region = self._region_mask(corrupted_anchors, test_points, threshold)
        iou_with_duplicates = self._iou(corrupted_region, reference_region)

        # Merge width coarser than the jitter (paper: 0.05 / 0.02 = 2.5x);
        # 1.5x is the ratio that over-merges genuine anchors without
        # erasing the whole recovery signal at this reduced anchor count.
        resolution = 1.5 * jitter
        policy = DeduplicationPolicy(
            resolution=(resolution, resolution), origin=(0.0, 0.0)
        )
        dedup_result = deduplicate_points(corrupted_anchors, policy)
        dedup_region = self._region_mask(dedup_result.points, test_points, threshold)
        iou_after_dedup = self._iou(dedup_region, reference_region)

        # Core, robust regression assertions: de-duplication meaningfully
        # improves agreement with the duplicate-free region. Margins are
        # kept wide (not tight to the measured 0.662/0.910) so this block
        # stays stable under minor numerical variation; the precise check
        # against the paper's figures is below.
        assert iou_with_duplicates < 0.8
        assert iou_after_dedup > 0.85
        assert iou_after_dedup - iou_with_duplicates > 0.15

        # Tight match to the paper's reported figures (measured at
        # seed=7: iou_with_duplicates=0.6623, iou_after_dedup=0.9097).
        # abs=0.03/0.05 give roughly 10x/2x headroom over the measured
        # gap to the paper's numbers (0.0027/0.0283) -- tight enough that
        # reverting the affinity formula (1-prod(k) instead of
        # 1-prod(1-k), which saturates dedup IoU to 1.000 and masks any
        # real degradation) or a similarly sized regression fails this.
        #
        # Primary regression target: this probe's OWN deterministic output.
        # Pinning the measured values rather than the paper's is what makes
        # this a regression test -- a band centered on 0.938 would let the
        # recovery drift *upward* toward it unnoticed, and upward is
        # precisely where the known failure mode goes.
        assert iou_with_duplicates == pytest.approx(0.6623, abs=0.01)
        assert iou_after_dedup == pytest.approx(0.9097, abs=0.01)

        # Secondary: agreement with the paper's annulus2d/calibrated
        # figures (0.665 / 0.938), reproduced here to 0.003 / 0.028. The
        # residual gap is the scale-down -- 150 anchors and a 1.5x
        # merge-to-jitter ratio against the paper's 1000 and 2.5x.
        assert abs(iou_with_duplicates - 0.665) < 0.03
        assert abs(iou_after_dedup - 0.938) < 0.05

    @staticmethod
    def test_duplicate_sensitivity_probe_permutation_invariant() -> None:
        """The recovered representatives do not depend on input row order.

        Multiplicity survives only in the provenance artifact
        (``multiplicities``/``groups``): the kernel/anchor count built
        from the representatives is the unique-cell count, never the
        raw row count, regardless of how the corrupted rows are ordered.
        """
        rng = np.random.default_rng(7)
        n_ref = 60
        theta = rng.uniform(0, 2 * np.pi, n_ref)
        radius = np.sqrt(rng.uniform(0, 1, n_ref))
        reference_anchors = np.stack(
            [radius * np.cos(theta), radius * np.sin(theta)], axis=1
        )
        duplicated_idx = rng.choice(n_ref, size=10, replace=False)
        repeated = np.repeat(reference_anchors[duplicated_idx], 8, axis=0)
        kept_idx = np.setdiff1d(np.arange(n_ref), duplicated_idx)
        corrupted_anchors = np.vstack([reference_anchors[kept_idx], repeated])
        record_ids = list(range(corrupted_anchors.shape[0]))

        policy = DeduplicationPolicy(resolution=(1e-6, 1e-6), origin=(0.0, 0.0))
        baseline = deduplicate_points(corrupted_anchors, policy, record_ids=record_ids)

        perm = rng.permutation(corrupted_anchors.shape[0])
        shuffled = deduplicate_points(
            corrupted_anchors[perm],
            policy,
            record_ids=[record_ids[i] for i in perm],
        )

        np.testing.assert_array_equal(baseline.points, shuffled.points)
        assert baseline.groups == shuffled.groups
        # Each duplicated support point is replaced by 8 exact repeats
        # (not kept alongside a 9th original), so multiplicity peaks at
        # 8; this is visible only in the provenance, never in the
        # anchor/kernel count.
        assert int(np.max(baseline.multiplicities)) == 8
        assert baseline.n_output == len(kept_idx) + len(duplicated_idx)
