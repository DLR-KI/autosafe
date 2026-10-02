# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the dataset-mode helpers: de-duplication, YAML specs, baselines."""

from pathlib import Path

import numpy as np
import pytest

from autosafe.deduplication import DeduplicationPolicy
from autosafe.exceptions import EmptyAnchorPoolError, OODAnchorCoincidenceError
from autosafe.tools.evaluate.comparison import FastHullApproximation
from autosafe.tools.evaluate.dataset.baselines import (
    _baseline_memberships,
    _BaselineEvaluationData,
    _compute_method_membership,
    _evaluate_monitor_membership,
    _hull_membership,
)
from autosafe.tools.evaluate.dataset.dedup_integration import run_dataset_deduplication
from autosafe.tools.evaluate.dataset.ground_truth import _ground_truth_labels_from_yaml
from autosafe.tools.evaluate.dataset.normalization import (
    _normalizer_from_yaml_bounds,
)
from autosafe.tools.evaluate.dataset.yaml_spec import (
    _legacy_limits_yaml_to_odd_config,
    _sampling_bounds_from_yaml,
)


def _write_csv(path: Path, points: np.ndarray) -> Path:
    header = ",".join(f"x{i}" for i in range(points.shape[1]))
    np.savetxt(path, points, delimiter=",", header=header, comments="")
    return path


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


_LEGACY_LIMITS = """\
limits:
  h:
    values: [-100.0, 0.0, 100.0]
  tau:
    values: [5.0, 0.0, 40.0]
"""


def _policy(dim: int = 2) -> DeduplicationPolicy:
    return DeduplicationPolicy(resolution=(1e-6,) * dim, origin=(0.0,) * dim)


def _id_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    return _write_csv(tmp_path / "id.csv", rng.normal(size=(12, 2)))


@pytest.mark.parametrize("reserved", [-1, 12])
def test_dedup_rejects_invalid_calibration_reservation(
    tmp_path: Path, reserved: int
) -> None:
    with pytest.raises(ValueError, match="n_calibration_reserved"):
        run_dataset_deduplication(
            _id_csv(tmp_path),
            policy=_policy(),
            ood_path=None,
            yaml_normalizer=None,
            normalize_data=True,
            n_calibration_reserved=reserved,
        )


def test_dedup_raises_when_every_row_is_ood(tmp_path: Path) -> None:
    ids = _id_csv(tmp_path)
    with pytest.raises(EmptyAnchorPoolError, match="no anchors"):
        run_dataset_deduplication(
            ids,
            policy=_policy(),
            ood_path=ids,
            yaml_normalizer=None,
            normalize_data=True,
        )


def test_dedup_applies_the_yaml_normalizer(tmp_path: Path) -> None:
    ids = _id_csv(tmp_path)
    normalizer = _normalizer_from_yaml_bounds(
        _write_yaml(
            tmp_path / "box.yaml",
            "type: box\ndim: 2\nlower_bounds: [-10.0, -10.0]\n"
            "upper_bounds: [10.0, 10.0]\n",
        )
    )
    assert normalizer is not None
    outcome = run_dataset_deduplication(
        ids,
        policy=_policy(),
        ood_path=None,
        yaml_normalizer=normalizer,
        normalize_data=True,
    )
    raw = np.loadtxt(ids, delimiter=",", skiprows=1)
    np.testing.assert_allclose(
        np.sort(outcome.id_points_normalized, axis=0),
        np.sort(raw / 10.0, axis=0),
    )


def test_dedup_catches_coincidence_created_by_normalization(tmp_path: Path) -> None:
    """Distinct raw points can collapse onto each other in float64.

    The OOD point 1.0 is not an anchor, but against an anchor range of
    1e18 it normalizes to exactly the same value as the anchor 0.0.
    """
    ids = _write_csv(
        tmp_path / "id.csv", np.array([[0.0], [2.5e17], [5e17], [7.5e17], [1e18]])
    )
    ood = _write_csv(tmp_path / "ood.csv", np.array([[1.0]]))
    with pytest.raises(OODAnchorCoincidenceError, match="coincide exactly"):
        run_dataset_deduplication(
            ids,
            policy=DeduplicationPolicy(resolution=(1.0,), origin=(0.0,)),
            ood_path=ood,
            yaml_normalizer=None,
            normalize_data=True,
        )


def test_legacy_limits_become_a_box(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path / "limits.yaml", _LEGACY_LIMITS)
    bounds = _sampling_bounds_from_yaml(yaml_path)
    assert bounds is not None
    np.testing.assert_array_equal(bounds[0], [-100.0, 0.0])
    np.testing.assert_array_equal(bounds[1], [100.0, 40.0])

    labels = _ground_truth_labels_from_yaml(
        yaml_path, np.array([[0.0, 20.0], [150.0, 20.0], [0.0, -1.0]])
    )
    np.testing.assert_array_equal(labels, [True, False, False])


def test_non_box_yaml_has_no_sampling_bounds(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path / "poly.yaml",
        """\
type: polytope
dim: 2
constraints:
  - type: linear
    coefficients: [1.0, -1.0]
    relation: ">="
    bound: 0.0
""",
    )
    assert _sampling_bounds_from_yaml(yaml_path) is None


@pytest.mark.parametrize(
    "config",
    [
        {"limits": {"h": 1.0}},
        {"limits": {"h": {"values": 1.0}}},
        {"limits": {"h": {"values": []}}},
    ],
    ids=["entry_not_mapping", "values_not_list", "values_empty"],
)
def test_malformed_legacy_limits_raise(config: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError), match="Limit entry for 'h'"):
        _legacy_limits_yaml_to_odd_config(config)


def test_empty_legacy_limits_are_returned_unchanged() -> None:
    config: dict[str, object] = {"limits": {}}
    assert _legacy_limits_yaml_to_odd_config(config) is config


def test_hull_fast_path_in_high_dimensions() -> None:
    """Above 3 dims the hull is approximated by the anchors' bounding ball."""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(50, 5))
    radius = np.max(np.linalg.norm(ref - ref.mean(axis=0), axis=1))
    center = ref.mean(axis=0)
    direction = np.eye(5)[0]
    test = np.vstack([
        center + 0.5 * radius * direction,
        center + 1.5 * radius * direction,
    ])
    np.testing.assert_array_equal(_hull_membership(ref, test), [True, False])


def test_chunked_monitor_evaluation_matches_single_batch() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(40, 2))
    test = rng.normal(scale=2.0, size=(6000, 2))
    monitor = FastHullApproximation()
    monitor.fit(ref.T)
    whole = np.asarray(monitor.evaluate_batch(test.T), dtype=bool)
    chunked = _evaluate_monitor_membership(monitor, test.T, 6000, chunk_size=1000)
    np.testing.assert_array_equal(chunked, whole)
    assert whole.any()
    assert not whole.all()

    data = _BaselineEvaluationData(
        reference_points=ref,
        test_points=test,
        ref_points_t=ref.T,
        test_points_t=test.T,
        n_test_points=6000,
    )
    via_dispatch = _compute_method_membership("fast_hull_approx", data)
    assert via_dispatch is not None
    np.testing.assert_array_equal(via_dispatch, whole)


def test_unknown_baseline_method_returns_no_labels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels = _baseline_memberships(np.zeros((3, 2)), np.zeros((2, 2)), ["bogus"])
    assert labels == {}
    assert (
        "Critical error in baseline membership computation" in capsys.readouterr().out
    )
