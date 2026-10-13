# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the dataset-mode helpers: de-duplication, YAML specs, baselines."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from autosafe.deduplication import DeduplicationPolicy
from autosafe.exceptions import (
    EmptyAnchorPoolError,
    EmptyParameterError,
    MissingColumnError,
    OODAnchorCoincidenceError,
    ParameterTypeError,
)
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
    _normalized_discrete_values,
    _normalizer_from_yaml_bounds,
)
from autosafe.tools.evaluate.dataset.sampling import _sample_points_with_bounds
from autosafe.tools.evaluate.dataset.yaml_spec import (
    _categories_from_yaml,
    _discrete_values_from_yaml,
    _legacy_limits_yaml_to_odd_config,
    _periodic_columns_from_yaml,
    _sampling_bounds_from_yaml,
    _wrapped_columns_from_yaml,
)
from autosafe.tools.experiments.utils import DatasetLoadOptions, load_dataset


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
    -   type: linear
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
    with pytest.raises((ParameterTypeError, EmptyParameterError), match=r"^h must"):
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


_DISCRETE_LIMITS = """\
limits:
    h:
        type: range
        values: [-100.0, 100.0]
    s_adv:
        type: discrete
        values: [0, 1, 2, 4]
"""


def test_discrete_values_are_read_from_limits(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path / "limits.yaml", _DISCRETE_LIMITS)
    discrete = _discrete_values_from_yaml(yaml_path)
    assert list(discrete) == [1]
    np.testing.assert_array_equal(discrete[1], [0.0, 1.0, 2.0, 4.0])
    assert (
        _discrete_values_from_yaml(_write_yaml(tmp_path / "range.yaml", _LEGACY_LIMITS))
        == {}
    )


def test_ground_truth_requires_a_valid_discrete_code(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path / "limits.yaml", _DISCRETE_LIMITS)
    points = np.array([
        [0.0, 2.0],  # valid code
        [0.0, 2.5],  # inside [0, 4] but not a code
        [0.0, 3.0],  # gap in the code set
        [0.0, -0.4],  # below the smallest code
        [0.0, 4.0 + 1e-9],  # round-trip noise on a valid code
    ])
    labels = _ground_truth_labels_from_yaml(yaml_path, points)
    np.testing.assert_array_equal(labels, [True, False, False, False, True])


def test_ground_truth_discrete_codes_survive_normalization(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path / "limits.yaml", _DISCRETE_LIMITS)
    normalizer = _normalizer_from_yaml_bounds(yaml_path)
    lower = np.array([-100.0, 0.0])
    codes = _normalized_discrete_values(yaml_path, lower, normalizer)[1]
    points = np.column_stack([np.zeros(codes.size + 1), np.append(codes, 0.1)])
    labels = _ground_truth_labels_from_yaml(yaml_path, points, normalizer=normalizer)
    np.testing.assert_array_equal(labels, [True] * codes.size + [False])


def test_bounded_sampler_keeps_discrete_coordinates_on_codes() -> None:
    rng = np.random.default_rng(1)
    codes = np.array([0.0, 1.0, 2.0, 4.0])
    anchors = np.column_stack([rng.uniform(-1, 1, 50), rng.choice(codes, 50)])
    lower, upper = np.array([-1.0, 0.0]), np.array([1.0, 4.0])
    plain = _sample_points_with_bounds(anchors, lower, upper, n_samples=400, seed=3)
    fixed = _sample_points_with_bounds(
        anchors, lower, upper, n_samples=400, seed=3, discrete_values={1: codes}
    )
    # The continuous coordinates come from the same stream either way.
    np.testing.assert_array_equal(plain[:, 0], fixed[:, 0])
    assert np.isin(fixed[:, 1], codes).all()
    assert not np.isin(plain[:, 1], codes).all()
    # Uniform half covers every code; local half keeps anchor codes.
    assert set(np.unique(fixed[:200, 1])) == set(codes)
    assert set(np.unique(fixed[200:, 1])) <= set(np.unique(anchors[:, 1]))


_WRAPPED_LIMITS = """\
limits:
    psi:
        type: wrapped_range
        values: [-180.0, 180.0]
    tau:
        type: range
        values: [0.0, 40.0]
"""


def test_wrapped_columns_are_read_from_limits(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path / "limits.yaml", _WRAPPED_LIMITS)
    assert _wrapped_columns_from_yaml(yaml_path) == [0]
    bounds = _sampling_bounds_from_yaml(yaml_path)
    assert bounds is not None
    np.testing.assert_array_equal(bounds[0], [-180.0, 0.0])
    assert (
        _wrapped_columns_from_yaml(_write_yaml(tmp_path / "plain.yaml", _LEGACY_LIMITS))
        == []
    )


def test_ground_truth_wraps_periodic_columns(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path / "limits.yaml", _WRAPPED_LIMITS)
    points = np.array([
        [180.3, 20.0],  # past the face of the wrapped range: inside
        [-541.0, 20.0],  # several periods away: still inside
        [0.0, 40.5],  # past the face of an ordinary range: outside
        [179.9999, 20.0],  # just below the upper end: inside
        [-180.0000001, 20.0],  # just below the lower end: inside
    ])
    labels = _ground_truth_labels_from_yaml(yaml_path, points)
    np.testing.assert_array_equal(labels, [True, True, False, True, True])
    # The caller's array is not modified by the wrapping.
    assert points[0, 0] == pytest.approx(180.3)


def test_bounded_sampler_wraps_periodic_coordinates() -> None:
    rng = np.random.default_rng(2)
    anchors = np.column_stack([rng.choice([-0.999, 0.999], 60), rng.uniform(-1, 1, 60)])
    lower, upper = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    plain = _sample_points_with_bounds(
        anchors, lower, upper, n_samples=2000, seed=4, local_noise_std=0.05
    )
    wrapped = _sample_points_with_bounds(
        anchors,
        lower,
        upper,
        n_samples=2000,
        seed=4,
        local_noise_std=0.05,
        wrapped_columns=[0],
    )
    np.testing.assert_array_equal(plain[:, 1], wrapped[:, 1])
    assert (plain[:, 0] < -1).any()
    assert (plain[:, 0] > 1).any()
    assert ((wrapped[:, 0] >= -1) & (wrapped[:, 0] < 1)).all()
    # The uniform half still covers one period uniformly.
    counts, _ = np.histogram(wrapped[:1000, 0], bins=4, range=(-1, 1))
    assert int(np.min(counts)) > 200


_FINITE_LIMITS = """\
limits:
    hour:
        type: wrapped_discrete
        values: [0, 6, 12, 18]
    vehicle:
        type: categorical
        values: [car, bus, airplane]
    weekday:
        type: wrapped_discrete
        values: [1, 2, 3, 4, 5]
        period: 7
"""


def test_finite_limit_types_give_codes_bounds_and_periods(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path / "limits.yaml", _FINITE_LIMITS)
    codes = _discrete_values_from_yaml(yaml_path)
    np.testing.assert_array_equal(codes[0], [0.0, 6.0, 12.0, 18.0])
    np.testing.assert_array_equal(codes[1], [0.0, 1.0, 2.0])
    assert _categories_from_yaml(yaml_path) == {"vehicle": ["car", "bus", "airplane"]}
    bounds = _sampling_bounds_from_yaml(yaml_path)
    assert bounds is not None
    np.testing.assert_array_equal(bounds[0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(bounds[1], [18.0, 2.0, 5.0])
    # Default period: span plus the code spacing; explicit period wins.
    assert _periodic_columns_from_yaml(yaml_path) == {0: (0.0, 24.0), 2: (1.0, 7.0)}


def test_ground_truth_wraps_discrete_codes(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path / "limits.yaml", _FINITE_LIMITS)
    points = np.array([
        [24.0, 0.0, 1.0],  # hour 24 == 0
        [-6.0, 1.0, 8.0],  # hour -6 == 18, weekday 8 == 1
        [-1e-12, 2.0, 3.0],  # round-trip noise below the lowest code
        [3.0, 0.0, 1.0],  # hour 3 is not a code
        [0.0, 1.5, 1.0],  # between two categories
        [0.0, 0.0, 6.0],  # weekday 6 is not a code (period 7)
    ])
    labels = _ground_truth_labels_from_yaml(yaml_path, points)
    np.testing.assert_array_equal(labels, [True, True, True, False, False, False])


def test_wrapped_range_period_rejects_degenerate_interval(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path / "limits.yaml",
        "limits:\n    a:\n        type: wrapped_range\n        values: [5.0, 5.0]\n",
    )
    with pytest.raises(ValueError, match="period > 0"):
        _periodic_columns_from_yaml(yaml_path)


def test_categories_are_encoded_on_load(tmp_path: Path) -> None:
    csv = tmp_path / "vehicles.csv"
    csv.write_text("speed,vehicle\n1.0,bus\n2.0,airplane\n3.0,car\n", encoding="utf-8")
    categories = {"vehicle": ["car", "bus", "airplane"]}
    df, _ = load_dataset(
        csv, DatasetLoadOptions(normalize=False, categories=categories)
    )
    assert df["vehicle"].to_list() == [1, 2, 0]
    assert df["vehicle"].dtype == pl.Int64
    csv.write_text("speed,vehicle\n1.0,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside its categories"):
        load_dataset(csv, DatasetLoadOptions(normalize=False, categories=categories))
    with pytest.raises(MissingColumnError):
        load_dataset(csv, DatasetLoadOptions(normalize=False, categories={"x": ["a"]}))
