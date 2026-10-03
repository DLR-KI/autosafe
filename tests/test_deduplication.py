# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Unit tests for src/autosafe/deduplication.py."""

import numpy as np
import pytest

from autosafe.deduplication import (
    DeduplicationPolicy,
    check_label_conflicts,
    deduplicate_points,
    exact_equality_policy,
)
from autosafe.exceptions import (
    DeduplicationDimensionMismatchError,
    DeduplicationLabelConflictError,
    InvalidResolutionVectorError,
    MissingRecordIdentifierError,
    NonFiniteCoordinateError,
    ResolutionCellOverflowError,
)


def test_exact_duplicates_collapse_and_preserve_ids() -> None:
    """Exact duplicates collapse to one point; all source ids survive."""
    pts = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [5.0, 5.0]])
    policy = DeduplicationPolicy(resolution=(1.0, 1.0), origin=(0.0, 0.0))
    result = deduplicate_points(pts, policy, record_ids=["a", "b", "c", "d"])

    assert result.n_output == 2
    assert result.n_input == 4
    assert result.n_duplicates == 2
    groups_by_size = sorted(result.groups, key=len)
    assert groups_by_size[0] == ("d",)
    assert set(groups_by_size[1]) == {"a", "b", "c"}


def test_near_duplicates_in_one_cell_collapse_others_do_not() -> None:
    """Near duplicates within a cell collapse; separate cells do not."""
    pts = np.array([[0.0, 0.0], [0.05, 0.0], [0.9, 0.0], [0.95, 0.0]])
    policy = DeduplicationPolicy(resolution=(0.5, 0.5), origin=(0.0, 0.0))
    result = deduplicate_points(pts, policy)

    assert result.n_output == 2
    assert result.multiplicities.tolist() == [2, 2]


def test_adjacent_cells_do_not_merge() -> None:
    """Two points straddling a cell boundary land in different cells."""
    pts = np.array([[0.49, 0.0], [0.51, 0.0]])
    policy = DeduplicationPolicy(resolution=(0.5, 0.5), origin=(0.0, 0.0))
    result = deduplicate_points(pts, policy)

    assert result.n_output == 2
    assert not np.array_equal(result.cell_ids[0], result.cell_ids[1])


def test_boundary_values_follow_floor_convention() -> None:
    """A point exactly on a cell boundary follows floor((x-o)/q)."""
    policy = DeduplicationPolicy(resolution=(1.0,), origin=(0.0,))
    result = deduplicate_points(np.array([[1.0], [0.999999]]), policy)
    # 1.0 floors to cell 1; 0.999999 floors to cell 0: different cells.
    assert result.n_output == 2
    assert sorted(int(c[0]) for c in result.cell_ids) == [0, 1]

    # A point exactly at the origin is cell 0, together with anything up
    # to (but excluding) the next boundary.
    result2 = deduplicate_points(np.array([[0.0], [0.999999]]), policy)
    assert result2.n_output == 1


def test_representative_is_always_an_observed_row() -> None:
    """The representative is a member point, never the synthetic mean."""
    pts = np.array([[0.0, 0.0], [0.2, 0.0], [1.0, 0.0]])
    policy = DeduplicationPolicy(resolution=(2.0, 2.0), origin=(0.0, 0.0))
    result = deduplicate_points(pts, policy)

    assert result.n_output == 1
    mean = pts.mean(axis=0)
    assert not np.allclose(result.points[0], mean)
    assert any(np.array_equal(result.points[0], row) for row in pts)


def test_representative_minimizes_resolution_scaled_distance() -> None:
    """The representative minimizes ||(x - mean) / resolution||_2."""
    # Anisotropic resolution: dimension 1 is "coarser" so distances along
    # it count for less. Wide enough that all three fall in one cell.
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0]])
    policy = DeduplicationPolicy(resolution=(2.0, 10.0), origin=(0.0, 0.0))
    result = deduplicate_points(pts, policy)
    assert result.n_output == 1

    mean = pts.mean(axis=0)
    scaled = (pts - mean) / np.array([1.0, 10.0])
    expected = pts[np.argmin(np.linalg.norm(scaled, axis=1))]
    assert np.array_equal(result.points[0], expected)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_permutation_invariance(seed: int) -> None:
    """All permutations give identical representatives, order, ids."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(40, 3))
    # Force real duplication so grouping is nontrivial.
    pts = np.vstack([base, base[:10], base[:3]])
    ids = list(range(pts.shape[0]))
    policy = DeduplicationPolicy(resolution=(0.3, 0.3, 0.3), origin=(0.0, 0.0, 0.0))

    baseline = deduplicate_points(pts, policy, record_ids=ids)

    perm = rng.permutation(pts.shape[0])
    shuffled = deduplicate_points(pts[perm], policy, record_ids=[ids[i] for i in perm])

    np.testing.assert_array_equal(baseline.points, shuffled.points)
    assert baseline.groups == shuffled.groups
    assert baseline.representative_ids == shuffled.representative_ids
    np.testing.assert_array_equal(baseline.multiplicities, shuffled.multiplicities)
    np.testing.assert_array_equal(baseline.cell_ids, shuffled.cell_ids)
    assert baseline.input_digest == shuffled.input_digest
    assert baseline.output_digest == shuffled.output_digest


def test_idempotence() -> None:
    """De-duplicating an already-de-duplicated point set is a no-op."""
    rng = np.random.default_rng(11)
    base = rng.normal(size=(25, 2))
    pts = np.vstack([base, base[:8]])
    policy = DeduplicationPolicy(resolution=(0.4, 0.4), origin=(0.0, 0.0))

    once = deduplicate_points(pts, policy)
    twice = deduplicate_points(once.points, policy)

    np.testing.assert_array_equal(once.points, twice.points)
    assert twice.n_duplicates == 0


def test_unit_transformation_gives_identical_grouping() -> None:
    """Rescaling coordinates+resolution+origin by the same factor agrees.

    Uses well-separated clusters (a clear nearest-to-mean winner in each
    cell) rather than continuous random data, so the comparison is not
    sensitive to a genuine near-tie flipping under the rescaled
    floating-point arithmetic---a distinct numerical-robustness concern
    from the property under test here (that grouping and representative
    identity are unit-invariant).
    """
    pts = np.array([
        [0.0, 0.0],
        [0.05, 0.02],
        [-0.03, 0.04],
        [3.0, 3.0],
        [3.02, 2.98],
    ])
    policy = DeduplicationPolicy(resolution=(0.5, 0.5), origin=(0.1, -0.2))
    result = deduplicate_points(pts, policy, record_ids=list(range(pts.shape[0])))

    scale = 1000.0  # e.g. meters -> millimeters
    scaled_pts = pts * scale
    scaled_policy = DeduplicationPolicy(
        resolution=(0.5 * scale, 0.5 * scale),
        origin=(0.1 * scale, -0.2 * scale),
    )
    scaled_result = deduplicate_points(
        scaled_pts, scaled_policy, record_ids=list(range(pts.shape[0]))
    )

    assert result.groups == scaled_result.groups
    assert result.representative_ids == scaled_result.representative_ids
    np.testing.assert_allclose(result.points * scale, scaled_result.points)


def test_equal_distance_ties_break_lexicographically() -> None:
    """Equal-distance representative ties resolve by (coords, record id)."""
    # Symmetric around the mean at (0.5, 0): both members are equidistant.
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    policy = DeduplicationPolicy(resolution=(2.0, 2.0), origin=(0.0, 0.0))

    result_a = deduplicate_points(pts, policy, record_ids=["z", "a"])
    # Lexicographically, (0.0, 0.0) < (1.0, 0.0), so it wins regardless of id.
    assert np.array_equal(result_a.points[0], [0.0, 0.0])
    assert result_a.representative_ids[0] == "z"


def test_identical_coordinate_ties_break_by_record_id() -> None:
    """Exact coordinate ties resolve by the stable record identifier."""
    pts = np.array([[0.0, 0.0], [0.0, 0.0]])
    policy = DeduplicationPolicy(resolution=(1.0, 1.0), origin=(0.0, 0.0))

    result = deduplicate_points(pts, policy, record_ids=["b", "a"])
    assert result.representative_ids[0] == "a"


def test_missing_record_identifiers_raise() -> None:
    """A None entry or wrong-length record_ids raises clearly."""
    pts = np.array([[0.0], [1.0], [2.0]])
    policy = DeduplicationPolicy(resolution=(1.0,), origin=(0.0,))

    with pytest.raises(MissingRecordIdentifierError):
        deduplicate_points(pts, policy, record_ids=[1, None, 3])  # ty: ignore[invalid-argument-type]

    with pytest.raises(MissingRecordIdentifierError):
        deduplicate_points(pts, policy, record_ids=[1, 2])


def test_label_conflicts_reported_not_raised_by_default() -> None:
    """Cell co-membership between ID and OOD is reported, not raised."""
    policy = DeduplicationPolicy(resolution=(1.0, 1.0), origin=(0.0, 0.0))
    id_result = deduplicate_points(np.array([[0.2, 0.2]]), policy)
    ood_result = deduplicate_points(np.array([[0.3, 0.3]]), policy)

    conflicts = check_label_conflicts(id_result, ood_result)
    assert conflicts == ((0, 0),)


def test_label_conflicts_raise_when_strict() -> None:
    """The strict flag turns cell co-membership into a hard error."""
    policy = DeduplicationPolicy(resolution=(1.0, 1.0), origin=(0.0, 0.0))
    id_result = deduplicate_points(np.array([[0.2, 0.2]]), policy)
    ood_result = deduplicate_points(np.array([[0.3, 0.3]]), policy)

    with pytest.raises(DeduplicationLabelConflictError):
        check_label_conflicts(id_result, ood_result, strict=True)


def test_no_label_conflict_when_cells_differ() -> None:
    """Disjoint cells produce an empty conflict report."""
    policy = DeduplicationPolicy(resolution=(1.0, 1.0), origin=(0.0, 0.0))
    id_result = deduplicate_points(np.array([[0.2, 0.2]]), policy)
    ood_result = deduplicate_points(np.array([[5.3, 5.3]]), policy)

    assert check_label_conflicts(id_result, ood_result) == ()


def test_invalid_resolution_rejected() -> None:
    """Non-positive or non-finite resolution components fail clearly."""
    with pytest.raises(InvalidResolutionVectorError):
        DeduplicationPolicy(resolution=(0.0, 1.0), origin=(0.0, 0.0))
    with pytest.raises(InvalidResolutionVectorError):
        DeduplicationPolicy(resolution=(-1.0, 1.0), origin=(0.0, 0.0))
    with pytest.raises(InvalidResolutionVectorError):
        DeduplicationPolicy(resolution=(float("nan"), 1.0), origin=(0.0, 0.0))
    with pytest.raises(InvalidResolutionVectorError):
        DeduplicationPolicy(resolution=(float("inf"), 1.0), origin=(0.0, 0.0))


def test_partial_resolution_vector_rejected() -> None:
    """A resolution/origin length mismatch is rejected as a partial vector."""
    with pytest.raises(DeduplicationDimensionMismatchError):
        DeduplicationPolicy(resolution=(1.0, 1.0, 1.0), origin=(0.0, 0.0))


def test_dimension_mismatch_between_policy_and_points() -> None:
    """A points array whose width disagrees with the policy is rejected."""
    policy = DeduplicationPolicy(resolution=(1.0, 1.0), origin=(0.0, 0.0))
    with pytest.raises(DeduplicationDimensionMismatchError):
        deduplicate_points(np.zeros((3, 3)), policy)


def test_non_finite_coordinates_rejected() -> None:
    """NaN or infinite coordinates fail clearly."""
    policy = DeduplicationPolicy(resolution=(1.0, 1.0), origin=(0.0, 0.0))
    with pytest.raises(NonFiniteCoordinateError):
        deduplicate_points(np.array([[0.0, float("nan")]]), policy)
    with pytest.raises(NonFiniteCoordinateError):
        deduplicate_points(np.array([[0.0, float("inf")]]), policy)


def test_points_must_be_2d() -> None:
    """A 1D or 3D input array is rejected."""
    policy = DeduplicationPolicy(resolution=(1.0,), origin=(0.0,))
    with pytest.raises(ValueError, match="2D"):
        deduplicate_points(np.array([0.0, 1.0, 2.0]), policy)


def test_exact_equality_policy_never_merges_distinct_records() -> None:
    """The no-op exact-equality policy merges only exact duplicates."""
    policy = exact_equality_policy(2)
    pts = np.array([[1.234567, 2.345678], [1.234567, 2.345678], [1.234568, 2.345678]])
    result = deduplicate_points(pts, policy)
    # First two are bit-identical and merge; the third differs at 1e-6 and
    # a resolution of 1e-12 (default epsilon) keeps it separate.
    assert result.n_output == 2


def test_empty_input() -> None:
    """An empty point set is handled without error."""
    policy = DeduplicationPolicy(resolution=(1.0, 1.0), origin=(0.0, 0.0))
    result = deduplicate_points(np.zeros((0, 2)), policy)
    assert result.n_output == 0
    assert result.n_input == 0
    assert result.groups == ()


def test_constant_dimension_recorded_explicitly() -> None:
    """A dimension where every row shares one coordinate is flagged."""
    pts = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    policy = DeduplicationPolicy(resolution=(0.5, 0.5), origin=(0.0, 0.0))
    result = deduplicate_points(pts, policy)
    assert result.constant_dims == (False, True)


def test_provenance_round_trips_through_digest() -> None:
    """Digests are stable and change when the underlying data changes."""
    policy = DeduplicationPolicy(resolution=(0.5, 0.5), origin=(0.0, 0.0))
    pts = np.array([[0.0, 0.0], [0.9, 0.9]])
    result_a = deduplicate_points(pts, policy)
    result_b = deduplicate_points(pts, policy)
    assert result_a.input_digest == result_b.input_digest
    assert result_a.output_digest == result_b.output_digest

    result_c = deduplicate_points(np.array([[0.0, 0.0], [0.95, 0.95]]), policy)
    assert result_a.input_digest != result_c.input_digest


def test_cell_count_distribution() -> None:
    """cell_count_distribution histograms multiplicities correctly."""
    pts = np.array([[0.0], [0.0], [1.0], [1.0], [1.0], [5.0]])
    policy = DeduplicationPolicy(resolution=(0.5,), origin=(0.0,))
    result = deduplicate_points(pts, policy)
    assert result.cell_count_distribution() == {2: 1, 3: 1, 1: 1}


def test_policy_digest_changes_with_policy() -> None:
    """The policy digest is sensitive to every field it captures."""
    base = DeduplicationPolicy(resolution=(0.5, 0.5), origin=(0.0, 0.0))
    other_resolution = DeduplicationPolicy(resolution=(0.6, 0.5), origin=(0.0, 0.0))
    other_origin = DeduplicationPolicy(resolution=(0.5, 0.5), origin=(0.1, 0.0))
    other_space = DeduplicationPolicy(
        resolution=(0.5, 0.5), origin=(0.0, 0.0), coordinate_space="normalized"
    )

    digests = {
        base.digest(),
        other_resolution.digest(),
        other_origin.digest(),
        other_space.digest(),
    }
    assert len(digests) == 4


def test_cell_index_overflow_raises_instead_of_merging() -> None:
    """Distinct points must never be merged by a wrapped int64 cell index."""
    pts = np.array([[0.0], [2.5e19], [5e19], [7.5e19], [1e20]])
    policy = DeduplicationPolicy(resolution=(1e-6,), origin=(0.0,))
    with pytest.raises(ResolutionCellOverflowError, match="dimension 0"):
        deduplicate_points(pts, policy)


def test_cell_index_just_inside_int64_is_accepted() -> None:
    pts = np.array([[0.0], [9.0e18]])
    result = deduplicate_points(
        pts, DeduplicationPolicy(resolution=(1.0,), origin=(0.0,))
    )
    assert result.points.shape == (2, 1)
