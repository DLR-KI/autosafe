# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for nearest-neighbor search and anchor subsampling."""

import numpy as np
import pytest

from autosafe.exceptions import (
    EmptyParameterError,
    NonPositiveParameterError,
    RowShapeMismatchError,
    ShapeMismatchError,
    TooFewAnchorsError,
)
from autosafe.neighbors import (
    distance_to_nearest_anchor,
    find_closest_vectors,
    find_closest_vectors_by_index,
    find_closest_vectors_by_index_per_dimension,
)
from autosafe.pointsets import subsample_indices


def _brute_force_nearest_other(points: np.ndarray) -> np.ndarray:
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(gaps, np.inf)
    return gaps.min(axis=1)


def test_find_closest_vectors_returns_float64_distances() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(200, 4))

    indices, distances = find_closest_vectors(points, disable_tqdm=True)

    assert indices.dtype == np.int64
    assert distances.dtype == np.float64
    assert np.all(indices != np.arange(points.shape[0]))
    np.testing.assert_array_equal(
        distances, np.linalg.norm(points[indices] - points, axis=1)
    )
    np.testing.assert_allclose(distances, _brute_force_nearest_other(points))
    np.testing.assert_array_equal(
        indices, find_closest_vectors_by_index(points, disable_tqdm=True)
    )


def test_duplicates_are_never_their_own_neighbor() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(50, 3))
    # Every row appears two to four times, so FAISS sees many exact ties
    # and may list a twin before the query itself.
    points = np.repeat(base, rng.integers(2, 5, size=base.shape[0]), axis=0)
    points = points[rng.permutation(points.shape[0])]

    indices, distances = find_closest_vectors(points, disable_tqdm=True)

    assert np.all(indices != np.arange(points.shape[0]))
    np.testing.assert_array_equal(points[indices], points)
    np.testing.assert_array_equal(distances, 0.0)


def test_per_dimension_search_excludes_self_with_ties() -> None:
    points = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 2.0], [1.0, 5.0]])

    per_dim = find_closest_vectors_by_index_per_dimension(points, disable_tqdm=True)

    assert per_dim.shape == (2, 4)
    assert np.all(per_dim != np.arange(4))
    np.testing.assert_array_equal(per_dim[0], [1, 0, 3, 2])


def test_find_closest_vectors_validates_input() -> None:
    with pytest.raises(TooFewAnchorsError):
        find_closest_vectors(np.zeros((1, 2)), disable_tqdm=True)
    with pytest.raises(ShapeMismatchError):
        find_closest_vectors(np.zeros(3), disable_tqdm=True)  # ty: ignore[invalid-argument-type]


def test_distance_to_nearest_anchor_matches_brute_force() -> None:
    rng = np.random.default_rng(2)
    anchors = rng.normal(size=(300, 5))
    queries = np.vstack([rng.normal(size=(100, 5)) * 2.0, anchors[:7]])

    distances = distance_to_nearest_anchor(queries, anchors)

    expected = np.linalg.norm(queries[:, None, :] - anchors[None, :, :], axis=2).min(
        axis=1
    )
    assert distances.dtype == np.float64
    assert distances.shape == (queries.shape[0],)
    np.testing.assert_allclose(distances, expected, rtol=1e-6)
    np.testing.assert_array_equal(distances[-7:], 0.0)


def test_distance_to_nearest_anchor_validates_input() -> None:
    np.testing.assert_array_equal(
        distance_to_nearest_anchor(np.zeros((0, 2)), np.ones((3, 2))),
        np.zeros(0),
    )
    with pytest.raises(EmptyParameterError):
        distance_to_nearest_anchor(np.zeros((2, 2)), np.zeros((0, 2)))
    with pytest.raises(RowShapeMismatchError):
        distance_to_nearest_anchor(np.zeros((2, 2)), np.zeros((2, 3)))


def test_subsample_indices_matches_inline_draw() -> None:
    count, size, seed = 1000, 37, 11
    expected = np.sort(
        np.random.default_rng(seed).choice(count, size=size, replace=False)
    )

    drawn = subsample_indices(count, size, seed)

    assert drawn.dtype == np.int64
    np.testing.assert_array_equal(drawn, expected)
    np.testing.assert_array_equal(subsample_indices(5, 5, seed), np.arange(5))
    np.testing.assert_array_equal(subsample_indices(5, 9, seed), np.arange(5))
    with pytest.raises(NonPositiveParameterError):
        subsample_indices(5, 0, seed)
