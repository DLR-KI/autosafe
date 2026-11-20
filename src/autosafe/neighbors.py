# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Nearest-neighbor search over anchor point sets.

Used to parameterize each anchor's kernel from the distance to its
closest neighbor, globally or per dimension, and to measure how far
query points lie from the anchors.

The search runs in float32 (FAISS ``IndexFlatL2``); every distance
returned here is recomputed in float64 from the found pairs. The pair
is therefore the nearest one under the float32 metric, which can differ
from the float64 nearest only between candidates whose distances agree
to float32 precision.
"""

import faiss
import numpy as np
import numpy.typing as npt
import tqdm.rich

from autosafe.exceptions import (
    EmptyParameterError,
    RowShapeMismatchError,
    ShapeMismatchError,
    TooFewAnchorsError,
)
from autosafe.typing import Matrix, NPMatrix

#: Matrices have two axes.
_MATRIX_NDIM = 2

#: A nearest *other* vector needs at least two vectors.
_MIN_VECTORS = 2


def _as_point_matrix(points: Matrix | NPMatrix, name: str) -> npt.NDArray[np.float64]:
    """Return points as a float64 matrix.

    Args:
        points (Matrix | NPMatrix): Points, shape (m, n).
        name (str): Parameter name used in error messages.

    Returns:
        npt.NDArray[np.float64]: The points as float64, shape (m, n).

    Raises:
        ShapeMismatchError: If ``points`` is not two-dimensional.
    """
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != _MATRIX_NDIM:
        raise ShapeMismatchError(name, result.shape, (None, None))
    return result


def _flat_l2_index(points_float32: npt.NDArray[np.float32]) -> faiss.IndexFlatL2:
    """Build an exact L2 FAISS index over float32 points.

    Args:
        points_float32 (npt.NDArray[np.float32]): C-contiguous points,
            shape (m, n).

    Returns:
        faiss.IndexFlatL2: Index holding all ``m`` points.
    """
    index = faiss.IndexFlatL2(points_float32.shape[1])
    index.add(points_float32)  # pyright: ignore[reportCallIssue]
    return index


def _closest_other_indices(
    sample_array: Matrix | NPMatrix,
    *,
    disable_tqdm: bool,
) -> npt.NDArray[np.int64]:
    """Index of every vector's nearest other vector.

    A ``k=2`` search returns the query and its nearest other vector,
    but not necessarily in that order: with exact duplicates (in
    float32) FAISS may list a twin before the query. At most one of
    the two hits is the query itself, so the first hit whose id differs
    from the query's row id is the nearest other vector. Ties between
    equidistant neighbors resolve arbitrarily.

    Args:
        sample_array (Matrix | NPMatrix): Matrix with m vectors of
            dimension n.
        disable_tqdm (bool): Whether to disable the tqdm progress bar.

    Returns:
        npt.NDArray[np.int64]: Shape (m,); index of the nearest vector
            other than the vector itself.

    Raises:
        TooFewAnchorsError: If fewer than two vectors are given.
    """
    sample_array_float32 = np.ascontiguousarray(sample_array, dtype=np.float32)
    n_vectors = sample_array_float32.shape[0]
    if n_vectors < _MIN_VECTORS:
        raise TooFewAnchorsError(_MIN_VECTORS, n_vectors)
    index = _flat_l2_index(sample_array_float32)

    closest_indices: npt.NDArray[np.int64] = np.array([], dtype=np.int64)
    # Fake tqdm progress bar for consistency
    for _ in tqdm.rich.tqdm(
        range(1),
        desc="Finding closest samples",
        disable=disable_tqdm,
    ):
        _, indices = index.search(sample_array_float32, k=2)  # pyright: ignore[reportCallIssue]
        indices = indices.astype(np.int64)
        own = np.arange(n_vectors, dtype=np.int64)
        first_is_self = indices[:, 0] == own
        closest_indices = np.where(first_is_self, indices[:, 1], indices[:, 0])
    return closest_indices


def find_closest_vectors(
    sample_array: Matrix | NPMatrix,
    *,
    disable_tqdm: bool = False,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Nearest other vector of every vector, with its float64 distance.

    The search runs in float32; the distance is recomputed in float64
    from the found pairs. Exact duplicates are each other's nearest
    other vector (distance 0); a vector is never returned as its own
    neighbor. Ties between equidistant neighbors resolve arbitrarily.
    A ``sample_array`` that is not a matrix raises
    :class:`~autosafe.exceptions.ShapeMismatchError`, one with fewer
    than two rows :class:`~autosafe.exceptions.TooFewAnchorsError`.

    Args:
        sample_array (Matrix | NPMatrix): Matrix with m >= 2 vectors of
            dimension n.
        disable_tqdm (bool): Whether to disable the tqdm progress bar.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: Shape
            (m,) index of the nearest other vector, and shape (m,)
            float64 L2 distance to it.
    """
    points = _as_point_matrix(sample_array, "sample_array")
    indices = _closest_other_indices(points, disable_tqdm=disable_tqdm)
    distances = np.linalg.norm(points[indices] - points, axis=1)
    return indices, np.asarray(distances, dtype=np.float64)


def find_closest_vectors_by_index(
    sample_array: Matrix | NPMatrix,
    disable_tqdm: bool = False,  # ruff:ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
) -> npt.NDArray[np.int64]:
    """Find the closest other vector for each vector in the matrix.

    Same search as :func:`find_closest_vectors`, without the distances.

    Args:
        sample_array (Matrix | NPMatrix): Matrix with m >= 2 vectors of
            dimension n.
        disable_tqdm (bool): Whether to disable the tqdm progress bar.

    Returns:
        npt.NDArray[np.int64]: Index of the closest other vector for
            each vector.
    """
    return _closest_other_indices(sample_array, disable_tqdm=disable_tqdm)


def find_closest_vectors_by_index_per_dimension(
    sample_array: Matrix | NPMatrix,
    *,
    disable_tqdm: bool = False,
) -> npt.NDArray[np.int64]:
    """Find the closest vector for each vector per dimension.

    For each dimension, return the index of the closest other vector
    based on that dimension alone.

    Args:
        sample_array (Matrix | NPMatrix): Matrix with m vectors of
            dimension n.
        disable_tqdm (bool): Whether to disable the tqdm progress bar.

    Returns:
        npt.NDArray[np.int64]: Shape (n, m); index of the closest other
            vector per dimension for each vector.
    """
    m, n = sample_array.shape
    closest_per_dim = np.zeros((n, m), dtype=np.int64)

    for dim in tqdm.rich.tqdm(
        range(n),
        desc="Finding closest samples per dimension",
        disable=disable_tqdm,
    ):
        closest_per_dim[dim, :] = find_closest_vectors_by_index(
            sample_array=sample_array[:, dim].reshape(-1, 1),
            disable_tqdm=True,
        )

    return closest_per_dim


def distance_to_nearest_anchor(
    queries: Matrix | NPMatrix,
    anchors: Matrix | NPMatrix,
    *,
    disable_tqdm: bool = True,
) -> npt.NDArray[np.float64]:
    """(M,) float64 L2 distance of every query to its nearest anchor.

    The nearest anchor is found by an exact float32 search; the
    distance is recomputed in float64 from the found pair. A query that
    coincides with an anchor has distance 0 (queries are not anchors,
    so there is no self-exclusion).

    Args:
        queries (Matrix | NPMatrix): Query points, shape (M, n).
        anchors (Matrix | NPMatrix): Anchor points, shape (N, n),
            N >= 1.
        disable_tqdm (bool): Whether to disable the tqdm progress bar.

    Returns:
        npt.NDArray[np.float64]: Shape (M,) distance of every query to
            its nearest anchor.

    Raises:
        EmptyParameterError: If ``anchors`` has no rows.
        RowShapeMismatchError: If queries and anchors differ in their
            number of columns. Inputs that are not matrices raise
            :class:`~autosafe.exceptions.ShapeMismatchError`.
    """
    query_points = _as_point_matrix(queries, "queries")
    anchor_points = _as_point_matrix(anchors, "anchors")
    if anchor_points.shape[0] == 0:
        raise EmptyParameterError("anchors")
    if query_points.shape[1] != anchor_points.shape[1]:
        raise RowShapeMismatchError(query_points.shape[1], anchor_points.shape[1])
    if query_points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    index = _flat_l2_index(np.ascontiguousarray(anchor_points, dtype=np.float32))
    nearest: npt.NDArray[np.int64] = np.array([], dtype=np.int64)
    for _ in tqdm.rich.tqdm(
        range(1),
        desc="Finding nearest anchors",
        disable=disable_tqdm,
    ):
        _, indices = index.search(  # pyright: ignore[reportCallIssue]
            np.ascontiguousarray(query_points, dtype=np.float32), k=1
        )
        nearest = indices[:, 0].astype(np.int64)
    distances = np.linalg.norm(query_points - anchor_points[nearest], axis=1)
    return np.asarray(distances, dtype=np.float64)


__all__ = [
    "distance_to_nearest_anchor",
    "find_closest_vectors",
    "find_closest_vectors_by_index",
    "find_closest_vectors_by_index_per_dimension",
]
