# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Reference (pre-vectorization) implementation of ``deduplicate_points``.

Kept verbatim so the regression test can check that the vectorized
implementation returns bit-identical results.
"""

import hashlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from autosafe.deduplication import (
    DeduplicationPolicy,
    DeduplicationResult,
    RecordId,
    _id_sort_key,
)
from autosafe.exceptions import (
    DeduplicationDimensionMismatchError,
    MissingRecordIdentifierError,
    NonFiniteCoordinateError,
    ResolutionCellOverflowError,
    ShapeMismatchError,
)

_INT64_CELL_LIMIT = float(2**63)


def reference_digest_rows(
    points: npt.NDArray[np.float64], ids: Sequence[RecordId]
) -> str:
    """Order-independent digest of a (points, ids) pair.

    Rows are canonically sorted by ``(coordinates, id)`` before hashing
    so permuting the input rows never changes the digest.

    Args:
        points (npt.NDArray[np.float64]): Shape ``(n, n_dims)``.
        ids (Sequence[RecordId]): Length-``n`` record identifiers.

    Returns:
        str: 64-character hex SHA-256 digest.
    """
    n = points.shape[0]
    if n == 0:
        return hashlib.sha256(b"dedup:empty").hexdigest()
    order = sorted(
        range(n),
        key=lambda i: (tuple(float(v) for v in points[i]), _id_sort_key(ids[i])),
    )
    sorted_points = np.ascontiguousarray(points[order], dtype=np.float64)
    id_repr = "|".join(repr(ids[i]) for i in order).encode("utf-8")
    payload = sorted_points.tobytes() + b":" + id_repr
    return hashlib.sha256(payload).hexdigest()


def reference_deduplicate_points(  # ruff:ignore[too-many-locals, too-many-statements]
    points: npt.ArrayLike,
    policy: DeduplicationPolicy,
    *,
    record_ids: Sequence[RecordId] | None = None,
) -> DeduplicationResult:
    """De-duplicate points into resolution cells, deterministically.

    Target complexity is ``O(L n log L)`` time and ``O(L n)`` space for
    ``L`` input records of dimension ``n``. The result is invariant to
    the input row order (permutation invariance): the canonical sort
    key is ``(cell_id tuple, coordinate tuple, record id)``, which
    never depends on positional index, and ties in representative
    distance are broken by the earliest element of that same
    canonical order---exactly lexicographic-by-coordinates-then-id.

    Args:
        points (npt.ArrayLike): Shape ``(n_points, n_dims)`` finite
            coordinates.
        policy (DeduplicationPolicy): Fully resolved de-duplication
            policy; ``policy.n_dims`` must match ``points``' column
            count.
        record_ids (Sequence[RecordId] | None): Stable source-record
            identifier per row. When ``None``, the row index is used.
            When supplied, every row must have a non-``None`` entry.

    Returns:
        DeduplicationResult: Representatives, groups, multiplicities,
            cell ids, the resolved policy, and audit digests.

    Raises:
        ShapeMismatchError: If ``points`` is not 2-dimensional.
        DeduplicationDimensionMismatchError: If ``points``' column count
            does not match ``policy.n_dims``.
        NonFiniteCoordinateError: If any coordinate is non-finite.
        ResolutionCellOverflowError: If a cell index does not fit into
            int64, i.e. the resolution is too fine for the coordinates.
        MissingRecordIdentifierError: If ``record_ids`` is supplied with
            the wrong length or a ``None`` entry.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ShapeMismatchError("points", pts.shape, (None, None))
    n_input, n_dims = pts.shape
    if n_dims != policy.n_dims:
        raise DeduplicationDimensionMismatchError(policy.n_dims, n_dims)

    if n_input and not np.all(np.isfinite(pts)):
        bad_mask = ~np.all(np.isfinite(pts), axis=1)
        raise NonFiniteCoordinateError(
            int(bad_mask.sum()), n_input, int(np.argmax(bad_mask))
        )

    if record_ids is None:
        ids: tuple[RecordId, ...] = tuple(range(n_input))
    else:
        ids = tuple(record_ids)
        n_usable = sum(1 for rid in ids if rid is not None)
        if len(ids) != n_input or n_usable != n_input:
            raise MissingRecordIdentifierError(n_input, n_usable)

    input_digest = reference_digest_rows(pts, ids)

    if n_input == 0:
        empty = np.zeros((0, n_dims), dtype=np.float64)
        return DeduplicationResult(
            points=empty,
            groups=(),
            representative_ids=(),
            multiplicities=np.zeros((0,), dtype=np.int64),
            cell_ids=np.zeros((0, n_dims), dtype=np.int64),
            policy=policy,
            input_digest=input_digest,
            output_digest=reference_digest_rows(empty, ()),
            n_input=0,
            constant_dims=tuple(False for _ in range(n_dims)),
        )

    resolution = np.asarray(policy.resolution, dtype=np.float64)
    origin = np.asarray(policy.origin, dtype=np.float64)

    constant_dims = tuple(bool(np.ptp(pts[:, d]) <= 0.0) for d in range(n_dims))

    cell_ids_float = np.floor((pts - origin) / resolution)
    overflow = np.abs(cell_ids_float) >= _INT64_CELL_LIMIT
    if overflow.any():
        dim = int(np.argmax(overflow.any(axis=0)))
        raise ResolutionCellOverflowError(
            dim, int(overflow[:, dim].sum()), float(resolution[dim])
        )
    cell_ids_all = cell_ids_float.astype(np.int64)

    # Canonical global order: (cell_id tuple, coordinate tuple, id).
    # Primarily sorting by cell id groups every cell's members
    # contiguously AND emits cells in lexicographic order in one pass.
    order = sorted(
        range(n_input),
        key=lambda i: (
            tuple(int(v) for v in cell_ids_all[i]),
            tuple(float(v) for v in pts[i]),
            _id_sort_key(ids[i]),
        ),
    )

    rep_points: list[np.ndarray] = []
    rep_cell_ids: list[tuple[int, ...]] = []
    groups: list[tuple[RecordId, ...]] = []
    representative_ids: list[RecordId] = []
    multiplicities: list[int] = []

    group_start = 0
    n_sorted = len(order)
    current_cell = tuple(int(v) for v in cell_ids_all[order[0]])
    for i in range(1, n_sorted + 1):
        next_cell = (
            tuple(int(v) for v in cell_ids_all[order[i]]) if i < n_sorted else None
        )
        if next_cell == current_cell:
            continue
        member_idx = order[group_start:i]
        # Members are already canonically sorted (coordinates, then
        # record id) within the cell, so the mean is a deterministic,
        # order-independent reduction and the earliest minimal-distance
        # member IS the documented tie-break.
        member_points = pts[member_idx]
        mean = member_points.mean(axis=0, dtype=np.float64)
        scaled = (member_points - mean) / resolution
        distances = np.linalg.norm(scaled, axis=1)
        best = int(np.argmin(distances))

        rep_points.append(member_points[best])
        rep_cell_ids.append(current_cell)
        groups.append(tuple(ids[j] for j in member_idx))
        representative_ids.append(ids[member_idx[best]])
        multiplicities.append(len(member_idx))

        group_start = i
        current_cell = next_cell if next_cell is not None else current_cell

    points_out = (
        np.stack(rep_points).astype(np.float64)
        if rep_points
        else np.zeros((0, n_dims), dtype=np.float64)
    )
    cell_ids_out = np.asarray(rep_cell_ids, dtype=np.int64)
    multiplicities_out = np.asarray(multiplicities, dtype=np.int64)
    groups_out = tuple(groups)
    representative_ids_out = tuple(representative_ids)

    output_digest = reference_digest_rows(points_out, representative_ids_out)

    return DeduplicationResult(
        points=points_out,
        groups=groups_out,
        representative_ids=representative_ids_out,
        multiplicities=multiplicities_out,
        cell_ids=cell_ids_out,
        policy=policy,
        input_digest=input_digest,
        output_digest=output_digest,
        n_input=n_input,
        constant_dims=constant_dims,
    )
