# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Deterministic, provenance-aware resolution-cell de-duplication.

Repeated visits to the same acquisition cell are coverage evidence, not
additional support for the operational design domain (ODD): letting
multiplicity leak into anchor selection lets sampling density distort
the noisy-OR affinity construction. This module groups records into
resolution cells fixed by documented acquisition metadata (never tuned
against evaluation results) and retains, per cell, the single actually
observed record nearest the cell mean.

This module is kernel-independent and pure: it does not construct
kernels, does not normalize data, and does not read or write files. See
``src/autosafe/tools/evaluate/dataset`` for the pipeline integration and
provenance-artifact serialization that build on it.

Fixed by Algorithm 1 of the paper (not a design choice made here):

-   Cell assignment is the source system's quantizer mapping when
    supplied, else componentwise ``floor((x - o) / q)`` for a positive
    resolution vector ``q`` and grid origin ``o``.
-   The representative is an actually observed cell member minimizing
    the dimensionless ``||(x - mean) / q||_2``; the synthetic cell mean
    is never emitted as an anchor.
-   Ties break lexicographically by coordinates, then by a stable
    source-record identifier.
-   Cell means are computed in canonical (lexicographically sorted)
    order, in float64.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from autosafe.exceptions import (
    DeduplicationDimensionMismatchError,
    DeduplicationLabelConflictError,
    InvalidResolutionVectorError,
    MissingRecordIdentifierError,
    NonFiniteCoordinateError,
    ResolutionCellOverflowError,
)

#: Stable source-record identifier. Ints and strings sort
#: deterministically relative to each other (see ``_id_sort_key``);
#: mixing in other hashable types is accepted but not given a
#: documented total order.
RecordId = int | str

#: Bump whenever the grouping/tie-break algorithm changes. Hashed into
#: the cache key downstream so a stale cache built with a different
#: algorithm is never silently reused (see odd_cache.py's
#: ``-dedup<digest>`` tagging).
DEDUP_ALGORITHM_VERSION = 1

#: Smallest |cell index| the int64 cast cannot represent (2**63).
_INT64_CELL_LIMIT = float(2**63)

CoordinateSpace = Literal["raw", "normalized"]
RepresentativeMetric = Literal["resolution_scaled_euclidean"]
TieBreak = Literal["lexicographic_coordinates_then_record_id"]


@dataclass(frozen=True)
class DeduplicationPolicy:
    """A fully resolved, documented de-duplication policy.

    ``resolution`` and ``origin`` are documented assurance inputs
    derived from acquisition metadata (sensor quantization, timestamp
    granularity, or another recorded acquisition resolution)---never
    hyperparameters tuned against evaluation data. When no coarser
    acquisition resolution is justified, use an exact-record-equality
    policy (a resolution far below the data's own precision; see
    :func:`exact_equality_policy`), which never merges distinct
    records.

    Attributes:
        resolution (tuple[float, ...]): Length-``n`` positive,
            finite per-dimension cell width.
        origin (tuple[float, ...]): Length-``n`` grid origin.
        coordinate_space (CoordinateSpace): Whether ``resolution``/
            ``origin`` are expressed in raw acquisition units or in
            already-normalized coordinates. Purely documentary here;
            the pipeline integration is responsible for applying the
            policy in the coordinate system it declares.
        representative_metric (RepresentativeMetric): The distance used
            to select a cell's representative. Fixed by the paper to
            the resolution-scaled Euclidean distance; the field exists
            so the choice is explicit and auditable, not implicit.
        tie_break (TieBreak): The canonical tie-break rule. Fixed by the
            paper to lexicographic-by-coordinates-then-record-id.
        n_dims (int): Number of dimensions the policy is defined over.

    Raises:
        DeduplicationDimensionMismatchError: If ``resolution`` and
            ``origin`` have different lengths.
        InvalidResolutionVectorError: If any ``resolution`` component is
            not a positive finite number.
    """

    resolution: tuple[float, ...]
    origin: tuple[float, ...]
    coordinate_space: CoordinateSpace = "raw"
    representative_metric: RepresentativeMetric = "resolution_scaled_euclidean"
    tie_break: TieBreak = "lexicographic_coordinates_then_record_id"

    def __post_init__(self) -> None:
        """Coerce and validate the resolved policy.

        Raises:
            DeduplicationDimensionMismatchError: If ``resolution`` and
                ``origin`` have different lengths.
            InvalidResolutionVectorError: If any ``resolution``
                component is not a positive finite number.
        """
        resolution = tuple(float(v) for v in self.resolution)
        origin = tuple(float(v) for v in self.origin)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "origin", origin)
        if len(resolution) != len(origin):
            raise DeduplicationDimensionMismatchError(len(resolution), len(origin))
        for index, value in enumerate(resolution):
            if not np.isfinite(value) or value <= 0.0:
                raise InvalidResolutionVectorError(index, value)

    @property
    def n_dims(self) -> int:
        """Number of dimensions the policy is defined over.

        Returns:
            int: Length of ``resolution`` (equivalently ``origin``).
        """
        return len(self.resolution)

    def digest(self) -> str:
        """Stable digest of the fully resolved policy.

        Returns:
            str: 64-character hex SHA-256 digest of the policy fields
                and the de-duplication algorithm version.
        """
        payload = json.dumps(
            {
                "algorithm_version": DEDUP_ALGORITHM_VERSION,
                "resolution": self.resolution,
                "origin": self.origin,
                "coordinate_space": self.coordinate_space,
                "representative_metric": self.representative_metric,
                "tie_break": self.tie_break,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_equality_policy(
    n_dims: int,
    *,
    coordinate_space: CoordinateSpace = "raw",
    epsilon: float = 1e-12,
) -> DeduplicationPolicy:
    """Build the no-op, exact-record-equality policy.

    Used when no coarser acquisition resolution is justified: only
    records that are exactly equal (up to ``epsilon``, far below any
    real sensor's precision) are merged, so no merge width is ever
    inferred or tuned from the evaluation data.

    Args:
        n_dims (int): Number of coordinate dimensions.
        coordinate_space (CoordinateSpace): Coordinate system the policy
            is declared in.
        epsilon (float): Cell width, far below sensor precision, used
            only to make exact float equality robust to representation
            noise (e.g. CSV round-tripping).

    Returns:
        DeduplicationPolicy: Policy that merges only (near-)exact
            duplicate records.
    """
    return DeduplicationPolicy(
        resolution=tuple(epsilon for _ in range(n_dims)),
        origin=tuple(0.0 for _ in range(n_dims)),
        coordinate_space=coordinate_space,
    )


@dataclass(frozen=True, eq=False)
class DeduplicationResult:
    """Result of applying a :class:`DeduplicationPolicy` to a point set.

    Attributes:
        points (npt.NDArray[np.float64]): Shape ``(n_cells, n_dims)``
            observed representatives, one per occupied cell, in
            canonical (lexicographic cell-id) order. Never a synthetic
            cell mean.
        groups (tuple[tuple[RecordId, ...], ...]): Per representative,
            the canonically ordered tuple of source-record identifiers
            that fell into its cell (length ``n_cells``).
        representative_ids (tuple[RecordId, ...]): Per cell, the
            source-record identifier of the chosen representative
            (the member of ``groups[i]`` matching ``points[i]``).
        multiplicities (npt.NDArray[np.int64]): Shape ``(n_cells,)``
            source-row count per representative.
        cell_ids (npt.NDArray[np.int64]): Shape ``(n_cells, n_dims)``
            integer cell coordinate per representative.
        policy (DeduplicationPolicy): The fully resolved policy applied.
        input_digest (str): Order-independent digest of the input
            (points, record ids); identical across row permutations.
        output_digest (str): Order-independent digest of the resulting
            representatives.
        n_input (int): Number of input rows.
        constant_dims (tuple[bool, ...]): Length ``n_dims``; ``True``
            for a dimension where every input row shared the exact
            same coordinate (recorded explicitly rather than silently
            collapsed).
        n_output (int): Number of retained representatives.
        n_duplicates (int): Number of input rows collapsed away.
    """

    points: npt.NDArray[np.float64]
    groups: tuple[tuple[RecordId, ...], ...]
    representative_ids: tuple[RecordId, ...]
    multiplicities: npt.NDArray[np.int64]
    cell_ids: npt.NDArray[np.int64]
    policy: DeduplicationPolicy
    input_digest: str
    output_digest: str
    n_input: int
    constant_dims: tuple[bool, ...]

    @property
    def n_output(self) -> int:
        """Number of retained representatives (occupied cells).

        Returns:
            int: ``points.shape[0]``.
        """
        return int(self.points.shape[0])

    @property
    def n_duplicates(self) -> int:
        """Number of input rows collapsed away.

        Returns:
            int: ``n_input - n_output``.
        """
        return self.n_input - self.n_output

    def cell_count_distribution(self) -> dict[int, int]:
        """Histogram of cell multiplicity.

        Returns:
            dict[int, int]: Mapping from group size (multiplicity) to
                the number of cells with that size.
        """
        counts: dict[int, int] = {}
        for multiplicity in self.multiplicities.tolist():
            key = int(multiplicity)
            counts[key] = counts.get(key, 0) + 1
        return counts


def _id_sort_key(record_id: RecordId) -> tuple[int, RecordId]:
    """Total-order sort key for a possibly-heterogeneous record id.

    Args:
        record_id (RecordId): Source-record identifier.

    Returns:
        tuple[int, RecordId]: A key comparable across ints and strings
            (all ints sort before all strings; within a type, natural
            ordering applies).
    """
    if isinstance(record_id, str):
        return (1, record_id)
    return (0, record_id)


def _digest_rows(points: npt.NDArray[np.float64], ids: Sequence[RecordId]) -> str:
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


def deduplicate_points(  # ruff:ignore[too-many-locals, too-many-statements]
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
        ValueError: If ``points`` is not 2-dimensional.
        DeduplicationDimensionMismatchError: If ``points``' column count
            does not match ``policy.n_dims``.
        NonFiniteCoordinateError: If any coordinate is non-finite.
        ResolutionCellOverflowError: If a cell index does not fit into
            int64, i.e. the resolution is too fine for the coordinates.
        MissingRecordIdentifierError: If ``record_ids`` is supplied with
            the wrong length or a ``None`` entry.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:  # ruff:ignore[magic-value-comparison]
        raise ValueError(
            f"points must be a 2D (n_points, n_dims) array, got ndim={pts.ndim}"
        )
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

    input_digest = _digest_rows(pts, ids)

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
            output_digest=_digest_rows(empty, ()),
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

    output_digest = _digest_rows(points_out, representative_ids_out)

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


def check_label_conflicts(
    id_result: DeduplicationResult,
    ood_result: DeduplicationResult,
    *,
    strict: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Report resolution cells shared between ID and OOD anchors.

    Exact coordinate coincidence between an ID and an OOD point is a
    separate, always-hard error (see
    :class:`~autosafe.exceptions.OODAnchorCoincidenceError`, checked by
    the OOD consistency loop). Cell co-membership without coordinate
    coincidence is weaker evidence of a labeling problem, so it is
    reported for data-owner adjudication by default and only raised as
    an error when ``strict`` is set.

    Args:
        id_result (DeduplicationResult): De-duplication result for the
            in-distribution anchor candidates.
        ood_result (DeduplicationResult): De-duplication result for the
            OOD constraint set, built under the same policy.
        strict (bool): Raise instead of only reporting when ``True``.

    Returns:
        tuple[tuple[int, ...], ...]: Cell-id tuples occupied by both
            results, in ascending order.

    Raises:
        DeduplicationLabelConflictError: If ``strict`` is ``True`` and
            at least one conflicting cell is found.
    """
    id_cells = {tuple(row) for row in id_result.cell_ids.tolist()}
    ood_cells = {tuple(row) for row in ood_result.cell_ids.tolist()}
    conflicts = tuple(sorted(id_cells & ood_cells))
    if strict and conflicts:
        raise DeduplicationLabelConflictError(len(conflicts), conflicts[0])
    return conflicts


__all__ = [
    "DEDUP_ALGORITHM_VERSION",
    "DeduplicationPolicy",
    "DeduplicationResult",
    "RecordId",
    "check_label_conflicts",
    "deduplicate_points",
    "exact_equality_policy",
]
