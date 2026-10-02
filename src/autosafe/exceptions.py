# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Custom exceptions and warnings raised across autoSAFE.

Each class builds its own message, so call sites raise them with the
offending values only. This keeps long message text out of the ``raise``
statements (ruff ``raise-vanilla-args``) and gives callers a specific
type to catch.

This module must not import from the rest of the package: everything
below depends on the standard library only, so any module can import it
without risking a cycle.
"""

from pathlib import Path


class RowShapeMismatchError(ValueError):
    """Raised when two point sets have different column counts."""

    def __init__(self, n_query_columns: int, n_reference_columns: int) -> None:
        super().__init__(
            f"column mismatch: query has {n_query_columns}, "
            f"reference has {n_reference_columns}"
        )


class OODDimensionMismatchError(ValueError):
    """Raised when OOD points do not match the anchor dimensionality."""

    def __init__(self, n_ood_columns: int, n_anchor_columns: int) -> None:
        super().__init__(
            f"ood_points has {n_ood_columns} columns but the anchors have "
            f"{n_anchor_columns}"
        )


class OODAnchorCoincidenceError(ValueError):
    """Raised when an OOD point coincides exactly with an anchor point.

    The affinity at an anchor is 1 for any covariance, so the OOD
    consistency loop has no reachable exit condition. See
    docs/ood-consistency.md.
    """

    def __init__(self, n_coincident: int, n_ood: int, first_index: int) -> None:
        super().__init__(
            f"{n_coincident} of {n_ood} OOD points coincide exactly with "
            f"anchor points (first at OOD index {first_index}); the affinity "
            "at an anchor is 1 for any covariance, so the OOD consistency "
            "loop cannot converge. Anchors must be drawn from the "
            "in-distribution set only, disjoint from the OOD set. See "
            "docs/ood-consistency.md."
        )


class KernelSaturationError(RuntimeError):
    """Raised when repeated shrinking exhausts the float64 range.

    See docs/ood-consistency.md.
    """

    def __init__(self, kernel_index: int, n_shrinks: int, ood_index: int) -> None:
        super().__init__(
            f"Kernel {kernel_index} saturated after {n_shrinks} shrinks: "
            "sigma underflowed to 0 or its inverse overflowed to inf in "
            "float64, and further iteration would compute 0*inf = NaN. The "
            f"OOD point at index {ood_index} is numerically indistinguishable "
            f"from anchor {kernel_index}. Check that the in-distribution and "
            "OOD sets are disjoint and not merely near-duplicate. See "
            "docs/ood-consistency.md."
        )


class OODConsistencyNotReachedError(RuntimeError):
    """Raised when the adjustment loop hits its iteration cap.

    See docs/ood-consistency.md.
    """

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        max_iterations: int,
        worst_alpha: float,
        xi: float,
        worst_index: int,
        n_adjusted: int,
        n_anchors: int,
    ) -> None:
        xi_n = 1.0 - (1.0 - xi) ** (1.0 / n_anchors)
        super().__init__(
            f"OOD consistency not reached after {max_iterations} iterations "
            f"(max affinity {worst_alpha:.6g} > xi={xi} at OOD index "
            f"{worst_index}; {n_adjusted} of {n_anchors} kernels adjusted). "
            "The worst-case number of adjustments grows with the anchor "
            "count, and for N="
            f"{n_anchors} with a per-kernel budget of {xi_n:.6g} the cap may "
            "simply be too low---raise max_iterations. If instead the "
            "affinity is pinned at 1.0, an OOD point is (near-)coincident "
            "with an anchor. See docs/ood-consistency.md."
        )


class EmptyAnchorPoolError(ValueError):
    """Raised when excluding the OOD rows leaves no anchor points."""

    def __init__(self, dataset_path: Path) -> None:
        super().__init__(
            f"every row of {dataset_path} is in the OOD set; no anchors "
            "remain after excluding it. The anchor set must be the "
            "complement of the OOD set, so these two inputs cannot be "
            "the same data. See docs/ood-consistency.md."
        )


class BaselinesOnlyRequiresCachedODDError(ValueError):
    """Raised when ``baselines_only`` is set without a cached ODD JSON.

    ``baselines_only`` exists to skip the expensive affinity-ODD
    build/calibration step and reuse a previously exported ODD; without
    ``odd_json`` there is nothing to reuse, so proceeding would either
    fail downstream or silently perform the full (expensive) rebuild it
    was meant to avoid.
    """

    def __init__(self, dataset_path: Path) -> None:
        super().__init__(
            f"baselines_only=True requires a cached odd_json for "
            f"{dataset_path}, since its entire purpose is to reuse an "
            "already-built affinity ODD instead of rebuilding one. Pass "
            "the odd_json path written by the original (non-baselines_only) "
            "run."
        )


class BaselinesOnlyCacheMismatchError(ValueError):
    """Raised when the cached ODD does not match the requested kernel.

    Silently refreshing kernels to match would recompute calibration
    and contradict the guarantee ``baselines_only`` exists to provide:
    that the autoSAFE affinity column stays byte-identical to the
    already-reported run.
    """

    def __init__(self, odd_json: Path) -> None:
        super().__init__(
            f"baselines_only=True but the cached ODD at {odd_json} does not "
            "match the requested closest_sample_mode/kernel_type/"
            "kernel_kwargs. Refreshing it would recompute kernel "
            "calibration, contradicting the byte-identical-autoSAFE "
            "guarantee baselines_only exists to provide. Rebuild the ODD "
            "for these kernel settings without baselines_only first, then "
            "point odd_json at the result."
        )


class DeduplicationDimensionMismatchError(ValueError):
    """Raised when a policy and a point array disagree on dimension.

    Covers both a wrong-length ``resolution``/``origin`` vector (a
    partial vector) and a point array whose column count does not match
    the policy. See ``src/autosafe/deduplication.py``.
    """

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(
            f"de-duplication policy is defined for {expected} dimension(s) "
            f"but the input has {got}; resolution, origin and the point "
            "array must all agree on dimensionality. Partial resolution/"
            "origin vectors are rejected rather than silently broadcast or "
            "truncated."
        )


class InvalidResolutionVectorError(ValueError):
    """Raised when a resolution component is not positive and finite.

    The acquisition resolution is a documented assurance input, never a
    tuned hyperparameter, so a zero, negative, ``NaN`` or infinite entry
    is rejected outright rather than silently clamped.
    """

    def __init__(self, index: int, value: float) -> None:
        super().__init__(
            f"resolution[{index}] = {value!r} is not a positive finite "
            "number; every dimension's acquisition resolution must be "
            "documented and strictly positive."
        )


class NonFiniteCoordinateError(ValueError):
    """Raised when de-duplication input has a non-finite coordinate."""

    def __init__(self, n_bad: int, n_total: int, first_index: int) -> None:
        super().__init__(
            f"{n_bad} of {n_total} input points contain a non-finite "
            f"coordinate (first at row {first_index}); de-duplication "
            "requires finite coordinates throughout."
        )


class ResolutionCellOverflowError(ValueError):
    """Raised when a resolution cell index does not fit into int64.

    Cell indices are ``floor((x - origin) / resolution)``. Beyond the
    int64 range the cast wraps, which would silently merge distinct
    points into one cell.
    """

    def __init__(self, dim: int, n_bad: int, resolution: float) -> None:
        super().__init__(
            f"{n_bad} points have a resolution-cell index beyond the int64 "
            f"range in dimension {dim} (|x - origin| / resolution exceeds "
            f"about 9.2e18 with resolution {resolution}); use a coarser "
            "resolution or an origin closer to the data."
        )


class MissingRecordIdentifierError(ValueError):
    """Raised when ``record_ids`` is incomplete or mismatched in length.

    Every input row must resolve to exactly one stable identifier:
    either every row supplies one explicitly, or none do (the row
    index is used instead). A partially supplied sequence, or one
    containing ``None``, is rejected rather than silently patched.
    """

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(
            f"record_ids must supply exactly one identifier per input row "
            f"({expected} expected, {got} usable entries found; a missing "
            "entry counts as unusable); omit record_ids entirely to fall "
            "back to the row index."
        )


class DeduplicationLabelConflictError(ValueError):
    """Raised when an ID/OOD resolution cell coincides, strict mode.

    Cell co-membership between the ID and OOD de-duplication results
    is reported for data-owner adjudication by default; this error is
    raised only when the caller opts into the strict label-conflict
    flag.
    """

    def __init__(self, n_conflicts: int, first_cell: tuple[int, ...]) -> None:
        super().__init__(
            f"{n_conflicts} resolution cell(s) contain both ID and OOD "
            f"representatives (first at cell {first_cell}); this is "
            "reported for data-owner adjudication and raised as an error "
            "only because strict label-conflict checking is enabled."
        )


class ConvexHullError(RuntimeError):
    """Raised when hull creation fails for every Qhull strategy."""

    def __init__(self) -> None:
        super().__init__(
            "Convex hull computation failed for all Qhull options. "
            "Input points are likely degenerate."
        )


class EmptyODDError(ValueError):
    """Raised when evaluating the affinity of an ODD without anchors.

    With no anchors the affinity is undefined in practice, and the ODD
    does not know its dimension, so it cannot even tell which axis of a
    query matrix indexes the points.
    """

    def __init__(self) -> None:
        super().__init__(
            "cannot evaluate an ODD that has no anchor points; add samples "
            "before evaluating it."
        )


class SigmaNotInvertibleError(ValueError):
    """Raised when a kernel's sigma cannot be repaired to be invertible.

    A sigma without a finite inverse is first repaired by flooring its
    eigenvalues at machine epsilon. That repair needs a finite
    eigendecomposition, which fails when every entry of sigma is
    subnormal, e.g. a scalar sigma below the smallest normal float64.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(
            f"the {dim}x{dim} sigma matrix is not invertible and could not "
            "be repaired: its inverse is still not finite after flooring the "
            "eigenvalues. Its entries are likely too small for float64 (e.g. "
            "a scalar sigma below the smallest normal float)."
        )


class NearAnchorOODWarning(UserWarning):
    """Warns that OOD points are numerically on top of an anchor.

    They are not exactly coincident---that raises
    :class:`OODAnchorCoincidenceError`---but they are close enough that
    convergence may need very many iterations.
    """

    def __init__(self, n_saturated: int) -> None:
        super().__init__(
            f"{n_saturated} OOD points have log-survival -inf (affinity "
            "numerically 1.0); they are not exactly coincident with an "
            "anchor but are close enough that convergence may need very "
            "many iterations. See docs/ood-consistency.md."
        )


__all__ = [
    "BaselinesOnlyCacheMismatchError",
    "BaselinesOnlyRequiresCachedODDError",
    "ConvexHullError",
    "DeduplicationDimensionMismatchError",
    "DeduplicationLabelConflictError",
    "EmptyAnchorPoolError",
    "InvalidResolutionVectorError",
    "KernelSaturationError",
    "MissingRecordIdentifierError",
    "NearAnchorOODWarning",
    "NonFiniteCoordinateError",
    "OODAnchorCoincidenceError",
    "OODConsistencyNotReachedError",
    "OODDimensionMismatchError",
    "RowShapeMismatchError",
    "SigmaNotInvertibleError",
]
