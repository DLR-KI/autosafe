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


class ConvexHullError(RuntimeError):
    """Raised when hull creation fails for every Qhull strategy."""

    def __init__(self) -> None:
        super().__init__(
            "Convex hull computation failed for all Qhull options. "
            "Input points are likely degenerate."
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
    "ConvexHullError",
    "EmptyAnchorPoolError",
    "KernelSaturationError",
    "NearAnchorOODWarning",
    "OODAnchorCoincidenceError",
    "OODConsistencyNotReachedError",
    "OODDimensionMismatchError",
    "RowShapeMismatchError",
]
