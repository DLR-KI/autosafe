# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Comparison methods and utilities for evaluation workflows."""

from typing import Any

import numpy as np
import numpy.typing as npt

from autosafe.odd.comparison import (
    ClusteredConvexHulls,
    ClusteredSuperlevelSetMonitor,
    DBSCANCluster,
    KMeansBoundaries,
    KNNMonitor,
    ODDBoundaryMethod,
    SuperlevelSetMonitor,
)
from autosafe.odd.comparison.base import DecisionBoundary
from autosafe.typing import FloatType, Matrix, NPMatrix, NPVector, Vector

# Threshold constants for hull approximation selection
_MIN_DIMS_FAST = 3
_MIN_POINTS_FAST = 200
_MIN_DIMS_HARD = 4
_LARGE_DATASET_THRESHOLD = 500

DEFAULT_FALLBACK_METHOD = "fast_hull_approx"  # For use in spec files


class FastHullApproximation(ODDBoundaryMethod):
    """Fast high-dimensional convex hull approximation.

    Uses a simple geometric approximation avoiding expensive ConvexHull
    computations. Suitable for datasets with many points or high
    dimensionality (e.g., WineQT with 1143 points in 12D).

    Particularly useful when defining specific comparison methods for
    different experiments.
    """

    def __init__(self) -> None:
        """Initialize fast hull approximation.

        No parameters needed. Uses simple geometric heuristics to
        approximate hull membership, making it fast and robust.
        """
        self._center: np.ndarray | None = None
        self._max_radius: float | None = None

    @property
    def method_type(self) -> str:
        """Method type identifier.

        Returns:
            str: Method name string.
        """
        return "fast_hull_approx"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """Return decision boundary information.

        Returns:
            DecisionBoundary: Structured decision-boundary metadata.
        """
        return DecisionBoundary(
            type="fast_hull_approx",
            parameters={},
            coverage={},
            conservatism=FloatType(0.1),  # Slightly conservative approximation
        )

    def fit(self, reference_points: Matrix | NPMatrix) -> "FastHullApproximation":
        """Fit approximation to reference points.

        Computes mean center and maximum radius of reference points
        to define the hull approximation boundary.

        Args:
            reference_points (Matrix | NPMatrix): Reference ODD points
                (shape: (n_features, n_samples)).

        Returns:
            FastHullApproximation: Self instance for method chaining.
        """
        self._center = np.mean(reference_points, axis=1)
        center_col = self._center[:, np.newaxis]
        distances = np.linalg.norm(reference_points - center_col, axis=0)
        self._max_radius = float(np.max(distances))
        return self

    def __call__(self, test_point: Vector | NPVector) -> bool:
        """Check if test point is in the approximated hull.

        Args:
            test_point (Vector | NPVector): Point to evaluate
                (shape: (n_features,)).

        Returns:
            bool: True when point is inside the hull approximation.

        Raises:
            RuntimeError: If method not yet fitted.
        """
        if self._center is None or self._max_radius is None:
            raise RuntimeError("FastHullApproximation not fitted. Call .fit() first.")

        test_point = np.asarray(test_point, dtype=float)
        distance_from_center = np.linalg.norm(test_point - self._center)

        # In high dims, add small epsilon to be inclusive
        epsilon = np.finfo(float).eps
        return bool(
            distance_from_center <= self._max_radius + epsilon
        )  # np.bool_ -> Python bool

    def evaluate_batch(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.bool_]:
        """Vectorized evaluation for multiple test points.

        Args:
            test_points (Matrix | NPMatrix): Matrix of points
                (shape: (n_features, n_samples)).

        Returns:
            npt.NDArray[np.bool_]: Boolean membership array
                (shape: (n_samples,)).

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if self._center is None or self._max_radius is None:
            raise RuntimeError("FastHullApproximation not fitted. Call .fit() first.")

        center_col = self._center[:, np.newaxis]
        distances = np.linalg.norm(test_points - center_col, axis=0)
        epsilon = np.finfo(float).eps
        return distances <= self._max_radius + epsilon


class HullAutoSelector:
    """Automatically select hull method based on data characteristics.

    Utility to determine when to use fast approximation vs. traditional
    ConvexHull based on size and dimensionality.
    """

    @staticmethod
    def is_fast_approximation_recommended(
        reference_points: Matrix | NPMatrix,
    ) -> bool:
        """Check if fast hull approximation should be used.

        Args:
            reference_points (Matrix | NPMatrix): Reference ODD points.

        Returns:
            bool: True if fast approximation is recommended.
        """
        n_points, n_dims = reference_points.shape

        # Use fast approximation for large or high-dimensional datasets
        # that would be too slow for ConvexHull computation
        if n_dims >= _MIN_DIMS_FAST:
            if n_points > _MIN_POINTS_FAST:
                return True
            if n_dims > _MIN_DIMS_HARD:
                return True
        return n_points > _LARGE_DATASET_THRESHOLD


def _resolve_comparison_methods(method_specs: list[str]) -> list[str]:
    """Resolve method specifications to actual monitor class names.

    Given method names from experiment spec (may include friendly
    name aliases), map them to actual implementation classes.

    Args:
        method_specs (list[str]): Method names from experiment spec.

    Returns:
        list[str]: Resolved method names.
    """
    return list(method_specs)


def create_comparison_monitor(method_name: str, **params: Any) -> ODDBoundaryMethod:  # noqa: ANN401
    """Create a monitor instance for a given comparison method.

    This factory function handles instantiation of different comparison
    methods based on names from experiment specifications.

    Args:
        method_name (str): Name of the comparison method.
        params (Any): Additional parameters for the method.

    Returns:
        ODDBoundaryMethod: Instantiated monitor instance.

    Raises:
        ValueError: If method name is unrecognized.
    """
    method_instances = {
        # Original methods from odd.comparison
        "hull_single": lambda: ClusteredConvexHulls(n_clusters=1),
        "hull_clustered": lambda: ClusteredConvexHulls(n_clusters=3),
        "knn": lambda: KNNMonitor(k=3, gamma=params.get("gamma")),
        "kmeans": lambda: KMeansBoundaries(n_clusters=3),
        "density_single": lambda: SuperlevelSetMonitor(gamma=params.get("gamma")),
        "density_clustered": lambda: ClusteredSuperlevelSetMonitor(
            n_clusters=3,
            gamma=params.get("gamma"),
            min_cluster_size=params.get("min_cluster_size", 3),
        ),
        "dbscan_cluster": lambda: DBSCANCluster(
            eps=params.get("eps"), min_samples=params.get("min_samples", 5)
        ),
        # New fast approximation method
        "fast_hull_approx": FastHullApproximation,
    }

    if method_name not in method_instances:
        raise ValueError(
            f"Unsupported comparison method: {method_name}. "
            f"Available methods: {list(method_instances.keys())}"
        )

    return method_instances[method_name]()


def get_available_method_names() -> list[str]:
    """Get list of all available comparison method names.

    Returns:
        list[str]: Method names available for experiment specs.
    """
    return [
        "hull_single",
        "hull_clustered",
        "knn",
        "kmeans",
        "density_single",
        "density_clustered",
        "dbscan_cluster",
        "fast_hull_approx",
    ]


def validate_method_names(method_names: list[str]) -> None:
    """Validate that method names are supported.

    Args:
        method_names (list[str]): Method names to validate.

    Raises:
        ValueError: If unsupported methods are provided.
    """
    available = get_available_method_names()
    for name in method_names:
        if name not in available:
            raise ValueError(
                f"Unsupported comparison method: {name}. Available methods: {available}"
            )


__all__ = [
    "DEFAULT_FALLBACK_METHOD",
    "FastHullApproximation",
    "HullAutoSelector",
    "_resolve_comparison_methods",
    "create_comparison_monitor",
    "get_available_method_names",
    "validate_method_names",
]
