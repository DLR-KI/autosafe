# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Base classes and interfaces for ODD comparison methods."""

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypedDict

from autosafe.typing import FloatType, Matrix, NPMatrix, NPVector, Vector


class DecisionBoundary(TypedDict):
    """Decision boundary information for visualization and analysis."""

    type: str  # "hull", "clusters", "density", "threshold", "knn"
    parameters: dict[str, Any]  # Method-specific parameters
    coverage: dict[str, FloatType]  # Area/volume metrics
    conservatism: FloatType | None  # Conservatism rating (0-1)


class ClusteringComparisonResult(TypedDict):
    """Results from clustering-based comparison methods."""

    method: str  # "kmeans", "hierarchical"
    cluster_assignments: npt.NDArray[np.int_]  # Assignment for each point
    centroids: Matrix  # Cluster centers
    silhouette_score: FloatType  # Quality metric


class DensityComparisonResult(TypedDict):
    """Results from density-based comparison methods."""

    method: str  # "kde", "gmm"
    pdf_values: npt.NDArray[np.float64]  # PDF at reference points
    threshold: FloatType  # Decision threshold gamma
    superlevel_mask: npt.NDArray[np.bool_]  # Binary membership


class KNNComparisonResult(TypedDict):
    """Results from KNN-based comparison methods."""

    method: str  # "knn"
    k: int  # Number of nearest neighbors
    gamma: FloatType  # Threshold distance
    consensus_radius: FloatType  # Average minimum distance
    conservatism: FloatType  # Conservatism metric (0 = liberal, 1 = conservative)


class ODIComparisonResult(TypedDict):
    """Overall ODD comparison result container."""

    improved_odd: Matrix  # Points in refined ODD
    original_affinity: Matrix  # Original affinity values
    comparisons: dict[str, Any]  # Results from all methods
    comparison_scores: dict[str, FloatType]  # Quality metrics


class ODDComparisonConfig(TypedDict):
    """Configuration for ODD comparison analysis."""

    methods: list[str]  # Which methods to use: ["hull", "knn", "kmeans", "density"]
    evaluate_point_grid: bool  # Generate evaluation grid
    grid_resolution: int  # Grid resolution for visualization
    conservatism_target: FloatType | None  # Target conservatism level (0-1)


class ODDBoundaryMethod(ABC):
    """Abstract base class for all ODD boundary comparison methods.

    All concrete boundary methods must implement this interface to be
    used in the comprehensive ODD evaluation framework.
    """

    @property
    @abstractmethod
    def method_type(self) -> str:
        """Return the type of this comparison method."""

    @property
    @abstractmethod
    def decision_boundary(self) -> DecisionBoundary:
        """Return the decision boundary information.

        Returns information useful for visualization, analysis, and
        comparison across different methods.

        Returns:
            DecisionBoundary: Structured information about the boundary
                shape, parameters, and metrics.
        """

    @abstractmethod
    def fit(self, reference_points: Matrix | NPMatrix) -> Self:
        """Fit the boundary method to reference ODD data.

        Args:
            reference_points (Matrix | NPMatrix): Matrix of points
                defining the ground truth ODD shape:
                (n_features, n_samples).

        Returns:
            Self: The fitted boundary method instance.
        """

    @abstractmethod
    def __call__(self, test_point: Vector | NPVector) -> bool:
        """Determine if a test point is inside the decision boundary.

        This implements the core "all(distances .< gamma)" logic for KNN
        and similar threshold-based membership testing.

        Args:
            test_point (Vector | NPVector): Point to evaluate.

        Returns:
            bool: True if the test point is considered inside the ODD
                boundary, False otherwise.
        """

    def evaluate_batch(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.bool_]:
        """Vectorized evaluation of multiple test points.

        Args:
            test_points (Matrix | NPMatrix): Matrix of points to
                evaluate. Shape (n_features, n_samples).

        Returns:
            npt.NDArray[np.bool_]: boolean array of shape (n_samples,)
                indicating membership.
        """
        return np.array([self(test_points[:, i]) for i in range(test_points.shape[1])])

    def get_conservatism_metric(self, reference_points: Matrix | NPMatrix) -> FloatType:
        """Calculate how conservative this boundary method is.

        Higher values indicate more conservative boundaries (less likely
        to include points that shouldn't be included).

        Args:
            reference_points (Matrix | NPMatrix): Original ODD points
                for comparison.

        Returns:
            FloatType: Conservatism score in range [0, 1]
        """
        # Default implementation: measure coverage relative to reference
        ref_results = self.evaluate_batch(reference_points)
        coverage = ref_results.mean() if len(ref_results) > 0 else 0.5
        return cast("FloatType", 1.0 - coverage)  # More coverage = less conservative

    def get_coverage_stats() -> dict[str, Any]:
        """Calculate area/volume coverage statistics.

        Returns:
            dict[str, Any]: Dictionary with area, volume, or size
                metrics.
        """
        # To be implemented by subclasses as needed
        return {"area": None, "volume": None, "coverage_ratio": None}


def validate_comparison_config(config: ODDComparisonConfig) -> None:
    """Validate comparison configuration parameters.

    Args:
        config (ODDComparisonConfig): Comparison configuration to
            validate.

    Raises:
        ValueError: If methods or numeric configuration values are
            invalid.
    """
    valid_methods = {"hull", "knn", "kmeans", "density", "all"}

    for method in config["methods"]:
        if method not in valid_methods:
            raise ValueError(
                f"Invalid comparison method: {method}. Valid methods: {valid_methods}"
            )

    conservatism_target = config.get("conservatism_target")
    if conservatism_target is not None and not (0 <= conservatism_target <= 1):
        raise ValueError("Conservatism target must be in range [0, 1]")

    if config.get("grid_resolution") and config["grid_resolution"] <= 0:
        raise ValueError("Grid resolution must be positive")


__all__ = [
    "ClusteringComparisonResult",
    "DecisionBoundary",
    "DensityComparisonResult",
    "KNNComparisonResult",
    "ODDBoundaryMethod",
    "ODDComparisonConfig",
    "ODIComparisonResult",
    "validate_comparison_config",
]
