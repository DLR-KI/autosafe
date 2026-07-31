# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""K-nearest-neighbor ODD boundary monitor."""

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from typing_extensions import Self

from autosafe.odd.comparison.base import (
    DecisionBoundary,
    ODDBoundaryMethod,
)
from autosafe.typing import FloatType, Matrix, NPFloatType, NPMatrix, NPVector, Vector

MIN_CLUSTER_POINTS = 3
VISUALIZATION_2D_DIMENSIONS = 2
MIN_CONSENSUS_EPSILON = 1e-10
MIN_HULL_POINTS = 3


class KNNMonitor(ODDBoundaryMethod):
    """KNN boundary method with threshold-based membership.

    Implements the Nearest Neighbors Representation:
    ODD = {x | all(k-nearest-neighbors(x, k) < gamma)}

    Attributes:
        k (int): Number of nearest neighbors to consider.
        gamma (FloatType | None): Threshold distance for membership
            decision.
        metric (str): Distance metric.
        leaf_size (int): KDTree optimization parameter.
        data (Matrix): Reference ODD points.
        tree (KDTree | None): KDTree used for nearest-neighbor queries.
        trained (bool): Whether the method has been fitted.
        consensus_radius (FloatType | None): Average minimum distance.
        method_type (str): Method identifier property.
        decision_boundary (DecisionBoundary): Decision-boundary metadata
            property.
    """

    def __init__(
        self,
        k: int = 3,
        gamma: FloatType | None = None,
        metric: str = "euclidean",
        leaf_size: int = 40,
    ) -> None:
        """Initialize KNNMonitor with default parameters.

        Args:
            k (int): Number of nearest neighbors
            gamma (FloatType | None):
                Distance threshold for membership (if None,
                auto-detected).
            metric (str): Distance metric for KDTree
            leaf_size (int): KDTree optimization parameter

        Note:
            If gamma is None, it will be automatically set based on data
            distribution via auto_detect_consensus_radius() method.
        """
        self.k = k  # k-nearest neighbors
        self.gamma = gamma  # distance threshold gamma
        self.metric = metric  # distance metric
        self.leaf_size = leaf_size  # KDTree leaf size
        self.data: Matrix | NPMatrix  # reference data (will be set in fit)
        self.tree: KDTree | None = None
        self.trained = False  # fitted flag
        self.consensus_radius: FloatType | None = None  # calculated in fit

    @property
    def method_type(self) -> str:
        """method_type property for the KNN comparison method.

        Returns:
            str: The method name.
        """
        return "knn"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """decision_boundary property for the KNN monitor.

        Returns:
            DecisionBoundary: Structured decision-boundary metadata.
        """
        if not self.trained:
            return DecisionBoundary(
                type="knn",
                parameters={"k": self.k, "gamma": self.gamma},
                coverage={},
                conservatism=None,
            )

        return DecisionBoundary(
            type="knn",
            parameters={
                "k": self.k,
                "gamma": float(self.gamma) if self.gamma is not None else None,
                "metric": self.metric,
                "leaf_size": self.leaf_size,
            },
            coverage=self._estimate_coverage(),
            conservatism=self.compute_conservatism_metric(),
        )

    def _estimate_coverage(self) -> dict[str, FloatType]:
        """Estimate coverage metrics for this KNN boundary.

        Returns:
            dict[str, FloatType]: Coverage metrics for the fitted
                boundary.
        """
        if not self.trained or self.data is None:
            return {}

        # Calculate coverage on the training data itself
        train_results = np.array([
            self(self.data[:, i]) for i in range(self.data.shape[1])
        ])
        coverage_ratio = train_results.mean() if len(train_results) > 0 else 0.5

        return {
            "coverage_ratio": NPFloatType(coverage_ratio),
            "n_reference_points": NPFloatType(len(train_results)),
            "consensus_radius": NPFloatType(self.consensus_radius)
            if self.consensus_radius
            else NPFloatType(0.0),
        }

    def auto_detect_consensus_radius(
        self, data: Matrix | NPMatrix | None = None
    ) -> FloatType:
        """Calculate appropriate gamma based on inter-point distances.

        This implements the "smaller gamma = more conservative"
        principle.
        Returns 80% of the average minimum distance to k-neighbors.

        Args:
            data (Matrix | NPMatrix | None): Optional reference data. If
                None, uses self.data.

        Returns:
            FloatType: recommended gamma threshold value

        Raises:
            ValueError: If no reference data is available.
        """
        data_to_use = data if data is not None else self.data
        if data_to_use is None:
            raise ValueError(
                "Data must be provided either during initialization or "
                "in this method call"
            )

        # Build KDTree on the data (transposed to n_samples x
        # n_features format).
        tree = KDTree(data_to_use.T, leafsize=self.leaf_size)

        # Query k+1 neighbors so we can drop the self-match (distance 0)
        # that KDTree returns when query points are in the tree.
        distances, _ = tree.query(data_to_use.T, k=self.k + 1)
        distances = distances[:, 1:]  # drop column 0 (self, distance = 0)

        # all(k_distances < gamma) tests the k-th (largest) NN distance.
        # Calibrate gamma against that same quantity so reference points
        # pass at the 80 % conservatism level.
        kth_distances = distances[:, -1]

        # 80 % of mean k-th NN distance across reference points
        return NPFloatType(0.8 * kth_distances.mean())

    def fit(self, reference_points: Matrix | NPMatrix) -> Self:
        """Fit KNNMonitor to reference ODD data using KDTree.

        Args:
            reference_points (Matrix | NPMatrix):
                Reference ODD points (shape: (n_features, n_samples))

        Returns:
            Self: self for method chaining
        """
        self.data = reference_points

        # Build tree once; derive consensus_radius and optionally gamma.
        self.tree = KDTree(
            reference_points.T,
            leafsize=self.leaf_size,
            compact_nodes=True,
            balanced_tree=True,
        )
        self.consensus_radius = self.auto_detect_consensus_radius(reference_points)
        if self.gamma is None:
            self.gamma = self.consensus_radius

        self.trained = True
        return self

    def compute_conservatism_metric(self) -> FloatType:
        """Calculate conservatism rating for this KNN boundary.

        Smaller gamma results in a more conservative representation in
        the sense of being less likely to include situations that we
        should not.

        Returns:
            FloatType: Conservatism score in range [0, 1] where 1 is
            most conservative.
        """
        if not self.trained or self.gamma is None or self.consensus_radius is None:
            return NPFloatType(0.5)

        # Conservatism increases as (consensus_radius - gamma)
        # increases. More aggressive (less conservative) as gamma
        # approaches consensus_radius.
        if self.consensus_radius <= MIN_CONSENSUS_EPSILON:
            return NPFloatType(1.0)

        delta = self.consensus_radius - self.gamma
        if delta <= 0:
            # gamma >= consensus_radius = minimal conservatism
            return NPFloatType(0.0)

        # Normalize to [0, 1] range
        # At consensus_radius: delta = 0 -> 0% conservatism
        # At gamma = 0: delta = consensus_radius -> 100% conservatism
        conservatism = float(delta / self.consensus_radius)
        return NPFloatType(min(max(conservatism, 0.0), 1.0))  # Clamp to [0, 1]

    def __call__(self, test_point: Vector | NPVector) -> bool:
        """Determine if test point belongs to the ODD.

        Implements: all(distances < gamma)

        Args:
            test_point (Vector | NPVector): Point to evaluate with shape
                (n_features,).

        Returns:
            bool: True if ALL k-nearest neighbors are within threshold
                gamma

        Raises:
            RuntimeError: If method not yet fitted
        """
        if not self.trained:
            raise RuntimeError("KNNMonitor not fitted yet. Call .fit() first.")

        if self.tree is None:
            raise RuntimeError(
                "KDTree not initialized. Check that fit() completed successfully."
            )

        test_point_reshaped = (
            test_point.reshape(1, -1) if test_point.ndim == 1 else test_point
        )

        try:
            distances, _ = self.tree.query(test_point_reshaped, k=self.k)
        except (TypeError, ValueError, RuntimeError) as e:
            raise RuntimeError(f"KDTree query failed: {e}") from e

        return (distances < self.gamma).all()

    def evaluate_batch(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.bool_]:
        """Vectorized evaluation for multiple test points.

        Args:
            test_points (Matrix | NPMatrix): Matrix of points with shape
                (n_features, n_samples).

        Returns:
            npt.NDArray[np.bool_]: boolean array indicating membership
                (shape: (n_samples,)).

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained:
            raise RuntimeError("Method not fitted")

        if self.tree is None:
            raise RuntimeError(
                "KDTree not initialized. Check that fit() completed successfully."
            )

        # Single batch KDTree query: (n_samples, n_features) input.
        distances, _ = self.tree.query(test_points.T, k=self.k)
        # distances: (n_samples, k): in ODD if all k distances < gamma
        return np.all(distances < self.gamma, axis=1)


__all__ = ["KNNMonitor"]
