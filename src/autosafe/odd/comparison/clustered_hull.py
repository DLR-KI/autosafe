# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Per-cluster convex hull ODD boundaries."""

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.spatial
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from typing_extensions import Self

from autosafe.odd.comparison.base import (
    DecisionBoundary,
    ODDBoundaryMethod,
)
from autosafe.typing import Matrix, NPFloatType, NPMatrix, NPVector, Vector

MIN_CLUSTER_POINTS = 3
VISUALIZATION_2D_DIMENSIONS = 2
MIN_CONSENSUS_EPSILON = 1e-10
MIN_HULL_POINTS = 3


class ClusteredConvexHulls(ODDBoundaryMethod):
    """Hierarchical approach using convex hulls of data sub-clusters.

    This is similar to KMeansBoundaries but uses different clustering
    methods and allows for nested hierarchical hulls.
    """

    def __init__(self, n_clusters: int = 3, method: str = "kmeans") -> None:
        self.n_clusters = n_clusters
        self.method = method
        self.labels_: npt.NDArray[np.int_] | None = None
        self.hulls: list[ConvexHull | None] = []
        self._cluster_balls: list[tuple[Any, float] | None] = []
        self.trained = False

    @property
    def method_type(self) -> str:
        """Method type.

        Returns:
            str: The method name.
        """
        return "clustered_hulls"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """Boundary information.

        Returns:
            DecisionBoundary: Structured decision-boundary metadata.
        """
        return DecisionBoundary(
            type="clustered_hulls",
            parameters={
                "n_clusters": self.n_clusters,
                "method": self.method,
            },
            coverage={},
            conservatism=NPFloatType(0.0),
        )

    def fit(self, reference_points: Matrix | NPMatrix) -> Self:
        """Fit hierarchical hulls.

        Args:
            reference_points (Matrix | NPMatrix): Reference points.

        Returns:
            Self: Self for method chaining.

        Raises:
            ValueError: If unsupported clustering method is requested.
        """
        if self.method != "kmeans":
            raise ValueError("Only 'kmeans' clustering is currently supported")

        data_t = reference_points.T
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels_ = kmeans.fit_predict(data_t)

        n_dims = reference_points.shape[0]
        self.hulls = []
        self._cluster_balls = []
        for cluster_id in range(self.n_clusters):
            cluster_points = data_t[self.labels_ == cluster_id]

            if cluster_points.shape[0] == 0:
                self.hulls.append(None)
                self._cluster_balls.append(None)
                continue

            # Always store a ball fallback.
            center = cluster_points.mean(axis=0)
            radius = float(np.max(np.linalg.norm(cluster_points - center, axis=1)))
            self._cluster_balls.append((center, radius))

            if cluster_points.shape[0] < n_dims + 1:
                self.hulls.append(None)
                continue
            try:
                self.hulls.append(ConvexHull(cluster_points))
            except (ValueError, np.linalg.LinAlgError, scipy.spatial.QhullError):
                self.hulls.append(None)

        self.trained = True
        return self

    def __call__(self, test_point: Vector | NPVector) -> bool:
        """Check if point is in any sub-cluster hull.

        Args:
            test_point (Vector | NPVector): Point to evaluate.

        Returns:
            bool: True if point is inside at least one cluster hull.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained:
            raise RuntimeError("Not fitted")

        point_array = np.asarray(test_point, dtype=float)
        for i, hull in enumerate(self.hulls):
            if hull is not None:
                try:
                    a_hull = hull.equations[:, :-1]
                    b_hull = hull.equations[:, -1]
                    eps = np.finfo(float).eps
                    if np.all(a_hull @ point_array + b_hull <= eps):
                        return True
                    continue
                except (ValueError, np.linalg.LinAlgError, scipy.spatial.QhullError):
                    pass

            # Hull unavailable: fall back to per-cluster bounding ball.
            if i < len(self._cluster_balls):
                ball = self._cluster_balls[i]
                if ball is not None:
                    center, radius = ball
                    if np.linalg.norm(point_array - center) <= radius:
                        return True

        return False

    def evaluate_batch(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.bool_]:
        """Evaluate clustered-hull membership for a batch of points.

        Args:
            test_points (Matrix | NPMatrix): Matrix of test points
                (shape: (n_features, n_samples)).

        Returns:
            npt.NDArray[np.bool_]: Boolean membership array
                (shape: (n_samples,)).

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained:
            raise RuntimeError("Not fitted")

        n_samples = test_points.shape[1]
        result = np.zeros(n_samples, dtype=bool)
        eps = np.finfo(float).eps
        pts_t = test_points.T  # (n_samples, n_features)

        for i, hull in enumerate(self.hulls):
            if hull is not None:
                try:
                    a_hull = hull.equations[:, :-1]
                    b_hull = hull.equations[:, -1]
                    in_hull = np.all(pts_t @ a_hull.T + b_hull <= eps, axis=1)
                    result |= in_hull
                    continue
                except (
                    ValueError,
                    np.linalg.LinAlgError,
                    scipy.spatial.QhullError,
                ):
                    pass
            if i < len(self._cluster_balls):
                ball = self._cluster_balls[i]
                if ball is not None:
                    center, radius = ball
                    in_ball = np.linalg.norm(pts_t - center, axis=1) <= radius
                    result |= in_ball

        return result


__all__ = ["ClusteredConvexHulls"]
