# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""DBSCAN-based ODD boundary clustering."""

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
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


class DBSCANCluster(ODDBoundaryMethod):
    """Optional alternative: Density-based clustering boundary.

    More robust to outliers than k-means but requires parameter tuning.
    """

    def __init__(self, eps: float | None = None, min_samples: int = 5) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.core_points_: npt.NDArray[np.float64] | None = None
        self.trained = False

    @property
    def method_type(self) -> str:
        """Method type.

        Returns:
            str: The method name.
        """
        return "dbscan"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """Boundary information.

        Returns:
            DecisionBoundary: Structured decision-boundary metadata.
        """
        return DecisionBoundary(
            type="dbscan",
            parameters={"eps": self.eps},
            coverage={},
            conservatism=NPFloatType(0.0),
        )

    def fit(self, reference_points: Matrix | NPMatrix) -> Self:
        """Fit DBSCAN clustering.

        Args:
            reference_points (Matrix | NPMatrix): Reference points.

        Returns:
            Self: Self for method chaining.
        """
        # When eps is unset, derive it from the k-th nearest neighbor
        # distance per reference point (standard DBSCAN elbow method).
        # Using the k-th column (not the mean of all k columns) matches
        # what DBSCAN actually tests: a core point needs min_samples
        # points within eps, so calibrate eps to the k-th NN distance.
        if self.eps is None:
            n_pts = reference_points.shape[1]
            k = min(self.min_samples, n_pts - 1)
            if k >= 1:
                tree = KDTree(reference_points.T)
                dists, _ = tree.query(reference_points.T, k=k + 1)
                self.eps = float(np.median(dists[:, -1]))
            else:
                self.eps = 0.5

        # Transpose for sklearn
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(reference_points.T)
        self.core_samples = dbscan.core_sample_indices_
        self.labels_ = labels

        if len(self.core_samples) > 0:
            self.core_points_ = np.asarray(
                reference_points.T[self.core_samples], dtype=float
            )
            self._core_tree = KDTree(self.core_points_)
        else:
            self.core_points_ = None
            self._core_tree = None

        self.trained = True
        return self

    def __call__(self, test_point: Vector | NPVector) -> bool:
        """Point is in ODD if it is near any core cluster point.

        Args:
            test_point (Vector | NPVector): Point to evaluate.

        Returns:
            bool: True if point is within DBSCAN core radius.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained:
            raise RuntimeError("Not fitted")
        if self.core_points_ is None or len(self.core_points_) == 0:
            return False

        point = np.asarray(test_point, dtype=float)
        distances = np.linalg.norm(self.core_points_ - point, axis=1)
        return bool(np.min(distances) <= self.eps)

    def evaluate_batch(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.bool_]:
        """Evaluate DBSCAN-core membership for a batch of points.

        Args:
            test_points (Matrix | NPMatrix): Matrix of test points
                (shape: (n_features, n_samples)).

        Returns:
            npt.NDArray[np.bool_]: Boolean membership array.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained:
            raise RuntimeError("Not fitted")
        if self._core_tree is None or self.core_points_ is None:
            return np.zeros(test_points.shape[1], dtype=bool)
        dists, _ = self._core_tree.query(test_points.T, k=1)
        # scipy KDTree returns shape (n,) for k=1,
        # normalise before indexing.
        return np.asarray(dists).ravel() <= self.eps


__all__ = ["DBSCANCluster"]
