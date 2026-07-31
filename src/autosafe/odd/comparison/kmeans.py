# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""K-means cluster ODD boundary method."""

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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


class KMeansBoundaries(ODDBoundaryMethod):
    """K-means clustering-based ODD boundary method.

    Uses cluster centroids and a per-cluster centroid-distance radius
    to define ODD boundaries. A point is in-ODD iff it falls within at
    least one cluster's radius, where the radius is the
    ``radius_quantile`` quantile of that cluster's member distances to
    its centroid (rejecting the farthest members as outliers).

    Attributes:
        n_clusters (int): Number of clusters to find.
        metric (str): Distance metric for clustering.
        min_cluster_size (int): Minimum points per cluster.
        radius_quantile (float): Quantile of the per-cluster
            centroid-distance distribution used as the inclusion
            radius.
        centroids (Matrix | None): Cluster centers after fitting.
        radii (list[float | None]): Per-cluster inclusion radius;
            ``None`` for empty clusters.
        silhouette (FloatType | None): Silhouette score.
        conservatism (FloatType | None): Conservatism score.
        method_type (str): Method identifier property.
        decision_boundary (DecisionBoundary): Decision-boundary metadata
            property.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        metric: str = "euclidean",
        min_cluster_size: int = 3,
        radius_quantile: float = 0.95,
    ) -> None:
        """Initialize k-means boundary detector.

        Args:
            n_clusters (int): Number of clusters to build.
            metric (str): Distance metric label used for metadata.
            min_cluster_size (int): Minimum cluster size to keep.
            radius_quantile (float): Quantile of the per-cluster
                centroid-distance distribution used as the inclusion
                radius (default 0.95).
        """
        self.n_clusters = n_clusters
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.radius_quantile = radius_quantile
        self.data: Matrix | NPMatrix | None = None
        self.centroids: Matrix | None = None
        self.labels_: npt.NDArray[np.int_] | None = None
        self.radii: list[float | None] = []
        self.silhouette: FloatType | None = None
        self.conservatism: FloatType | None = None
        self.trained = False

    @property
    def method_type(self) -> str:
        """method_type property for the k-means comparison method.

        Returns:
            Method name.
        """
        return "kmeans"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """decision_boundary property for the k-means monitor.

        Returns:
            DecisionBoundary: Structured decision-boundary metadata.
        """
        if not self.trained:
            return DecisionBoundary(
                type="kmeans",
                parameters={
                    "n_clusters": self.n_clusters,
                    "metric": self.metric,
                    "radius_quantile": self.radius_quantile,
                },
                coverage={},
                conservatism=None,
            )

        return DecisionBoundary(
            type="kmeans",
            parameters={
                "n_clusters": self.n_clusters,
                "metric": self.metric,
                "silhouette": float(self.silhouette) if self.silhouette else None,
                "cluster_sizes": self.cluster_sizes,
                "min_cluster_size": self.min_cluster_size,
                "radius_quantile": self.radius_quantile,
                "cluster_radii": self.radii,
            },
            coverage=self._estimate_coverage(),
            conservatism=self.conservatism,
        )

    def _estimate_coverage(self) -> dict[str, FloatType]:
        """Estimate coverage metrics for this k-means boundary.

        Returns:
            dict[str, FloatType]: Coverage metrics for the fitted
                boundary.
        """
        if not self.trained or self.data is None:
            return {}

        train_results = np.array([
            self(self.data[:, i]) for i in range(self.data.shape[1])
        ])
        coverage_ratio = train_results.mean() if len(train_results) > 0 else 0.5

        return {
            "coverage_ratio": NPFloatType(coverage_ratio),
            "n_reference_points": NPFloatType(len(train_results)),
        }

    def _fit_clustering(self, data: Matrix | NPMatrix) -> None:
        """Fit k-means clustering.

        Args:
            data (Matrix): Reference ODD points
                (shape: (n_features, n_samples))
        """
        # Initial fit
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels_ = kmeans.fit_predict(data.T)
        self.centroids = (
            kmeans.cluster_centers_.T
        )  # Transpose back to (n_features, n_clusters)

        # Calculate cluster metrics
        unique, counts = np.unique(self.labels_, return_counts=True)
        self.cluster_sizes = dict(zip(unique, counts, strict=False))

        # Validate cluster sizes
        min_size = self.min_cluster_size
        valid_clusters = [i for i, count in enumerate(counts) if count >= min_size]

        if len(valid_clusters) < self.n_clusters / 2:
            warnings.warn(
                f"Only {len(valid_clusters)}/{self.n_clusters} clusters "
                f"have >= {min_size} points. Consider reducing min_cluster_size.",
                stacklevel=2,
            )

        # Compute the centroid-distance outlier-rejection radius per
        # cluster.
        self._compute_cluster_radii(data, self.centroids)

    def _compute_cluster_radii(
        self, data: Matrix | NPMatrix, centroids: Matrix
    ) -> None:
        """Compute the centroid-distance inclusion radius per cluster.

        The radius is the ``radius_quantile`` quantile of each
        cluster's member distances to its centroid, so the farthest
        ``1 - radius_quantile`` fraction of members are rejected as
        outliers.

        Args:
            data (Matrix): Reference ODD points
                (shape: (n_features, n_samples))
            centroids (Matrix): Cluster centers (shape:
                (n_features, n_clusters)).
        """
        self.radii = []

        for cluster_id in range(self.n_clusters):
            cluster_points = data[:, self.labels_ == cluster_id]

            if cluster_points.shape[1] == 0:
                self.radii.append(None)
                continue

            centroid = centroids[:, cluster_id]
            distances = np.linalg.norm(cluster_points.T - centroid, axis=1)
            self.radii.append(float(np.quantile(distances, self.radius_quantile)))

    def _calculate_silhouette_score(self, data: Matrix | NPMatrix) -> FloatType:
        """Calculate silhouette score for cluster quality.

        Args:
            data (Matrix | NPMatrix): Reference ODD points.

        Returns:
            FloatType: Silhouette score or 0.0 if the score cannot be
                computed.
        """
        try:
            score = silhouette_score(data.T, self.labels_)
            return NPFloatType(score)
        except (ValueError, TypeError):
            return NPFloatType(0.0)

    def fit(self, reference_points: Matrix | NPMatrix) -> Self:
        """Fit k-means clustering to reference data.

        Args:
            reference_points (Matrix | NPMatrix):
                Reference ODD points (shape: (n_features, n_samples))

        Returns:
            Self: Self for method chaining
        """
        self.data = reference_points

        # Cluster the data
        self._fit_clustering(reference_points)

        # Calculate metrics
        self.silhouette = self._calculate_silhouette_score(reference_points)
        self.conservatism = self._calculate_conservatism()

        self.trained = True
        return self

    def _calculate_conservatism(self) -> FloatType:
        """Calculate conservatism based on cluster sizes and quality.

        Smaller, tighter clusters = more conservative. Larger, sparse
        clusters = more liberal.

        Returns:
            FloatType: Conservatism score in the range [0, 1].
        """
        if self.silhouette is None:
            return NPFloatType(0.5)

        # Higher silhouette = better separation = generally more
        # conservative.
        silhouette_factor = self.silhouette / 0.5  # Normalize around 0.5

        # More clusters relative to data size = more conservative
        if hasattr(self, "cluster_sizes") and len(self.cluster_sizes) > 0:
            avg_cluster_size = sum(self.cluster_sizes.values()) / len(
                self.cluster_sizes
            )
            n_samples = sum(self.cluster_sizes.values())
            size_factor = 0.5 * (1 - (avg_cluster_size / (n_samples / 10)))
        else:
            size_factor = 0.5

        # Combine factors
        conservatism = NPFloatType((silhouette_factor * 0.5 + size_factor) / 2)
        return NPFloatType(max(min(conservatism, 1.0), 0.1))

    def __call__(self, test_point: Vector | NPVector) -> bool:
        """Determine if test point is within any cluster's radius.

        Args:
            test_point (Vector | NPVector): Point to evaluate with shape
                (n_features,).

        Returns:
            bool: True iff ``min_i(||x - c_i|| - r_i) <= 0`` over
                clusters with a radius.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained or self.centroids is None:
            raise RuntimeError("KMeansBoundaries not fitted yet")

        point_array = np.asarray(test_point, dtype=float)

        for cluster_id, radius in enumerate(self.radii):
            if radius is None:
                continue
            centroid = self.centroids[:, cluster_id]
            if np.linalg.norm(point_array - centroid) - radius <= 0:
                return True

        return False

    def evaluate_batch(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.bool_]:
        """Vectorized evaluation for multiple test points.

        Args:
            test_points (Matrix | NPMatrix): Points with shape
                (n_features, n_samples).

        Returns:
            npt.NDArray[np.bool_]: Boolean membership array
                (shape: (n_samples,)).

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained or self.centroids is None:
            raise RuntimeError("KMeansBoundaries not fitted yet")

        n_samples = test_points.shape[1]
        result = np.zeros(n_samples, dtype=bool)
        pts_t = test_points.T  # (n_samples, n_features)

        for cluster_id, radius in enumerate(self.radii):
            if radius is None:
                continue
            centroid = self.centroids[:, cluster_id]
            in_ball = np.linalg.norm(pts_t - centroid, axis=1) - radius <= 0
            result |= in_ball

        return result

    def get_cluster_info(self) -> dict[str, Any]:
        """Return detailed cluster information.

        Returns:
            dict[str, Any]: Cluster summary dictionary.
        """
        if not hasattr(self, "centroids"):
            return {}

        return {
            "centroids": self.centroids,
            "cluster_sizes": self.cluster_sizes,
            "centroid_distances": self._calculate_centroid_distances(),
            "silhouette": self.silhouette or None,
        }

    def _calculate_centroid_distances(self) -> npt.NDArray[np.float64]:
        """Calculate pairwise distances between centroids.

        Returns:
            npt.NDArray[np.float64]: Symmetric centroid-distance matrix.
        """
        if self.centroids is None:
            return np.zeros((0, 0), dtype=float)

        n_clusters = self.centroids.shape[1]
        distances = np.zeros((n_clusters, n_clusters))

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(self.centroids[:, i] - self.centroids[:, j])
                distances[i, j] = distances[j, i] = dist

        return distances


def auto_detect_optimal_k(data: Matrix | NPMatrix, max_k_upper: int = 10) -> int:
    """Suggest optimal k value based on data characteristics.

    Args:
        data (Matrix | NPMatrix): Reference ODD points.
        max_k_upper (int): Maximum k to consider.

    Returns:
        int: recommended k value
    """
    n_samples = (
        data.shape[1] if hasattr(data, "shape") and len(data.shape) > 1 else len(data)
    )

    # Rule of thumb: k = sqrt(n_samples) for moderate size datasets
    k_sqrt = int(np.sqrt(n_samples))

    # Cap at reasonable upper limit
    k_suggestion = min(k_sqrt, max_k_upper, 20)

    return max(k_suggestion, 1)  # Ensure at least 1


__all__ = ["KMeansBoundaries", "auto_detect_optimal_k"]
