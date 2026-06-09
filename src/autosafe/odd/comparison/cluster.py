# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Clustering-based ODD boundary comparison methods.

Includes KNN-based boundaries and k-means clustering approaches for ODD
validation.
"""

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.spatial
from scipy.spatial import ConvexHull, Delaunay, KDTree
from sklearn.cluster import DBSCAN, KMeans
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
POINT_DIMENSIONS_2D = 2


class KNNMonitor(ODDBoundaryMethod):
    """KNN boundary method with threshold-based membership.

    Implements the algorithm from Chapter 12.1.1 Nearest Neighbors
    Representation:
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


class KMeansBoundaries(ODDBoundaryMethod):
    """K-means clustering-based ODD boundary method.

    Uses cluster centroids and cluster convex hulls to define ODD
    boundaries. Outliers are rejected by requiring points to be
    sufficiently close to cluster centers based on validation metrics.

    Attributes:
        n_clusters (int): Number of clusters to find.
        metric (str): Distance metric for clustering.
        min_cluster_size (int): Minimum points per cluster.
        centroids (Matrix | None): Cluster centers after fitting.
        hulls (list[Any]): Convex hulls per cluster.
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
    ) -> None:
        """Initialize k-means boundary detector.

        Args:
            n_clusters (int): Number of clusters to build.
            metric (str): Distance metric label used for metadata.
            min_cluster_size (int): Minimum cluster size to keep.
        """
        self.n_clusters = n_clusters
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.data: Matrix | NPMatrix | None = None
        self.centroids: Matrix | None = None
        self.labels_: npt.NDArray[np.int_] | None = None
        self.hulls: list[ConvexHull | None] = []
        self._cluster_balls: list[tuple[Any, float] | None] = []
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
                parameters={"n_clusters": self.n_clusters, "metric": self.metric},
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

        # Create convex hulls per cluster
        self._create_cluster_convex_hulls(data)

    def _create_cluster_convex_hulls(self, data: Matrix | NPMatrix) -> None:
        """Create convex hull for each cluster.

        Args:
            data (Matrix): Reference ODD points
                (shape: (n_features, n_samples))
        """
        self.hulls = []
        self._cluster_balls = []
        n_dims = data.shape[0]

        for cluster_id in range(self.n_clusters):
            cluster_points = data[:, self.labels_ == cluster_id]

            if cluster_points.shape[1] == 0:
                self.hulls.append(None)
                self._cluster_balls.append(None)
                continue

            # Always store a ball (center + radius) as fallback.
            pts_t = cluster_points.T
            center = pts_t.mean(axis=0)
            radius = float(np.max(np.linalg.norm(pts_t - center, axis=1)))
            self._cluster_balls.append((center, radius))

            # ConvexHull in D dimensions needs at least D+1 points.
            if cluster_points.shape[1] < n_dims + 1:
                self.hulls.append(None)
                continue

            try:
                hull = ConvexHull(cluster_points.T)
                self.hulls.append(hull)
            except (ValueError, np.linalg.LinAlgError, scipy.spatial.QhullError):
                self.hulls.append(None)

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
        """Determine if test point is in any cluster's convex hull.

        Args:
            test_point (Vector | NPVector): Point to evaluate with shape
                (n_features,).

        Returns:
            bool: True if point is inside any valid cluster's convex
                hull.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained:
            raise RuntimeError("KMeansBoundaries not fitted yet")

        point_array = np.asarray(test_point, dtype=float)

        # Check each cluster: prefer hull, fall back to bounding ball.
        for i, hull in enumerate(self.hulls):
            if hull is not None:
                try:
                    a_hull = hull.equations[:, :-1]
                    b_hull = hull.equations[:, -1]
                    eps = np.finfo(float).eps
                    if np.all(a_hull @ point_array + b_hull <= eps):
                        return True
                    continue
                except (
                    ValueError,
                    TypeError,
                    np.linalg.LinAlgError,
                    scipy.spatial.QhullError,
                ):
                    pass

            # Hull unavailable: use per-cluster bounding ball.
            if i < len(self._cluster_balls):
                ball = self._cluster_balls[i]
                if ball is not None:
                    center, radius = ball
                    if np.linalg.norm(point_array - center) <= radius:
                        return True

        return False

    @staticmethod
    def _point_in_hull(
        point: npt.NDArray[np.float64],
        hull: ConvexHull,
    ) -> bool:
        """Check if point is inside a convex hull in 2D/3D.

        Args:
            point (npt.NDArray[np.float64]): Test point.
            hull (ConvexHull): Convex hull object.

        Returns:
            bool: True when point is considered inside the hull.
        """
        try:
            # For 2D points only (for now)
            if hull.points.shape[1] == POINT_DIMENSIONS_2D:
                tri = Delaunay(hull.points)
                return bool(tri.find_simplex(point) >= 0)
            # 3D point-in-hull requires more complex geometry
            # For now, use distance to centroid as proxy
            centroid = hull.points.mean(axis=0)
            distance = np.linalg.norm(point - centroid)
            max_dist = np.max(np.linalg.norm(hull.points - centroid, axis=1))
            return distance < max_dist

        except (ValueError, TypeError, np.linalg.LinAlgError, scipy.spatial.QhullError):
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
        if not self.trained:
            raise RuntimeError("KMeansBoundaries not fitted yet")

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
                except (ValueError, np.linalg.LinAlgError, scipy.spatial.QhullError):
                    pass

            # Hull unavailable: fall back to per-cluster bounding ball.
            if i < len(self._cluster_balls):
                ball = self._cluster_balls[i]
                if ball is not None:
                    center, radius = ball
                    in_ball = np.linalg.norm(pts_t - center, axis=1) <= radius
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
        """Return method type.

        Returns:
            str: The method name.
        """
        return "clustered_hulls"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """Return boundary information.

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
        """Return method type.

        Returns:
            str: The method name.
        """
        return "dbscan"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """Return boundary information.

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


__all__ = [
    "ClusteredConvexHulls",
    "DBSCANCluster",
    "KMeansBoundaries",
    "KNNMonitor",
    "auto_detect_optimal_k",
]
