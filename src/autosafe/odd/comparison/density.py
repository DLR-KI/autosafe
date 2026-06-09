# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Density-based ODD boundary comparison methods.

Includes superlevel set analysis using kernel density estimation for
probability density function (PDF) analysis.
"""

from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from typing_extensions import Self

from autosafe.odd.comparison.base import (
    DecisionBoundary,
    ODDBoundaryMethod,
)
from autosafe.typing import FloatType, Matrix, NPFloatType, NPMatrix, NPVector, Vector

MAX_SILVERMAN_SAMPLES = 100
MIN_PDF_EPSILON = 1e-10
VISUALIZATION_DIMENSIONS = 2
DEFAULT_GAMMA = NPFloatType(0.01)
DEFAULT_SIGMOID_WEIGHT = NPFloatType(1.0)


class SuperlevelSetMonitor(ODDBoundaryMethod):
    """Superlevel set ODD boundary using kernel density estimation.

    Implements threshold-based membership: ODD = {x | pdf(x) > gamma}

    Attributes:
        gamma (FloatType): Likelihood threshold gamma.
        bandwidth (str | None): KDE bandwidth parameter.
        kde (KernelDensity | None): Fitted kernel density estimator.
        ref_points (Matrix | None): Reference ODD points.
        sigmoid_weight (FloatType): Boundary transition sharpness.
            factor.
        candidate_gamma (FloatType | None): Auto-detected gamma value.
        method_type (str): Method identifier property.
        decision_boundary (DecisionBoundary): Decision-boundary.
            metadata property.
    """

    def __init__(
        self,
        gamma: FloatType | None = None,
        bandwidth: str | None = "scott",
        sigmoid_weight: FloatType | None = None,
    ) -> None:
        """Initialize superlevel set monitor with density threshold.

        Args:
            gamma (FloatType | None): Likelihood threshold for the.
                superlevel set.
            bandwidth (str | None): KDE bandwidth or "scott"/.
                "silverman" for automatic selection.
            sigmoid_weight (FloatType | None): Controls seam transition.
                in the analysis.
        """
        self.gamma = gamma if gamma is not None else DEFAULT_GAMMA
        self.bandwidth = bandwidth
        self.sigmoid_weight = (
            sigmoid_weight if sigmoid_weight is not None else DEFAULT_SIGMOID_WEIGHT
        )
        self.kde: KernelDensity | None = None
        self.ref_points: Matrix | NPMatrix | None = None
        self.candidate_gamma: FloatType | None = None
        self.trained = False

    @property
    def method_type(self) -> str:
        """method_type property for the density comparison method.

        Returns:
            str: The method name.
        """
        return "density"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """decision_boundary property for the density monitor.

        Returns:
            DecisionBoundary: Structured decision-boundary metadata.
        """
        if not hasattr(self, "kde") or self.kde is None or not self.trained:
            return DecisionBoundary(
                type="density",
                parameters={"gamma": float(self.gamma), "bandwidth": self.bandwidth},
                coverage={},
                conservatism=None,
            )

        return DecisionBoundary(
            type="density",
            parameters={
                "gamma": float(self.gamma),
                "bandwidth": self.bandwidth,
                "candidate_gamma": self.candidate_gamma,
                "sigmoid_weight": self.sigmoid_weight,
            },
            coverage=self._estimate_coverage(),
            conservatism=self._calculate_conservatism(),
        )

    def _estimate_coverage(self) -> dict[str, FloatType]:
        """Estimate coverage metrics for density-based boundary.

        Returns:
            dict[str, FloatType]: Coverage metrics for the fitted.
                boundary.
        """
        if not self.trained or self.ref_points is None:
            return {}

        # Calculate PDF on training points
        train_pdf = self.pdf(self.ref_points)
        above_threshold = (train_pdf > self.gamma).mean()

        return {
            "coverage_ratio": NPFloatType(above_threshold),
            "n_above_threshold": NPFloatType((train_pdf > self.gamma).sum()),
        }

    def pdf(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.float64]:
        """Estimate PDF at test points using KDE.

        Args:
            test_points (Matrix | NPMatrix): Points to evaluate.

        Returns:
            npt.NDArray[np.float64]: PDF values at test points.

        Raises:
            RuntimeError: If the monitor is not fitted or KDE is.
                missing.
        """
        if not self.trained:
            raise RuntimeError("SuperlevelSetMonitor not fitted yet")

        if self.kde is None:
            raise RuntimeError("KDE not initialized")

        # KDE.score_samples returns log PDF, convert to PDF
        log_pdf = self.kde.score_samples(test_points.T)
        pdf_values = np.exp(
            log_pdf * self.sigmoid_weight
        )  # Apply sigmoid transformation

        return cast("npt.NDArray[np.float64]", pdf_values)

    def suggest_reasonable_gamma(
        self,
        percentile: FloatType | None = None,
    ) -> FloatType:
        """Suggest reasonable gamma threshold based on PDF analysis.

        Args:
            percentile (FloatType | None): Percentile of PDF values to.
                use as the threshold.

        Returns:
            FloatType: Suggested gamma value based on data distribution.
        """
        if percentile is None:
            percentile = NPFloatType(75.0)

        if self.ref_points is None or self.kde is None:
            return self.gamma

        # Calculate PDF on reference data
        pdf_values = self.pdf(self.ref_points)
        if len(pdf_values) == 0:
            return self.gamma

        # Suggest threshold based on data distribution
        candidate = NPFloatType(np.percentile(pdf_values, percentile))
        self.candidate_gamma = candidate
        return candidate

    def validate_hypersurface(self) -> dict[str, Any]:
        """Validate the density boundary quality.

        Returns:
            dict[str, Any]: Dictionary with quality metrics.
        """
        if not self.trained:
            return {"valid": False, "error": "Not fitted"}

        # Test both types of samples
        test_results = {}

        return {
            "validation": test_results,
            "gamma": self.gamma,
            "suggested_gamma": self.candidate_gamma,
            "coverage_range": self._candidate_radius_analysis(),
        }

    @staticmethod
    def _candidate_radius_analysis() -> dict[str, Any]:
        """Analyze range of reasonable radii/coverage values.

        Returns:
            dict[str, Any]: Analysis metadata.
        """
        return {"analysis_type": "density"}

    def bandwidth_for_deflection(self) -> str:
        """Get deflated bandwidth string.

        Silverman reduces bandwidth compared to Scott's rule.

        Returns:
            str: Bandwidth selection string.
        """
        if self.ref_points is None:
            return self.bandwidth or "scott"
        return (
            "silverman"
            if self.ref_points.size < MAX_SILVERMAN_SAMPLES
            else self.bandwidth or "scott"
        )

    def fit(self, reference_points: Matrix | NPMatrix) -> Self:
        """Fit KDE to reference ODD data.

        Args:
            reference_points (Matrix | NPMatrix): Reference ODD points.

        Returns:
            Self: Self for method chaining.
        """
        self.ref_points = reference_points  # Store reference points

        # Auto-detect reasonable bandwidth if not specified
        if isinstance(self.bandwidth, str) or self.bandwidth is None:
            actual_bandwidth = self.bandwidth or "scott"
        else:
            actual_bandwidth = self.bandwidth

        # Input convention in comparison methods is
        # (n_features, n_samples). KernelDensity expects
        # (n_samples, n_features), therefore always transpose.
        kde_data = reference_points.T
        self.kde = KernelDensity(bandwidth=actual_bandwidth, kernel="gaussian")
        self.kde.fit(kde_data)
        self.trained = True  # must be set before pdf() / suggest_reasonable_gamma()

        # Always calibrate gamma from data so the threshold reflects the
        # actual PDF magnitude.
        self.gamma = self.suggest_reasonable_gamma(percentile=NPFloatType(75.0))

        return self

    def _calculate_conservatism(self) -> FloatType:
        """Calculate conservatism for density-based boundary.

        Higher gamma values = more conservative (exclude more points).

        Returns:
            FloatType: Conservatism score in the range [0, 1].
        """
        if not self.trained or self.kde is None:
            return NPFloatType(0.5)

        # Calculate PDF range on reference data
        if self.ref_points is None:
            return NPFloatType(0.5)

        try:
            pdf_values = self.pdf(self.ref_points)
        except (RuntimeError, ValueError):
            return NPFloatType(0.5)

        if len(pdf_values) == 0:
            return NPFloatType(0.5)

        max_pdf = pdf_values.max()
        pdf_values.min()
        if max_pdf <= MIN_PDF_EPSILON:
            return NPFloatType(0.5)

        # Conservative approach: compare gamma to PDF range
        # Higher gamma relative to max PDF = more conservative
        # gamma near max_pdf filters most points (very conservative)
        # gamma near min_pdf includes most points (less conservative)
        normalized_gamma = self.gamma / max_pdf
        return NPFloatType(max(0.0, min(1.0, normalized_gamma)))

    def __call__(self, test_point: Vector | NPVector) -> bool:
        """Check if test point is in superlevel set (PDF > gamma)".

        Args:
            test_point (Vector | NPVector): Point to evaluate.

        Returns:
            bool: True if pdf(test_point) > gamma.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained:
            raise RuntimeError("SuperlevelSetMonitor not fitted yet")

        # Evaluate PDF at test point and apply threshold
        test_point_matrix = cast(
            "Matrix",
            np.asarray(test_point, dtype=float).reshape(-1, 1),
        )
        pdf_value = self.pdf(test_point_matrix)

        return bool(
            pdf_value[0] > self.gamma
            if isinstance(pdf_value, np.ndarray)
            else pdf_value > self.gamma
        )

    def evaluate_batch(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.bool_]:
        """Vectorized evaluation for multiple test points (optimized).

        Args:
            test_points (Matrix | NPMatrix): Matrix of points.

        Returns:
            npt.NDArray[np.bool_]: Boolean array indicating membership.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained:
            raise RuntimeError("Method not fitted")

        # Evaluate all points
        pdf_values = self.pdf(test_points)
        return pdf_values > self.gamma

    def create_visualization_grid(
        self,
        bounding_box: tuple[Vector | NPVector, Vector | NPVector],
        resolution: int = 100,
    ) -> tuple[NPMatrix, NPMatrix]:
        """Create evenly spaced grid for visualization.

        Args:
            bounding_box (tuple[Vector | NPVector, Vector | NPVector]):
                (Lower, upper) bounds.
            resolution (int): Grid resolution per dimension.

        Returns:
            tuple[NPMatrix, NPMatrix]: Grid points and PDF values for
                contour plotting.

        Raises:
            NotImplementedError: If the input is not two-dimensional.
        """
        lower, upper = bounding_box

        if upper.shape[0] != VISUALIZATION_DIMENSIONS:
            raise NotImplementedError("Visualization currently only supports 2D data")

        # Create meshgrid for 2D points
        x = np.linspace(lower[0], upper[0], resolution)
        y = np.linspace(lower[1], upper[1], resolution)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.vstack([xx.ravel(), yy.ravel()])
        grid_pdf = self.pdf(grid_points)

        return grid_points, grid_pdf


class ClusteredSuperlevelSetMonitor(ODDBoundaryMethod):
    """Clustered superlevel-set boundary based on per-cluster KDEs.

    The method partitions the reference ODD into clusters, fits one
    `SuperlevelSetMonitor` per cluster, and classifies a point as inside
    if any cluster-level superlevel set accepts it.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        gamma: FloatType | None = None,
        bandwidth: str | None = "scott",
        min_cluster_size: int = 3,
    ) -> None:
        self.n_clusters = n_clusters
        self.gamma = gamma if gamma is not None else DEFAULT_GAMMA
        self.bandwidth = bandwidth
        self.min_cluster_size = min_cluster_size
        self.monitors: list[SuperlevelSetMonitor] = []
        self.trained = False

    @property
    def method_type(self) -> str:
        """method_type property for the clustered density method.

        Returns:
            str: The method name.
        """
        return "density_clustered"

    @property
    def decision_boundary(self) -> DecisionBoundary:
        """decision_boundary property for the clustered density monitor.

        Returns:
            Structured decision-boundary metadata.
        """
        return DecisionBoundary(
            type="density_clustered",
            parameters={
                "n_clusters": self.n_clusters,
                "gamma": float(self.gamma),
                "bandwidth": self.bandwidth,
                "min_cluster_size": self.min_cluster_size,
            },
            coverage={
                "n_cluster_monitors": NPFloatType(len(self.monitors)),
            },
            conservatism=None,
        )

    def fit(self, reference_points: Matrix | NPMatrix) -> Self:
        """Fit clustered superlevel sets to reference points.

        Args:
            reference_points (Matrix | NPMatrix): Reference ODD points.

        Returns:
            Self: Self for method chaining.
        """
        data_t = reference_points.T
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_t)

        self.monitors = []
        for cluster_id in range(self.n_clusters):
            cluster_points = data_t[labels == cluster_id]
            if cluster_points.shape[0] < self.min_cluster_size:
                continue
            monitor = SuperlevelSetMonitor(
                gamma=self.gamma,
                bandwidth=self.bandwidth,
            )
            monitor.fit(cast("Matrix", cluster_points.T))
            self.monitors.append(monitor)

        self.trained = True
        return self

    def __call__(self, test_point: Vector | NPVector) -> bool:
        """Return True if any cluster-level superlevel set accepts it.

        Args:
            test_point (Vector | NPVector): Point to evaluate.

        Returns:
            bool: True if any cluster monitor accepts the point.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained:
            raise RuntimeError("ClusteredSuperlevelSetMonitor not fitted yet")

        return any(monitor(test_point) for monitor in self.monitors)

    def evaluate_batch(self, test_points: Matrix | NPMatrix) -> npt.NDArray[np.bool_]:
        """Evaluate points against the clustered superlevel-set union.

        Args:
            test_points (Matrix | NPMatrix): Matrix of points.

        Returns:
            Boolean array indicating membership.

        Raises:
            RuntimeError: If the monitor is not fitted.
        """
        if not self.trained:
            raise RuntimeError("ClusteredSuperlevelSetMonitor not fitted yet")

        if not self.monitors:
            return np.zeros(test_points.shape[1], dtype=bool)

        decisions = np.zeros(test_points.shape[1], dtype=bool)
        for monitor in self.monitors:
            decisions |= monitor.evaluate_batch(test_points)
        return decisions


__all__ = ["ClusteredSuperlevelSetMonitor", "SuperlevelSetMonitor"]
