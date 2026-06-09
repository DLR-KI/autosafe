# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Robust data preprocessing and normalization utilities."""

from typing import Literal

import jax.numpy as jnp
import numpy as np

from autosafe import _jax_config  # noqa: F401
from autosafe.typing import FloatType, Matrix, NPFloatType, NPMatrix, NPVector


class RangeNormalizer:
    """Robust normalization to target ranges, preserving outliers.

    This normalizer maps data to [target_low, target_high] range while:

    1. Ignoring outliers for range calculation (using robust statistics)
    2. Preserving outliers (values can exceed target range)
    3. Handling zero-variance dimensions gracefully
    4. Supporting different normalization methods
    """

    def __init__(
        self,
        target_range: tuple[float, float] = (-1.0, 1.0),
        method: Literal["minmax", "iqr", "zscore"] = "iqr",
        iqr_factor: float = 1.5,
        eps: float = 1e-10,
    ) -> None:
        """Initialize the range normalizer.

        Args:
            target_range (tuple[float, float]): Desired output range.
                (for example [-1, 1] for kernelaffinity).
            method (Literal["minmax", "iqr", "zscore"]):
                Normalization method. Supported values are 'minmax',
                'iqr', and 'zscore' (default 'iqr').
            iqr_factor (float): Multiplier for IQR to define "normal".
                range bounds.
            eps (float): Small constant to prevent division by zero.
        """
        self.target_range = target_range
        self.target_min, self.target_high = target_range
        self.target_span = self.target_high - self.target_min
        self.method = method
        self.iqr_factor = iqr_factor
        self.eps = eps

        # Statistics that will be fit to data (stored as NumPy)
        self.ref_min_: NPVector | None = None
        self.ref_max_: NPVector | None = None
        self.ref_q1_: NPVector | None = None
        self.ref_q3_: NPVector | None = None
        self.ref_mean_: NPVector | None = None
        self.ref_std_: NPVector | None = None
        self.ref_bounds_: NPVector | None = None
        self.ref_data_range_: NPVector | None = None

        self.n_features_: int = 0  # Will be set during fit
        self.initialized_: bool = False

    def fit(self, x: Matrix | NPMatrix) -> "RangeNormalizer":
        """Fit normalizer to data using selected method.

        Args:
            x (Matrix | NPMatrix): Input data matrix with shape
                (n_points, n_features).

        Returns:
            RangeNormalizer: Self for chaining.

        Raises:
            ValueError: If input data is not 2D or if required.
                statistics cannot be computed for the selected method.
        """
        if x.ndim != 2:  # noqa: PLR2004
            raise ValueError(f"Expected 2D data, got {x.ndim}D shape: {x.shape}")

        x_j = jnp.asarray(x, dtype=FloatType)
        self.n_features_ = x_j.shape[1]

        if self.method == "minmax":
            self.ref_min_ = np.asarray(jnp.min(x_j, axis=0), NPFloatType)
            self.ref_max_ = np.asarray(jnp.max(x_j, axis=0), NPFloatType)

        elif self.method == "iqr":
            # Use robust statistics (25th and 75th percentiles)
            self.ref_q1_ = np.asarray(jnp.percentile(x_j, 25, axis=0), NPFloatType)
            self.ref_q3_ = np.asarray(jnp.percentile(x_j, 75, axis=0), NPFloatType)
            self.ref_data_range_ = np.asarray(
                jnp.max(x_j, axis=0) - jnp.min(x_j, axis=0), NPFloatType
            )

            # Extended bounds:
            # Q1 - iqr_factor*IQR  to  Q3 + iqr_factor*IQR.
            iqr = self.ref_q3_ - self.ref_q1_ + self.eps
            lower_bound = self.ref_q1_ - self.iqr_factor * iqr
            upper_bound = self.ref_q3_ + self.iqr_factor * iqr
            self.ref_bounds_ = np.asarray(
                np.vstack([lower_bound, upper_bound]), NPFloatType
            )

        elif self.method == "zscore":
            self.ref_mean_ = np.asarray(jnp.mean(x_j, axis=0), NPFloatType)
            self.ref_std_ = np.asarray(jnp.std(x_j, axis=0), NPFloatType) + self.eps

        self.initialized_ = True
        return self

    def transform(self, x: Matrix | NPMatrix) -> "Matrix":
        """Normalize data using fitted parameters.

        Args:
            x (Matrix | NPMatrix): Input data matrix for batch+
                processing.

        Returns:
            Matrix: Normalized JAX array, preserving outliers (values
                can exceed target range).

        Raises:
            ValueError: If normalizer is not initialized or if required.
                statistics are not available for the selected method.
        """
        if not self.initialized_:
            raise ValueError("Normalizer must be fit to data before transforming.")

        x_j = jnp.asarray(x, dtype=FloatType)

        if self.method == "minmax":
            if self.ref_min_ is None or self.ref_max_ is None:
                raise ValueError("Min-max normalizer requires fitted min/max values.")

            normalized = (x_j - self.ref_min_) / (
                (self.ref_max_ - self.ref_min_) + self.eps
            )
            return normalized * self.target_span + self.target_min

        if self.method == "iqr":
            if self.ref_q1_ is None or self.ref_q3_ is None or self.ref_bounds_ is None:
                raise ValueError(
                    "IQR-based normalizer requires fitted Q1/Q3/bounds values."
                )

            lower = self.ref_bounds_[0]
            upper = self.ref_bounds_[1]
            min_denom = jnp.maximum(
                1e-3
                * jnp.asarray(
                    self.ref_data_range_
                    if self.ref_data_range_ is not None
                    else np.ones(self.n_features_)
                ),
                self.eps,
            )
            span = jnp.maximum(upper - lower, min_denom)
            normalized = (x_j - lower) / span
            return normalized * self.target_span + self.target_min

        if self.method == "zscore":
            if self.ref_mean_ is None or self.ref_std_ is None:
                raise ValueError("Z-score normalizer requires fitted mean/std values.")

            z_scores = (x_j - self.ref_mean_) / self.ref_std_
            return ((z_scores + 3.0) / 6.0) * self.target_span + self.target_min

        raise ValueError(f"Unknown method: {self.method}")

    def fit_transform(self, x: Matrix | NPMatrix) -> "Matrix":
        """Fit and transform in one step.

        Args:
            x (Matrix | NPMatrix): Input data to fit and transform.

        Returns:
            Matrix: Normalized data after fitting.
        """
        return self.fit(x).transform(x)

    def inverse_transform(self, x: Matrix | NPMatrix) -> "Matrix":
        """Attempt to reverse normalization.

        Note: Outliers that exceeded normalization bounds during
        transform may not be restored to their exact original values.

        Args:
            x (Matrix | NPMatrix): Normalized data to inverse transform.

        Returns:
            Matrix: Inverse transformed data as a JAX array.

        Raises:
            ValueError: If normalizer is not initialized or if inverse.
                transform is not implemented for the selected method.
        """
        if not self.initialized_:
            raise ValueError("Normalizer must be fit before inverse transform.")

        x_j = jnp.asarray(x, dtype=FloatType)

        if self.method == "minmax":
            if self.ref_min_ is None or self.ref_max_ is None:
                raise ValueError("Min-max normalizer requires fitted min/max values.")
            return (x_j - self.target_min) / self.target_span * (
                self.ref_max_ - self.ref_min_
            ) + self.ref_min_
        if self.method == "iqr":
            if self.ref_q1_ is None or self.ref_q3_ is None or self.ref_bounds_ is None:
                raise ValueError("IQR normalizer requires fitted Q1/Q3/bounds values.")
            lower = self.ref_bounds_[0]
            upper = self.ref_bounds_[1]
            min_denom = jnp.maximum(
                1e-3
                * jnp.asarray(
                    self.ref_data_range_
                    if self.ref_data_range_ is not None
                    else np.ones(self.n_features_)
                ),
                self.eps,
            )
            span = jnp.maximum(upper - lower, min_denom)
            x_normalized = (x_j - self.target_min) / self.target_span
            return x_normalized * span + lower
        raise ValueError(f"Inverse transform not implemented for method: {self.method}")

    @property
    def range_bounds_(self) -> tuple[NPVector, NPVector]:
        """Get the effective range bounds used for normalization.

        Returns:
            tuple[NPVector, NPVector]: (Lower_bounds, upper_bounds) used
                for normalization, based on the selected method.

        Raises:
            ValueError: If range bounds are not available for the.
                selected method.
        """
        if (
            self.method == "minmax"
            and self.ref_min_ is not None
            and self.ref_max_ is not None
        ):
            return self.ref_min_, self.ref_max_
        if self.method == "iqr" and self.ref_bounds_ is not None:
            return self.ref_bounds_[0], self.ref_bounds_[1]  # lower, upper
        raise ValueError(f"Range bounds not available for method: {self.method}")


def create_robust_normalization_pipeline(
    target_range: tuple[float, float] = (-1.0, 1.0),
    method: Literal["iqr", "minmax"] = "iqr",
) -> RangeNormalizer:
    """Create a robust normalization pipeline with specified parameters.

    Args:
        target_range (tuple[float, float]): Desired output range for.
            normalization (default (-1.0, 1.0)).
        method (Literal["iqr", "minmax"]): Normalization method.
            (default 'iqr'). Supported values are 'iqr' (robust to
            outliers) and 'minmax' (includes outliers).

    Returns:
        RangeNormalizer: Configured normalizer instance ready for.
            fitting and transforming data.
    """
    return RangeNormalizer(target_range=target_range, method=method)
