# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""High-level autoSAFE ODD model and fitting pipeline."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from autosafe.kernels.rbf import calibrate_rbf_scale_d_tilde
from autosafe.neighbors import find_closest_vectors_by_index
from autosafe.odd.config import (
    AutoSafeConfig,
    CalibratedRBFConfig,
    ConformalMembershipConfig,
    FixedMembershipConfig,
    ManualRBFConfig,
    ResolvedAutoSafeConfig,
    ResolvedKernelConfig,
    ResolvedMembershipConfig,
)
from autosafe.odd.membership import conformal_membership_threshold
from autosafe.preprocessing import RangeNormalizer
from autosafe.sample import Sample
from autosafe.samples import Samples
from autosafe.typing import ClosestSampleModeType, NPMatrix, ScaleValue


def _as_matrix(
    value: npt.ArrayLike,
    name: str,
    *,
    allow_empty: bool = False,
) -> NPMatrix:
    """Validate external tabular data and return float64 rows.

    Returns:
        NPMatrix: Validated two-dimensional float64 data.

    Raises:
        ValueError: If the input is empty, non-finite, or not a matrix.
    """
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:  # ruff:ignore[magic-value-comparison]
        raise ValueError(f"{name} must have shape (n_samples, n_features)")
    if not allow_empty and result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if result.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_feature_count(
    data: NPMatrix | None,
    expected: int,
    name: str,
) -> None:
    """Validate an optional matrix against the anchor feature count.

    Raises:
        ValueError: If the matrix has a different feature count.
    """
    if data is not None and data.shape[1] != expected:
        raise ValueError(f"{name} has {data.shape[1]} features; expected {expected}")


def _kernel_scale(
    value: ScaleValue,
    feature_count: int,
    name: str,
) -> float | npt.NDArray[np.float64]:
    """Convert immutable configuration scales to kernel arguments.

    Returns:
        float | npt.NDArray[np.float64]: Scalar or feature vector.

    Raises:
        ValueError: If a vector has the wrong feature count.
    """
    if isinstance(value, tuple):
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (feature_count,):
            raise ValueError(
                f"{name} has {result.size} values; expected {feature_count}"
            )
        return result
    return value


def _relative_floor(
    maximum_variance: ScaleValue,
    ratio: float,
) -> ScaleValue:
    """Derive lambda from kappa and a relative floor.

    Returns:
        ScaleValue: Derived absolute variance floor.
    """
    if isinstance(maximum_variance, tuple):
        return tuple(ratio * value for value in maximum_variance)
    return ratio * maximum_variance


def _validate_floor(
    maximum_variance: ScaleValue,
    variance_floor: ScaleValue,
    feature_count: int,
) -> None:
    """Validate component-wise ``0 < lambda < kappa``.

    Raises:
        ValueError: If any floor is not below its maximum variance.
    """
    kappa = np.broadcast_to(
        np.asarray(maximum_variance, dtype=np.float64),
        (feature_count,),
    )
    lam = np.broadcast_to(
        np.asarray(variance_floor, dtype=np.float64),
        (feature_count,),
    )
    if not np.all(lam < kappa):
        raise ValueError("variance_floor (lambda) must be below maximum_variance")


def _normalizer_from_config(config: AutoSafeConfig) -> RangeNormalizer | None:
    """Create the configured normalizer, if enabled.

    Returns:
        RangeNormalizer | None: Unfitted normalizer or ``None``.
    """
    settings = config.normalization
    if not settings.enabled:
        return None
    return RangeNormalizer(
        target_range=settings.target_range,
        method=settings.method,
        iqr_factor=settings.iqr_factor,
        eps=settings.epsilon,
    )


def _transform(
    normalizer: RangeNormalizer | None,
    data: NPMatrix | None,
) -> NPMatrix | None:
    """Apply a fitted normalizer while preserving ``None``.

    Returns:
        NPMatrix | None: Transformed data or ``None``.
    """
    if data is None:
        return None
    if normalizer is None:
        return data.copy()
    return np.asarray(normalizer.transform(data), dtype=np.float64)


def _resolve_kernel(
    anchors: NPMatrix,
    config: AutoSafeConfig,
) -> tuple[
    ResolvedKernelConfig,
    dict[str, Any],
    ClosestSampleModeType,
]:
    """Resolve kernel parameters against anchor geometry.

    Returns:
        tuple: Resolved configuration, kernel arguments, and neighbor
        mode.

    Raises:
        TypeError: If an unsupported kernel configuration is provided.
    """
    feature_count = anchors.shape[1]
    kernel = config.kernel
    if isinstance(kernel, CalibratedRBFConfig):
        neighbor_indices = find_closest_vectors_by_index(
            anchors,
            disable_tqdm=True,
        )
        distances = np.linalg.norm(anchors[neighbor_indices] - anchors, axis=1)
        kappa, eta = calibrate_rbf_scale_d_tilde(
            distances,
            gamma=kernel.decay_per_median_gap,
            s=kernel.width_in_median_gaps,
        )
        positive = distances[distances > 0.0]
        median_distance = float(np.median(positive))
        lam = kernel.relative_variance_floor * kappa
        resolved = ResolvedKernelConfig(
            mode="calibrated",
            maximum_variance=kappa,
            distance_decay_rate=eta,
            variance_floor=lam,
            median_nearest_neighbor_distance=median_distance,
            decay_per_median_gap=kernel.decay_per_median_gap,
            width_in_median_gaps=kernel.width_in_median_gaps,
            relative_variance_floor=kernel.relative_variance_floor,
        )
        return resolved, {"kappa": kappa, "eta": eta, "lam": lam}, "global"

    if not isinstance(kernel, ManualRBFConfig):  # pragma: no cover
        raise TypeError(f"unsupported kernel configuration {type(kernel)}")
    floor = kernel.variance_floor
    if floor is None:
        floor = _relative_floor(
            kernel.maximum_variance,
            kernel.relative_variance_floor,
        )
    _validate_floor(kernel.maximum_variance, floor, feature_count)
    kappa_arg = _kernel_scale(
        kernel.maximum_variance,
        feature_count,
        "maximum_variance",
    )
    eta_arg = _kernel_scale(
        kernel.distance_decay_rate,
        feature_count,
        "distance_decay_rate",
    )
    lam_arg = _kernel_scale(floor, feature_count, "variance_floor")
    resolved = ResolvedKernelConfig(
        mode="manual",
        maximum_variance=kernel.maximum_variance,
        distance_decay_rate=kernel.distance_decay_rate,
        variance_floor=floor,
        relative_variance_floor=(
            kernel.relative_variance_floor if kernel.variance_floor is None else None
        ),
    )
    return (
        resolved,
        {"kappa": kappa_arg, "eta": eta_arg, "lam": lam_arg},
        kernel.nearest_neighbor_mode,
    )


def _fixed_membership(config: FixedMembershipConfig) -> ResolvedMembershipConfig:
    """Resolve a fixed zeta to its authoritative log-space threshold.

    Returns:
        ResolvedMembershipConfig: Fixed threshold in both spaces.
    """
    zeta = config.affinity_threshold
    return ResolvedMembershipConfig(
        mode="fixed",
        affinity_threshold=zeta,
        log_survival_threshold=float(np.log1p(-zeta)),
    )


def _conformal_membership(
    config: ConformalMembershipConfig,
    samples: Samples,
    calibration_data: NPMatrix | None,
) -> ResolvedMembershipConfig:
    """Resolve epsilon after OOD adjustment.

    Returns:
        ResolvedMembershipConfig: Calibrated membership threshold.

    Raises:
        ValueError: If held-out calibration data is absent.
    """
    if calibration_data is None:
        raise ValueError("conformal membership requires held-out calibration_data")
    _, survival = samples.affinity_dual(calibration_data)
    zeta, log_threshold = conformal_membership_threshold(
        np.asarray(survival, dtype=np.float64),
        config.target_false_exclusion_rate,
    )
    return ResolvedMembershipConfig(
        mode="conformal",
        affinity_threshold=zeta,
        log_survival_threshold=log_threshold,
        target_false_exclusion_rate=config.target_false_exclusion_rate,
        calibration_size=calibration_data.shape[0],
    )


@dataclass(slots=True)
class AutoSafeODD:
    """A fitted autoSAFE ODD including ID and observed OOD data."""

    in_distribution_data: NPMatrix
    out_of_distribution_data: NPMatrix | None
    calibration_data: NPMatrix | None
    normalized_in_distribution_data: NPMatrix
    normalized_out_of_distribution_data: NPMatrix | None
    normalized_calibration_data: NPMatrix | None
    samples: Samples
    config: AutoSafeConfig
    resolved_config: ResolvedAutoSafeConfig
    normalizer: RangeNormalizer | None
    feature_names: tuple[str, ...] | None = None
    ood_consistency_result: dict[str, object] | None = None

    @classmethod
    def fit(  # ruff:ignore[complex-structure,too-many-branches,too-many-locals]
        cls,
        in_distribution_data: npt.ArrayLike,
        *,
        config: AutoSafeConfig | Mapping[str, object],
        out_of_distribution_data: npt.ArrayLike | None = None,
        calibration_data: npt.ArrayLike | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "AutoSafeODD":
        """Fit a complete kernel ODD from ID and optional OOD samples.

        Args:
            in_distribution_data (npt.ArrayLike): ID anchor rows.
            config (AutoSafeConfig | Mapping[str, object]): Typed
                configuration or an alias mapping.
            out_of_distribution_data (npt.ArrayLike | None): Known OOD
                rows to constrain.
            calibration_data (npt.ArrayLike | None): Held-out ID rows
                used only when membership is conformal.
            feature_names (Sequence[str] | None): Optional ordered
                names.

        Returns:
            AutoSafeODD: Fitted, queryable ODD.

        Raises:
            ValueError: If data and configuration are inconsistent.
            RuntimeError: If an internal invariant is violated.
        """
        if not isinstance(config, AutoSafeConfig):
            config = AutoSafeConfig.from_mapping(config)

        id_data = _as_matrix(in_distribution_data, "in_distribution_data")
        if id_data.shape[0] < 2:  # ruff:ignore[magic-value-comparison]
            raise ValueError("at least two ID anchor rows are required")
        ood_data = (
            None
            if out_of_distribution_data is None
            else _as_matrix(
                out_of_distribution_data,
                "out_of_distribution_data",
            )
        )
        held_out = (
            None
            if calibration_data is None
            else _as_matrix(calibration_data, "calibration_data")
        )
        feature_count = id_data.shape[1]
        _validate_feature_count(ood_data, feature_count, "out_of_distribution_data")
        _validate_feature_count(held_out, feature_count, "calibration_data")

        names: tuple[str, ...] | None = None
        if feature_names is not None:
            names = tuple(feature_names)
            if len(names) != feature_count:
                raise ValueError(
                    f"feature_names has {len(names)} entries; expected {feature_count}"
                )
            if len(set(names)) != len(names):
                raise ValueError("feature_names must be unique")
            if any(not name for name in names):
                raise ValueError("feature_names must not contain empty names")

        if ood_data is None and config.ood is not None:
            raise ValueError(
                "OOD consistency configuration requires out_of_distribution_data"
            )
        if ood_data is not None and config.ood is None:
            raise ValueError(
                "out_of_distribution_data requires OOD consistency configuration"
            )

        normalizer = _normalizer_from_config(config)
        if normalizer is not None:
            normalizer.fit(id_data)
        id_normalized = _transform(normalizer, id_data)
        ood_normalized = _transform(normalizer, ood_data)
        calibration_normalized = _transform(normalizer, held_out)
        if id_normalized is None:  # pragma: no cover
            raise RuntimeError("ID normalization unexpectedly returned None")

        resolved_kernel, kernel_kwargs, neighbor_mode = _resolve_kernel(
            id_normalized,
            config,
        )
        samples = Samples(
            [Sample(row) for row in id_normalized],
            closest_sample_mode=neighbor_mode,
            kernel_cls="RBF",
            kernel_kwargs=kernel_kwargs,
        )

        ood_result: dict[str, object] | None = None
        if ood_normalized is not None:
            ood_config = config.ood
            if ood_config is None:  # pragma: no cover
                raise RuntimeError("validated OOD config disappeared")
            ood_result = samples.enforce_ood_consistency(
                ood_normalized,
                xi=ood_config.max_affinity,
                shrink_factor=ood_config.covariance_shrink_factor,
                max_iterations=ood_config.max_iterations,
                refresh_interval=ood_config.refresh_interval,
                log_interval=ood_config.log_interval,
                batch_jump=ood_config.batch_jump,
            )

        membership = config.membership
        if isinstance(membership, FixedMembershipConfig):
            resolved_membership = _fixed_membership(membership)
        else:
            resolved_membership = _conformal_membership(
                membership,
                samples,
                calibration_normalized,
            )
        if (
            config.ood is not None
            and config.ood.max_affinity >= resolved_membership.affinity_threshold
        ):
            raise ValueError("resolved zeta must be greater than OOD max affinity xi")

        return cls(
            in_distribution_data=id_data,
            out_of_distribution_data=ood_data,
            calibration_data=held_out,
            normalized_in_distribution_data=id_normalized,
            normalized_out_of_distribution_data=ood_normalized,
            normalized_calibration_data=calibration_normalized,
            samples=samples,
            config=config,
            resolved_config=ResolvedAutoSafeConfig(
                requested=config,
                kernel=resolved_kernel,
                membership=resolved_membership,
            ),
            normalizer=normalizer,
            feature_names=names,
            ood_consistency_result=ood_result,
        )

    def _query_matrix(
        self,
        data: npt.ArrayLike,
    ) -> tuple[NPMatrix, bool]:
        """Validate and normalize query rows.

        Returns:
            tuple[NPMatrix, bool]: Query matrix and single-row marker.

        Raises:
            RuntimeError: If normalization violates its contract.
        """
        raw = np.asarray(data, dtype=np.float64)
        is_single = raw.ndim == 1
        matrix = raw[None, :] if is_single else raw
        matrix = _as_matrix(matrix, "query_data")
        _validate_feature_count(
            matrix,
            self.in_distribution_data.shape[1],
            "query_data",
        )
        transformed = _transform(self.normalizer, matrix)
        if transformed is None:  # pragma: no cover
            raise RuntimeError("query normalization unexpectedly returned None")
        return transformed, is_single

    def affinity(
        self,
        data: npt.ArrayLike,
    ) -> float | npt.NDArray[np.float64]:
        """Evaluate linear-space affinity for one or more rows.

        Returns:
            float | npt.NDArray[np.float64]: Affinity value or vector.
        """
        transformed, is_single = self._query_matrix(data)
        value = np.asarray(self.samples(transformed), dtype=np.float64)
        if is_single:
            return float(value.reshape(-1)[0])
        return value

    def log_survival(
        self,
        data: npt.ArrayLike,
    ) -> float | npt.NDArray[np.float64]:
        """Evaluate stable ``log(1 - affinity)`` for query rows.

        Returns:
            float | npt.NDArray[np.float64]: Log-survival value or
            vector.
        """
        transformed, is_single = self._query_matrix(data)
        _, survival = self.samples.affinity_dual(transformed)
        value = np.asarray(survival, dtype=np.float64)
        if is_single:
            return float(value.reshape(-1)[0])
        return value

    def contains(
        self,
        data: npt.ArrayLike,
    ) -> bool | npt.NDArray[np.bool_]:
        """Classify query rows using the authoritative log threshold.

        Returns:
            bool | npt.NDArray[np.bool_]: Membership decision(s).
        """
        survival = self.log_survival(data)
        threshold = self.resolved_config.membership.log_survival_threshold
        if isinstance(survival, float):
            return survival <= threshold
        return np.asarray(survival <= threshold, dtype=np.bool_)


__all__ = ["AutoSafeODD"]
