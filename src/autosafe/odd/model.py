# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""High-level autoSAFE ODD model and fitting pipeline."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from autosafe.exceptions import (
    CalibrationDataRequiredError,
    DuplicateFeatureNameError,
    EmptyFeatureNameError,
    EmptyParameterError,
    LengthMismatchError,
    NonFiniteParameterError,
    OODConfigurationMismatchError,
    ParameterOrderError,
    ParameterTypeError,
    ShapeMismatchError,
    TooFewAnchorsError,
    UnexpectedNoneError,
)
from autosafe.kernels.rbf import calibrate_rbf_scale_d_tilde, resolve_width_multiple
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

    Args:
        value (npt.ArrayLike): Tabular input data.
        name (str): Parameter name used in error messages.
        allow_empty (bool): Whether a matrix without rows is accepted.

    Returns:
        NPMatrix: Validated two-dimensional float64 data.

    Raises:
        ShapeMismatchError: If the input is not a matrix.
        EmptyParameterError: If the input has no rows (unless allowed)
            or no columns.
        NonFiniteParameterError: If the input is non-finite.
    """
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:  # ruff:ignore[magic-value-comparison]
        raise ShapeMismatchError(name, result.shape, (None, None))
    if (not allow_empty and result.shape[0] == 0) or result.shape[1] == 0:
        raise EmptyParameterError(name)
    if not np.isfinite(result).all():
        raise NonFiniteParameterError(name)
    return result


def _validate_feature_count(
    data: NPMatrix | None,
    expected: int,
    name: str,
) -> None:
    """Validate an optional matrix against the anchor feature count.

    Args:
        data (NPMatrix | None): Matrix to check, or ``None`` to skip the
            check.
        expected (int): Feature count of the anchors.
        name (str): Parameter name used in error messages.

    Raises:
        ShapeMismatchError: If the matrix has a different feature count.
    """
    if data is not None and data.shape[1] != expected:
        raise ShapeMismatchError(name, data.shape, (None, expected))


def _kernel_scale(
    value: ScaleValue,
    feature_count: int,
    name: str,
) -> float | npt.NDArray[np.float64]:
    """Convert immutable configuration scales to kernel arguments.

    Args:
        value (ScaleValue): Scalar or per-feature scale from the
            configuration.
        feature_count (int): Number of features.
        name (str): Parameter name used in error messages.

    Returns:
        float | npt.NDArray[np.float64]: Scalar or feature vector.

    Raises:
        LengthMismatchError: If a vector has the wrong feature count.
    """
    if isinstance(value, tuple):
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (feature_count,):
            raise LengthMismatchError(name, result.size, feature_count)
        return result
    return value


def _relative_floor(
    maximum_variance: ScaleValue,
    ratio: float,
) -> ScaleValue:
    """Derive lambda from kappa and a relative floor.

    Args:
        maximum_variance (ScaleValue): Kappa, a scalar or per-feature
            tuple.
        ratio (float): Relative floor lambda_rel.

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

    Args:
        maximum_variance (ScaleValue): Kappa, a scalar or per-feature
            tuple.
        variance_floor (ScaleValue): Lambda, a scalar or per-feature
            tuple.
        feature_count (int): Number of features.

    Raises:
        ParameterOrderError: If any floor is not below its maximum
            variance.
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
        raise ParameterOrderError("variance_floor", "maximum_variance")


def _normalizer_from_config(config: AutoSafeConfig) -> RangeNormalizer | None:
    """Create the configured normalizer, if enabled.

    Args:
        config (AutoSafeConfig): Configuration holding the normalization
            settings.

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

    Args:
        normalizer (RangeNormalizer | None): Fitted normalizer, or
            ``None`` for identity.
        data (NPMatrix | None): Data to transform, or ``None``.

    Returns:
        NPMatrix | None: Transformed data or ``None``.
    """
    if data is None:
        return None
    if normalizer is None:
        return data.copy()
    return np.asarray(normalizer.transform(data), dtype=np.float64)


#: Resolved kernel configuration, kernel arguments, and neighbor mode.
ResolvedKernel = tuple[ResolvedKernelConfig, dict[str, Any], ClosestSampleModeType]


def _resolve_kernel(
    anchors: NPMatrix,
    config: AutoSafeConfig,
) -> ResolvedKernel:
    """Resolve kernel parameters against anchor geometry.

    Args:
        anchors (NPMatrix): Normalized ID anchor points.
        config (AutoSafeConfig): Configuration holding the kernel
            settings.

    Returns:
        ResolvedKernel: Resolved configuration, kernel arguments, and
            neighbor mode.

    Raises:
        ParameterTypeError: If an unsupported kernel configuration is
            provided.
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
            width_in_median_gaps=resolve_width_multiple(
                kernel.width_in_median_gaps, anchors.shape[0]
            ),
            relative_variance_floor=kernel.relative_variance_floor,
        )
        return resolved, {"kappa": kappa, "eta": eta, "lam": lam}, "global"

    if not isinstance(kernel, ManualRBFConfig):  # pragma: no cover
        raise ParameterTypeError(
            "kernel", kernel, (CalibratedRBFConfig, ManualRBFConfig)
        )
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

    Args:
        config (FixedMembershipConfig): Fixed membership configuration.

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

    Args:
        config (ConformalMembershipConfig): Conformal membership
            configuration.
        samples (Samples): Fitted (and OOD-adjusted) ODD.
        calibration_data (NPMatrix | None): Normalized held-out ID
            calibration points.

    Returns:
        ResolvedMembershipConfig: Calibrated membership threshold.

    Raises:
        CalibrationDataRequiredError: If held-out calibration data is
            absent.
    """
    if calibration_data is None:
        raise CalibrationDataRequiredError
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
            TooFewAnchorsError: If fewer than two ID rows are given.
            LengthMismatchError: If ``feature_names`` does not match the
                feature count.
            DuplicateFeatureNameError: If ``feature_names`` repeats a
                name.
            EmptyFeatureNameError: If ``feature_names`` contains an
                empty name.
            OODConfigurationMismatchError: If OOD data and OOD
                configuration are not given together.
            ParameterOrderError: If the resolved zeta is not above xi.
            UnexpectedNoneError: If an internal invariant is violated.
        """
        if not isinstance(config, AutoSafeConfig):
            config = AutoSafeConfig.from_mapping(config)

        id_data = _as_matrix(in_distribution_data, "in_distribution_data")
        if id_data.shape[0] < 2:  # ruff:ignore[magic-value-comparison]
            raise TooFewAnchorsError(2, id_data.shape[0])
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
                raise LengthMismatchError("feature_names", len(names), feature_count)
            if len(set(names)) != len(names):
                raise DuplicateFeatureNameError(names)
            if any(not name for name in names):
                raise EmptyFeatureNameError

        if ood_data is None and config.ood is not None:
            raise OODConfigurationMismatchError(has_configuration=True)
        if ood_data is not None and config.ood is None:
            raise OODConfigurationMismatchError(has_configuration=False)

        normalizer = _normalizer_from_config(config)
        if normalizer is not None:
            normalizer.fit(id_data)
        id_normalized = _transform(normalizer, id_data)
        ood_normalized = _transform(normalizer, ood_data)
        calibration_normalized = _transform(normalizer, held_out)
        if id_normalized is None:  # pragma: no cover
            raise UnexpectedNoneError("id_normalized")

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
                raise UnexpectedNoneError("config.ood")
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
            raise ParameterOrderError("xi", "zeta")

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

        Args:
            data (npt.ArrayLike): One query point or a matrix of query
                points.

        Returns:
            tuple[NPMatrix, bool]: Query matrix and single-row marker.

        Raises:
            UnexpectedNoneError: If normalization violates its contract.
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
            raise UnexpectedNoneError("transformed")
        return transformed, is_single

    def affinity(
        self,
        data: npt.ArrayLike,
    ) -> float | npt.NDArray[np.float64]:
        """Evaluate linear-space affinity for one or more rows.

        The value saturates at exactly 1.0 in float64 deep inside the
        ODD; ``contains`` and ``log_survival`` use the log-space rule
        and should be used for decisions and rankings.

        Args:
            data (npt.ArrayLike): One row or a matrix of query rows.

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

        Args:
            data (npt.ArrayLike): One row or a matrix of query rows.

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

        Args:
            data (npt.ArrayLike): One row or a matrix of query rows.

        Returns:
            bool | npt.NDArray[np.bool_]: Membership decision(s).
        """
        survival = self.log_survival(data)
        threshold = self.resolved_config.membership.log_survival_threshold
        if isinstance(survival, float):
            return survival <= threshold
        return np.asarray(survival <= threshold, dtype=np.bool_)


__all__ = ["AutoSafeODD"]
