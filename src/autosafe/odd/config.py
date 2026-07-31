# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Configuration types for high-level autoSAFE ODD construction.

The public configuration uses descriptive names. The
:meth:`AutoSafeConfig.from_mapping` method also accepts notation from
the autoSAFE paper. All aliases resolve to one canonical field before a
model is built.
"""

import difflib
import math
import re
import unicodedata
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from autosafe.kernels.rbf import SIGMA_FLOOR_RATIO
from autosafe.typing import KernelConfig, MembershipConfig, ScaleValue


def _finite_float(value: object, name: str) -> float:
    """Return a finite float.

    Args:
        value (object): Value to convert.
        name (str): Parameter name used in error messages.

    Returns:
        float: Converted finite value.

    Raises:
        ValueError: If the value is not finite.
        TypeError: If the value is not numeric.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    """Return a strictly positive finite float.

    Args:
        value (object): Value to convert.
        name (str): Parameter name used in error messages.

    Returns:
        float: Converted positive value.

    Raises:
        ValueError: If the converted value is not positive.
    """
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than 0")
    return result


def _positive_integer(value: object, name: str) -> int:
    """Return a strictly positive integer.

    Args:
        value (object): Value to validate.
        name (str): Parameter name used in error messages.

    Returns:
        int: Validated positive integer.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return result


def _unit_interval(value: object, name: str) -> float:
    """Return a finite float strictly inside the unit interval.

    Args:
        value (object): Value to convert.
        name (str): Parameter name used in error messages.

    Returns:
        float: Converted value inside ``(0, 1)``.

    Raises:
        ValueError: If the converted value is outside ``(0, 1)``.
    """
    result = _finite_float(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return result


def _scale_value(value: object, name: str) -> ScaleValue:
    """Normalize a positive scalar or one-dimensional sequence.

    Args:
        value (object): Scalar or sequence to normalize.
        name (str): Parameter name used in error messages.

    Returns:
        ScaleValue: A float or immutable tuple of floats.

    Raises:
        ValueError: If a sequence is empty or any value is invalid.
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _positive_float(np.ravel(value)[0], name)
        if value.ndim != 1:
            raise ValueError(f"{name} must be scalar or one-dimensional")
        raw: object = list(value)
    else:
        raw = value
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        result = tuple(_positive_float(item, name) for item in raw)
        if not result:
            raise ValueError(f"{name} must not be empty")
        return result
    return _positive_float(raw, name)


def _scale_to_json(value: ScaleValue) -> float | list[float]:
    """Convert an immutable scale value to JSON-compatible data.

    Args:
        value (ScaleValue): Immutable scalar or vector.

    Returns:
        float | list[float]: JSON-compatible scalar or list.
    """
    return list(value) if isinstance(value, tuple) else value


@dataclass(frozen=True, slots=True)
class CalibratedRBFConfig:
    r"""Data-calibrated RBF parameters.

    Attributes:
        decay_per_median_gap (float): Dimensionless decay constant
            :math:`\\gamma`.
        width_in_median_gaps (float): Width multiple :math:`s`.
        relative_variance_floor (float): Relative floor
            :math:`\\lambda_\\mathrm{rel}`.
    """

    decay_per_median_gap: float = 1.0
    width_in_median_gaps: float = 3.0
    relative_variance_floor: float = SIGMA_FLOOR_RATIO

    def __post_init__(self) -> None:
        """Validate calibrated kernel parameters."""
        object.__setattr__(
            self,
            "decay_per_median_gap",
            _positive_float(self.decay_per_median_gap, "decay_per_median_gap"),
        )
        object.__setattr__(
            self,
            "width_in_median_gaps",
            _positive_float(self.width_in_median_gaps, "width_in_median_gaps"),
        )
        object.__setattr__(
            self,
            "relative_variance_floor",
            _unit_interval(
                self.relative_variance_floor,
                "relative_variance_floor",
            ),
        )


@dataclass(frozen=True, slots=True)
class ManualRBFConfig:
    """Manually specified RBF parameters.

    ``variance_floor`` takes precedence when supplied. Otherwise the
    floor is derived as
    ``relative_variance_floor * maximum_variance``.
    """

    maximum_variance: ScaleValue = 1.0
    distance_decay_rate: ScaleValue = 1.0
    variance_floor: ScaleValue | None = None
    relative_variance_floor: float = SIGMA_FLOOR_RATIO
    nearest_neighbor_mode: Literal["global", "per_dimension"] = "global"

    def __post_init__(self) -> None:
        """Validate and normalize manual kernel parameters.

        Raises:
            ValueError: If a parameter is outside its valid range.
        """
        object.__setattr__(
            self,
            "maximum_variance",
            _scale_value(self.maximum_variance, "maximum_variance"),
        )
        object.__setattr__(
            self,
            "distance_decay_rate",
            _scale_value(self.distance_decay_rate, "distance_decay_rate"),
        )
        if self.variance_floor is not None:
            object.__setattr__(
                self,
                "variance_floor",
                _scale_value(self.variance_floor, "variance_floor"),
            )
        object.__setattr__(
            self,
            "relative_variance_floor",
            _unit_interval(
                self.relative_variance_floor,
                "relative_variance_floor",
            ),
        )
        if self.nearest_neighbor_mode not in {"global", "per_dimension"}:
            raise ValueError(
                "nearest_neighbor_mode must be 'global' or 'per_dimension'"
            )


@dataclass(frozen=True, slots=True)
class FixedMembershipConfig:
    r"""A fixed affinity threshold :math:`\zeta`."""

    affinity_threshold: float

    def __post_init__(self) -> None:
        """Validate the membership threshold."""
        object.__setattr__(
            self,
            "affinity_threshold",
            _unit_interval(self.affinity_threshold, "affinity_threshold"),
        )


@dataclass(frozen=True, slots=True)
class ConformalMembershipConfig:
    r"""Split-conformal calibration of :math:`\zeta`.

    Attributes:
        target_false_exclusion_rate (float): Desired marginal
            false-exclusion bound :math:`\varepsilon`.
    """

    target_false_exclusion_rate: float

    def __post_init__(self) -> None:
        """Validate the conformal target."""
        object.__setattr__(
            self,
            "target_false_exclusion_rate",
            _unit_interval(
                self.target_false_exclusion_rate,
                "target_false_exclusion_rate",
            ),
        )


@dataclass(frozen=True, slots=True)
class OODConsistencyConfig:
    """Configuration for the observed-OOD consistency adjustment."""

    max_affinity: float
    covariance_shrink_factor: float = 0.9
    max_iterations: int = 1_000_000
    refresh_interval: int = 1000
    log_interval: int = 1000
    batch_jump: bool = False

    def __post_init__(self) -> None:
        """Validate OOD consistency controls."""
        object.__setattr__(
            self,
            "max_affinity",
            _unit_interval(self.max_affinity, "max_affinity"),
        )
        object.__setattr__(
            self,
            "covariance_shrink_factor",
            _unit_interval(
                self.covariance_shrink_factor,
                "covariance_shrink_factor",
            ),
        )
        object.__setattr__(
            self,
            "max_iterations",
            _positive_integer(self.max_iterations, "max_iterations"),
        )
        object.__setattr__(
            self,
            "refresh_interval",
            _positive_integer(self.refresh_interval, "refresh_interval"),
        )
        object.__setattr__(
            self,
            "log_interval",
            _positive_integer(self.log_interval, "log_interval"),
        )


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Configuration for the common ID/OOD coordinate transform."""

    enabled: bool = True
    method: Literal["minmax", "iqr", "zscore"] = "iqr"
    target_range: tuple[float, float] = (-1.0, 1.0)
    iqr_factor: float = 1.5
    epsilon: float = 1e-10

    def __post_init__(self) -> None:
        """Validate normalization settings.

        Raises:
            ValueError: If a setting is outside its valid range.
        """
        if self.method not in {"minmax", "iqr", "zscore"}:
            raise ValueError("normalization method must be minmax, iqr, or zscore")
        if len(self.target_range) != 2:  # ruff:ignore[magic-value-comparison]
            raise ValueError("target_range must contain two values")
        low = _finite_float(self.target_range[0], "target_range")
        high = _finite_float(self.target_range[1], "target_range")
        if low >= high:
            raise ValueError("target_range lower bound must be below upper bound")
        object.__setattr__(self, "target_range", (low, high))
        object.__setattr__(
            self,
            "iqr_factor",
            _positive_float(self.iqr_factor, "iqr_factor"),
        )
        object.__setattr__(
            self,
            "epsilon",
            _positive_float(self.epsilon, "epsilon"),
        )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Evaluation-only controls that do not alter the fitted ODD."""

    local_noise_multiple: float = 3.0

    def __post_init__(self) -> None:
        """Validate evaluation controls."""
        object.__setattr__(
            self,
            "local_noise_multiple",
            _positive_float(self.local_noise_multiple, "local_noise_multiple"),
        )


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One canonical parameter and its accepted spellings."""

    canonical_name: str
    symbol: str | None
    aliases: tuple[str, ...]
    description: str
    deprecated_aliases: tuple[str, ...] = ()


PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec(
        "kernel.mode",
        None,
        ("kernel_mode", "calibration"),
        "Manual or calibrated RBF parameterization.",
    ),
    ParameterSpec(
        "kernel.decay_per_median_gap",
        "γ",  # ruff:ignore[ambiguous-unicode-character-string]
        (
            "gamma",
            "calibration_gamma",
            "decay_per_median_gap",
            "kernel.gamma",
            "kernel.decay",
        ),
        "Dimensionless calibrated decay constant.",
        ("calibration_c",),
    ),
    ParameterSpec(
        "kernel.width_in_median_gaps",
        "s",
        (
            "s",
            "calibration_s",
            "width_in_median_gaps",
            "kernel.s",
            "kernel.width_multiple",
        ),
        "Calibrated width in median nearest-neighbor gaps.",
    ),
    ParameterSpec(
        "kernel.relative_variance_floor",
        "λ_rel",
        (
            "lambda_rel",
            "lambda_relative",
            "relative_variance_floor",
            "λrel",
            "kernel.lambda_rel",
            "kernel.relative_floor",
        ),
        "Variance floor relative to kappa.",
    ),
    ParameterSpec(
        "kernel.maximum_variance",
        "κ",
        ("kappa", "maximum_variance", "kernel.kappa"),
        "Manual maximum kernel variance.",
    ),
    ParameterSpec(
        "kernel.distance_decay_rate",
        "η",
        ("eta", "distance_decay_rate", "kernel.eta"),
        "Manual inverse-length decay rate.",
    ),
    ParameterSpec(
        "kernel.variance_floor",
        "λ",
        (
            "lambda",
            "lambda_",
            "lam",
            "variance_floor",
            "kernel.lambda",
            "kernel.lam",
        ),
        "Manual absolute variance floor.",
    ),
    ParameterSpec(
        "kernel.nearest_neighbor_mode",
        None,
        (
            "nearest_neighbor_mode",
            "closest_sample_mode",
            "kernel.closest_sample_mode",
        ),
        "Global or per-dimension nearest-neighbor assignment.",
    ),
    ParameterSpec(
        "membership.mode",
        None,
        ("membership_mode",),
        "Fixed or conformal membership-threshold mode.",
    ),
    ParameterSpec(
        "membership.affinity_threshold",
        "ζ",
        (
            "zeta",
            "affinity_threshold",
            "membership.zeta",
            "membership.threshold",
        ),
        "Fixed inside/outside affinity threshold.",
    ),
    ParameterSpec(
        "membership.target_false_exclusion_rate",
        "ε",
        (
            "epsilon",
            "eps",
            "false_exclusion_rate",
            "target_false_exclusion_rate",
            "membership.epsilon",
        ),
        "Conformal target false-exclusion rate.",
    ),
    ParameterSpec(
        "ood.max_affinity",
        "ξ",
        ("xi", "ood_xi", "max_affinity", "max_ood_affinity", "ood.xi"),
        "Maximum affinity at observed OOD points.",
    ),
    ParameterSpec(
        "ood.covariance_shrink_factor",
        "c",
        (
            "c",
            "covariance_shrink_factor",
            "shrink_factor",
            "ood_shrink_factor",
            "ood.c",
        ),
        "Covariance shrink factor for OOD consistency.",
    ),
    ParameterSpec(
        "ood.max_iterations",
        None,
        ("ood_max_iterations",),
        "Safety cap for OOD adjustment iterations.",
    ),
    ParameterSpec(
        "ood.refresh_interval",
        None,
        ("ood_refresh_interval",),
        "Exact log-survival refresh interval.",
    ),
    ParameterSpec(
        "ood.log_interval",
        None,
        ("ood_log_interval",),
        "OOD progress reporting interval.",
    ),
    ParameterSpec(
        "ood.batch_jump",
        None,
        ("ood_batch_jump",),
        "Enable approximate closed-form OOD shrink jumps.",
    ),
    ParameterSpec(
        "normalization.enabled",
        None,
        ("normalize", "normalize_data"),
        "Whether to normalize inputs before fitting.",
    ),
    ParameterSpec(
        "normalization.method",
        None,
        ("normalization_method",),
        "Input normalization method.",
    ),
    ParameterSpec(
        "normalization.target_range",
        None,
        ("normalization_range",),
        "Target coordinate range.",
    ),
    ParameterSpec(
        "normalization.iqr_factor",
        None,
        ("iqr_factor",),
        "IQR expansion factor.",
    ),
    ParameterSpec(
        "normalization.epsilon",
        None,
        ("normalization_epsilon", "normalization_eps"),
        "Numerical normalization regularizer.",
    ),
    ParameterSpec(
        "evaluation.local_noise_multiple",
        "m",
        (
            "m",
            "local_noise_multiple",
            "local_noise_multiplier",
            "evaluation.m",
        ),
        "Validation sampling displacement in median gaps.",
    ),
)
"""Registry for alias resolution, documentation, and CLI generation."""


def _normalize_alias(value: str) -> str:
    """Normalize an alias without erasing Greek notation.

    Args:
        value (str): User-provided parameter name.

    Returns:
        str: Unicode-normalized alias lookup key.
    """
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"[\s-]+", "_", value)


def _all_spec_names(spec: ParameterSpec) -> tuple[str, ...]:
    names = (spec.canonical_name, *spec.aliases, *spec.deprecated_aliases)
    if spec.symbol is not None:
        names = (*names, spec.symbol)
    return names


_ALIAS_INDEX: dict[str, ParameterSpec] = {}
_DEPRECATED_ALIASES: set[str] = set()
for _spec in PARAMETER_SPECS:
    for _alias in _all_spec_names(_spec):
        _normalized_alias = _normalize_alias(_alias)
        _existing = _ALIAS_INDEX.get(_normalized_alias)
        if _existing is not None and _existing != _spec:  # pragma: no cover
            raise RuntimeError(
                f"parameter alias {_alias!r} maps to multiple canonical fields"
            )
        _ALIAS_INDEX[_normalized_alias] = _spec
    _DEPRECATED_ALIASES.update(
        _normalize_alias(alias) for alias in _spec.deprecated_aliases
    )


def _flatten_mapping(
    mapping: Mapping[str, object],
    prefix: str = "",
) -> list[tuple[str, object]]:
    """Flatten nested configuration mappings into dotted names.

    Args:
        mapping (Mapping[str, object]): Mapping to flatten.
        prefix (str): Prefix inherited from a containing mapping.

    Returns:
        list[tuple[str, object]]: Flattened names and values.

    Raises:
        TypeError: If a parameter name is not a string.
    """
    flattened: list[tuple[str, object]] = []
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError("configuration parameter names must be strings")
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.extend(
                _flatten_mapping(
                    cast("Mapping[str, object]", value),
                    dotted,
                )
            )
        else:
            flattened.append((dotted, value))
    return flattened


def _resolve_mapping(
    mapping: Mapping[str, object],
) -> tuple[dict[str, object], tuple[tuple[str, str], ...]]:
    """Resolve aliases and reject duplicate definitions.

    Args:
        mapping (Mapping[str, object]): User configuration.

    Returns:
        tuple: Canonical values and their original source names.

    Raises:
        ValueError: If a name is unknown or defined more than once.
    """
    resolved: dict[str, object] = {}
    sources: list[tuple[str, str]] = []
    known_names = sorted(_ALIAS_INDEX)
    for raw_name, value in _flatten_mapping(mapping):
        normalized = _normalize_alias(raw_name)
        spec = _ALIAS_INDEX.get(normalized)
        if spec is None:
            suggestions = difflib.get_close_matches(normalized, known_names, n=3)
            suffix = f"; did you mean {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"unknown autoSAFE parameter {raw_name!r}{suffix}")
        canonical = spec.canonical_name
        if canonical in resolved:
            previous = next(source for target, source in sources if target == canonical)
            raise ValueError(f"{raw_name!r} and {previous!r} both define {canonical!r}")
        if normalized in _DEPRECATED_ALIASES:
            warnings.warn(
                f"{raw_name!r} is deprecated; use 'gamma', 'γ', or "  # ruff:ignore[ambiguous-unicode-character-string]
                "'kernel.decay_per_median_gap'",
                DeprecationWarning,
                stacklevel=3,
            )
        resolved[canonical] = value
        sources.append((canonical, raw_name))
    return resolved, tuple(sources)


def _take_prefix(
    values: dict[str, object],
    prefix: str,
) -> dict[str, object]:
    """Remove and return values belonging to a dotted section.

    Args:
        values (dict[str, object]): Canonical values to partition.
        prefix (str): Dotted section prefix.

    Returns:
        dict[str, object]: Values inside the requested section.
    """
    section: dict[str, object] = {}
    dotted = f"{prefix}."
    for key in tuple(values):
        if key.startswith(dotted):
            section[key.removeprefix(dotted)] = values.pop(key)
    return section


@dataclass(frozen=True, slots=True)
class AutoSafeConfig:
    """Complete high-level autoSAFE configuration."""

    membership: MembershipConfig
    kernel: KernelConfig = field(default_factory=CalibratedRBFConfig)
    ood: OODConsistencyConfig | None = None
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    parameter_sources: tuple[tuple[str, str], ...] = field(
        default=(),
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate constraints available before fitting.

        Raises:
            ValueError: If OOD and membership thresholds are
                inconsistent.
        """
        if (
            self.ood is not None
            and isinstance(self.membership, FixedMembershipConfig)
            and self.ood.max_affinity >= self.membership.affinity_threshold
        ):
            raise ValueError(
                "ood.max_affinity (xi) must be below "
                "membership.affinity_threshold (zeta)"
            )

    @classmethod
    def from_mapping(  # ruff:ignore[complex-structure,too-many-branches,too-many-locals,too-many-statements]
        cls,
        mapping: Mapping[str, object],
    ) -> "AutoSafeConfig":
        """Build a configuration from descriptive or paper notation.

        Args:
            mapping (Mapping[str, object]): Flat or nested
                configuration.

        Returns:
            AutoSafeConfig: Validated canonical configuration.

        Raises:
            ValueError: On unknown, conflicting, or incomplete
                parameters.
        """
        values, sources = _resolve_mapping(mapping)

        kernel_values = _take_prefix(values, "kernel")
        raw_mode = str(kernel_values.pop("mode", "")).casefold()
        mode_aliases = {
            "": "",
            "auto": "calibrated",
            "calibrated": "calibrated",
            "manual": "manual",
        }
        if raw_mode not in mode_aliases:
            raise ValueError("kernel.mode must be 'calibrated'/'auto' or 'manual'")
        mode = mode_aliases[raw_mode]
        calibrated_fields = {
            "decay_per_median_gap",
            "width_in_median_gaps",
        }
        manual_fields = {
            "maximum_variance",
            "distance_decay_rate",
            "variance_floor",
            "nearest_neighbor_mode",
        }
        has_calibrated = bool(calibrated_fields & kernel_values.keys())
        has_manual = bool(manual_fields & kernel_values.keys())
        if has_calibrated and has_manual:
            raise ValueError("manual and calibrated kernel parameters cannot be mixed")
        if mode == "manual" and has_calibrated:
            raise ValueError("calibrated parameters cannot be used in manual mode")
        if mode == "calibrated" and has_manual:
            raise ValueError("manual parameters cannot be used in calibrated mode")
        if not mode:
            mode = "manual" if has_manual else "calibrated"
        if (
            mode == "manual"
            and "variance_floor" in kernel_values
            and "relative_variance_floor" in kernel_values
        ):
            raise ValueError(
                "define either lambda/variance_floor or "
                "lambda_rel/relative_variance_floor, not both"
            )

        if mode == "calibrated":
            kernel: KernelConfig = CalibratedRBFConfig(
                **cast("dict[str, Any]", kernel_values)
            )
        else:
            kernel = ManualRBFConfig(**cast("dict[str, Any]", kernel_values))

        membership_values = _take_prefix(values, "membership")
        raw_membership_mode = str(membership_values.pop("mode", "")).casefold()
        if raw_membership_mode not in {"", "auto", "fixed", "conformal"}:
            raise ValueError("membership.mode must be 'fixed', 'conformal', or 'auto'")
        zeta = membership_values.pop("affinity_threshold", None)
        epsilon = membership_values.pop("target_false_exclusion_rate", None)
        if membership_values:  # pragma: no cover - alias registry is exhaustive
            raise ValueError(
                f"unsupported membership fields: {sorted(membership_values)}"
            )
        if zeta is not None and epsilon is not None:
            raise ValueError("define either zeta or epsilon, not both")
        if zeta is None and epsilon is None:
            raise ValueError("membership requires zeta or epsilon")
        inferred_membership_mode = "fixed" if zeta is not None else "conformal"
        if raw_membership_mode not in {
            "",
            "auto",
            inferred_membership_mode,
        }:
            raise ValueError(
                f"membership.mode={raw_membership_mode!r} conflicts with "
                f"{'zeta' if zeta is not None else 'epsilon'}"
            )
        membership: MembershipConfig
        if zeta is not None:
            membership = FixedMembershipConfig(affinity_threshold=cast("float", zeta))
        else:
            membership = ConformalMembershipConfig(
                target_false_exclusion_rate=cast("float", epsilon),
            )

        ood_values = _take_prefix(values, "ood")
        ood: OODConsistencyConfig | None = None
        if ood_values:
            if "max_affinity" not in ood_values:
                raise ValueError("OOD controls require xi/max_ood_affinity")
            ood = OODConsistencyConfig(**cast("dict[str, Any]", ood_values))

        normalization_values = _take_prefix(values, "normalization")
        normalization = NormalizationConfig(
            **cast("dict[str, Any]", normalization_values)
        )

        evaluation_values = _take_prefix(values, "evaluation")
        evaluation = EvaluationConfig(**cast("dict[str, Any]", evaluation_values))

        if values:  # pragma: no cover - alias registry is exhaustive
            raise ValueError(f"unhandled autoSAFE parameters: {sorted(values)}")
        return cls(
            membership=membership,
            kernel=kernel,
            ood=ood,
            normalization=normalization,
            evaluation=evaluation,
            parameter_sources=sources,
        )

    def to_mapping(  # ruff:ignore[complex-structure,too-many-branches]
        self,
        *,
        style: Literal["canonical", "paper"] = "canonical",
    ) -> dict[str, object]:
        """Serialize parameters using canonical or paper names.

        Returns:
            dict[str, object]: JSON-compatible configuration mapping.

        Raises:
            ValueError: If ``style`` is unsupported.
        """
        if style == "paper":
            result: dict[str, object] = {}
            if isinstance(self.kernel, CalibratedRBFConfig):
                result.update({
                    "γ": self.kernel.decay_per_median_gap,  # ruff:ignore[ambiguous-unicode-character-string]
                    "s": self.kernel.width_in_median_gaps,
                    "λ_rel": self.kernel.relative_variance_floor,
                })
            else:
                result.update({
                    "κ": _scale_to_json(self.kernel.maximum_variance),
                    "η": _scale_to_json(self.kernel.distance_decay_rate),
                })
                if self.kernel.variance_floor is not None:
                    result["λ"] = _scale_to_json(self.kernel.variance_floor)
                else:
                    result["λ_rel"] = self.kernel.relative_variance_floor
            if isinstance(self.membership, FixedMembershipConfig):
                result["ζ"] = self.membership.affinity_threshold
            else:
                result["ε"] = self.membership.target_false_exclusion_rate
            if self.ood is not None:
                result["ξ"] = self.ood.max_affinity
                result["c"] = self.ood.covariance_shrink_factor
                result["ood"] = {
                    "max_iterations": self.ood.max_iterations,
                    "refresh_interval": self.ood.refresh_interval,
                    "log_interval": self.ood.log_interval,
                    "batch_jump": self.ood.batch_jump,
                }
            if isinstance(self.kernel, ManualRBFConfig):
                result["nearest_neighbor_mode"] = self.kernel.nearest_neighbor_mode
            result["normalization"] = {
                "enabled": self.normalization.enabled,
                "method": self.normalization.method,
                "target_range": list(self.normalization.target_range),
                "iqr_factor": self.normalization.iqr_factor,
                "epsilon": self.normalization.epsilon,
            }
            result["m"] = self.evaluation.local_noise_multiple
            return result
        if style != "canonical":
            raise ValueError("style must be 'canonical' or 'paper'")

        kernel_mapping: dict[str, object]
        if isinstance(self.kernel, CalibratedRBFConfig):
            kernel_mapping = {
                "mode": "calibrated",
                "decay_per_median_gap": self.kernel.decay_per_median_gap,
                "width_in_median_gaps": self.kernel.width_in_median_gaps,
                "relative_variance_floor": self.kernel.relative_variance_floor,
            }
        else:
            kernel_mapping = {
                "mode": "manual",
                "maximum_variance": _scale_to_json(self.kernel.maximum_variance),
                "distance_decay_rate": _scale_to_json(self.kernel.distance_decay_rate),
                "nearest_neighbor_mode": self.kernel.nearest_neighbor_mode,
            }
            if self.kernel.variance_floor is not None:
                kernel_mapping["variance_floor"] = _scale_to_json(
                    self.kernel.variance_floor
                )
            else:
                kernel_mapping["relative_variance_floor"] = (
                    self.kernel.relative_variance_floor
                )

        membership_mapping: dict[str, object]
        if isinstance(self.membership, FixedMembershipConfig):
            membership_mapping = {
                "mode": "fixed",
                "affinity_threshold": self.membership.affinity_threshold,
            }
        else:
            membership_mapping = {
                "mode": "conformal",
                "target_false_exclusion_rate": (
                    self.membership.target_false_exclusion_rate
                ),
            }
        result = {
            "kernel": kernel_mapping,
            "membership": membership_mapping,
            "normalization": {
                "enabled": self.normalization.enabled,
                "method": self.normalization.method,
                "target_range": list(self.normalization.target_range),
                "iqr_factor": self.normalization.iqr_factor,
                "epsilon": self.normalization.epsilon,
            },
            "evaluation": {
                "local_noise_multiple": self.evaluation.local_noise_multiple,
            },
        }
        if self.ood is not None:
            result["ood"] = {
                "max_affinity": self.ood.max_affinity,
                "covariance_shrink_factor": self.ood.covariance_shrink_factor,
                "max_iterations": self.ood.max_iterations,
                "refresh_interval": self.ood.refresh_interval,
                "log_interval": self.ood.log_interval,
                "batch_jump": self.ood.batch_jump,
            }
        return result


@dataclass(frozen=True, slots=True)
class ResolvedKernelConfig:
    """Concrete kernel values used to construct an ODD."""

    mode: Literal["calibrated", "manual"]
    maximum_variance: ScaleValue
    distance_decay_rate: ScaleValue
    variance_floor: ScaleValue
    median_nearest_neighbor_distance: float | None = None
    decay_per_median_gap: float | None = None
    width_in_median_gaps: float | None = None
    relative_variance_floor: float | None = None


@dataclass(frozen=True, slots=True)
class ResolvedMembershipConfig:
    """Concrete log-space membership decision."""

    mode: Literal["fixed", "conformal"]
    affinity_threshold: float
    log_survival_threshold: float
    target_false_exclusion_rate: float | None = None
    calibration_size: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedAutoSafeConfig:
    """Requested configuration and all data-derived parameter values."""

    requested: AutoSafeConfig
    kernel: ResolvedKernelConfig
    membership: ResolvedMembershipConfig


__all__ = [
    "PARAMETER_SPECS",
    "AutoSafeConfig",
    "CalibratedRBFConfig",
    "ConformalMembershipConfig",
    "EvaluationConfig",
    "FixedMembershipConfig",
    "KernelConfig",
    "ManualRBFConfig",
    "MembershipConfig",
    "NormalizationConfig",
    "OODConsistencyConfig",
    "ParameterSpec",
    "ResolvedAutoSafeConfig",
    "ResolvedKernelConfig",
    "ResolvedMembershipConfig",
    "ScaleValue",
]
