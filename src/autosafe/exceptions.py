# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Custom exceptions and warnings raised across autoSAFE.

Each class builds its own message, so call sites raise them with the
offending values only. This keeps long message text out of the ``raise``
statements (ruff ``raise-vanilla-args``) and gives callers a specific
type to catch.

This module must not import from the rest of the package: everything
below depends on the standard library only, so any module can import it
without risking a cycle.
"""

from collections.abc import Iterable
from pathlib import Path

# Generic parameter validation. These cover checks that recur across
# modules. Callers pass parameter names, offending values and types
# only; every message is built here.


def _type_names(expected: type | tuple[type, ...]) -> str:
    types = expected if isinstance(expected, tuple) else (expected,)
    return " or ".join(t.__name__ for t in types)


def _shape(shape: Iterable[int | None]) -> str:
    dims = ["*" if d is None else str(d) for d in shape]
    return f"({dims[0]},)" if len(dims) == 1 else f"({', '.join(dims)})"


class UnknownOptionError(ValueError):
    """Raised when a parameter takes a value outside its allowed set."""

    def __init__(
        self, parameter: str, value: object, choices: Iterable[object] = ()
    ) -> None:
        allowed = sorted(map(str, choices))
        suffix = f"; expected one of {allowed}" if allowed else ""
        super().__init__(f"unknown {parameter} {value!r}{suffix}")


class ParameterTypeError(TypeError):
    """Raised when a parameter has the wrong type."""

    def __init__(
        self, parameter: str, value: object, expected: type | tuple[type, ...]
    ) -> None:
        super().__init__(
            f"{parameter} must be of type {_type_names(expected)}, "
            f"got {type(value).__name__}"
        )


class ArrayDtypeError(TypeError):
    """Raised when an array has the wrong dtype."""

    def __init__(self, parameter: str, got: object, expected: object) -> None:
        super().__init__(f"{parameter} must have dtype {expected}, got {got}")


class NonPositiveParameterError(ValueError):
    """Raised when a parameter that must be strictly positive is not."""

    def __init__(self, parameter: str, value: object) -> None:
        super().__init__(f"{parameter} must be greater than 0, got {value!r}")


class ParameterOutOfRangeError(ValueError):
    """Raised when a parameter lies outside its admissible interval.

    The interval is open unless ``lower_closed`` / ``upper_closed`` say
    otherwise.
    """

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        parameter: str,
        value: object,
        lower: float,
        upper: float,
        *,
        lower_closed: bool = False,
        upper_closed: bool = False,
    ) -> None:
        left = "[" if lower_closed else "("
        right = "]" if upper_closed else ")"
        super().__init__(
            f"{parameter} must be in {left}{lower}, {upper}{right}, got {value!r}"
        )


class ParameterOrderError(ValueError):
    """Raised when one parameter must lie below another.

    The order is strict unless ``strict=False``.
    """

    def __init__(self, smaller: str, larger: str, *, strict: bool = True) -> None:
        relation = "strictly less than" if strict else "less than or equal to"
        super().__init__(f"{smaller} must be {relation} {larger}")


class NonFiniteParameterError(ValueError):
    """Raised when a parameter contains NaN or infinite values."""

    def __init__(self, parameter: str) -> None:
        super().__init__(f"{parameter} must contain only finite values")


class EmptyParameterError(ValueError):
    """Raised when a parameter that must hold data is empty."""

    def __init__(self, parameter: str) -> None:
        super().__init__(f"{parameter} must not be empty")


class ShapeMismatchError(ValueError):
    """Raised when an array has the wrong shape.

    ``None`` in ``expected`` matches any extent and prints as ``*``.
    """

    def __init__(
        self,
        parameter: str,
        got: Iterable[int],
        expected: Iterable[int | None],
    ) -> None:
        super().__init__(
            f"{parameter} has shape {_shape(got)}; expected {_shape(expected)}"
        )


class LengthMismatchError(ValueError):
    """Raised when a sequence has the wrong number of entries."""

    def __init__(self, parameter: str, got: int, expected: int) -> None:
        super().__init__(f"{parameter} has {got} entries; expected {expected}")


class ConflictingParametersError(ValueError):
    """Raised when two mutually exclusive parameters are both given."""

    def __init__(self, first: str, second: str) -> None:
        super().__init__(f"define either {first} or {second}, not both")


class MissingParameterError(ValueError):
    """Raised when a required parameter was not given.

    ``unless`` names a parameter whose presence would have made this
    one optional.
    """

    def __init__(self, parameter: str, *, unless: str | None = None) -> None:
        suffix = f" unless {unless} is given" if unless is not None else ""
        super().__init__(f"{parameter} is required{suffix}")


class DeprecatedAliasConflictError(ValueError):
    """Raised when a parameter and its deprecated alias disagree."""

    def __init__(self, parameter: str, alias: str) -> None:
        super().__init__(
            f"{parameter} and deprecated {alias} cannot define different values"
        )


class NotPositiveSemidefiniteError(ValueError):
    """Raised when a matrix must be positive semidefinite but is not."""

    def __init__(self, parameter: str) -> None:
        super().__init__(f"{parameter} must be positive semidefinite")


class TooFewAnchorsError(ValueError):
    """Raised when an operation gets fewer anchors than it needs."""

    def __init__(self, required: int, got: int) -> None:
        super().__init__(f"at least {required} anchors are required, got {got}")


class NotFittedError(RuntimeError, ValueError):
    """Raised when an estimator is used before it was fitted.

    It subclasses both :class:`RuntimeError` and :class:`ValueError`
    because the estimators raised one or the other before this class
    existed, and callers may still catch either.
    """

    def __init__(self, estimator: str) -> None:
        super().__init__(f"{estimator} is not fitted yet; call fit() first")


class InputFileNotFoundError(FileNotFoundError):
    """Raised when an input file or directory does not exist."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"input file not found: {path}")


class NoInputFilesError(FileNotFoundError):
    """Raised when an evaluation is started without any input file."""

    def __init__(self, folder: Path | None = None) -> None:
        where = f" in {folder}" if folder is not None else ""
        super().__init__(f"no input files found{where}")


class UnsupportedFileFormatError(ValueError):
    """Raised when a file extension does not name a supported format."""

    def __init__(self, suffix: str) -> None:
        super().__init__(f"unsupported file format {suffix!r}")


# Specific errors.


class KernelParametersNotSetError(ValueError):
    """Raised when a kernel is used before its parameters are set.

    For the RBF kernel this is the covariance ``sigma`` and its
    inverse; for the Laplacian kernel it is the scale ``alpha``.
    """

    def __init__(self, parameter: str) -> None:
        super().__init__(f"kernel parameter {parameter} has not been set")


class ZeroNearestNeighborDistancesError(ValueError):
    """Raised when every calibration nearest-neighbor distance is 0.

    The bandwidth calibration divides by a median distance, so it
    needs at least one positive distance. All-zero distances mean every
    anchor is an exact duplicate (or, per dimension, every coordinate
    is).
    """

    def __init__(self, *, per_dimension: bool) -> None:
        which = (
            "per-dimension nn-distance medians"
            if per_dimension
            else "full-space nn distances"
        )
        super().__init__(
            f"all {which} are zero; the bandwidth calibration needs at least "
            "one positive distance"
        )


class RowShapeMismatchError(ValueError):
    """Raised when two point sets have different column counts."""

    def __init__(self, n_query_columns: int, n_reference_columns: int) -> None:
        super().__init__(
            f"column mismatch: query has {n_query_columns}, "
            f"reference has {n_reference_columns}"
        )


class OODDimensionMismatchError(ValueError):
    """Raised when OOD points do not match the anchor dimensionality."""

    def __init__(self, n_ood_columns: int, n_anchor_columns: int) -> None:
        super().__init__(
            f"ood_points has {n_ood_columns} columns but the anchors have "
            f"{n_anchor_columns}"
        )


class OODAnchorCoincidenceError(ValueError):
    """Raised when an OOD point coincides exactly with an anchor point.

    The affinity at an anchor is 1 for any covariance, so the OOD
    consistency loop has no reachable exit condition. See
    docs/ood-consistency.md.
    """

    def __init__(self, n_coincident: int, n_ood: int, first_index: int) -> None:
        super().__init__(
            f"{n_coincident} of {n_ood} OOD points coincide exactly with "
            f"anchor points (first at OOD index {first_index}); the affinity "
            "at an anchor is 1 for any covariance, so the OOD consistency "
            "loop cannot converge. Anchors must be drawn from the "
            "in-distribution set only, disjoint from the OOD set. See "
            "docs/ood-consistency.md."
        )


class KernelSaturationError(RuntimeError):
    """Raised when repeated shrinking exhausts the float64 range.

    See docs/ood-consistency.md.
    """

    def __init__(self, kernel_index: int, n_shrinks: int, ood_index: int) -> None:
        super().__init__(
            f"Kernel {kernel_index} saturated after {n_shrinks} shrinks: "
            "sigma underflowed to 0 or its inverse overflowed to inf in "
            "float64, and further iteration would compute 0*inf = NaN. The "
            f"OOD point at index {ood_index} is numerically indistinguishable "
            f"from anchor {kernel_index}. Check that the in-distribution and "
            "OOD sets are disjoint and not merely near-duplicate. See "
            "docs/ood-consistency.md."
        )


class OODConsistencyNotReachedError(RuntimeError):
    """Raised when the adjustment loop hits its iteration cap.

    See docs/ood-consistency.md.
    """

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        max_iterations: int,
        worst_alpha: float,
        xi: float,
        worst_index: int,
        n_adjusted: int,
        n_anchors: int,
    ) -> None:
        xi_n = 1.0 - (1.0 - xi) ** (1.0 / n_anchors)
        super().__init__(
            f"OOD consistency not reached after {max_iterations} iterations "
            f"(max affinity {worst_alpha:.6g} > xi={xi} at OOD index "
            f"{worst_index}; {n_adjusted} of {n_anchors} kernels adjusted). "
            "The worst-case number of adjustments grows with the anchor "
            "count, and for N="
            f"{n_anchors} with a per-kernel budget of {xi_n:.6g} the cap may "
            "simply be too low---raise max_iterations. If instead the "
            "affinity is pinned at 1.0, an OOD point is (near-)coincident "
            "with an anchor. See docs/ood-consistency.md."
        )


class EmptyAnchorPoolError(ValueError):
    """Raised when excluding the OOD rows leaves no anchor points."""

    def __init__(self, dataset_path: Path) -> None:
        super().__init__(
            f"every row of {dataset_path} is in the OOD set; no anchors "
            "remain after excluding it. The anchor set must be the "
            "complement of the OOD set, so these two inputs cannot be "
            "the same data. See docs/ood-consistency.md."
        )


class BaselinesOnlyRequiresCachedODDError(ValueError):
    """Raised when ``baselines_only`` is set without a cached ODD JSON.

    ``baselines_only`` exists to skip the expensive affinity-ODD
    build/calibration step and reuse a previously exported ODD; without
    ``odd_json`` there is nothing to reuse, so proceeding would either
    fail downstream or silently perform the full (expensive) rebuild it
    was meant to avoid.
    """

    def __init__(self, dataset_path: Path) -> None:
        super().__init__(
            f"baselines_only=True requires a cached odd_json for "
            f"{dataset_path}, since its entire purpose is to reuse an "
            "already-built affinity ODD instead of rebuilding one. Pass "
            "the odd_json path written by the original (non-baselines_only) "
            "run."
        )


class BaselinesOnlyCacheMismatchError(ValueError):
    """Raised when the cached ODD does not match the requested kernel.

    Silently refreshing kernels to match would recompute calibration
    and contradict the guarantee ``baselines_only`` exists to provide:
    that the autoSAFE affinity column stays byte-identical to the
    already-reported run.
    """

    def __init__(self, odd_json: Path) -> None:
        super().__init__(
            f"baselines_only=True but the cached ODD at {odd_json} does not "
            "match the requested closest_sample_mode/kernel_type/"
            "kernel_kwargs. Refreshing it would recompute kernel "
            "calibration, contradicting the byte-identical-autoSAFE "
            "guarantee baselines_only exists to provide. Rebuild the ODD "
            "for these kernel settings without baselines_only first, then "
            "point odd_json at the result."
        )


class DeduplicationDimensionMismatchError(ValueError):
    """Raised when a policy and a point array disagree on dimension.

    Covers both a wrong-length ``resolution``/``origin`` vector (a
    partial vector) and a point array whose column count does not match
    the policy. See ``src/autosafe/deduplication.py``.
    """

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(
            f"de-duplication policy is defined for {expected} dimension(s) "
            f"but the input has {got}; resolution, origin and the point "
            "array must all agree on dimensionality. Partial resolution/"
            "origin vectors are rejected rather than silently broadcast or "
            "truncated."
        )


class InvalidResolutionVectorError(ValueError):
    """Raised when a resolution component is not positive and finite.

    The acquisition resolution is a documented assurance input, never a
    tuned hyperparameter, so a zero, negative, ``NaN`` or infinite entry
    is rejected outright rather than silently clamped.
    """

    def __init__(self, index: int, value: float) -> None:
        super().__init__(
            f"resolution[{index}] = {value!r} is not a positive finite "
            "number; every dimension's acquisition resolution must be "
            "documented and strictly positive."
        )


class NonFiniteCoordinateError(ValueError):
    """Raised when de-duplication input has a non-finite coordinate."""

    def __init__(self, n_bad: int, n_total: int, first_index: int) -> None:
        super().__init__(
            f"{n_bad} of {n_total} input points contain a non-finite "
            f"coordinate (first at row {first_index}); de-duplication "
            "requires finite coordinates throughout."
        )


class ResolutionCellOverflowError(ValueError):
    """Raised when a resolution cell index does not fit into int64.

    Cell indices are ``floor((x - origin) / resolution)``. Beyond the
    int64 range the cast wraps, which would silently merge distinct
    points into one cell.
    """

    def __init__(self, dim: int, n_bad: int, resolution: float) -> None:
        super().__init__(
            f"{n_bad} points have a resolution-cell index beyond the int64 "
            f"range in dimension {dim} (|x - origin| / resolution exceeds "
            f"about 9.2e18 with resolution {resolution}); use a coarser "
            "resolution or an origin closer to the data."
        )


class MissingRecordIdentifierError(ValueError):
    """Raised when ``record_ids`` is incomplete or mismatched in length.

    Every input row must resolve to exactly one stable identifier:
    either every row supplies one explicitly, or none do (the row
    index is used instead). A partially supplied sequence, or one
    containing ``None``, is rejected rather than silently patched.
    """

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(
            f"record_ids must supply exactly one identifier per input row "
            f"({expected} expected, {got} usable entries found; a missing "
            "entry counts as unusable); omit record_ids entirely to fall "
            "back to the row index."
        )


class DeduplicationLabelConflictError(ValueError):
    """Raised when an ID/OOD resolution cell coincides, strict mode.

    Cell co-membership between the ID and OOD de-duplication results
    is reported for data-owner adjudication by default; this error is
    raised only when the caller opts into the strict label-conflict
    flag.
    """

    def __init__(self, n_conflicts: int, first_cell: tuple[int, ...]) -> None:
        super().__init__(
            f"{n_conflicts} resolution cell(s) contain both ID and OOD "
            f"representatives (first at cell {first_cell}); this is "
            "reported for data-owner adjudication and raised as an error "
            "only because strict label-conflict checking is enabled."
        )


class ConvexHullError(RuntimeError):
    """Raised when hull creation fails for every Qhull strategy."""

    def __init__(self) -> None:
        super().__init__(
            "Convex hull computation failed for all Qhull options. "
            "Input points are likely degenerate."
        )


class EmptyODDError(ValueError):
    """Raised when evaluating the affinity of an ODD without anchors.

    With no anchors the affinity is undefined in practice, and the ODD
    does not know its dimension, so it cannot even tell which axis of a
    query matrix indexes the points.
    """

    def __init__(self) -> None:
        super().__init__(
            "cannot evaluate an ODD that has no anchor points; add samples "
            "before evaluating it."
        )


class SigmaNotInvertibleError(ValueError):
    """Raised when a kernel's sigma cannot be repaired to be invertible.

    A sigma without a finite inverse is first repaired by flooring its
    eigenvalues at machine epsilon. That repair needs a finite
    eigendecomposition, which fails when every entry of sigma is
    subnormal, e.g. a scalar sigma below the smallest normal float64.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(
            f"the {dim}x{dim} sigma matrix is not invertible and could not "
            "be repaired: its inverse is still not finite after flooring the "
            "eigenvalues. Its entries are likely too small for float64 (e.g. "
            "a scalar sigma below the smallest normal float)."
        )


class NearAnchorOODWarning(UserWarning):
    """Warns that OOD points are numerically on top of an anchor.

    They are not exactly coincident---that raises
    :class:`OODAnchorCoincidenceError`---but they are close enough that
    convergence may need very many iterations.
    """

    def __init__(self, n_saturated: int) -> None:
        super().__init__(
            f"{n_saturated} OOD points have log-survival -inf (affinity "
            "numerically 1.0); they are not exactly coincident with an "
            "anchor but are close enough that convergence may need very "
            "many iterations. See docs/ood-consistency.md."
        )


class LinearAffinityDecisionWarning(UserWarning):
    """Warns that membership decisions threshold the linear affinity.

    The linear affinity rounds to exactly 1.0 in float64 once the
    accumulated kernel mass is large, so thresholds near one cannot
    separate those points and rankings collapse into ties. Decisions
    should use the log-survival rule
    ``log(1 - alpha) <= log(1 - zeta)``.
    """

    def __init__(self, context: str) -> None:
        super().__init__(
            f"{context}: no log-survival values are available, so thresholds "
            "are applied to the linear affinity, which saturates at 1.0 in "
            "float64. Store the survival from Samples.affinity_dual to use the "
            "log-space decision rule. See docs/log-space-affinity.md."
        )


class MissingSpecFieldError(ValueError):
    """Raised when an experiment-spec item lacks a field it needs."""

    def __init__(self, mode: str, field: str) -> None:
        super().__init__(f"{mode} mode requires a non-empty {field!r} field")


class KernelNotDefinedError(ValueError):
    """Raised when a sample has no kernel attached."""

    def __init__(self, index: int | None = None) -> None:
        which = "this sample" if index is None else f"sample {index}"
        super().__init__(f"no kernel is defined for {which}")


class UnsupportedKernelError(TypeError):
    """Raised when an operation supports only the RBF kernel."""

    def __init__(self, index: int, kernel: object) -> None:
        super().__init__(
            f"kernel {index} is a {type(kernel).__name__}; OOD consistency "
            "only supports RBFKernel"
        )


class ClosestSampleNotFoundError(ValueError):
    """Raised when a sample has no nearest neighbor assigned."""

    def __init__(self, sample: object) -> None:
        super().__init__(f"no closest sample found for sample {sample}")


class RangeBoundsUnavailableError(ValueError):
    """Raised when a normalizer has no finite range bounds to report.

    Only fitted min-max and IQR normalizers have range bounds; the
    z-score normalizer is unbounded.
    """

    def __init__(self, method: str) -> None:
        super().__init__(f"range bounds are not available for method {method!r}")


class NeighborQueryError(RuntimeError):
    """Raised when a nearest-neighbor tree query fails.

    The underlying error is attached as ``__cause__``.
    """

    def __init__(self) -> None:
        super().__init__("nearest-neighbor tree query failed")


class EmptyHyperparameterGridError(RuntimeError):
    """Raised when a hyperparameter grid search yields no candidate."""

    def __init__(self) -> None:
        super().__init__("hyperparameter grid search produced no candidate")


class ConfigAliasCollisionError(RuntimeError):
    """Raised at import time when an alias maps to two parameters."""

    def __init__(self, alias: str) -> None:
        super().__init__(f"parameter alias {alias!r} maps to multiple canonical fields")


class UnknownConfigParameterError(ValueError):
    """Raised when a configuration names an unknown parameter."""

    def __init__(self, name: str, suggestions: Iterable[str] = ()) -> None:
        close = list(suggestions)
        suffix = f"; did you mean {', '.join(close)}?" if close else ""
        super().__init__(f"unknown autoSAFE parameter {name!r}{suffix}")


class UnhandledConfigParametersError(ValueError):
    """Raised when resolved parameters are left over after parsing.

    The alias registry is exhaustive, so this indicates a bug in the
    configuration parser rather than in the user's configuration.
    """

    def __init__(self, names: Iterable[str]) -> None:
        super().__init__(f"unhandled autoSAFE parameters: {sorted(names)}")


class DuplicateConfigParameterError(ValueError):
    """Raised when two aliases in a configuration set one parameter."""

    def __init__(self, name: str, previous: str, canonical: str) -> None:
        super().__init__(f"{name!r} and {previous!r} both define {canonical!r}")


class MixedKernelModesError(ValueError):
    """Raised when calibrated and manual kernel parameters are mixed.

    ``mode`` is the explicitly requested kernel mode, or ``None`` when
    no mode was given and both kinds of parameters appear.
    """

    def __init__(self, mode: str | None) -> None:
        if mode is None:
            detail = "manual and calibrated kernel parameters cannot be mixed"
        else:
            other = "manual" if mode == "calibrated" else "calibrated"
            detail = f"{other} parameters cannot be used in {mode} mode"
        super().__init__(detail)


class MembershipModeConflictError(ValueError):
    """Raised when ``membership.mode`` contradicts its threshold."""

    def __init__(self, mode: str, parameter: str) -> None:
        super().__init__(f"membership.mode={mode!r} conflicts with {parameter}")


class InvalidLogSurvivalError(ValueError):
    """Raised when log-survival values contain NaN or ``+inf``.

    ``log(1 - alpha)`` lies in ``[-inf, 0]``; ``-inf`` is a valid
    saturated value, but NaN and ``+inf`` are not.
    """

    def __init__(self) -> None:
        super().__init__("log_survival contains NaN or +inf values")


class CalibrationSetTooSmallError(ValueError):
    """Raised when a conformal calibration set cannot certify a rate.

    The order statistic of rank ``floor(epsilon * (n + 1))`` must exist,
    which needs ``n >= ceil(1 / epsilon) - 1`` held-out points.
    """

    def __init__(self, n_points: int, false_exclusion_rate: float) -> None:
        super().__init__(
            f"calibration set of {n_points} points is too small for the "
            f"requested false-exclusion rate {false_exclusion_rate}"
        )


class UncertifiableThresholdError(ValueError):
    """Raised when conformal calibration yields ``zeta = 0``."""

    def __init__(self) -> None:
        super().__init__(
            "conformal calibration produced zeta=0; no strict membership "
            "threshold is certifiable"
        )


class CalibrationDataRequiredError(ValueError):
    """Raised when conformal membership lacks calibration data."""

    def __init__(self) -> None:
        super().__init__("conformal membership requires held-out calibration_data")


class DuplicateFeatureNameError(ValueError):
    """Raised when ``feature_names`` contains a name twice."""

    def __init__(self, names: Iterable[str]) -> None:
        super().__init__(f"feature_names must be unique, got {list(names)}")


class EmptyFeatureNameError(ValueError):
    """Raised when ``feature_names`` contains an empty name."""

    def __init__(self) -> None:
        super().__init__("feature_names must not contain empty names")


class OODConfigurationMismatchError(ValueError):
    """Raised when OOD data and OOD configuration are not both given.

    ``has_configuration`` tells which of the two is present.
    """

    def __init__(self, *, has_configuration: bool) -> None:
        if has_configuration:
            detail = "OOD consistency configuration requires out_of_distribution_data"
        else:
            detail = "out_of_distribution_data requires OOD consistency configuration"
        super().__init__(detail)


class UnexpectedNoneError(RuntimeError):
    """Raised when an internal value that must be set is ``None``.

    This signals a violated internal invariant, i.e. a bug.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is unexpectedly None")


class NotPositiveDefiniteError(ValueError):
    """Raised when a matrix must be positive definite but is not."""

    def __init__(self, parameter: str) -> None:
        super().__init__(f"{parameter} must be positive definite")


class ParameterDependencyError(ValueError):
    """Raised when a parameter is given without one it depends on."""

    def __init__(self, parameter: str, requires: str) -> None:
        super().__init__(f"{parameter} requires {requires}")


class InvalidIdentifierError(ValueError):
    """Raised when an OpenODD identifier has an invalid form.

    Identifiers start with a letter and contain only letters, digits,
    dots, hyphens, or underscores.
    """

    def __init__(self, parameter: str, value: str) -> None:
        super().__init__(
            f"{parameter} {value!r} must start with a letter and contain only "
            "letters, digits, dots, hyphens, or underscores"
        )


class NegativeIndexError(ValueError):
    """Raised when a feature index is negative or not an integer."""

    def __init__(self, parameter: str, value: object) -> None:
        super().__init__(
            f"{parameter} must contain non-negative integers, got {value!r}"
        )


class DuplicateIndexError(ValueError):
    """Raised when a feature-index sequence repeats an index."""

    def __init__(self, parameter: str, indices: Iterable[int]) -> None:
        super().__init__(
            f"{parameter} must not contain duplicate indices, got {list(indices)}"
        )


class WhitespaceInUnitError(ValueError):
    """Raised when an OpenODD unit contains whitespace."""

    def __init__(self, unit: str) -> None:
        super().__init__(f"unit must not contain whitespace, got {unit!r}")


class ReservedMetadataKeyError(ValueError):
    """Raised when extra export metadata uses a reserved key."""

    def __init__(self, key: str) -> None:
        super().__init__(f"extra metadata must not replace reserved {key!r}")


class DuplicateConceptIdError(ValueError):
    """Raised when OpenODD feature mappings repeat a concept ID."""

    def __init__(self, concept_ids: Iterable[str]) -> None:
        super().__init__(
            f"OpenODD feature concept IDs must be unique, got {list(concept_ids)}"
        )


class FeatureMappingCoverageError(ValueError):
    """Raised when OpenODD features do not cover each dimension once."""

    def __init__(self, covered: Iterable[int], expected: Iterable[int]) -> None:
        super().__init__(
            "OpenODD features must map every model dimension exactly once; "
            f"received {sorted(covered)!r}, expected {list(expected)!r}"
        )


class UnrepresentableODDError(ValueError):
    """Raised when no ODD region fits the supplied feature mapping."""

    def __init__(self) -> None:
        super().__init__(
            "no OpenODD region can be represented by the supplied feature mapping"
        )


class ValueBelowMinimumError(ValueError):
    """Raised when a count or size is below its minimum."""

    def __init__(self, parameter: str, value: object, minimum: object) -> None:
        super().__init__(f"{parameter} must be at least {minimum}, got {value!r}")


class TooManySamplesError(ValueError):
    """Raised when an in-memory evaluation gets too many samples."""

    def __init__(self, n_samples: int, maximum: int) -> None:
        super().__init__(
            f"{n_samples} evaluation samples are too many for in-memory "
            f"evaluation; use at most {maximum} or split the run"
        )


class MissingColumnError(ValueError):
    """Raised when a data frame lacks a required column."""

    def __init__(self, frame: str, column: str) -> None:
        super().__init__(f"{frame} must contain a {column!r} column")


class NoReferenceLabelsError(ValueError):
    """Raised when an evaluation has no reference labels."""

    def __init__(self) -> None:
        super().__init__("no reference labels are available for evaluation")


class AutoCalibrationModeError(ValueError):
    """Raised when automatic calibration is used without global mode.

    Per-dimension 1-D projection distances are a degenerate bandwidth
    scale; see docs/bandwidth-calibration.md.
    """

    def __init__(self, closest_sample_mode: str) -> None:
        super().__init__(
            "calibration: auto requires closest_sample_mode: global, got "
            f"{closest_sample_mode!r} (per-dimension 1D projection distances "
            "are a degenerate bandwidth scale; see "
            "docs/bandwidth-calibration.md)"
        )


class NonPositivePeriodError(ValueError):
    """Raised when a periodic column has a non-positive period."""

    def __init__(self, column: int, period: float) -> None:
        super().__init__(
            f"periodic limit in column {column} needs period > 0, got {period}"
        )


class MissingConfigFieldError(ValueError):
    """Raised when an ODD configuration lacks required fields."""

    def __init__(self, context: str, fields: Iterable[str]) -> None:
        names = ", ".join(repr(name) for name in fields)
        super().__init__(f"{context} configuration must specify {names}")


class ConfigFileNotMappingError(ValueError):
    """Raised when a YAML configuration file does not hold a mapping."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"expected a mapping in YAML config file {path}")


class NotADirectoryParameterError(ValueError):
    """Raised when a parameter must name a directory but does not."""

    def __init__(self, parameter: str, path: Path) -> None:
        super().__init__(f"{parameter} is not a directory: {path}")


class DimensionLengthError(IndexError):
    """Raised when a per-dimension list does not match the dimension.

    A list of length 1 is broadcast; any other length must equal the
    dimension.
    """

    def __init__(self, parameter: str, length: int, dim: int) -> None:
        super().__init__(f"{parameter} has length {length}; expected 1 or dim={dim}")


class PolytopeCreationError(ValueError):
    """Raised when constraints do not define a valid polytope.

    The underlying error is attached as ``__cause__``.
    """

    def __init__(self) -> None:
        super().__init__("failed to create a polytope from the constraints")


class UnknownCategoryValuesError(ValueError):
    """Raised when a categorical column holds unlisted values."""

    def __init__(self, column: str, values: Iterable[object]) -> None:
        super().__init__(
            f"column {column!r} holds values outside its categories: {list(values)}"
        )


__all__ = [
    "ArrayDtypeError",
    "AutoCalibrationModeError",
    "BaselinesOnlyCacheMismatchError",
    "BaselinesOnlyRequiresCachedODDError",
    "CalibrationDataRequiredError",
    "CalibrationSetTooSmallError",
    "ClosestSampleNotFoundError",
    "ConfigAliasCollisionError",
    "ConfigFileNotMappingError",
    "ConflictingParametersError",
    "ConvexHullError",
    "DeduplicationDimensionMismatchError",
    "DeduplicationLabelConflictError",
    "DeprecatedAliasConflictError",
    "DimensionLengthError",
    "DuplicateConceptIdError",
    "DuplicateConfigParameterError",
    "DuplicateFeatureNameError",
    "DuplicateIndexError",
    "EmptyAnchorPoolError",
    "EmptyFeatureNameError",
    "EmptyHyperparameterGridError",
    "EmptyODDError",
    "EmptyParameterError",
    "FeatureMappingCoverageError",
    "InputFileNotFoundError",
    "InvalidIdentifierError",
    "InvalidLogSurvivalError",
    "InvalidResolutionVectorError",
    "KernelNotDefinedError",
    "KernelParametersNotSetError",
    "KernelSaturationError",
    "LengthMismatchError",
    "LinearAffinityDecisionWarning",
    "MembershipModeConflictError",
    "MissingColumnError",
    "MissingConfigFieldError",
    "MissingParameterError",
    "MissingRecordIdentifierError",
    "MissingSpecFieldError",
    "MixedKernelModesError",
    "NearAnchorOODWarning",
    "NegativeIndexError",
    "NeighborQueryError",
    "NoInputFilesError",
    "NoReferenceLabelsError",
    "NonFiniteCoordinateError",
    "NonFiniteParameterError",
    "NonPositiveParameterError",
    "NonPositivePeriodError",
    "NotADirectoryParameterError",
    "NotFittedError",
    "NotPositiveDefiniteError",
    "NotPositiveSemidefiniteError",
    "OODAnchorCoincidenceError",
    "OODConfigurationMismatchError",
    "OODConsistencyNotReachedError",
    "OODDimensionMismatchError",
    "ParameterDependencyError",
    "ParameterOrderError",
    "ParameterOutOfRangeError",
    "ParameterTypeError",
    "PolytopeCreationError",
    "RangeBoundsUnavailableError",
    "ReservedMetadataKeyError",
    "ResolutionCellOverflowError",
    "RowShapeMismatchError",
    "ShapeMismatchError",
    "SigmaNotInvertibleError",
    "TooFewAnchorsError",
    "TooManySamplesError",
    "UncertifiableThresholdError",
    "UnexpectedNoneError",
    "UnhandledConfigParametersError",
    "UnknownCategoryValuesError",
    "UnknownConfigParameterError",
    "UnknownOptionError",
    "UnrepresentableODDError",
    "UnsupportedFileFormatError",
    "UnsupportedKernelError",
    "ValueBelowMinimumError",
    "WhitespaceInUnitError",
    "ZeroNearestNeighborDistancesError",
]
