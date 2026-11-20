# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Shared threshold/confusion/metric evaluation helpers."""

import operator
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from autosafe import _jax_config  # ruff:ignore[unused-import]
from autosafe.exceptions import (
    LengthMismatchError,
    LinearAffinityDecisionWarning,
    MissingColumnError,
    NonPositiveParameterError,
    ParameterOutOfRangeError,
    ShapeMismatchError,
    UnknownOptionError,
    ValueBelowMinimumError,
)
from autosafe.tools.evaluate.core import calculate_performance_metrics
from autosafe.tools.monte_carlo.dicts import ConfusionMatrixDict
from autosafe.typing import AffinityVector

if TYPE_CHECKING:
    import numpy.typing as npt

# Minimum grid depth (in decades) used at both tails of the adaptive
# threshold grid when the observed data does not require going deeper.
# 16 decades comfortably exceeds float64's ~15-16 significant digits.
_MIN_GRID_DEPTH = 16.0
_LN10 = float(np.log(10.0))


def build_affinity_thresholds(mode: str = "linear", count: int = 100) -> AffinityVector:
    """Build affinity threshold grid.

    Args:
        mode (str): Threshold spacing mode ("linear" or "log").
        count (int): Number of threshold values.

    Returns:
        AffinityVector: array of affinity threshold values.

    Raises:
        NonPositiveParameterError: If `count` is not positive.
        UnknownOptionError: If mode is invalid.
    """
    if count <= 0:
        raise NonPositiveParameterError("count", count)
    if mode == "log":
        return jnp.logspace(-3, 0, num=count, base=10.0)
    if mode == "linear":
        return jnp.linspace(0.0, 1.0, num=count)
    raise UnknownOptionError("mode", mode, ("linear", "log"))


_MIN_THRESHOLD_COUNT = 2


def build_threshold_pairs(  # ruff:ignore[too-many-locals]
    count: int,
    affinities: np.ndarray,
    survivals: np.ndarray,
) -> list[tuple[float, float]]:
    """Build the adaptive two-sided (affinity, survival) threshold grid.

    Each entry is an exact analytic pair ``(zeta, S)`` with
    ``S = log1p(-zeta)``, so the log-space decision rule stays exact
    to any depth even once ``zeta`` itself has rounded to 1.0 in
    float64. The grid has two fixed anchors (``zeta=0`` and
    ``zeta=1``/``S=-inf``) plus three adaptive sections: a lower edge
    geometrically approaching 0, a linear middle section over
    ``[0.1, 0.9]``, and an upper edge geometrically approaching 1
    (mirroring the lower edge in survival space). The two edges are
    only as deep as the data requires: the lower edge reaches the
    smallest positive observed affinity, and the upper edge reaches
    the most negative observed survival value (both floored at
    ``_MIN_GRID_DEPTH`` decades).

    Args:
        count (int): Total number of threshold pairs to return
            (includes both anchors).
        affinities (np.ndarray): Observed affinity values, used only to
            calibrate how deep the lower edge must go.
        survivals (np.ndarray): Observed survival values, used only to
            calibrate how deep the upper edge must go.

    Returns:
        list[tuple[float, float]]: ``(affinity_threshold,
            survival_threshold)`` pairs sorted by descending survival
            (equivalently, ascending affinity; ties among rows whose
            affinity rounds to the same float are resolved by the exact
            survival value). May contain fewer than ``count`` entries:
            the lower/middle and middle/upper sections meet at exact
            (bit-identical) ``zeta = 0.1`` and ``zeta = 0.9`` pairs,
            which are de-duplicated.

    Raises:
        ValueBelowMinimumError: If `count` is smaller than 2 (the two
            anchors).
    """
    if count < _MIN_THRESHOLD_COUNT:
        raise ValueBelowMinimumError("count", count, _MIN_THRESHOLD_COUNT)

    affinities = np.asarray(affinities, dtype=float)
    survivals = np.asarray(survivals, dtype=float)

    t_lo = _MIN_GRID_DEPTH
    positive = affinities[affinities > 0.0]
    if positive.size > 0:
        smallest_positive = float(np.min(positive))
        if smallest_positive > 0.0 and np.isfinite(smallest_positive):
            t_lo = max(t_lo, abs(np.log10(smallest_positive)))

    t_hi = _MIN_GRID_DEPTH
    finite_survivals = survivals[np.isfinite(survivals)]
    if finite_survivals.size > 0:
        min_survival = float(np.min(finite_survivals))
        t_hi = max(t_hi, abs(min_survival) / _LN10)

    remaining = count - 2
    n_lower = round(remaining * 0.25)
    n_mid = round(remaining * 0.25)
    n_upper = remaining - n_lower - n_mid

    pairs: list[tuple[float, float]] = [(0.0, 0.0)]

    if n_lower > 0:
        t_lower = np.linspace(1.0, t_lo, num=n_lower)
        zeta_lower = np.power(10.0, -t_lower)
        survival_lower = np.log1p(-zeta_lower)
        pairs.extend(zip(zeta_lower.tolist(), survival_lower.tolist(), strict=True))

    if n_mid > 0:
        zeta_mid = np.linspace(0.1, 0.9, num=n_mid)
        survival_mid = np.log1p(-zeta_mid)
        pairs.extend(zip(zeta_mid.tolist(), survival_mid.tolist(), strict=True))

    if n_upper > 0:
        t_upper = np.linspace(1.0, t_hi, num=n_upper)
        survival_upper = -t_upper * _LN10
        zeta_upper = -np.expm1(survival_upper)
        pairs.extend(zip(zeta_upper.tolist(), survival_upper.tolist(), strict=True))

    with np.errstate(divide="ignore"):
        pairs.append((1.0, float(np.log1p(-1.0))))

    # The lower/middle and middle/upper sections meet at exact
    # (bit-identical) zeta=0.1 and zeta=0.9 pairs; de-duplicate by the
    # exact survival value, which is the finer-grained of the two.
    deduped = list({pair[1]: pair for pair in pairs}.values())

    # Descending survival <=> ascending affinity, but with the full
    # (unrounded) resolution needed to order rows whose printed
    # affinity_threshold has saturated to the same float.
    deduped.sort(key=operator.itemgetter(1), reverse=True)
    return deduped


def _as_threshold_pairs(
    thresholds: np.ndarray | jax.Array | list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Normalize a thresholds argument into (affinity, survival) pairs.

    Accepts either the new pair-grid output (``build_threshold_pairs``)
    or a plain sequence of affinity thresholds
    (``build_affinity_thresholds``, still used by the Monte-Carlo
    results path). Plain thresholds get their survival counterpart
    derived via ``log1p(-zeta)``, matching the previous per-threshold
    behavior exactly.

    Args:
        thresholds (np.ndarray | jax.Array | list[tuple[float, float]]):
            Either affinity thresholds or (affinity, survival) pairs.

    Returns:
        list[tuple[float, float]]: (affinity_threshold,
            survival_threshold) pairs.
    """
    pairs: list[tuple[float, float]] = []
    for value in thresholds:
        if isinstance(value, (tuple, list)) and len(value) == 2:  # ruff:ignore[magic-value-comparison]
            pairs.append((float(value[0]), float(value[1])))
        else:
            affinity_threshold = float(value)
            with np.errstate(divide="ignore"):
                survival_threshold = float(np.log1p(-affinity_threshold))
            pairs.append((affinity_threshold, survival_threshold))
    return pairs


def evaluate_affinity_metrics(  # ruff:ignore[too-many-locals]
    samples_df: pl.DataFrame,
    reference_labels: dict[str, np.ndarray],
    thresholds: np.ndarray | jax.Array | list[tuple[float, float]],
    source: str,
) -> pl.DataFrame:
    """Evaluate affinity predictions for multiple reference label sets.

    Confusion counts are computed with a sort + ``searchsorted`` sweep:
    per reference, the affinity/survival values are split into
    positive/negative masks and sorted once, then every threshold is
    answered with a binary search instead of a full-table scan. This
    is O(refs * n log n) overall, independent of the number of
    thresholds.

    Args:
        samples_df (pl.DataFrame): Table containing an 'affinity' column
            and optionally a 'survival' column (= log(1 - affinity)).
            When 'survival' is present, every decision uses the
            log-space rule and all rows have affinity_space "log".
            Without it, the linear affinity is thresholded (rows
            "linear") and a ``LinearAffinityDecisionWarning`` is
            emitted, since the linear affinity saturates at 1.0.
        reference_labels (dict[str, np.ndarray]):
            Mapping of reference name to boolean labels.
        thresholds (np.ndarray | jax.Array | list[tuple[float, float]]):
            Either plain affinity thresholds (legacy Monte-Carlo path)
            or (affinity, survival) pairs from
            ``build_threshold_pairs``.
        source (str): Human-readable source identifier.

    Returns:
        pl.DataFrame: DataFrame with one row per threshold, reference,
            and affinity_space, including a ``survival_threshold``
            column. Sorted by descending survival within each
            source/reference/affinity_space group so that rows whose
            printed ``affinity_threshold`` has saturated to the same
            float stay in the correct (deepest-last) order.

    Raises:
        MissingColumnError: If the affinity column is missing.
        LengthMismatchError: If a reference label vector does not
            match the sample count.
    """
    if "affinity" not in samples_df.columns:
        raise MissingColumnError("samples_df", "affinity")

    has_survival = "survival" in samples_df.columns
    if not has_survival:
        warnings.warn(
            LinearAffinityDecisionWarning(f"evaluate_affinity_metrics({source})"),
            stacklevel=2,
        )
    threshold_pairs = _as_threshold_pairs(thresholds)

    affinity_all = samples_df["affinity"].to_numpy()
    survival_all = samples_df["survival"].to_numpy() if has_survival else None

    rows: list[dict[str, float | int | str]] = []

    for reference_name, labels in reference_labels.items():
        if len(labels) != samples_df.height:
            raise LengthMismatchError(reference_name, len(labels), samples_df.height)

        labels_arr = np.asarray(labels, dtype=bool)
        affinity_pos = np.sort(affinity_all[labels_arr])
        affinity_neg = np.sort(affinity_all[~labels_arr])
        n_pos = affinity_pos.size
        n_neg = affinity_neg.size

        if survival_all is not None:
            survival_pos = np.sort(survival_all[labels_arr])
            survival_neg = np.sort(survival_all[~labels_arr])

        spaces = ["log"] if has_survival else ["linear"]
        for space in spaces:
            for affinity_threshold, survival_threshold in threshold_pairs:
                if space == "linear":
                    # affinity >= affinity_threshold, ascending array:
                    # count from the leftmost matching index onward.
                    true_positive = n_pos - int(
                        np.searchsorted(affinity_pos, affinity_threshold, side="left")
                    )
                    false_positive = n_neg - int(
                        np.searchsorted(affinity_neg, affinity_threshold, side="left")
                    )
                else:
                    # survival <= survival_threshold, ascending array:
                    # count everything up to (and including) the limit.
                    true_positive = int(
                        np.searchsorted(survival_pos, survival_threshold, side="right")
                    )
                    false_positive = int(
                        np.searchsorted(survival_neg, survival_threshold, side="right")
                    )
                false_negative = n_pos - true_positive
                true_negative = n_neg - false_positive

                confusion = ConfusionMatrixDict(
                    true_positive=true_positive,
                    false_positive=false_positive,
                    true_negative=true_negative,
                    false_negative=false_negative,
                )
                metrics = calculate_performance_metrics(confusion)
                rows.append({
                    "source": source,
                    "reference": reference_name,
                    "affinity_space": space,
                    "affinity_threshold": affinity_threshold,
                    "survival_threshold": survival_threshold,
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "true_negative": true_negative,
                    "false_negative": false_negative,
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1_score": float(metrics["f1_score"]),
                    "specificity": float(metrics["specificity"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "iou": float(metrics["iou"]),
                    "pr_product": float(metrics["pr_product"]),
                    "prevalence": float(metrics["prevalance"]),
                })

    return pl.DataFrame(rows).sort(
        ["source", "reference", "affinity_space", "survival_threshold"],
        descending=[False, False, False, True],
    )


def save_metrics_csv(results: pl.DataFrame, output_path: Path) -> Path:
    """Persist evaluation metrics to CSV.

    Args:
        results (pl.DataFrame): Metrics table.
        output_path (Path): Destination CSV file path.

    Returns:
        Path: The written CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output_path)
    return output_path


@dataclass(frozen=True)
class MembershipAgreement:
    """Agreement of a membership decision with a reference decision.

    A ratio whose denominator is empty counts as 1.0: an empty decision
    makes no wrong claim.

    Attributes:
        precision (float): ``|R and O| / |O|``, the share of the
            other decision's members that the reference also accepts.
        recall (float): ``|R and O| / |R|``, the share of the
            reference's members that the other decision also accepts.
        iou (float): ``|R and O| / |R or O|``.
        n_reference (int): ``|R|``, points the reference accepts.
        n_other (int): ``|O|``, points the other decision accepts.
        n_both (int): ``|R and O|``, points both accept.
    """

    precision: float
    recall: float
    iou: float
    n_reference: int
    n_other: int
    n_both: int


@dataclass(frozen=True)
class SubsetAgreement:
    """Symmetric agreement verdict of two membership decisions.

    Neither decision is ground truth, so each serves once as the
    reference of the other.

    Attributes:
        first_vs_second (MembershipAgreement): The first decision
            against the second as reference.
        second_vs_first (MembershipAgreement): The second decision
            against the first as reference.
        first_precision (float): ``|A and B| / |A|``, the share of the
            first decision's members that the second also accepts
            (``first_vs_second.precision``).
        second_precision (float): ``|A and B| / |B|``
            (``second_vs_first.precision``).
        iou (float): ``|A and B| / |A or B|`` (symmetric).
        tolerance (float): Allowed shortfall of each ratio below 1.
        agree (bool): Whether both precisions and the IoU are at least
            ``1 - tolerance``.
        passed (bool): Same as ``agree``.
    """

    first_vs_second: MembershipAgreement
    second_vs_first: MembershipAgreement
    first_precision: float
    second_precision: float
    iou: float
    tolerance: float
    agree: bool
    passed: bool


def _membership_vector(values: "npt.ArrayLike", name: str) -> "npt.NDArray[np.bool_]":
    """Return a one-dimensional boolean membership vector.

    Args:
        values (npt.ArrayLike): Membership decisions.
        name (str): Parameter name used in error messages.

    Returns:
        npt.NDArray[np.bool_]: The decisions as a boolean vector.

    Raises:
        ShapeMismatchError: If ``values`` is not one-dimensional.
    """
    result = np.asarray(values, dtype=np.bool_)
    if result.ndim != 1:
        raise ShapeMismatchError(name, result.shape, (None,))
    return result


def _ratio(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator``, 1.0 for an empty denominator.

    Args:
        numerator (int): Count in the numerator.
        denominator (int): Count in the denominator.

    Returns:
        float: The ratio, or 1.0 if ``denominator`` is 0.
    """
    return numerator / denominator if denominator > 0 else 1.0


def membership_agreement(
    reference: "npt.ArrayLike",
    other: "npt.ArrayLike",
) -> MembershipAgreement:
    """Precision, recall and IoU of a decision against a reference.

    Both arguments are membership decisions over the same points.
    Unlike ``calculate_performance_metrics`` (which reports 0.0), an
    empty denominator counts as 1.0.

    Args:
        reference (npt.ArrayLike): Shape (n,) reference decisions.
        other (npt.ArrayLike): Shape (n,) decisions to compare.

    Returns:
        MembershipAgreement: Ratios and the underlying counts.

    Raises:
        LengthMismatchError: If the decisions differ in length.
    """
    ref = _membership_vector(reference, "reference")
    oth = _membership_vector(other, "other")
    if ref.shape != oth.shape:
        raise LengthMismatchError("other", oth.shape[0], ref.shape[0])
    n_reference = int(np.count_nonzero(ref))
    n_other = int(np.count_nonzero(oth))
    n_both = int(np.count_nonzero(ref & oth))
    n_either = n_reference + n_other - n_both
    return MembershipAgreement(
        precision=_ratio(n_both, n_other),
        recall=_ratio(n_both, n_reference),
        iou=_ratio(n_both, n_either),
        n_reference=n_reference,
        n_other=n_other,
        n_both=n_both,
    )


def subset_agreement(
    first: "npt.ArrayLike",
    second: "npt.ArrayLike",
    tolerance: float = 0.005,
) -> SubsetAgreement:
    """Decide whether two membership decisions agree within a tolerance.

    The decisions agree when both precisions (each decision measured
    against the other) and the IoU are at least ``1 - tolerance``.
    Empty denominators count as 1.0, as in :func:`membership_agreement`.

    Args:
        first (npt.ArrayLike): Shape (n,) decisions.
        second (npt.ArrayLike): Shape (n,) decisions over the same
            points.
        tolerance (float): Allowed shortfall below 1, in [0, 1].

    Returns:
        SubsetAgreement: Both directions, both precisions, the IoU and
            the verdict.

    Raises:
        ParameterOutOfRangeError: If ``tolerance`` is outside [0, 1].
    """
    if not 0.0 <= tolerance <= 1.0:
        raise ParameterOutOfRangeError(
            "tolerance", tolerance, 0.0, 1.0, lower_closed=True, upper_closed=True
        )
    first_vs_second = membership_agreement(second, first)
    second_vs_first = membership_agreement(first, second)
    verdict = (
        min(first_vs_second.precision, second_vs_first.precision, second_vs_first.iou)
        >= 1.0 - tolerance
    )
    return SubsetAgreement(
        first_vs_second=first_vs_second,
        second_vs_first=second_vs_first,
        first_precision=first_vs_second.precision,
        second_precision=second_vs_first.precision,
        iou=second_vs_first.iou,
        tolerance=tolerance,
        agree=verdict,
        passed=verdict,
    )


__all__ = [
    "MembershipAgreement",
    "SubsetAgreement",
    "build_affinity_thresholds",
    "build_threshold_pairs",
    "evaluate_affinity_metrics",
    "membership_agreement",
    "save_metrics_csv",
    "subset_agreement",
]
