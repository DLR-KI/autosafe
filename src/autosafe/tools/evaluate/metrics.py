# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Shared threshold/confusion/metric evaluation helpers."""

import operator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from autosafe import _jax_config  # ruff:ignore[unused-import]
from autosafe.tools.evaluate.core import calculate_performance_metrics
from autosafe.tools.monte_carlo.dicts import ConfusionMatrixDict
from autosafe.typing import AffinityVector

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
        ValueError: If `count` is not positive or mode is invalid.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if mode == "log":
        return jnp.logspace(-3, 0, num=count, base=10.0)
    if mode == "linear":
        return jnp.linspace(0.0, 1.0, num=count)
    raise ValueError("mode must be 'linear' or 'log'")


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
        ValueError: If `count` is smaller than 2 (the two anchors).
    """
    if count < _MIN_THRESHOLD_COUNT:
        raise ValueError("count must be at least 2 to hold both anchors")

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
            When 'survival' is present, rows are emitted for both
            affinity_space values ("linear" and "log").
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
        ValueError: If required columns are missing or labels mismatch.
    """
    if "affinity" not in samples_df.columns:
        raise ValueError("samples_df must contain an 'affinity' column")

    has_survival = "survival" in samples_df.columns
    threshold_pairs = _as_threshold_pairs(thresholds)

    affinity_all = samples_df["affinity"].to_numpy()
    survival_all = samples_df["survival"].to_numpy() if has_survival else None

    rows: list[dict[str, float | int | str]] = []

    for reference_name, labels in reference_labels.items():
        if len(labels) != samples_df.height:
            raise ValueError(
                "reference label length does not match sample count "
                f"for '{reference_name}'"
            )

        labels_arr = np.asarray(labels, dtype=bool)
        affinity_pos = np.sort(affinity_all[labels_arr])
        affinity_neg = np.sort(affinity_all[~labels_arr])
        n_pos = affinity_pos.size
        n_neg = affinity_neg.size

        if survival_all is not None:
            survival_pos = np.sort(survival_all[labels_arr])
            survival_neg = np.sort(survival_all[~labels_arr])

        spaces = ["linear", "log"] if has_survival else ["linear"]
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
        The written CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output_path)
    return output_path


__all__ = [
    "build_affinity_thresholds",
    "build_threshold_pairs",
    "evaluate_affinity_metrics",
    "save_metrics_csv",
]
