# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Shared threshold/confusion/metric evaluation helpers."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from autosafe import _jax_config  # noqa: F401
from autosafe.tools.evaluate.core import (
    calculate_confusion_matrix,
    calculate_confusion_matrix_log,
    calculate_performance_metrics,
)
from autosafe.typing import AffinityVector


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


def evaluate_affinity_metrics(
    samples_df: pl.DataFrame,
    reference_labels: dict[str, np.ndarray],
    thresholds: np.ndarray | jax.Array,
    source: str,
) -> pl.DataFrame:
    """Evaluate affinity predictions for multiple reference label sets.

    Args:
        samples_df (pl.DataFrame): Table containing an 'affinity' column
            and optionally a 'survival' column (= log(1 - affinity)).
            When 'survival' is present, rows are emitted for both
            affinity_space values ("linear" and "log").
        reference_labels (dict[str, np.ndarray]):
            Mapping of reference name to boolean labels.
        thresholds (np.ndarray | jax.Array): Thresholds applied to
            affinity values.
        source (str): Human-readable source identifier.

    Returns:
        pl.DataFrame: DataFrame with one row per threshold, reference,
            and affinity_space.

    Raises:
        ValueError: If required columns are missing or labels mismatch.
    """
    if "affinity" not in samples_df.columns:
        raise ValueError("samples_df must contain an 'affinity' column")

    has_survival = "survival" in samples_df.columns

    rows: list[dict[str, float | int | str]] = []

    for reference_name, labels in reference_labels.items():
        if len(labels) != samples_df.height:
            raise ValueError(
                "reference label length does not match sample count "
                f"for '{reference_name}'"
            )

        labeled = samples_df.with_columns(
            pl.Series("_reference", labels, dtype=pl.Boolean),
        )
        actually_positive = labeled.filter(pl.col("_reference"))
        actually_negative = labeled.filter(pl.col("_reference") == False)  # noqa: E712

        spaces = ["linear", "log"] if has_survival else ["linear"]
        for space in spaces:
            for threshold in thresholds:
                if space == "linear":
                    confusion = calculate_confusion_matrix(
                        actually_positive=actually_positive,
                        actually_negative=actually_negative,
                        affinity_limit=np.float64(threshold),
                    )
                else:
                    with np.errstate(divide="ignore"):
                        limit = float(np.log1p(-np.float64(threshold)))
                    confusion = calculate_confusion_matrix_log(
                        actually_positive=actually_positive,
                        actually_negative=actually_negative,
                        log_survival_limit=limit,
                    )
                metrics = calculate_performance_metrics(confusion)
                rows.append({
                    "source": source,
                    "reference": reference_name,
                    "affinity_space": space,
                    "affinity_threshold": float(threshold),
                    "true_positive": int(confusion["true_positive"]),
                    "false_positive": int(confusion["false_positive"]),
                    "true_negative": int(confusion["true_negative"]),
                    "false_negative": int(confusion["false_negative"]),
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

    return pl.DataFrame(rows).sort([
        "source",
        "reference",
        "affinity_space",
        "affinity_threshold",
    ])


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
    "evaluate_affinity_metrics",
    "save_metrics_csv",
]
