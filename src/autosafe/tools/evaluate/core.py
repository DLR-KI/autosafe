# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Generic evaluation helpers used by evaluate, comparison, and.

Monte Carlo flows.
"""

import ast
import re
from pathlib import Path

import numpy as np
import polars as pl
import scipy.spatial
import tqdm.rich

from autosafe.tools.monte_carlo.dicts import (
    ConfusionMatrixDict,
    PerformanceMetricsDict,
)


class ConvexHullError(RuntimeError):
    """Raised when convex hull creation fails for all Qhull.

    strategies.
    """


def calculate_confusion_matrix(
    actually_positive: pl.DataFrame,
    actually_negative: pl.DataFrame,
    affinity_limit: float,
) -> ConfusionMatrixDict:
    """Calculate confusion matrix for a set of samples.

    Args:
        actually_positive (pl.DataFrame): Samples that should be
            positive.
        actually_negative (pl.DataFrame): Samples that should be
            negative.
        affinity_limit (float): Affinity threshold.

    Returns:
        Confusion-matrix counts.
    """
    true_positives = actually_positive.filter(
        pl.col("affinity") >= affinity_limit
    ).height
    false_positives = actually_negative.filter(
        pl.col("affinity") >= affinity_limit
    ).height

    false_negatives = actually_positive.height - true_positives
    true_negatives = actually_negative.height - false_positives

    return ConfusionMatrixDict(
        true_positive=true_positives,
        false_positive=false_positives,
        true_negative=true_negatives,
        false_negative=false_negatives,
    )


def calculate_performance_metrics(
    confusion_matrix: ConfusionMatrixDict,
) -> PerformanceMetricsDict:
    """Calculate core and extended performance metrics.

    Args:
        confusion_matrix (ConfusionMatrixDict): Confusion-matrix counts.

    Returns:
        Derived performance metrics.
    """
    true_positives = confusion_matrix["true_positive"]
    false_positives = confusion_matrix["false_positive"]
    true_negatives = confusion_matrix["true_negative"]
    false_negatives = confusion_matrix["false_negative"]

    total_samples = true_positives + false_positives + true_negatives + false_negatives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    prevalence = (
        (true_positives + false_negatives) / total_samples if total_samples > 0 else 0.0
    )
    accuracy = (
        (true_positives + true_negatives) / total_samples if total_samples > 0 else 0.0
    )

    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2.0
    iou = (
        true_positives / (true_positives + false_positives + false_negatives)
        if (true_positives + false_positives + false_negatives) > 0
        else 0.0
    )
    pr_product = precision * recall if (precision > 0 and recall > 0) else 0.0

    return PerformanceMetricsDict(
        accuracy=accuracy,
        prevalance=prevalence,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        specificity=specificity,
        balanced_accuracy=balanced_accuracy,
        iou=iou,
        pr_product=pr_product,
    )


def create_convex_hull(data: pl.DataFrame) -> scipy.spatial.ConvexHull:
    """Create a convex hull around anchor points.

    Args:
        data (pl.DataFrame): Input dataframe containing anchors.

    Returns:
        scipy.spatial.ConvexHull: The convex hull.

    Raises:
        ConvexHullError: If all Qhull strategies fail.
    """
    try:
        dim = int(data.select(pl.col("config").struct.field("dim")).to_numpy()[0][0])
    except pl.exceptions.StructFieldNotFoundError:
        dim = len(
            data.select(
                pl.col("sampling_results").list[0].struct.field("coordinates"),
            ).to_numpy()[0][0]
        )

    try:
        anchor_points = np.concatenate(
            data.select(pl.col("anchors")).to_numpy()[0][0]
        ).reshape((-1, dim))
    except ValueError:
        matches = re.findall(
            r"array\(\s*(\[[^\]]*\])\s*\)", data["autosafe_odd"][0], flags=re.DOTALL
        )
        anchors = [np.array(ast.literal_eval(m), dtype=float) for m in matches]
        anchor_points = np.array(anchors).reshape((-1, dim))

    anchor_points = np.unique(anchor_points, axis=0)
    hull = None
    qhull_attempts = [None, "QJ", "Qbb Qx", "QJ Qbb"]
    for opt in tqdm.rich.tqdm(qhull_attempts, desc="Trying Convex Hull options"):
        try:  # noqa: PLW0717
            if opt is None:
                hull = scipy.spatial.ConvexHull(anchor_points)
            else:
                hull = scipy.spatial.ConvexHull(anchor_points, qhull_options=opt)
            if hull is not None:
                break
        except scipy.spatial.QhullError:
            hull = None

    if hull is None:
        raise ConvexHullError(
            "Convex hull computation failed for all Qhull options. "
            "Input points are likely degenerate."
        )

    return hull


def process_files(file: str) -> list[Path]:
    """Process a JSON file path or folder of JSON files.

    Args:
        file (str): Path to a JSON file or directory.

    Returns:
        list[Path]: Matching JSON file paths.

    Raises:
        FileNotFoundError: If the directory contains no JSON files.
    """
    files: list[Path] = []
    file_ = Path(file)
    if file_.is_file():
        files.append(file_)
    elif file_.is_dir():
        config_files = list(file_.glob("*.json"))
        if not config_files:
            raise FileNotFoundError(
                f"No configuration files found in the provided folder: {file_}",
            )
        files.extend(config_files)
    return files


__all__ = [
    "ConvexHullError",
    "calculate_confusion_matrix",
    "calculate_performance_metrics",
    "create_convex_hull",
    "process_files",
]
