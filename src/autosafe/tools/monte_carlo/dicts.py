# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Dicts module for the Monte Carlo sampling."""

from pathlib import Path
from typing import Any, Literal, TypedDict

from autosafe.samples import Samples
from autosafe.typing import BoundSpec, KernelType, NPMatrix, NPVector, Vector


class KernelConfig(TypedDict):
    """Configuration for a kernel.

    Args:
        type (KernelType): The type of the kernel.
        params (dict): The parameters for the kernel.
    """

    type: KernelType
    params: dict[str, Any]


class MonteCarloConfig(TypedDict):
    """Settings for the Monte Carlo sampling.

    Args:
        dim (int): Number of dimensions.
        odd_type (Literal["box"]): Polytope type of the ODD. All
            shapes other than 'box' are not implemented yet.
        odd_lower_limits (BoundSpec): Lower limits of the ODD.
        odd_upper_limits (BoundSpec): Upper limits of the ODD.
        box_lower_limits (BoundSpec): Lower limits of the sampling box.
        box_upper_limits (BoundSpec): Upper limits of the sampling box.
        odd_anchors (int): Number of anchors to use for the autoSAFE
            ODD.
        kernel_config (KernelConfig): Configuration of the kernel to
            use.
        samples (int): Number of samples to draw.
        filename (Path): Path to the output file.
        custom_odd_config (str | dict | None): Optional custom ODD
            configuration. Can be a path to YAML file or inline
            configuration dict for polytope definitions.
    """

    dim: int
    odd_type: Literal["box"]
    odd_lower_limits: BoundSpec
    odd_upper_limits: BoundSpec
    box_lower_limits: BoundSpec
    box_upper_limits: BoundSpec
    odd_anchors: int
    kernel_config: KernelConfig
    samples: int
    filename: Path
    custom_odd_config: str | dict[str, Any] | None


class PolytopeDict(TypedDict):
    """Dictionary representation of a polytope.

    Args:
        A (NPMatrix): The polytope's half-space matrix.
        b (NPVector): The polytope's half-space vector.
    """

    A: NPMatrix
    b: NPVector


class RegionDict(TypedDict):
    """Dictionary representation of a polytope region.

    Args:
        list_poly (list[PolytopeDict]): List of polytopes defining the
            region.
    """

    list_poly: list[PolytopeDict]


class BoxBounds(TypedDict):
    """Bounds of the sampling box.

    Args:
        lower_bounds (Vector): Lower bounds of the box.
        upper_bounds (Vector): Upper bounds of the box.
    """

    lower_bounds: Vector
    upper_bounds: Vector


class ODDBounds(TypedDict):
    """Bounds of the ODD box.

    Args:
        lower_bounds (Vector): Lower bounds of the ODD.
        upper_bounds (Vector): Upper bounds of the ODD.
    """

    lower_bounds: Vector
    upper_bounds: Vector


class SamplingResult(TypedDict):
    """Result of a single Monte Carlo sample.

    Args:
        coordinates (Vector): Coordinates of the sample.
        in_odd (bool): Whether the sample is inside the ODD.
        affinity (float): Affinity of the sample.
    """

    coordinates: Vector
    in_odd: bool
    affinity: float


class ResultStats(TypedDict):
    """Statistics of the Monte Carlo sampling.

    This includes not only the results of the sampling, but also the
    definitions of the sampling box and the ODD.

    Args:
        box (RegionDict): The sampling box.
        odd (RegionDict): The ODD box.
        anchors (list): The anchor points used for the autoSAFE ODD.
        config (MonteCarloConfig): Configuration of the Monte Carlo
            sampling.
        autosafe_odd (Samples): The autoSAFE ODD.
        total_samples (int): Total number of samples drawn.
        sampling_results (list[SamplingResult]): List of sampling
            results.
    """

    box: RegionDict  # BoxBounds
    odd: RegionDict  # ODDBounds
    anchors: list[Any]
    config: MonteCarloConfig
    autosafe_odd: Samples
    total_samples: int
    sampling_results: list[SamplingResult]


class PerformanceMetricsDict(TypedDict):
    """Performance metrics for Monte Carlo sampling evaluation.

    Args:
        accuracy (float): The accuracy of the sampling.
        prevalance (float): The prevalence of positive samples.
        precision (float): The precision of the sampling.
        recall (float): The recall of the sampling.
        f1_score (float): The F1 score (harmonic mean of precision and
            recall).
        specificity (float): The specificity/true negative rate.
        balanced_accuracy (float): The balanced accuracy score.
        iou (float): The Intersection-over-Union score.
        pr_product (float): The point-wise product of precision and
            recall at this threshold (precision x recall). This is NOT
            a PR-AUC; compute the true AUC from the precision/recall
            columns using the trapezoidal rule.
    """

    accuracy: float
    prevalance: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    balanced_accuracy: float
    iou: float
    pr_product: float


class ConfusionMatrixDict(TypedDict):
    """Confusion matrix for Monte Carlo sampling evaluation.

    Args:
        true_positive (int): Number of true positive samples.
        false_positive (int): Number of false positive samples.
        true_negative (int): Number of true negative samples.
        false_negative (int): Number of false negative samples.
    """

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


class EvaluationResultsDict(TypedDict):
    """Data class for evaluation results.

    Args:
        file (str): The name of the evaluated file.
        lim (float): The affinity limit used for evaluation.
        total_samples (int): Total number of samples evaluated.
        confusion_matrix_odd (ConfusionMatrixDict): Confusion matrix
            for the ODD evaluation.
        confusion_matrix_hull (ConfusionMatrixDict): Confusion matrix
            for the hull evaluation.
        performance_odd (PerformanceMetricsDict): Performance metrics
            for the ODD evaluation.
        performance_hull (PerformanceMetricsDict): Performance metrics
            for the hull evaluation.
    """

    file: str
    lim: float
    total_samples: int
    confusion_matrix_odd: ConfusionMatrixDict
    confusion_matrix_hull: ConfusionMatrixDict
    performance_odd: PerformanceMetricsDict
    performance_hull: PerformanceMetricsDict
