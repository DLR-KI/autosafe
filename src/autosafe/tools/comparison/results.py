# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Result containers for ODD comparison runs."""

from typing import Any, TypedDict

from autosafe.odd.comparison.base import DecisionBoundary
from autosafe.typing import MethodName


class ComparisonMethodResult(TypedDict):
    """Result for a single comparison method.

    Args:
        method (MethodName): Name of the comparison method used.
        coverage_ratio (float): Proportion of test points covered by the
            method's decision boundary (between 0 and 1).
        conservatism (float): A measure of how conservative the method
            is, with higher values indicating more conservative
            boundaries (between 0 and 1).
        parameters (dict[str, Any]): Dictionary of parameters used for
            the method (e.g., k for knn, n_clusters for kmeans).
        decision_boundary (DecisionBoundary): Detailed information about
            the decision boundary, including type, parameters, coverage
            metrics, and conservatism score.
    """

    method: MethodName
    coverage_ratio: float
    conservatism: float
    parameters: dict[str, Any]
    decision_boundary: DecisionBoundary


class ComparisonSummary(TypedDict):
    """Summary of comparison method evaluation.

    Args:
        most_conservative (str): Name of the method with the highest.
            conservatism score.
        best_coverage (str): Name of the method with the best coverage
            ratio.
        method_count (int): Total number of comparison methods.
            evaluated.
    """

    most_conservative: str
    best_coverage: str
    method_count: int


class ComparisonEvaluationResults(TypedDict):
    """Full comparison evaluation output.

    Args:
        dataset (str): Name of the dataset used for evaluation.
        reference_points (int): Number of reference points.
        test_points (int): Number of test points.
        comparison_methods (dict[str, ComparisonMethodResult]): Mapping
            of method names to their respective results.
        summary (ComparisonSummary): Summary of the evaluation results.
    """

    dataset: str
    reference_points: int
    test_points: int
    comparison_methods: dict[str, ComparisonMethodResult]
    summary: ComparisonSummary


__all__ = [
    "ComparisonEvaluationResults",
    "ComparisonMethodResult",
    "ComparisonSummary",
]
