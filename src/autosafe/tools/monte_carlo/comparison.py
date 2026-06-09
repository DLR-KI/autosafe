# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Compatibility bridge from Monte Carlo path to comparison module.

Comparison ownership is in `autosafe.tools.comparison`. This module
exists only to preserve legacy import paths.
"""

import dataclasses
import pathlib

from autosafe.tools.comparison import (
    COMP_APP,
    ComparisonEvaluationResults,
    _evaluate_comparison_methods,
    create_comparison_test_grid,
)
from autosafe.tools.comparison.core import MethodName
from autosafe.typing import Matrix, NPMatrix


def _create_test_grid(
    ref_points: Matrix | NPMatrix, resolution: int = 50
) -> Matrix | NPMatrix:
    """Compatibility alias for legacy Monte Carlo test-grid function.

    Args:
        ref_points (Matrix | NPMatrix): Reference points to define the
            grid bounds.
        resolution (int): Number of points per dimension in the
            generated grid. Higher values lead to finer grids but
            increased computational cost.

    Returns:
        Matrix | NPMatrix: Generated test grid.
    """
    return create_comparison_test_grid(ref_points=ref_points, resolution=resolution)


@dataclasses.dataclass(frozen=True)
class ComparisonExperimentConfig:
    """Configuration for legacy comparison evaluation.

    Attributes:
        methods (list[str] | None): Optional list of comparison methods
            to evaluate. If None, all methods are evaluated.
        knn_k (int): Number of neighbors for KNN-based method.
        knn_gamma (float): Gamma parameter for KNN-based method.
        kmeans_clusters (int): Number of clusters for KMeans-based
            method.
        density_gamma (float): Gamma parameter for density-based method.
        export_path (str | None): Optional path to export evaluation
            results.
    """

    methods: list[MethodName] | None = None
    knn_k: int = 3
    knn_gamma: float = 0.5
    kmeans_clusters: int = 3
    density_gamma: float = 0.01
    export_path: str | None = None


def evaluate_comparison(
    dataset_path: pathlib.Path,
    request: ComparisonExperimentConfig | None = None,
) -> ComparisonEvaluationResults:
    """Evaluate comparison methods through the comparison core API.

    Args:
        dataset_path (pathlib.Path): Path to input dataset.
        request (ComparisonExperimentConfig | None): Optional comparison
            configuration.

    Returns:
        ComparisonEvaluationResults: Complete comparison evaluation
            result.
    """
    request = request or ComparisonExperimentConfig()
    return _evaluate_comparison_methods(
        dataset_path=dataset_path,
        methods=request.methods,
        knn_k=request.knn_k,
        knn_gamma=request.knn_gamma,
        kmeans_clusters=request.kmeans_clusters,
        density_gamma=request.density_gamma,
        export_path=request.export_path,
    )


__all__ = [
    "COMP_APP",
    "ComparisonExperimentConfig",
    "_create_test_grid",
    "_evaluate_comparison_methods",
    "evaluate_comparison",
]
