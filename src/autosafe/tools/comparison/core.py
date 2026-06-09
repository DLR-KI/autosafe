# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Core comparison evaluation logic.

This module owns all comparison-related logic, including dataset setup,
test-grid generation, method evaluation, and result dict/dataframe
creation.
"""

import json
import operator
import pathlib
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import polars as pl
import scipy.spatial
from scipy.stats import qmc

from autosafe.odd.comparison import (
    ClusteredConvexHulls,
    ClusteredSuperlevelSetMonitor,
    DBSCANCluster,
    KMeansBoundaries,
    KNNMonitor,
    SuperlevelSetMonitor,
)
from autosafe.odd.comparison.base import DecisionBoundary
from autosafe.tools.experiments.utils import load_dataset
from autosafe.typing import Matrix, NPMatrix

MethodName = Literal[
    "hull_single",
    "knn",
    "kmeans",
    "density_single",
    "hull_clustered",
    "density_clustered",
    "dbscan_cluster",
]


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


class KNNMethodKwargs(TypedDict, total=False):
    """Keyword arguments for KNN comparison method.

    Args:
        k (int): Number of nearest neighbors.
        gamma (float | None): Distance threshold for membership (if
            None, auto-detected).
        metric (str): Distance metric for KDTree (default: "euclidean").
        leaf_size (int): KDTree optimization parameter (default: 40).
    """

    k: int
    gamma: np.float64 | None
    metric: str
    leaf_size: int


class KMeansMethodKwargs(TypedDict, total=False):
    """Keyword arguments for k-means comparison method.

    Args:
        n_clusters (int): Number of clusters for k-means.
        metric (str): Distance metric for clustering (default:
            "euclidean").
        min_cluster_size (int): Minimum cluster size to consider for
            boundary construction.
    """

    n_clusters: int
    metric: str
    min_cluster_size: int


class DensityMethodKwargs(TypedDict, total=False):
    """Keyword arguments for density-based comparison method.

    Args:
        gamma (float): Density threshold for superlevel set membership.
        bandwidth (Literal["scott", "silverman"] | None): Bandwidth
            method for KDE (None for auto).
        sigmoid_weight (float): Weighting factor for sigmoid
            transformation of density scores.
    """

    gamma: np.float64
    bandwidth: Literal["scott", "silverman"] | None
    sigmoid_weight: np.float64


class ClusteredHullMethodKwargs(TypedDict, total=False):
    """Keyword arguments for clustered convex hull comparison method.

    Args:
        n_clusters (int): Number of clusters for partitioning reference
            points before hull construction.
        method (Literal["kmeans", "dbscan"]): Clustering method to use.
            for partitioning.
    """

    n_clusters: int
    method: Literal["kmeans", "dbscan"]


class ClusteredDensityMethodKwargs(TypedDict, total=False):
    """Keyword arguments for clustered density comparison method.

    Args:
        n_clusters (int): Number of clusters for partitioning reference
            points before density estimation.
        gamma (float): Density threshold for superlevel set membership.
        bandwidth (Literal["scott", "silverman"] | None): Bandwidth
            method for KDE (None for auto).
        min_cluster_size (int): Minimum cluster size to consider for.
            boundary construction.
    """

    n_clusters: int
    gamma: np.float64
    bandwidth: Literal["scott", "silverman"] | None
    min_cluster_size: int


class DBSCANMethodKwargs(TypedDict, total=False):
    """Keyword arguments for DBSCAN-based comparison method.

    Args:
        eps (np.float64): Maximum distance between two samples for them
            to be considered as in the same neighborhood.
        min_samples (int): Minimum number of samples in a neighborhood
            for a point to be considered as a core point.
    """

    eps: np.float64
    min_samples: int


def _hull_membership(
    reference_points: Matrix | NPMatrix,
    test_points: Matrix | NPMatrix,
) -> NPMatrix:
    """Compute single convex-hull membership for test points.

    Args:
        reference_points (Matrix | NPMatrix):
            (N_features, n_reference_samples).
            array of reference points.
        test_points (Matrix | NPMatrix): (N_features, n_test_samples)
            array of test points.

    Returns:
        NPMatrix: Boolean array indicating membership of each test
            point.
    """
    hull = scipy.spatial.ConvexHull(reference_points.T)
    a_hull, b_hull = hull.equations[:, :-1], hull.equations[:, -1]
    eps = np.finfo(float).eps
    return np.all(test_points.T @ a_hull.T + b_hull.T <= eps, axis=1)  # noqa: SIM300


def setup_evaluation_framework(dataset_path: str | Path, **kwargs: object) -> dict:
    """Set up evaluation framework for comparison methods.

    Args:
        dataset_path (str | Path): Path to dataset file.
        kwargs (object): Additional configuration values to include in
            the returned configuration mapping.

    Returns:
        Dictionary with dataset metadata, loaded frame, and references.
    """
    df, dataset_type = load_dataset(Path(dataset_path))
    ref_points = df.to_numpy().T

    return {
        "dataset_type": dataset_type,
        "dataframe": df,
        "reference_points": ref_points,
        "config": kwargs,
    }


def create_comparison_test_grid(
    ref_points: Matrix | NPMatrix, resolution: int = 50
) -> Matrix | NPMatrix:
    """Create Sobol test grid in (n_features, n_samples) layout.

    Args:
        ref_points (Matrix | NPMatrix): Reference points shaped as
            (n_features, n_reference_samples).
        resolution (int): Number of test points requested.

    Returns:
        Matrix | NPMatrix: Sobol test points shaped as.
            (n_features, n_test_samples).
    """
    lower = ref_points.min(axis=1) - 0.05 * (
        ref_points.max(axis=1) - ref_points.min(axis=1)
    )
    upper = ref_points.max(axis=1) + 0.05 * (
        ref_points.max(axis=1) - ref_points.min(axis=1)
    )

    dim = ref_points.shape[0]
    sobol_engine = qmc.Sobol(dim, scramble=True)

    sobol_points = 2 ** int(np.ceil(np.log2(resolution)))
    test_points = sobol_engine.random_base2(sobol_points)

    if sobol_points > resolution:
        indices = np.linspace(0, sobol_points - 1, resolution, dtype=int)
        test_points = test_points[indices]

    test_points = np.array(lower + (upper - lower) * test_points)
    return test_points.T


def evaluate_dataset_with_comparison_methods(
    ref_points: Matrix | NPMatrix,
    test_points: Matrix | NPMatrix,
    method: MethodName = "knn",
    **method_kwargs: object,
) -> ComparisonMethodResult:
    """Evaluate a single comparison method.

    Args:
        ref_points (Matrix | NPMatrix):
            (N_features, n_reference_samples) array of reference points.
        test_points (Matrix | NPMatrix): (N_features, n_test_samples)
            array of test points.
        method (MethodName): Comparison method to evaluate.
        method_kwargs (object): Additional keyword arguments for the
            selected method, such as k or n_clusters.

    Returns:
        Typed comparison method result with coverage, conservatism,
        parameters, and decision-boundary metadata.

    Raises:
        ValueError: If an unknown method is specified.
    """
    if method == "hull_single":
        method_results = _hull_membership(ref_points, test_points)
        coverage_ratio = method_results.mean()
        return ComparisonMethodResult(
            method=method,
            coverage_ratio=float(coverage_ratio),
            conservatism=float(1.0 - coverage_ratio),
            parameters={},
            decision_boundary=DecisionBoundary(
                type="hull_single",
                parameters={},
                coverage={"coverage_ratio": np.float64(coverage_ratio)},
                conservatism=np.float64(1.0 - coverage_ratio),
            ),
        )

    if method == "knn":
        knn_kwargs = cast("KNNMethodKwargs", method_kwargs)
        monitor = KNNMonitor(
            k=knn_kwargs.get("k", 3),
            gamma=knn_kwargs.get("gamma"),
            metric=knn_kwargs.get("metric", "euclidean"),
            leaf_size=knn_kwargs.get("leaf_size", 40),
        )
    elif method == "kmeans":
        kmeans_kwargs = cast("KMeansMethodKwargs", method_kwargs)
        monitor = KMeansBoundaries(
            n_clusters=kmeans_kwargs.get("n_clusters", 3),
            metric=kmeans_kwargs.get("metric", "euclidean"),
            min_cluster_size=kmeans_kwargs.get("min_cluster_size", 10),
        )
    elif method in {"density", "density_single"}:
        density_kwargs = cast("DensityMethodKwargs", method_kwargs)
        monitor = SuperlevelSetMonitor(
            gamma=density_kwargs.get("gamma", np.float64(0.01)),
            bandwidth=density_kwargs.get("bandwidth", "scott"),
            sigmoid_weight=density_kwargs.get("sigmoid_weight", np.float64(1.0)),
        )
    elif method == "hull_clustered":
        hull_kwargs = cast("ClusteredHullMethodKwargs", method_kwargs)
        monitor = ClusteredConvexHulls(
            n_clusters=hull_kwargs.get("n_clusters", 3),
            method=hull_kwargs.get("method", "kmeans"),
        )
    elif method == "density_clustered":
        clustered_density_kwargs = cast("ClusteredDensityMethodKwargs", method_kwargs)
        monitor = ClusteredSuperlevelSetMonitor(
            n_clusters=clustered_density_kwargs.get("n_clusters", 3),
            gamma=clustered_density_kwargs.get("gamma", np.float64(0.01)),
            bandwidth=clustered_density_kwargs.get("bandwidth", "scott"),
            min_cluster_size=clustered_density_kwargs.get("min_cluster_size", 10),
        )
    elif method == "dbscan_cluster":
        dbscan_kwargs = cast("DBSCANMethodKwargs", method_kwargs)
        monitor = DBSCANCluster(
            eps=dbscan_kwargs.get("eps", np.float64(0.5)),
            min_samples=dbscan_kwargs.get("min_samples", 5),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    monitor.fit(ref_points)

    method_results = monitor.evaluate_batch(test_points)
    coverage_ratio = method_results.mean()
    conservatism_fn = getattr(monitor, "compute_conservatism_metric", None)
    conservatism = conservatism_fn() if callable(conservatism_fn) else 0.0

    return ComparisonMethodResult(
        method=method,
        coverage_ratio=float(coverage_ratio),
        conservatism=float(conservatism),
        parameters=method_kwargs,
        decision_boundary=monitor.decision_boundary,
    )


def _find_most_conservative(results: dict) -> str:
    """Find most conservative method based on conservatism score.

    Args:
        results (dict): Mapping of method names to method result
            dictionaries.

    Returns:
        str: Summary for the most conservative method.
    """
    conservatism_scores = {
        method_name: method_data.get("conservatism", 0)
        for method_name, method_data in results.items()
    }

    if not conservatism_scores:
        return "unknown"

    most_conservative = max(conservatism_scores.items(), key=operator.itemgetter(1))
    return f"{most_conservative[0]} (conservatism: {most_conservative[1]:.4f})"


def _find_best_coverage(results: dict) -> str:
    """Find method with best coverage.

    Args:
        results (dict): Mapping of method names to method result
            dictionaries.

    Returns:
        str: Summary for the best coverage method.
    """
    coverage_scores = {
        method_name: method_data.get("coverage_ratio", 0)
        for method_name, method_data in results.items()
    }

    if not coverage_scores:
        return "unknown"

    best_coverage = max(coverage_scores.items(), key=operator.itemgetter(1))
    return f"{best_coverage[0]} (coverage: {best_coverage[1]:.4f})"


def build_comparison_results_dataframe(results: dict) -> pl.DataFrame:
    """Build a comparison summary dataframe from result dictionary.

    Args:
        results (dict): Full comparison result dictionary.

    Returns:
        pl.DataFrame: Dataframe containing one row per method.
    """
    rows: list[dict[str, Any]] = []
    for method_name, method_data in results.get("comparison_methods", {}).items():
        rows.append({
            "dataset": results.get("dataset"),
            "method": method_name,
            "coverage_ratio": method_data.get("coverage_ratio"),
            "conservatism": method_data.get("conservatism"),
            "parameters": method_data.get("parameters", {}),
        })
    return pl.DataFrame(rows)


def evaluate_comparison_methods(  # noqa: C901 PLR0913 PLR0917
    dataset_path: pathlib.Path,
    methods: list[MethodName] | None = None,
    knn_k: int = 3,
    knn_gamma: float = 0.5,
    kmeans_clusters: int = 3,
    density_gamma: float = 0.01,
    export_path: str | None = None,
) -> ComparisonEvaluationResults:
    """Evaluate comparison methods and return a consolidated result.

    Args:
        dataset_path (pathlib.Path): Path to dataset file.
        methods (list[MethodName] | None): Optional list of methods to
            evaluate.
        knn_k (int): Number of nearest neighbors for KNN.
        knn_gamma (float): Threshold value for KNN monitor.
        kmeans_clusters (int): Number of clusters for k-means variants.
        density_gamma (float): Density threshold for superlevel methods.
        export_path (str | None): Optional JSON export path.

    Returns:
        Complete comparison evaluation result dictionary.

    Raises:
        ValueError: If an unknown comparison method is requested.
    """
    if methods is None:
        methods = [
            "hull_single",
            "knn",
            "kmeans",
            "density_single",
            "hull_clustered",
            "density_clustered",
            "dbscan_cluster",
        ]

    eval_framework = setup_evaluation_framework(dataset_path)
    ref_points = eval_framework["reference_points"]
    test_points = create_comparison_test_grid(ref_points, resolution=50)

    comparison_results: dict[str, ComparisonMethodResult] = {}
    for method in methods:
        if method == "hull_single":
            method_results = evaluate_dataset_with_comparison_methods(
                ref_points,
                test_points,
                method,
            )
        elif method == "knn":
            method_results = evaluate_dataset_with_comparison_methods(
                ref_points, test_points, method, k=knn_k, gamma=knn_gamma
            )
        elif method == "kmeans":
            method_results = evaluate_dataset_with_comparison_methods(
                ref_points, test_points, method, n_clusters=kmeans_clusters
            )
        elif method in {"density", "density_single"}:
            method_results = evaluate_dataset_with_comparison_methods(
                ref_points,
                test_points,
                "density_single",
                gamma=density_gamma,
            )
        elif method == "hull_clustered":
            method_results = evaluate_dataset_with_comparison_methods(
                ref_points,
                test_points,
                method,
                n_clusters=kmeans_clusters,
            )
        elif method == "density_clustered":
            method_results = evaluate_dataset_with_comparison_methods(
                ref_points,
                test_points,
                method,
                n_clusters=kmeans_clusters,
                gamma=density_gamma,
            )
        elif method == "dbscan_cluster":
            method_results = evaluate_dataset_with_comparison_methods(
                ref_points,
                test_points,
                method,
            )
        else:
            raise ValueError(f"Unknown comparison method: {method}")

        comparison_results[method_results["method"]] = method_results

    total_results: ComparisonEvaluationResults = {
        "dataset": str(dataset_path),
        "reference_points": ref_points.shape[0],
        "test_points": test_points.shape[0],
        "comparison_methods": comparison_results,
        "summary": {
            "most_conservative": _find_most_conservative(comparison_results),
            "best_coverage": _find_best_coverage(comparison_results),
            "method_count": len(comparison_results),
        },
    }

    if export_path:
        with pathlib.Path(export_path).open("w", encoding="utf-8") as f:
            json.dump(total_results, f, indent=2)

    return total_results


def _evaluate_comparison_methods(  # noqa: PLR0913 PLR0917
    dataset_path: pathlib.Path,
    methods: list[MethodName] | None = None,
    knn_k: int = 3,
    knn_gamma: float = 0.5,
    kmeans_clusters: int = 3,
    density_gamma: float = 0.01,
    export_path: str | None = None,
) -> ComparisonEvaluationResults:
    """Compatibility wrapper for legacy private API name.

    Args:
        dataset_path (pathlib.Path): Path to dataset file.
        methods (list[MethodName] | None): Optional list of methods to
            evaluate.
        knn_k (int): Number of nearest neighbors for KNN.
        knn_gamma (float): Threshold value for KNN monitor.
        kmeans_clusters (int): Number of clusters for k-means variants.
        density_gamma (float): Density threshold for superlevel methods.
        export_path (str | None): Optional JSON export path.

    Returns:
        Complete comparison evaluation result dictionary.
    """
    return evaluate_comparison_methods(
        dataset_path=dataset_path,
        methods=methods,
        knn_k=knn_k,
        knn_gamma=knn_gamma,
        kmeans_clusters=kmeans_clusters,
        density_gamma=density_gamma,
        export_path=export_path,
    )


__all__ = [
    "_evaluate_comparison_methods",
    "build_comparison_results_dataframe",
    "create_comparison_test_grid",
    "evaluate_comparison_methods",
    "evaluate_dataset_with_comparison_methods",
    "setup_evaluation_framework",
]
