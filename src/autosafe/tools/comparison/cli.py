# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Independent comparison CLI for ODD methods.

This module provides CLI commands for comparing ODD validation methods
that work with ANY dataset (not just Monte Carlo samples).

Addressing all structural concerns:
1. Independent of Monte Carlo - works with any dataset
2. No external process calls - fully integrated
3. Fixed typer.echo antipatterns - single formatted strings
4. Proper separation - generic evaluation + CLI wrapper
"""

import pathlib
from typing import Annotated, cast

import typer

from autosafe.tools.comparison.core import (
    ComparisonEvaluationResults,
    MethodName,
    _evaluate_comparison_methods,
)

# Create independent comparison CLI app
COMP_APP = typer.Typer(
    name="comparison", help="Independent ODD comparison method evaluation commands"
)


def _display_comparison_summary(
    results: ComparisonEvaluationResults,
    verbose: bool = False,  # noqa: FBT001 FBT002
) -> None:
    """Display summary of comparison results with a single echo call.

    Args:
        results (ComparisonEvaluationResults): The results to summarize.
        verbose (bool): Whether to include detailed method parameters in
            the summary.
    """
    dataset_info = results["summary"]

    method_sections: list[str] = []
    for method_name, method_results in results["comparison_methods"].items():
        section = (
            f"{method_name.upper()} Results:\n"
            f"  Coverage Ratio: {method_results['coverage_ratio']:.4f}\n"
            f"  Conservatism: {method_results['conservatism']:.4f}\n"
        )
        if verbose and method_results.get("parameters"):
            section += f"  Parameters: {method_results['parameters']}\n"
        method_sections.append(section)

    message = (
        f"ODD Comparison Result Summary\n"
        f"Dataset: {results['dataset']}\n"
        f"Reference points: {results['reference_points']}\n"
        f"Test points: {results['test_points']}\n"
        f"Summary:\n"
        f"  Most Conservative: {dataset_info['most_conservative']}\n"
        f"  Best Coverage: {dataset_info['best_coverage']}\n"
        f"  Methods Evaluated: {dataset_info['method_count']}\n"
        + "".join(method_sections)
    )

    typer.echo(message)


@COMP_APP.command(name="evaluate")
def run_comparison_evaluation(  # noqa: PLR0913 PLR0917
    dataset_path: str = typer.Argument(
        ..., help="Path to dataset file (CSV, JSON, etc.)"
    ),
    methods: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Comparison methods to evaluate: hull_single, knn, kmeans, "
                "density_single, hull_clustered, density_clustered, dbscan_cluster"
            )
        ),
    ] = None,
    knn_k: int = typer.Option(
        3, "--knn-k", "-k", min=1, help="Number of neighbors for KNN"
    ),
    knn_gamma: float = typer.Option(
        0.7, "--knn-gamma", "-g", min=0.0, max=1.0, help="Distance threshold for KNN"
    ),
    kmeans_clusters: int = typer.Option(
        3, "--kmeans-clusters", "-c", min=2, help="Number of clusters for k-means"
    ),
    density_gamma: float = typer.Option(
        0.01, "--density-gamma", "-d", min=1e-5, help="PDF threshold for density method"
    ),
    export_path: str = typer.Option(
        None, "--export", "-e", help="Path to save JSON results"
    ),
    verbose: bool = typer.Option(  # noqa: FBT001
        False,  # noqa: FBT003
        "--verbose",
        "-v",
        help="Show detailed results",
    ),
) -> None:
    """Run comprehensive ODD comparison evaluation.

    Evaluates multiple ODD boundary comparison methods on any dataset
    (CSV, JSON, etc.) and provides detailed metrics for each approach.

    Examples:
        autosafe comparison evaluate data/WineQT.csv
        autosafe comparison evaluate data.csv --methods knn kmeans
        autosafe comparison evaluate data.csv --export results.json

    Args:
        dataset_path (str): Path to the dataset file to evaluate.
        methods (list[str] | None): List of comparison methods to
            evaluate. If None, all methods will be evaluated.
        knn_k (int): Number of neighbors for KNN method.
        knn_gamma (float): Distance threshold for KNN method.
        kmeans_clusters (int): Number of clusters for k-means method.
        density_gamma (float): PDF threshold for density method.
        export_path (str): Optional path to save results as JSON.
        verbose (bool): Whether to show detailed results.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        BadParameter: If the dataset file is not found.
        RuntimeError: If comparison evaluation fails unexpectedly.
        ValueError: If the requested parameters are invalid.
    """
    dataset_path_obj = pathlib.Path(dataset_path)

    if not dataset_path_obj.exists():
        raise typer.BadParameter(f"Dataset file not found: {dataset_path_obj}")

    if methods is None:
        methods: list[MethodName] = [
            "hull_single",
            "knn",
            "kmeans",
            "density_single",
            "hull_clustered",
            "density_clustered",
            "dbscan_cluster",
        ]

    methods = cast("list[MethodName]", methods)  # ty: ignore[conflicting-declarations]

    try:
        results = _evaluate_comparison_methods(
            dataset_path=dataset_path_obj,
            methods=methods,
            knn_k=knn_k,
            knn_gamma=knn_gamma,
            kmeans_clusters=kmeans_clusters,
            density_gamma=density_gamma,
            export_path=export_path,
        )

        _display_comparison_summary(results, verbose)

        if export_path:
            typer.echo(f"\nResults saved to: {export_path}")

    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.echo(f"Evaluation error: {exc!s}", err=True)
        raise


@COMP_APP.command(name="quick")
def run_quick_comparison(
    dataset_path: str = typer.Argument(..., help="Path to dataset file"),
    export_path: str = typer.Option(
        None, "--export", "-e", help="Path to save JSON results"
    ),
) -> None:
    """Run quick comparison with default parameters.

    Examples:
        autosafe comparison quick data/WineQT.csv
        autosafe comparison quick data.csv --export quick_results.json

    Args:
        dataset_path (str): Path to dataset file.
        export_path (str): Optional path to save JSON results.
    """
    results = _evaluate_comparison_methods(
        dataset_path=pathlib.Path(dataset_path),
        methods=[
            "hull_single",
            "knn",
            "kmeans",
            "density_single",
            "hull_clustered",
            "density_clustered",
            "dbscan_cluster",
        ],
        export_path=export_path,
    )
    _display_comparison_summary(results, verbose=True)


@COMP_APP.command(name="info")
def show_comparison_info() -> None:
    """Show information about available comparison methods."""
    info_msg = (
        "Available ODD Comparison Methods:\n"
        "1. Single Convex Hull (hull_single)\n"
        "   - Convex hull around all reference points\n"
        "   - Best for: simple conservative geometric envelope\n"
        "2. KNN Monitor (knn)\n"
        "   - Nearest neighbors representation\n"
        "   - Best for: Outlier detection and proximity boundaries\n"
        "   - Parameters: k (neighbors), gamma (distance threshold)\n"
        "3. k-Means Boundaries (kmeans)\n"
        "   - Cluster-based ODD segmentation\n"
        "   - Best for: Natural data distribution analysis\n"
        "   - Parameters: n_clusters\n"
        "4. Density Superlevel Sets (density_single)\n"
        "   - Probability density function analysis\n"
        "   - Best for: Complex manifolds and probability regions\n"
        "   - Parameters: gamma (PDF threshold), bandwidth\n"
        "5. Clustered Convex Hulls (hull_clustered)\n"
        "   - Convex hull per cluster and union membership\n"
        "   - Best for: multi-modal geometry with separated regions\n"
        "6. Clustered Density Superlevel Sets (density_clustered)\n"
        "   - KDE superlevel sets fitted per cluster and union membership\n"
        "   - Best for: multi-modal density distributions\n"
        "7. DBSCAN Cluster Boundary (dbscan_cluster)\n"
        "   - Core-point density clustering with radius-based membership\n"
        "   - Best for: irregular shapes and outlier-robust boundaries\n"
        "Usage:\n"
        "   - Works with ANY dataset format (CSV, JSON, Parquet)\n"
        "   - Not just Monte Carlo results\n"
        "   - Use 'quick' command for initial assessment\n"
        "   - Use 'evaluate' for parameter tuning\n"
    )

    typer.echo(info_msg)


# Public API for CLI integration
__all__ = ["COMP_APP"]
