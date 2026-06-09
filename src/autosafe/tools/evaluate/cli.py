# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""CLI commands for evaluation workflows."""

from pathlib import Path
from typing import Annotated

import typer

from autosafe.tools.evaluate.workflows import (
    collect_monte_carlo_files,
    evaluate_dataset_mode,
    evaluate_monte_carlo_results,
)

EVAL_APP = typer.Typer()


@EVAL_APP.command(
    name="sampling-results",
    help="Evaluate Monte-Carlo sampling result files.",
)
def evaluate_sampling_results(
    file: Annotated[
        list[str] | None,
        typer.Option(
            help=("Path(s) to JSON file(s) containing Monte-Carlo sampling results."),
        ),
    ] = None,
    references: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Reference sets to compare against. "
                "Supported: ground_truth, hull_single, hull_clustered, "
                "knn, kmeans, density_single, density_clustered, dbscan_cluster "
                "(aliases: hull -> hull_single, density -> density_single)"
            )
        ),
    ] = None,
    threshold_count: Annotated[
        int,
        typer.Option(help="Number of evaluated affinity thresholds."),
    ] = 100,
    csv_output: Annotated[
        str | None,
        typer.Option(help="Optional output CSV path for metric rows."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(help="Affinity threshold range mode: 'linear' or 'log'."),
    ] = "linear",
) -> None:
    """Evaluate one or more Monte-Carlo sampling result files.

    Args:
        file (list[str] | None): JSON sampling result files to evaluate.
        references (list[str] | None): Optional baselines to compare
            against.
        threshold_count (int): Number of affinity thresholds.
        csv_output (str | None): Optional output CSV path.
        mode (str): Affinity threshold range mode.

    Raises:
        FileNotFoundError: If no file paths are provided.
    """
    if file is None:
        raise FileNotFoundError("No file(s) provided for evaluation.")

    files = collect_monte_carlo_files(file)
    evaluate_monte_carlo_results(
        files,
        threshold_mode=mode,
        threshold_count=threshold_count,
        references=references,
        csv_output=Path(csv_output) if csv_output else None,
    )


@EVAL_APP.command(
    name="dataset",
    help="Evaluate autoSAFE on a real dataset from data/ or an explicit path.",
)
def evaluate_dataset(
    dataset_path: Annotated[
        str,
        typer.Argument(help="Path to dataset (CSV/JSON/Parquet)."),
    ],
    odd_json: Annotated[
        str | None,
        typer.Option(help="Optional existing affinity ODD JSON to reuse."),
    ] = None,
    ground_truth_yaml: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional YAML ground-truth ODD specification. "
                "If omitted, a sibling dataset .yml/.yaml file is used when present."
            )
        ),
    ] = None,
    references: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Requested baseline comparisons. "
                "Supported: hull_single, hull_clustered, knn, kmeans, "
                "density_single, density_clustered, dbscan_cluster "
                "(aliases: hull -> hull_single, density -> density_single). "
                "If YAML ground truth exists, all baselines are evaluated "
                "in addition to ground_truth."
            )
        ),
    ] = None,
    csv_output: Annotated[
        str | None,
        typer.Option(help="Optional output CSV path for metrics."),
    ] = None,
) -> None:
    """Evaluate dataset mode with optional ground truth and ODD reuse.

    Args:
        dataset_path (str): Path to the dataset file.
        odd_json (str | None): Optional existing affinity ODD JSON.
        ground_truth_yaml (str | None): Optional YAML ground-truth ODD
            specification.
        references (list[str] | None): Requested baseline comparisons.
        csv_output (str | None): Optional output CSV path.

    Raises:
        BadParameter: If the dataset file does not exist.
    """
    dataset = Path(dataset_path)
    if not dataset.exists():
        raise typer.BadParameter(f"Dataset file not found: {dataset}")

    results, csv_path, odd_path = evaluate_dataset_mode(
        dataset_path=dataset,
        odd_json=Path(odd_json) if odd_json else None,
        ground_truth_yaml=Path(ground_truth_yaml) if ground_truth_yaml else None,
        references=references,
        csv_output=Path(csv_output) if csv_output else None,
    )
    typer.echo(
        f"Evaluation completed: {dataset.name} | "
        f"rows={results.height} | csv={csv_path} | odd_json={odd_path}"
    )


__all__ = ["EVAL_APP", "evaluate_dataset", "evaluate_sampling_results"]
