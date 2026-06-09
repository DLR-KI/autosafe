# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""CLI commands for experiments."""

import pathlib
from typing import Annotated, Any, cast

import typer

from autosafe.tools.evaluate.workflows import (
    collect_monte_carlo_files,
    evaluate_dataset_mode,
    evaluate_monte_carlo_results,
)
from autosafe.tools.experiments.core import (
    DatasetConfig,
    ExperimentManager,
    ExperimentType,
    KernelExperimentConfig,
)
from autosafe.tools.monte_carlo._sample import create_config
from autosafe.tools.monte_carlo.sample import run_single_sampling
from autosafe.typing import ClosestSampleModeType, KernelType

# Create experiments CLI app
EXP_APP = typer.Typer(name="experiments", help="Experiment management commands")
SETTINGS: dict[str, bool] = {"verbose": False}


@EXP_APP.callback()
def _configure_experiments_cli(
    verbose_count: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Verbose output"),
    ] = 0,
) -> None:
    """Configure shared CLI state for experiment commands."""
    SETTINGS["verbose"] = verbose_count > 0


def _validate_dataset_path(dataset_path: str) -> pathlib.Path:
    """Validate that the dataset path exists.

    Args:
        dataset_path (str): Path to the dataset file to validate.

    Returns:
        pathlib.Path: The validated dataset path.

    Raises:
        BadParameter: If the dataset file does not exist.
    """
    path = pathlib.Path(dataset_path)
    if not path.exists():
        raise typer.BadParameter(f"Dataset file not found: {path}")
    return path


@EXP_APP.command(name="evaluate")
def run_evaluation_command(
    dataset_path: Annotated[str, typer.Argument(help="Path to dataset file")],
    kernel_type: Annotated[
        KernelType,
        typer.Option("--kernel", "-k", help="Kernel type to use"),
    ] = "RBF",
    closest_sample_mode: Annotated[
        ClosestSampleModeType,
        typer.Option("--sample-mode", "-s", help="Closest sample mode"),
    ] = "per_dimension",
    n_samples: Annotated[
        int,
        typer.Option("--samples", "-n", help="Number of Monte Carlo samples"),
    ] = 200_000,
    export_dir: Annotated[
        str | None,
        typer.Option("--export", "-e", help="Export directory for results"),
    ] = None,
) -> None:
    """Run a Monte Carlo evaluation experiment.

    This command wraps the eval_experiments.py functionality and
    provides it through the CLI with additional parameter control.

    Args:
        dataset_path (str): Path to dataset file.
        kernel_type (KernelType): Kernel type used for evaluation.
        closest_sample_mode (ClosestSampleModeType): Strategy for
            nearest-sample selection.
        n_samples (int): Number of Monte Carlo samples.
        export_dir (str | None): Optional export directory.
    """
    dataset = _validate_dataset_path(dataset_path)
    verbose = SETTINGS["verbose"]

    if verbose:
        typer.echo(
            "Starting evaluation experiment...\n"
            f"Dataset: {dataset.name}\n"
            f"Kernel: {kernel_type}\n"
            f"Sample mode: {closest_sample_mode}\n"
            f"Samples: {n_samples}"
        )

    # Create experiment manager and run evaluation
    manager = ExperimentManager(
        config=KernelExperimentConfig(
            kernel_type=kernel_type,
            kernel_kwargs={},
            n_samples=n_samples,
            evaluation_samples=n_samples,
        )
    )

    if verbose:
        typer.echo("Running evaluation...")

    # Run the evaluation
    dataset_config = DatasetConfig(file_path=dataset)
    result = manager.run_evaluation_experiment(
        dataset_config=dataset_config,
        export_dir=pathlib.Path(export_dir) if export_dir else None,
    )

    if verbose:
        typer.echo(
            f"Evaluation completed in {result.processing_time:.2f} seconds\n"
            f"Experiment ID: {result.experiment_id}\n"
            f"Results saved to: {result.export_paths or 'Not exported'}"
        )
    else:
        typer.echo(f"Evaluation completed: {result.experiment_id}")


@EXP_APP.command(name="benchmark")
def run_benchmark_command(
    dataset_path: Annotated[str, typer.Argument(help="Path to dataset or ODD file")],
    kernel_type: Annotated[
        KernelType,
        typer.Option("--kernel", "-k", help="Kernel type to use"),
    ] = "RBF",
    n_samples: Annotated[
        int,
        typer.Option("--samples", "-n", help="Number of benchmark samples"),
    ] = 10_000_000,
    export_dir: Annotated[
        str | None,
        typer.Option("--export", "-e", help="Export directory for results"),
    ] = None,
) -> None:
    """Run a kernel benchmarking experiment.

    This command wraps the run_experiments.py functionality and provides
    performance testing with timing analysis.

    Args:
        dataset_path (str): Path to dataset or ODD file.
        kernel_type (KernelType): Kernel type used for benchmarking.
        n_samples (int): Number of benchmark samples.
        export_dir (str | None): Optional export directory.
    """
    dataset = _validate_dataset_path(dataset_path)
    verbose = SETTINGS["verbose"]

    if verbose:
        typer.echo(
            "Starting benchmark experiment...\n"
            f"Dataset: {dataset.name}\n"
            f"Kernel: {kernel_type}\n"
            f"Samples: {n_samples}"
        )

    # Create experiment manager and run benchmark
    manager = ExperimentManager(
        config=KernelExperimentConfig(
            kernel_type=kernel_type,
            kernel_kwargs={},
            n_samples=n_samples,
            evaluation_samples=n_samples // 10,  # Smaller for eval
        )
    )

    if verbose:
        typer.echo("Running benchmark...")

    # Run the benchmark
    dataset_config = DatasetConfig(file_path=dataset)
    result = manager.run_benchmark_experiment(
        dataset_config=dataset_config,
        export_dir=pathlib.Path(export_dir) if export_dir else None,
    )

    if verbose:
        typer.echo(
            f"Benchmark completed in {result.processing_time:.2f} seconds\n"
            f"Experiment ID: {result.experiment_id}\n"
            f"Results saved to: {result.export_paths or 'Not exported'}"
        )
    else:
        typer.echo(f"Benchmark completed: {result.experiment_id}")


@EXP_APP.command(name="pipeline")
def run_pipeline_command(
    datasets: Annotated[list[str], typer.Argument(help="List of dataset files")],
    kernel_type: Annotated[
        KernelType,
        typer.Option("--kernel", "-k", help="Kernel type to use"),
    ] = "RBF",
    n_samples: Annotated[
        int,
        typer.Option("--samples", "-n", help="Number of samples per evaluation"),
    ] = 200_000,
) -> None:
    """Run evaluation pipeline on multiple datasets.

    This command runs Monte Carlo evaluation on multiple datasets
    sequentially and provides a summary of all results.

    Args:
        datasets (list[str]): List of dataset file paths.
        kernel_type (KernelType): Kernel type used for each run.
        n_samples (int): Number of samples per dataset evaluation.
    """
    verbose = SETTINGS["verbose"]
    dataset_paths = [_validate_dataset_path(dp) for dp in datasets]

    if verbose:
        typer.echo(f"Running pipeline on {len(dataset_paths)} datasets...")
        for i, dataset_path in enumerate(dataset_paths, 1):
            typer.echo(f"  {i}. {dataset_path.name}")

    # Run evaluation for each dataset
    for dataset_path in dataset_paths:
        if verbose:
            typer.echo(f"Processing {dataset_path.name}...")

        try:  # noqa: PLW0717
            manager = ExperimentManager(
                config=KernelExperimentConfig(
                    kernel_type=kernel_type,
                    kernel_kwargs={},
                    n_samples=n_samples,
                    evaluation_samples=n_samples // 10,
                )
            )
            result = manager.run_benchmark_experiment(
                dataset_config=DatasetConfig(file_path=dataset_path)
            )

            if verbose:
                typer.echo(f"Processed in {result.processing_time:.1f}s")
            else:
                typer.echo(f"Processed: {dataset_path.name}")

        except (RuntimeError, ValueError) as e:
            typer.echo(f"Error processing {dataset_path.name}: {e}", err=True)


@EXP_APP.command(name="info")
def show_info() -> None:
    """Show information about available experiment types."""
    typer.echo("Available experiment types:")
    for exp_type in ExperimentType:
        typer.echo(f"  - {exp_type.value}")

    typer.echo(
        "\nSDK Usage:\n"
        "  from autosafe.tools.experiments.core import ExperimentManager\n"
        "  from autosafe.tools.experiments.evaluation import evaluate_experiment\n"
        "  from autosafe.tools.experiments.utils import load_dataset"
    )


@EXP_APP.command(name="list")
def list_datasets(
    directory: Annotated[
        str,
        typer.Argument(help="Directory to scan for datasets"),
    ] = ".",
) -> None:
    """List available dataset files in a directory.

    Args:
        directory (str): Directory to scan.

    Raises:
        BadParameter: If the provided directory does not exist.
    """
    path = pathlib.Path(directory)
    if not path.exists():
        raise typer.BadParameter(f"Directory not found: {path}")

    supported_extensions = {".csv", ".json", ".parquet", ".npy"}
    datasets = sorted([
        f.name for f in path.iterdir() if f.suffix.lower() in supported_extensions
    ])

    if not datasets:
        typer.echo("No datasets found.")
        return

    typer.echo(f"Available datasets in {path}:")
    for i, dataset in enumerate(datasets, 1):
        typer.echo(f"  {i}. {dataset}")


def glob_run_mc_sample(item: dict[str, Any]) -> dict[str, Any]:
    config_file = item.get("config_file")
    if config_file is None:
        raise ValueError("mc-sample mode requires 'config_file'")

    samples_from_spec = item.get("samples", item.get("n_samples"))
    kernel_type_from_spec = item.get("kernel_type")
    kernel_kwargs_from_spec = item.get("kernel_kwargs")

    if kernel_kwargs_from_spec is not None and not isinstance(
        kernel_kwargs_from_spec,
        dict,
    ):
        raise ValueError(
            f"kernel_kwargs must be a mapping, got {type(kernel_kwargs_from_spec)}"
        )

    config = create_config(config_file=str(config_file))

    if samples_from_spec is not None:
        config["samples"] = int(samples_from_spec)

    if kernel_type_from_spec is not None:
        kernel_type_str = str(kernel_type_from_spec)
        if kernel_type_str not in {"RBF", "Laplacian"}:
            raise ValueError(
                "kernel_type must be one of {'RBF', 'Laplacian'} in run-spec"
            )
        config["kernel_config"]["type"] = cast("KernelType", kernel_type_str)

    if kernel_kwargs_from_spec is not None:
        config["kernel_config"]["params"] = cast(
            "dict[str, Any]",
            kernel_kwargs_from_spec,
        )

    if any(x is not None for x in (samples_from_spec, kernel_type_from_spec)):
        typer.echo(
            "INFO: run-spec overrides active: using samples/kernel settings "
            "from spec item"
        )

    run_single_sampling(config)
    return {
        "mode": "mc-sample",
        "config_file": str(config_file),
        "output": str(config["filename"]),
    }


def glob_run_dataset(item: dict[str, Any]) -> dict[str, Any]:
    dataset_path = pathlib.Path(str(item["dataset_path"]))
    comparison_methods = item.get("comparison_methods")
    if comparison_methods is not None:
        if not isinstance(comparison_methods, list):
            raise ValueError(
                f"comparison_methods must be a list, got {type(comparison_methods)}"
            )
        references = list(comparison_methods)
    else:
        references = item.get("references")

    kernel_kwargs = item.get("kernel_kwargs", {})
    if not isinstance(kernel_kwargs, dict):
        raise ValueError(f"kernel_kwargs must be a mapping, got {type(kernel_kwargs)}")

    mode_str = str(item.get("closest_sample_mode", "per_dimension"))
    if mode_str == "global":
        closest_mode: ClosestSampleModeType = "global"
    else:
        closest_mode = "per_dimension"
    kernel_type: KernelType = (
        "Laplacian" if str(item.get("kernel_type", "RBF")) == "Laplacian" else "RBF"
    )
    evaluation_samples = int(
        item.get("evaluation_samples", item.get("n_samples", 200_000))
    )

    _, csv_path, odd_path = evaluate_dataset_mode(
        dataset_path=dataset_path,
        odd_json=pathlib.Path(item["odd_json"]) if item.get("odd_json") else None,
        odd_json_out=pathlib.Path(item["odd_json_out"])
        if item.get("odd_json_out")
        else None,
        ground_truth_yaml=pathlib.Path(item["ground_truth_yaml"])
        if item.get("ground_truth_yaml")
        else None,
        threshold_mode=str(item.get("threshold_mode", "linear")),
        threshold_count=int(item.get("threshold_count", 100)),
        references=(
            [str(ref) for ref in references] if isinstance(references, list) else None
        ),
        n_samples=evaluation_samples,
        closest_sample_mode=closest_mode,
        kernel_type=kernel_type,
        kernel_kwargs=kernel_kwargs,
        seed=int(item.get("seed", 0)),
        csv_output=pathlib.Path(item["csv_output"]) if item.get("csv_output") else None,
    )
    return {
        "mode": "dataset",
        "csv": str(csv_path),
        "odd_json": str(odd_path),
    }


def glob_run_mc_results(item: dict[str, Any]) -> dict[str, Any]:
    inputs = item.get("inputs", [])
    if not isinstance(inputs, list) or len(inputs) == 0:
        raise ValueError("mc-results mode requires a non-empty 'inputs' list")
    files = collect_monte_carlo_files([str(inp) for inp in inputs])
    result_df = evaluate_monte_carlo_results(
        files=files,
        threshold_mode=str(item.get("threshold_mode", "linear")),
        threshold_count=int(item.get("threshold_count", 100)),
        references=list(item.get("references", []))
        if isinstance(item.get("references"), list)
        else None,
        csv_output=pathlib.Path(item["csv_output"]) if item.get("csv_output") else None,
    )
    return {
        "mode": "mc-results",
        "rows": int(result_df.height),
    }


@EXP_APP.command(name="run-spec")
def run_spec(
    spec_path: Annotated[str, typer.Argument(help="YAML experiment spec file")],
    state_path: Annotated[
        str | None,
        typer.Option("--state", help="Optional state file for resume support"),
    ] = None,
    *,
    resume: Annotated[bool, typer.Option(help="Resume from state file")] = True,
    stop_on_error: Annotated[
        bool,
        typer.Option(help="Stop on first failure"),
    ] = False,
) -> None:
    """Run a batch of experiments from a YAML spec file.

    Args:
        spec_path (str): Path to YAML experiment specification.
        state_path (str | None): Optional state file used for resume
            support.
        resume (bool): Whether to resume from an existing state file.
        stop_on_error (bool): Whether to stop batch execution on first
            failure.

    Raises:
        BadParameter: If the spec file does not exist.
    """
    spec = pathlib.Path(spec_path)
    if not spec.exists():
        raise typer.BadParameter(f"Spec file not found: {spec}")

    manager = ExperimentManager(config=KernelExperimentConfig())

    def _runner(item: dict[str, Any]) -> dict[str, Any]:
        mode = item.get("mode")
        if mode == "mc-sample":
            return glob_run_mc_sample(item)
        if mode == "dataset":
            return glob_run_dataset(item)
        if mode == "mc-results":
            return glob_run_mc_results(item)
        raise ValueError(f"Unsupported spec mode: {mode}")

    summary = manager.run_batch_spec(
        spec_path=spec,
        run_item=_runner,
        state_path=pathlib.Path(state_path) if state_path else None,
        resume=resume,
        stop_on_error=stop_on_error,
    )

    typer.echo(
        f"Batch finished | completed={summary['completed_count']} "
        f"failed={summary['failed_count']} state={summary['state_path']}"
    )


def get_app() -> typer.Typer:
    """Get the experiments CLI application.

    Returns:
        typer.Typer: The experiments CLI app instance.
    """
    return EXP_APP


__all__ = ["EXP_APP", "get_app"]
