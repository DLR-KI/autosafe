# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""In-process experiment evaluation APIs.

This module exposes high-level experiment helpers without spawning child
processes.
"""

import dataclasses
import datetime
import pathlib
import secrets
from typing import Any, Protocol

import numpy as np

import autosafe
from autosafe.samples import Samples
from autosafe.tools.experiments.core import (
    BenchmarkResult,
    EvaluationResult,
    ExperimentType,
)
from autosafe.tools.experiments.utils import (
    find_dataset_bounds,
    load_dataset,
    save_results,
)
from autosafe.typing import (
    ClosestSampleModeType,
    KernelType,
    Matrix,
    NPAffinityVector,
    NPMatrix,
    NPVector,
)


class _HasSamples(Protocol):
    @property
    def samples(self) -> list[Any]:
        """Return the sample collection."""

    def __call__(self, points: Matrix | NPMatrix) -> NPAffinityVector:
        """Evaluate the ODD on a batch of points.

        Returns affinity values corresponding to the
        input points. The shape of the output should be (n_samples,)
        where n_samples is the number of rows in the input points
        matrix.

        Args:
            points (Matrix | NPMMatrix): Array of shape
                (n_samples, n_features) to evaluate.
        """


@dataclasses.dataclass(frozen=True)
class EvaluationExperimentConfig:
    """Configuration for in-process evaluation experiments.

    Attributes:
        closest_sample_mode (ClosestSampleModeType): Mode for selecting
            the closest sample when constructing the ODD from a dataset.
        kernel_type (KernelType): Type of kernel to use when
            constructing the ODD from a dataset.
        kernel_kwargs (dict[str, Any]): Additional keyword arguments to
            pass to the kernel constructor when building the ODD.
        n_samples (int): Number of samples to evaluate in the
            experiment.
        export_dir (pathlib.Path | None): Optional directory to export
            results to. If None, results will not be saved to disk.
    """

    closest_sample_mode: ClosestSampleModeType = "per_dimension"
    kernel_type: KernelType = "RBF"
    kernel_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    n_samples: int = 200_000
    export_dir: pathlib.Path | None = None


@dataclasses.dataclass(frozen=True)
class BenchmarkExperimentConfig:
    """Configuration for in-process benchmark experiments.

    Attributes:
        kernel_type (KernelType): Type of kernel to use when
            constructing the ODD from a dataset.
        kernel_kwargs (dict[str, Any]): Additional keyword arguments to
            pass to the kernel constructor when building the ODD.
        n_samples (int): Total number of samples to evaluate for
            benchmarking.
        export_dir (pathlib.Path | None): Optional directory to export
            results to. If None, results will not be saved to disk.
    """

    kernel_type: KernelType = "RBF"
    kernel_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    n_samples: int = 10_000_000
    export_dir: pathlib.Path | None = None


def _compute_odd_bounds(
    odd: Samples,
) -> tuple[NPVector, NPVector]:
    """Compute per-dimension bounds from ODD samples.

    Args:
        odd (Samples): ODD samples to compute bounds from.

    Returns:
        tuple[NPVector, NPVector]: A tuplecontaining the lower and upper
            bounds.
    """
    points = np.array([np.array(sample.x, dtype=float) for sample in odd.samples])
    return points.min(axis=0), points.max(axis=0)


def _load_dataset_context(
    dataset_path: pathlib.Path,
    *,
    kernel_type: KernelType,
    kernel_kwargs: dict[str, Any],
    closest_sample_mode: ClosestSampleModeType,
) -> tuple[
    Samples,
    int,
    int,
    NPVector,
    NPVector,
]:
    """Load the ODD and derived dataset context for an experiment.

    Args:
        dataset_path (pathlib.Path): Path to the dataset file or ODD
            JSON file.
        kernel_type (KernelType): Type of kernel to use.
        kernel_kwargs (dict[str, Any]): Additional keyword arguments for
            the kernel.
        closest_sample_mode (ClosestSampleModeType): Mode for selecting
            the closest sample.

    Returns:
        tuple[Samples, int, int, npt.NDArray, npt.NDArray]: ODD object,
            dataset size, dataset dimensions, and bounds.
    """
    if dataset_path.suffix.lower() == ".json":
        odd = autosafe.from_json(dataset_path)
        dataset_size = int(odd.shape[0]) if len(odd.shape) > 0 else 0
        dataset_dimensions = int(odd.shape[1]) if len(odd.shape) > 1 else 0
        lower, upper = _compute_odd_bounds(odd)
        return odd, dataset_size, dataset_dimensions, lower, upper

    df, _ = load_dataset(dataset_path)
    dataset_size, dataset_dimensions = int(df.height), int(df.width)
    odd = autosafe.from_polars(
        df,
        closest_sample_mode=closest_sample_mode,
        kernel_cls=kernel_type,
        kernel_kwargs=kernel_kwargs,
    )
    mins, maxs = find_dataset_bounds(df)
    lower = np.array(mins, dtype=float)
    upper = np.array(maxs, dtype=float)
    return odd, dataset_size, dataset_dimensions, lower, upper


def _sample_affinities(
    odd: Samples,
    lower: NPVector,
    upper: NPVector,
    *,
    dataset_dimensions: int,
    n_samples: int,
) -> NPVector:
    """Sample points and evaluate affinities in one step.

    Args:
        odd (Samples): The ODD samples object to evaluate.
        lower (NPVector): Lower bounds for sampling.
        upper (NPVector): Upper bounds for sampling.
        dataset_dimensions (int): Number of dimensions in the dataset.
        n_samples (int): Number of samples to draw.

    Returns:
        NPVector: Affinity values for the sampled points.
    """
    span = np.maximum(upper - lower, 1e-9)
    lower -= 0.05 * span
    upper += 0.05 * span
    rng = np.random.default_rng(0)
    points = rng.uniform(lower, upper, size=(n_samples, dataset_dimensions))
    return np.asarray(odd(points))


def _benchmark_timings(
    odd: Samples,
    lower: NPVector,
    upper: NPVector,
    *,
    dataset_dimensions: int,
    n_samples: int,
) -> dict[str, float]:
    """Measure benchmark timings for multiple sample sizes.

    Args:
        odd (Samples): The ODD samples object to evaluate.
        lower (NPVector): Lower bounds for sampling.
        upper (NPVector): Upper bounds for sampling.
        dataset_dimensions (int): Number of dimensions in the dataset.
        n_samples (int): Total number of samples to draw for
            benchmarking.

    Returns:
        dict[str, float]: Mapping of sample size to elapsed seconds.
    """
    rng = np.random.default_rng(0)
    timings: dict[str, float] = {}
    for size in [1, 10, 100, 1_000, 10_000]:
        if size > n_samples:
            continue
        points = rng.uniform(lower, upper, size=(size, dataset_dimensions))
        t0 = datetime.datetime.now(datetime.timezone.utc)
        _ = odd(points)
        t1 = datetime.datetime.now(datetime.timezone.utc)
        timings[str(size)] = (t1 - t0).total_seconds()
    return timings


def evaluate_experiment(
    dataset_path: pathlib.Path,
    request: EvaluationExperimentConfig | None = None,
) -> EvaluationResult:
    """Run an evaluation experiment directly in-process.

    Args:
        dataset_path (pathlib.Path): Path to the dataset file.
        request (EvaluationExperimentConfig | None): Optional evaluation
            configuration.

    Returns:
        EvaluationResult: Evaluation result containing experiment
            metadata and results.
    """
    request = request or EvaluationExperimentConfig()
    start_time = datetime.datetime.now(datetime.timezone.utc)
    odd, dataset_size, dataset_dimensions, lower, upper = _load_dataset_context(
        dataset_path,
        kernel_type=request.kernel_type,
        kernel_kwargs=request.kernel_kwargs,
        closest_sample_mode=request.closest_sample_mode,
    )

    eval_samples = int(max(1, min(request.n_samples, 50_000)))
    affinities = _sample_affinities(
        odd,
        lower,
        upper,
        dataset_dimensions=dataset_dimensions,
        n_samples=eval_samples,
    )

    end_time = datetime.datetime.now(datetime.timezone.utc)
    processing_time = (end_time - start_time).total_seconds()

    # Create experiment ID and timestamp
    experiment_id = generate_experiment_id()
    timestamp = end_time

    # Create and return evaluation result
    return EvaluationResult(
        experiment_id=experiment_id,
        experiment_type=ExperimentType.EVALUATION,
        timestamp=timestamp,
        dataset_path=dataset_path,
        dataset_size=dataset_size,
        dataset_dimensions=dataset_dimensions,
        config={
            "closest_sample_mode": request.closest_sample_mode,
            "kernel_type": request.kernel_type,
            "kernel_kwargs": request.kernel_kwargs,
            "n_samples": request.n_samples,
        },
        total_samples=eval_samples,
        processing_time=processing_time,
        affinity_statistics={
            "min": float(np.min(affinities)),
            "max": float(np.max(affinities)),
            "mean": float(np.mean(affinities)),
        },
    )


def _save_requested_result(
    result: EvaluationResult | BenchmarkResult,
    export_dir: pathlib.Path | None,
) -> None:
    """Persist a result when an export directory is provided.

    Args:
        result (EvaluationResult | BenchmarkResult): The result to save.
        export_dir (pathlib.Path | None): Optional directory to save the
            result in.
    """
    if export_dir is not None:
        save_results(result, export_dir / f"{result.experiment_id}.json")


def run_benchmark_experiment(
    dataset_path: pathlib.Path,
    request: BenchmarkExperimentConfig | None = None,
) -> BenchmarkResult:
    """Run a benchmark experiment directly in-process.

    Args:
        dataset_path (pathlib.Path): Path to the dataset file or ODD
            JSON file.
        request (BenchmarkExperimentConfig | None): Optional benchmark
            configuration.

    Returns:
        BenchmarkResult: Benchmark result containing metadata and
            performance metrics.
    """
    request = request or BenchmarkExperimentConfig()
    start_time = datetime.datetime.now(datetime.timezone.utc)
    odd, dataset_size, dataset_dimensions, lower, upper = _load_dataset_context(
        dataset_path,
        kernel_type=request.kernel_type,
        kernel_kwargs=request.kernel_kwargs,
        closest_sample_mode="per_dimension",
    )

    timings = _benchmark_timings(
        odd,
        lower,
        upper,
        dataset_dimensions=dataset_dimensions,
        n_samples=request.n_samples,
    )

    end_time = datetime.datetime.now(datetime.timezone.utc)
    processing_time = (end_time - start_time).total_seconds()
    samples_per_second = (
        request.n_samples / processing_time if processing_time > 0 else None
    )

    # Create experiment ID and timestamp
    experiment_id = generate_experiment_id()
    timestamp = end_time

    # Create and return benchmark result
    return BenchmarkResult(
        experiment_id=experiment_id,
        experiment_type=ExperimentType.BENCHMARK,
        timestamp=timestamp,
        dataset_path=dataset_path,
        dataset_size=dataset_size,
        dataset_dimensions=dataset_dimensions,
        config={
            "kernel_type": request.kernel_type,
            "kernel_kwargs": request.kernel_kwargs,
            "n_samples": request.n_samples,
        },
        total_samples=request.n_samples,
        processing_time=processing_time,
        samples_per_second=samples_per_second,
        timing_by_sample_size=timings,
    )


def run_evaluation_pipeline(
    datasets: list[pathlib.Path],
    request: EvaluationExperimentConfig | None = None,
) -> list[ExperimentType]:
    """Run a complete evaluation pipeline on multiple datasets.

    This orchestrates the execution of multiple experiments across
    different datasets and metric types, collecting all results.

    Args:
        datasets (list[pathlib.Path]): List of dataset paths to
            evaluate.
        request (EvaluationExperimentConfig | None): Optional evaluation
            configuration passed to each run.

    Returns:
        list[ExperimentType]: List of experiment types that were run.
    """
    results: list[ExperimentType] = []
    for dataset_path in datasets:
        _ = evaluate_experiment(dataset_path=dataset_path, request=request)
        results.append(ExperimentType.EVALUATION)
    return results


def run_monte_carlo_evaluation(
    dataset_path: pathlib.Path,
    n_samples: int = 200_000,
    closest_sample_mode: ClosestSampleModeType = "per_dimension",
    export_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Run Monte Carlo-style evaluation and return a dictionary summary.

    Args:
        dataset_path (pathlib.Path): Path to the dataset file.
        n_samples (int): Number of samples to evaluate.
        closest_sample_mode (ClosestSampleModeType): Sampling mode.
        export_dir (pathlib.Path | None): Optional export directory.

    Returns:
        dict[str, Any]: Dictionary summary of the evaluation run.
    """
    result = evaluate_experiment(
        dataset_path=dataset_path,
        request=EvaluationExperimentConfig(
            closest_sample_mode=closest_sample_mode,
            n_samples=n_samples,
            export_dir=export_dir,
        ),
    )

    return {
        "success": True,
        "experiment_id": result.experiment_id,
        "dataset_path": str(result.dataset_path),
        "dataset_size": result.dataset_size,
        "dataset_dimensions": result.dataset_dimensions,
        "total_samples": result.total_samples,
        "processing_time": result.processing_time,
        "affinity_statistics": result.affinity_statistics,
        "config": result.config,
    }


def generate_experiment_id() -> str:
    """Generate a unique experiment identifier with timestamp.

    Returns:
        str: Generated experiment identifier.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    random_suffix = f"{secrets.randbelow(900) + 100:03d}"
    return f"exp_{timestamp}_{random_suffix}"


__all__ = [
    "BenchmarkExperimentConfig",
    "EvaluationExperimentConfig",
    "evaluate_experiment",
    "generate_experiment_id",
    "run_benchmark_experiment",
    "run_evaluation_pipeline",
    "run_monte_carlo_evaluation",
]
