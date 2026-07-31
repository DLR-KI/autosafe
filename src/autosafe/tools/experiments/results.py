# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Result containers for experiment runs.

Parallel variants of one concept, kept together so they can be
compared side by side.
"""

import dataclasses
import datetime
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt

from autosafe.tools.experiments.types import ExperimentType


@dataclasses.dataclass
class ExperimentResult:
    """Container for experiment results.

    Attributes:
        experiment_id (str): Unique identifier for the experiment.
        experiment_type (ExperimentType): Type of the experiment.
        timestamp (datetime.datetime): Timestamp of when the experiment
            was run.
        dataset_path (pathlib.Path): Path to the dataset used in the
            experiment.
        dataset_size (int): Number of samples in the dataset.
        dataset_dimensions (int): Number of dimensions in the dataset.
        config (dict[str, Any]): Configuration parameters used in the
            experiment.
        total_samples (int): Total number of samples processed.
        processing_time (float): Total processing time in seconds.
        affinity_statistics (dict[str, Any] | None): Optional affinity
            statistics collected during evaluation.
        performance_metrics (dict[str, Any] | None): Optional
            performance metrics calculated from the experiment.
        kernel_matrices (npt.NDArray[np.float64] | None): Optional
            kernel matrices computed during the experiment.
        export_paths (list[pathlib.Path]): List of file paths where
            results have been exported.
    """

    experiment_id: str
    experiment_type: ExperimentType
    timestamp: datetime.datetime

    # Dataset information
    dataset_path: pathlib.Path
    dataset_size: int
    dataset_dimensions: int

    # Configuration
    config: dict[str, Any]

    # Results
    total_samples: int
    processing_time: float  # seconds
    affinity_statistics: dict[str, Any] | None = None
    performance_metrics: dict[str, Any] | None = None
    kernel_matrices: npt.NDArray[np.float64] | None = None

    # Output
    export_paths: list[pathlib.Path] = dataclasses.field(default_factory=list)

    def add_export(self, file_path: pathlib.Path) -> None:
        """Add an exported result file to the experiment results.

        Args:
            file_path (pathlib.Path): Exported file path.
        """
        self.export_paths.append(file_path)

    def to_dict(self) -> dict[str, Any]:
        """Convert experiment results to a dictionary.

        Returns:
            dict[str, Any]: Dictionary representation of the experiment
                result.
        """
        return {
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type.value,
            "timestamp": self.timestamp.isoformat(),
            "dataset_path": str(self.dataset_path),
            "dataset_size": self.dataset_size,
            "dataset_dimensions": self.dataset_dimensions,
            "config": self.config,
            "total_samples": self.total_samples,
            "processing_time": self.processing_time,
            "affinity_statistics": self.affinity_statistics,
            "performance_metrics": self.performance_metrics,
            "export_paths": [str(p) for p in self.export_paths],
        }


@dataclasses.dataclass
class EvaluationResult(ExperimentResult):
    """Specialized result for evaluation experiments.

    Attributes:
        points_in_odd (int | None): Number of samples that fall within
            the ODD.
        coverage_ratio (float | None): Ratio of points in ODD to total
            samples.
        mean_affinity (float | None): Mean affinity of samples to the
            ODD.
    """

    # Monte Carlo specific statistics
    points_in_odd: int | None = None
    coverage_ratio: float | None = None
    mean_affinity: float | None = None

    def __post_init__(self) -> None:
        """Initialize as evaluation type."""
        self.experiment_type = ExperimentType.EVALUATION


@dataclasses.dataclass
class BenchmarkResult(ExperimentResult):
    """Specialized result for benchmarking experiments.

    Attributes:
        samples_per_second (float | None): Processing speed in samples
            per second.
        memory_usage (float | None): Peak memory usage in megabytes.
        timing_by_sample_size (dict[str, float]): Timing breakdown by
            sample size (e.g., {"100k": 1.2, "1M": 10.5}).
    """

    # Performance metrics
    samples_per_second: float | None = None
    memory_usage: float | None = None  # MB
    timing_by_sample_size: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize as benchmark type."""
        self.experiment_type = ExperimentType.BENCHMARK


__all__ = [
    "BenchmarkResult",
    "EvaluationResult",
    "ExperimentResult",
]
