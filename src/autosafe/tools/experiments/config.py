# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Configuration objects for an experiment run."""

import dataclasses
import pathlib
from typing import Any

from autosafe.tools.experiments.types import DatasetType
from autosafe.typing import KernelType, Vector


@dataclasses.dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing.

    Attributes:
        file_path (pathlib.Path): Path to the dataset file.
        dataset_type (DatasetType): Type of the dataset file.
        normalization (dict[str, Any] | None): Optional normalization
            parameters.
        filters (dict[str, Any] | None): Optional filtering parameters.
        min_values (Vector | None): Optional minimum values for each
            dimension (for range extension).
        max_values (Vector | None): Optional maximum values for each
            dimension (for range extension).
        range_extension (float): Fraction to extend the data range
            beyond the observed min/max values (default: 0.5).
    """

    file_path: pathlib.Path
    dataset_type: DatasetType = DatasetType.CSV
    normalization: dict[str, Any] | None = None
    filters: dict[str, Any] | None = None

    # Data boundaries
    min_values: Vector | None = None
    max_values: Vector | None = None
    range_extension: float = 0.5  # Extend beyond min/max boundaries

    def validate(self) -> None:
        """Validate the dataset configuration.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the file extension is unsupported.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        if self.file_path.suffix.lower() not in {".csv", ".json", ".npy", ".parquet"}:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")


@dataclasses.dataclass
class KernelExperimentConfig:
    """Configuration for kernel experiments.

    Attributes:
        kernel_type (KernelType): Type of kernel to use in the
            experiment.
        kernel_kwargs (dict[str, Any]): Additional parameters for the
            kernel.
        n_samples (int): Number of samples to use for benchmarking.
        evaluation_samples (int): Number of samples to use for
            evaluation.
    """

    kernel_type: KernelType = "RBF"
    kernel_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Monte Carlo sampling configuration
    n_samples: int = 10_000_000  # For benchmarking
    evaluation_samples: int = 200_000  # For evaluation

    def validate(self) -> None:
        """Validate the kernel configuration.

        Raises:
            ValueError: If the sample count or kernel type is invalid.
        """
        if self.n_samples <= 0:
            raise ValueError("Number of samples must be positive")

        if self.kernel_type not in {"RBF", "Laplacian", "Gaussian"}:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")


__all__ = [
    "DatasetConfig",
    "KernelExperimentConfig",
]
