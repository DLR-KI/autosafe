# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Enumerations naming experiment and dataset kinds."""

import enum


class ExperimentType(enum.Enum):
    """Type of experiment to run."""

    EVALUATION = "evaluation"  # Monte Carlo evaluation with results
    BENCHMARK = "benchmark"  # Kernel performance benchmarking
    CUSTOM = "custom"  # Custom experiment configuration


class DatasetType(enum.Enum):
    """Supported dataset types."""

    CSV = "csv"
    JSON = "json"
    NUMPY = "numpy"
    POLARS = "polars"


__all__ = [
    "DatasetType",
    "ExperimentType",
]
