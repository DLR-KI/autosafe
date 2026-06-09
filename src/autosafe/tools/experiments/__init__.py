# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Unified experiment management package."""

from autosafe.tools.experiments.core import (
    BenchmarkResult,
    DatasetConfig,
    DatasetType,
    EvaluationResult,
    ExperimentManager,
    ExperimentType,
    KernelExperimentConfig,
    create_experiment_manager,
)
from autosafe.tools.experiments.evaluation import (
    evaluate_experiment,
    generate_experiment_id,
    run_benchmark_experiment,
    run_evaluation_pipeline,
    run_monte_carlo_evaluation,
)
from autosafe.tools.experiments.utils import (
    calculate_dataset_statistics,
    construct_experiment_id,
    create_dataset_config,
    find_dataset_bounds,
    get_dataset_type,
    load_dataset,
    load_experiment_results,
    save_results,
    setup_logging,
)

__all__ = [
    "BenchmarkResult",
    "DatasetConfig",
    "DatasetType",
    "EvaluationResult",
    "ExperimentManager",
    "ExperimentType",
    "KernelExperimentConfig",
    "calculate_dataset_statistics",
    "construct_experiment_id",
    "create_dataset_config",
    "create_experiment_manager",
    "evaluate_experiment",
    "find_dataset_bounds",
    "generate_experiment_id",
    "get_dataset_type",
    "load_dataset",
    "load_experiment_results",
    "run_benchmark_experiment",
    "run_evaluation_pipeline",
    "run_monte_carlo_evaluation",
    "save_results",
    "setup_logging",
]
