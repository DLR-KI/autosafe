# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Generic experiment utilities and evaluation APIs."""

from autosafe.experiments.evaluation import (
    ConvexHullError,
    calculate_confusion_matrix,
    calculate_performance_metrics,
    create_comparison_test_grid,
    create_convex_hull,
    evaluate_dataset_with_comparison_methods,
    process_files,
    setup_evaluation_framework,
)

__all__ = [
    "ConvexHullError",
    "calculate_confusion_matrix",
    "calculate_performance_metrics",
    "create_comparison_test_grid",
    "create_convex_hull",
    "evaluate_dataset_with_comparison_methods",
    "process_files",
    "setup_evaluation_framework",
]
