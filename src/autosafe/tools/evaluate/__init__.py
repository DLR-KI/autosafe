# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Evaluation tooling package."""

from autosafe.tools.evaluate.cli import (
    EVAL_APP,
    evaluate_dataset,
    evaluate_sampling_results,
)
from autosafe.tools.evaluate.core import (
    ConvexHullError,
    calculate_confusion_matrix,
    calculate_performance_metrics,
    create_convex_hull,
    process_files,
)

__all__ = [
    "EVAL_APP",
    "ConvexHullError",
    "calculate_confusion_matrix",
    "calculate_performance_metrics",
    "create_convex_hull",
    "evaluate_dataset",
    "evaluate_sampling_results",
    "process_files",
]
