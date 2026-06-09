# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Compatibility wrappers for historical experiments evaluation imports.

Ownership of generic evaluation helpers lives in
`autosafe.tools.evaluate` while comparison-specific logic lives in
`autosafe.tools.comparison`.
"""

from autosafe.tools.comparison.core import (
    create_comparison_test_grid,
    evaluate_dataset_with_comparison_methods,
    setup_evaluation_framework,
)
from autosafe.tools.evaluate.core import (
    ConvexHullError,
    calculate_confusion_matrix,
    calculate_performance_metrics,
    create_convex_hull,
    process_files,
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
