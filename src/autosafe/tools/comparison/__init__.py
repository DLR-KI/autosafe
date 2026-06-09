# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Independent comparison CLI module.

This module provides comparison functionality that works with any
dataset format, not just Monte Carlo sampling results. It is separate
from the Monte Carlo framework and can be used with regular datasets
like WineQT.csv.
"""

from autosafe.tools.comparison.cli import COMP_APP
from autosafe.tools.comparison.core import (
    ComparisonEvaluationResults,
    _evaluate_comparison_methods,
    build_comparison_results_dataframe,
    create_comparison_test_grid,
    evaluate_comparison_methods,
    evaluate_dataset_with_comparison_methods,
    setup_evaluation_framework,
)

__all__ = [
    "COMP_APP",
    "ComparisonEvaluationResults",
    "_evaluate_comparison_methods",
    "build_comparison_results_dataframe",
    "create_comparison_test_grid",
    "evaluate_comparison_methods",
    "evaluate_dataset_with_comparison_methods",
    "setup_evaluation_framework",
]
