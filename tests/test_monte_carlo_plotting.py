# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Coverage tests for autosafe.tools.monte_carlo.plotting package."""

import numpy as np
import polars as pl
import pytest

from autosafe.tools.monte_carlo.plotting import (
    _plot_accuracy_comparison,
    _plot_precision_recall_curve,
    _plot_precision_recall_curve_mean,
    create_evaluation_plots,
)


def test_create_evaluation_plots(monkeypatch: pytest.MonkeyPatch):
    """Test create_evaluation_plots generates expected plots."""
    # Mock plt.savefig to avoid writing files
    monkeypatch.setattr("matplotlib.pyplot.savefig", lambda _path: None)

    # Create DataFrame with proper struct columns
    data = pl.DataFrame({
        "file": [
            "file1.json",
            "file1.json",
            "file1.json",
            "file2.json",
            "file2.json",
            "file2.json",
        ],
        "lim": [0.1, 0.5, 0.9, 0.1, 0.5, 0.9],
    })

    # Add struct columns properly
    perf_odd = pl.Series([
        {"precision": 0.8, "recall": 0.7, "prevalance": 0.5, "accuracy": 0.75},
        {"precision": 0.85, "recall": 0.75, "prevalance": 0.55, "accuracy": 0.8},
        {"precision": 0.9, "recall": 0.8, "prevalance": 0.6, "accuracy": 0.85},
        {"precision": 0.7, "recall": 0.6, "prevalance": 0.4, "accuracy": 0.65},
        {"precision": 0.75, "recall": 0.65, "prevalance": 0.45, "accuracy": 0.7},
        {"precision": 0.8, "recall": 0.7, "prevalance": 0.5, "accuracy": 0.75},
    ])
    perf_hull = pl.Series([
        {"precision": 0.75, "recall": 0.65, "prevalance": 0.45, "accuracy": 0.7},
        {"precision": 0.8, "recall": 0.7, "prevalance": 0.5, "accuracy": 0.75},
        {"precision": 0.85, "recall": 0.75, "prevalance": 0.55, "accuracy": 0.8},
        {"precision": 0.65, "recall": 0.55, "prevalance": 0.35, "accuracy": 0.6},
        {"precision": 0.7, "recall": 0.6, "prevalance": 0.4, "accuracy": 0.65},
        {"precision": 0.75, "recall": 0.65, "prevalance": 0.45, "accuracy": 0.7},
    ])

    data = data.with_columns(perf_odd.alias("performance_odd"))
    data = data.with_columns(perf_hull.alias("performance_hull"))

    # Should not raise
    create_evaluation_plots(data)


def test_plot_precision_recall_curve(monkeypatch: pytest.MonkeyPatch):
    """Test _plot_precision_recall_curve saves plot."""
    monkeypatch.setattr("matplotlib.pyplot.savefig", lambda _path: None)

    lim_x = [0.1, 0.5, 0.9]
    recall_y = np.array([[0.7, 0.75, 0.8], [0.6, 0.65, 0.7]])
    precision_y = np.array([[0.8, 0.85, 0.9], [0.7, 0.75, 0.8]])

    _plot_precision_recall_curve(lim_x, recall_y, precision_y, "True ODD")


def test_plot_precision_recall_curve_mean(monkeypatch: pytest.MonkeyPatch):
    """Test _plot_precision_recall_curve_mean saves plot."""
    monkeypatch.setattr("matplotlib.pyplot.savefig", lambda _path: None)

    lim_x = [0.1, 0.5, 0.9]
    recall_y = np.array([[0.7, 0.75, 0.8], [0.6, 0.65, 0.7]])
    precision_y = np.array([[0.8, 0.85, 0.9], [0.7, 0.75, 0.8]])

    _plot_precision_recall_curve_mean(lim_x, recall_y, precision_y, "Convex Hull")


def test_plot_accuracy_comparison(monkeypatch: pytest.MonkeyPatch):
    """Test _plot_accuracy_comparison saves plot."""
    monkeypatch.setattr("matplotlib.pyplot.savefig", lambda _path: None)

    lim_x = [0.1, 0.5, 0.9]
    accuracy_odd_y = np.array([[0.75, 0.8, 0.85], [0.65, 0.7, 0.75]])
    accuracy_hull_y = np.array([[0.7, 0.75, 0.8], [0.6, 0.65, 0.7]])

    _plot_accuracy_comparison(lim_x, accuracy_odd_y, accuracy_hull_y)
