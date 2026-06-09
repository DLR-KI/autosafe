# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Monte Carlo-specific visualization and plotting functions.

This module contains plotting-specific code that is tied to the Monte
Carlo sampling workflow.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from autosafe import ROOT_FOLDER


def create_evaluation_plots(data: pl.DataFrame) -> None:
    """Create evaluation plots for Monte Carlo sampling results.

    This function generates multiple plots showing precision/recall
    curves, accuracy comparisons, and other performance metrics for
    both true ODD and convex hull comparisons.

    Args:
        data (pl.DataFrame): DataFrame containing evaluation results.
    """
    filenames = data["file"].unique().to_list()
    lim_x = data["lim"].unique().sort().to_list()

    # Extract metrics arrays for plot processing
    precision_odd_y = np.zeros((len(filenames), len(lim_x)))
    recall_odd_y = np.zeros((len(filenames), len(lim_x)))
    prevalence_odd_y = np.zeros((len(filenames), len(lim_x)))
    accuracy_odd_y = np.zeros((len(filenames), len(lim_x)))

    precision_hull_y = np.zeros((len(filenames), len(lim_x)))
    recall_hull_y = np.zeros((len(filenames), len(lim_x)))
    prevalence_hull_y = np.zeros((len(filenames), len(lim_x)))
    accuracy_hull_y = np.zeros((len(filenames), len(lim_x)))

    for i, fname in enumerate(filenames):
        df_f = data.filter(pl.col("file") == fname).sort("lim")
        precision_odd_y[i, :] = (
            df_f["performance_odd"].struct.field("precision").to_numpy()
        )
        recall_odd_y[i, :] = df_f["performance_odd"].struct.field("recall").to_numpy()
        prevalence_odd_y[i, :] = (
            df_f["performance_odd"].struct.field("prevalance").to_numpy()
        )
        accuracy_odd_y[i, :] = (
            df_f["performance_odd"].struct.field("accuracy").to_numpy()
        )

    for i, fname in enumerate(filenames):
        df_f = data.filter(pl.col("file") == fname).sort("lim")
        precision_hull_y[i, :] = (
            df_f["performance_hull"].struct.field("precision").to_numpy()
        )
        recall_hull_y[i, :] = df_f["performance_hull"].struct.field("recall").to_numpy()
        prevalence_hull_y[i, :] = (
            df_f["performance_hull"].struct.field("prevalance").to_numpy()
        )
        accuracy_hull_y[i, :] = (
            df_f["performance_hull"].struct.field("accuracy").to_numpy()
        )

    # Create plots
    _plot_precision_recall_curve(lim_x, recall_odd_y, precision_odd_y, "True ODD")
    _plot_precision_recall_curve_mean(lim_x, recall_odd_y, precision_odd_y, "True ODD")
    _plot_precision_recall_curve(lim_x, recall_hull_y, precision_hull_y, "Convex Hull")
    _plot_precision_recall_curve_mean(
        lim_x, recall_hull_y, precision_hull_y, "Convex Hull"
    )
    _plot_accuracy_comparison(lim_x, accuracy_odd_y, accuracy_hull_y)


def _plot_precision_recall_curve(
    lim_x: list, recall_y: np.ndarray, precision_y: np.ndarray, title: str
) -> None:
    """Plot precision-recall curves for all files.

    Args:
        lim_x (list): List of affinity limit values.
        recall_y (np.ndarray): 2D array of recall values.
            (files, x limits).
        precision_y (np.ndarray): 2D array of precision values.
            (files, x limits).
        title (str): Title for the plot.
    """
    _, ax = plt.subplots()
    ax.plot(lim_x, recall_y.T, "#1f77b4")
    ax.plot(lim_x, precision_y.T, "#ff7f0e")
    ax.set_xlabel("Affinity Limit")
    handles = [
        Line2D([0], [0], color="#1f77b4", lw=2),
        Line2D([0], [0], color="#ff7f0e", lw=2),
    ]
    ax.legend(handles, ["Recall", "Precision"])
    ax.set_title(title)
    plt.savefig(ROOT_FOLDER / f"pr_{title.lower().replace(' ', '_')}.png")


def _plot_precision_recall_curve_mean(
    lim_x: list,
    recall_y: np.ndarray,
    precision_y: np.ndarray,
    title: str,
) -> None:
    """Plot mean precision-recall curves with standard deviation.

    Args:
        lim_x (list): List of affinity limit values.
        recall_y (np.ndarray): 2D array of recall values.
            (files, x limits).
        precision_y (np.ndarray): 2D array of precision values.
            (files, x limits).
        title (str): Title for the plot.
    """
    _, ax = plt.subplots()
    ax.plot(lim_x, recall_y.mean(axis=0))
    ax.fill_between(
        lim_x,
        recall_y.mean(axis=0) - recall_y.std(axis=0),
        recall_y.mean(axis=0) + recall_y.std(axis=0),
        alpha=0.3,
        label="_nolegend_",
    )
    ax.plot(lim_x, precision_y.mean(axis=0))
    ax.fill_between(
        lim_x,
        precision_y.mean(axis=0) - precision_y.std(axis=0),
        precision_y.mean(axis=0) + precision_y.std(axis=0),
        alpha=0.3,
        label="_nolegend_",
    )
    ax.set_xlabel("Affinity Limit")
    ax.legend(["Recall", "Precision"])
    ax.set_title(title)
    plt.savefig(ROOT_FOLDER / f"pr_mean_{title.lower().replace(' ', '_')}.png")


def _plot_accuracy_comparison(
    lim_x: list,
    accuracy_odd_y: np.ndarray,
    accuracy_hull_y: np.ndarray,
) -> None:
    """Plot accuracy comparison between true ODD and convex hull.

    Args:
        lim_x (list): List of affinity limit values.
        accuracy_odd_y (np.ndarray): 2D array of accuracy values for.
            true ODD (files, x limits).
        accuracy_hull_y (np.ndarray): 2D array of accuracy values for.
            convex hull (files, x limits).
    """
    _, ax = plt.subplots()
    ax.plot(lim_x, accuracy_odd_y.mean(axis=0))
    ax.fill_between(
        lim_x,
        accuracy_odd_y.mean(axis=0) - accuracy_odd_y.std(axis=0),
        accuracy_odd_y.mean(axis=0) + accuracy_odd_y.std(axis=0),
        alpha=0.3,
        label="_nolegend_",
    )
    ax.plot(lim_x, accuracy_hull_y.mean(axis=0))
    ax.fill_between(
        lim_x,
        accuracy_hull_y.mean(axis=0) - accuracy_hull_y.std(axis=0),
        accuracy_hull_y.mean(axis=0) + accuracy_hull_y.std(axis=0),
        alpha=0.3,
        label="_nolegend_",
    )
    ax.set_xlabel("Affinity Limit")
    rsquared = (
        np.corrcoef(
            accuracy_odd_y.mean(axis=0),
            accuracy_hull_y.mean(axis=0),
        )[0, 1]
        ** 2
    )
    ax.set_title(f"Accuracy Comparison; $R^2$={rsquared:.4f}")
    ax.legend(["True ODD", "Convex Hull"])
    plt.savefig(ROOT_FOLDER / "acc_comparison.png")
