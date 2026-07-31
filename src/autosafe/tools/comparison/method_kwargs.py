# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Per-method keyword-argument shapes.

Parallel variants of one concept, kept together so they can be
compared side by side.
"""

from typing import Literal, TypedDict

import numpy as np


class KNNMethodKwargs(TypedDict, total=False):
    """Keyword arguments for KNN comparison method.

    Args:
        k (int): Number of nearest neighbors.
        gamma (float | None): Distance threshold for membership (if
            None, auto-detected).
        metric (str): Distance metric for KDTree (default: "euclidean").
        leaf_size (int): KDTree optimization parameter (default: 40).
    """

    k: int
    gamma: np.float64 | None
    metric: str
    leaf_size: int


class KMeansMethodKwargs(TypedDict, total=False):
    """Keyword arguments for k-means comparison method.

    Args:
        n_clusters (int): Number of clusters for k-means.
        metric (str): Distance metric for clustering (default:
            "euclidean").
        min_cluster_size (int): Minimum cluster size to consider for
            boundary construction.
    """

    n_clusters: int
    metric: str
    min_cluster_size: int


class DensityMethodKwargs(TypedDict, total=False):
    """Keyword arguments for density-based comparison method.

    Args:
        gamma (float): Density threshold for superlevel set membership.
        bandwidth (Literal["scott", "silverman"] | None): Bandwidth
            method for KDE (None for auto).
        sigmoid_weight (float): Weighting factor for sigmoid
            transformation of density scores.
    """

    gamma: np.float64
    bandwidth: Literal["scott", "silverman"] | None
    sigmoid_weight: np.float64


class ClusteredHullMethodKwargs(TypedDict, total=False):
    """Keyword arguments for clustered convex hull comparison method.

    Args:
        n_clusters (int): Number of clusters for partitioning reference
            points before hull construction.
        method (Literal["kmeans", "dbscan"]): Clustering method to use.
            for partitioning.
    """

    n_clusters: int
    method: Literal["kmeans", "dbscan"]


class ClusteredDensityMethodKwargs(TypedDict, total=False):
    """Keyword arguments for clustered density comparison method.

    Args:
        n_clusters (int): Number of clusters for partitioning reference
            points before density estimation.
        gamma (float): Density threshold for superlevel set membership.
        bandwidth (Literal["scott", "silverman"] | None): Bandwidth
            method for KDE (None for auto).
        min_cluster_size (int): Minimum cluster size to consider for.
            boundary construction.
    """

    n_clusters: int
    gamma: np.float64
    bandwidth: Literal["scott", "silverman"] | None
    min_cluster_size: int


class DBSCANMethodKwargs(TypedDict, total=False):
    """Keyword arguments for DBSCAN-based comparison method.

    Args:
        eps (np.float64): Maximum distance between two samples for them
            to be considered as in the same neighborhood.
        min_samples (int): Minimum number of samples in a neighborhood
            for a point to be considered as a core point.
    """

    eps: np.float64
    min_samples: int


__all__ = [
    "ClusteredDensityMethodKwargs",
    "ClusteredHullMethodKwargs",
    "DBSCANMethodKwargs",
    "DensityMethodKwargs",
    "KMeansMethodKwargs",
    "KNNMethodKwargs",
]
