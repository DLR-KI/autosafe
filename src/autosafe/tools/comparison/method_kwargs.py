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


class GMMMethodKwargs(TypedDict, total=False):
    """Keyword arguments for the Gaussian-mixture comparison method.

    Args:
        random_state (int): Seed for ``GaussianMixture``'s EM
            initialization. Required by ``GaussianMixtureBoundary``
            itself (no default there); a default is only supplied here
            so an omitted spec key does not crash the factory.
        n_components_range (tuple[int, int]): Inclusive
            ``(min, max)`` component count searched by BIC.
        gamma (float | None): Log-likelihood threshold for superlevel-
            set membership (None for auto-calibration).
    """

    random_state: int
    n_components_range: tuple[int, int]
    gamma: np.float64 | None


class OneClassSVMMethodKwargs(TypedDict, total=False):
    """Keyword arguments for the one-class SVM comparison method.

    Args:
        gamma (float | Literal["scale"]): RBF kernel gamma.
        nu (float): Upper bound on the fraction of margin errors /
            lower bound on the fraction of support vectors.
        auto_select (bool): Select (gamma, nu) from a small internal
            grid by best mean margin on the reference points.
    """

    gamma: np.float64 | Literal["scale"]
    nu: np.float64
    auto_select: bool


class SVDDMethodKwargs(TypedDict, total=False):
    """Keyword arguments for the SVDD comparison method.

    Args:
        kernel (Literal["rbf", "poly", "linear"]): Kernel name.
        gamma (float | Literal["scale"]): Kernel gamma for "rbf"/
            "poly".
        nu (float): Nu parameter, same convention as
            ``OneClassSVMMethodKwargs.nu``.
        degree (int): Polynomial degree for "poly".
        coef0 (float): Polynomial offset for "poly".
    """

    kernel: Literal["rbf", "poly", "linear"]
    gamma: np.float64 | Literal["scale"]
    nu: np.float64
    degree: int
    coef0: np.float64


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
    "GMMMethodKwargs",
    "KMeansMethodKwargs",
    "KNNMethodKwargs",
    "OneClassSVMMethodKwargs",
    "SVDDMethodKwargs",
]
