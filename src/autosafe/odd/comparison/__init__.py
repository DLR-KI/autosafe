# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""ODD comparison methods for extended validation.

This module provides multiple ODD boundary comparison approaches for
rigorous validation beyond simple convex hulls, including:
- KNN boundaries with threshold-based membership
- k-means clustering boundaries
- Density-based superlevel sets
- Hierarchical convex hulls of sub-clusters
"""

from autosafe.odd.comparison.base import (
    ClusteringComparisonResult,
    DecisionBoundary,
    DensityComparisonResult,
    KNNComparisonResult,
    ODDBoundaryMethod,
    ODDComparisonConfig,
    ODIComparisonResult,
    validate_comparison_config,
)
from autosafe.odd.comparison.clustered_hull import ClusteredConvexHulls
from autosafe.odd.comparison.dbscan import DBSCANCluster
from autosafe.odd.comparison.density import (
    ClusteredSuperlevelSetMonitor,
    SuperlevelSetMonitor,
)
from autosafe.odd.comparison.kmeans import KMeansBoundaries, auto_detect_optimal_k
from autosafe.odd.comparison.knn import KNNMonitor

__all__ = [
    "ClusteredConvexHulls",
    "ClusteredSuperlevelSetMonitor",
    "ClusteringComparisonResult",
    "DBSCANCluster",
    "DecisionBoundary",
    "DensityComparisonResult",
    "KMeansBoundaries",
    "KNNComparisonResult",
    "KNNMonitor",
    "ODDBoundaryMethod",
    "ODDComparisonConfig",
    "ODIComparisonResult",
    "SuperlevelSetMonitor",
    "auto_detect_optimal_k",
    "validate_comparison_config",
]
