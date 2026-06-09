# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Coverage tests for autosafe.tools.evaluate.comparison package."""

import math

import numpy as np
import pytest

from autosafe.tools.evaluate.comparison import (
    FastHullApproximation,
    HullAutoSelector,
    _resolve_comparison_methods,
    create_comparison_monitor,
    get_available_method_names,
    validate_method_names,
)


def test_fast_hull_approximation_init():
    """Test FastHullApproximation initialization."""
    approx = FastHullApproximation()
    assert approx.method_type == "fast_hull_approx"
    assert approx._center is None
    assert approx._max_radius is None


def test_fast_hull_approximation_fit_and_call():
    """Test FastHullApproximation fit and call."""
    approx = FastHullApproximation()
    ref_points = np.array([[0.0, 1.0], [0.0, 1.0]])  # (n_features=2, n_samples=2)
    approx.fit(ref_points)

    # Point inside should return True
    inside_point = np.array([0.5, 0.5])
    assert approx(inside_point) is True

    # Point outside should return False
    outside_point = np.array([5.0, 5.0])
    assert approx(outside_point) is False


def test_fast_hull_approximation_call_not_fitted():
    """Test FastHullApproximation raises RuntimeError when not fitted."""
    approx = FastHullApproximation()
    test_point = np.array([0.5, 0.5])

    with pytest.raises(RuntimeError, match="not fitted"):
        approx(test_point)


def test_fast_hull_approximation_evaluate_batch():
    """Test FastHullApproximation evaluate_batch."""
    approx = FastHullApproximation()
    ref_points = np.array([[0.0, 1.0], [0.0, 1.0]])
    approx.fit(ref_points)

    test_points = np.array([[0.5, 0.5, 5.0], [0.5, 0.5, 5.0]])  # 3 test points
    result = approx.evaluate_batch(test_points)

    assert result.shape == (3,)
    assert result[0]  # Inside
    assert result[1]  # Inside
    assert not result[2]  # Outside


def test_fast_hull_approximation_evaluate_batch_not_fitted():
    """Test FastHullApproximation evaluate_batch raises RuntimeError when not fitted."""
    approx = FastHullApproximation()
    test_points = np.array([[0.5, 0.5], [0.5, 0.5]])

    with pytest.raises(RuntimeError, match="not fitted"):
        approx.evaluate_batch(test_points)


def test_fast_hull_approximation_decision_boundary():
    """Test FastHullApproximation decision_boundary property."""
    approx = FastHullApproximation()
    boundary = approx.decision_boundary

    # decision_boundary returns a dict-like DecisionBoundary
    assert boundary["type"] == "fast_hull_approx"
    assert math.isclose(boundary["conservatism"] or 0.0, 0.1)


def test_hull_auto_selector_is_fast_approximation_recommended():
    """Test HullAutoSelector.is_fast_approximation_recommended."""
    # Small dataset - should not recommend fast approx
    small_points = np.array([[0.0, 1.0], [0.0, 1.0]])  # 2 points, 2D
    assert HullAutoSelector.is_fast_approximation_recommended(small_points) is False

    # Large dataset - should recommend fast approx
    large_points = np.zeros((2, 600))  # 600 points, 2D
    assert HullAutoSelector.is_fast_approximation_recommended(large_points) is True

    # High dimensional (>= 3 dims, > 200 points)
    high_dim_points = np.zeros((4, 300))  # 300 points, 4D
    assert HullAutoSelector.is_fast_approximation_recommended(high_dim_points) is True

    # Very high dimensional (> 4 dims)
    very_high_dim_points = np.zeros((5, 100))  # 100 points, 5D
    assert (
        HullAutoSelector.is_fast_approximation_recommended(very_high_dim_points) is True
    )


def test_resolve_comparison_methods():
    """Test _resolve_comparison_methods."""
    methods = ["hull_single", "knn", "density"]
    result = _resolve_comparison_methods(methods)
    assert result == methods


def test_create_comparison_monitor_hull_single():
    """Test create_comparison_monitor for hull_single."""
    monitor = create_comparison_monitor("hull_single")
    assert monitor is not None


def test_create_comparison_monitor_hull_clustered():
    """Test create_comparison_monitor for hull_clustered."""
    monitor = create_comparison_monitor("hull_clustered")
    assert monitor is not None


def test_create_comparison_monitor_knn():
    """Test create_comparison_monitor for knn."""
    monitor = create_comparison_monitor("knn", gamma=0.5)
    assert monitor is not None


def test_create_comparison_monitor_kmeans():
    """Test create_comparison_monitor for kmeans."""
    monitor = create_comparison_monitor("kmeans")
    assert monitor is not None


def test_create_comparison_monitor_density_single():
    """Test create_comparison_monitor for density_single."""
    monitor = create_comparison_monitor("density_single", gamma=0.01)
    assert monitor is not None


def test_create_comparison_monitor_density_clustered():
    """Test create_comparison_monitor for density_clustered."""
    monitor = create_comparison_monitor(
        "density_clustered", gamma=0.01, min_cluster_size=5
    )
    assert monitor is not None


def test_create_comparison_monitor_dbscan_cluster():
    """Test create_comparison_monitor for dbscan_cluster."""
    monitor = create_comparison_monitor("dbscan_cluster", eps=0.3, min_samples=3)
    assert monitor is not None


def test_create_comparison_monitor_fast_hull_approx():
    """Test create_comparison_monitor for fast_hull_approx."""
    monitor = create_comparison_monitor("fast_hull_approx")
    assert isinstance(monitor, FastHullApproximation)


def test_create_comparison_monitor_invalid_method():
    """Test create_comparison_monitor raises ValueError for invalid method."""
    with pytest.raises(ValueError, match="Unsupported comparison method"):
        create_comparison_monitor("invalid_method")


def test_get_available_method_names():
    """Test get_available_method_names."""
    names = get_available_method_names()
    assert isinstance(names, list)
    assert "hull_single" in names
    assert "knn" in names
    assert "fast_hull_approx" in names


def test_validate_method_names_valid():
    """Test validate_method_names with valid names."""
    # Should not raise
    validate_method_names(["hull_single", "knn"])
    validate_method_names([])


def test_validate_method_names_invalid():
    """Test validate_method_names with invalid names."""
    with pytest.raises(ValueError, match="Unsupported comparison method"):
        validate_method_names(["invalid_method"])
