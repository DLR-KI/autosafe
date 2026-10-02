# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for GaussianMixtureBoundary (odd.comparison.mixture)."""

import numpy as np
import pytest

from autosafe.odd.comparison.base import ODDBoundaryMethod
from autosafe.odd.comparison.mixture import GaussianMixtureBoundary


def _two_blob_cloud(seed: int = 3) -> np.ndarray:
    """Two well-separated 2D Gaussian blobs, shape (2, 120).

    Returns:
        np.ndarray: Reference points, shape (2, 120).
    """
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=[-3.0, -3.0], scale=0.3, size=(60, 2))
    blob_b = rng.normal(loc=[3.0, 3.0], scale=0.3, size=(60, 2))
    return np.vstack([blob_a, blob_b]).T


def test_gaussian_mixture_boundary_is_odd_boundary_method():
    monitor = GaussianMixtureBoundary(random_state=0)
    assert isinstance(monitor, ODDBoundaryMethod)


def test_gaussian_mixture_boundary_shape_and_dtype_contract():
    data = _two_blob_cloud()
    monitor = GaussianMixtureBoundary(random_state=0, n_components_range=(1, 6))
    fitted = monitor.fit(data)

    # fit returns self, matching the ABC contract.
    assert fitted is monitor
    assert monitor.method_type == "gmm"

    test_points = np.array([[-3.0, 3.0, 100.0], [-3.0, 3.0, 100.0]])
    membership = monitor.evaluate_batch(test_points)
    assert membership.shape == (3,)
    assert membership.dtype == np.bool_
    # In-blob points are inside; a far outlier is not.
    assert membership[0]
    assert membership[1]
    assert not membership[2]

    assert isinstance(monitor(test_points[:, 0]), bool)

    db = monitor.decision_boundary
    assert db["type"] == "gmm"
    assert db["parameters"]["n_components"] == monitor.n_components
    conservatism = db["conservatism"]
    assert conservatism is not None
    assert 0.0 <= conservatism <= 1.0


def test_gaussian_mixture_boundary_bic_recovers_two_components():
    data = _two_blob_cloud()
    monitor = GaussianMixtureBoundary(random_state=0, n_components_range=(1, 8))
    monitor.fit(data)
    assert monitor.n_components == 2


def test_gaussian_mixture_boundary_determinism():
    data = _two_blob_cloud()
    monitor_a = GaussianMixtureBoundary(random_state=7, n_components_range=(1, 6))
    monitor_b = GaussianMixtureBoundary(random_state=7, n_components_range=(1, 6))
    monitor_a.fit(data)
    monitor_b.fit(data)

    test_points = np.array([[-3.0, 0.0, 3.0], [-3.0, 0.0, 3.0]])
    scores_a = monitor_a.pdf(test_points)
    scores_b = monitor_b.pdf(test_points)
    assert np.array_equal(scores_a, scores_b)
    assert monitor_a.n_components == monitor_b.n_components
    assert monitor_a.gamma == monitor_b.gamma


def test_gaussian_mixture_boundary_different_random_state_can_differ():
    # Not asserting inequality (EM can coincidentally agree); this just
    # exercises the random_state plumbing without crashing.
    data = _two_blob_cloud()
    monitor_a = GaussianMixtureBoundary(random_state=1, n_components_range=(1, 6))
    monitor_b = GaussianMixtureBoundary(random_state=2, n_components_range=(1, 6))
    monitor_a.fit(data)
    monitor_b.fit(data)
    assert monitor_a.random_state == 1
    assert monitor_b.random_state == 2


def test_gaussian_mixture_boundary_not_fitted_raises():
    monitor = GaussianMixtureBoundary(random_state=0)
    with pytest.raises(RuntimeError):
        monitor(np.array([0.0, 0.0]))
    with pytest.raises(RuntimeError):
        monitor.evaluate_batch(np.array([[0.0], [0.0]]))
    db = monitor.decision_boundary
    assert db["coverage"] == {}
    assert db["conservatism"] is None


def test_gaussian_mixture_boundary_requires_random_state():
    with pytest.raises(TypeError):
        GaussianMixtureBoundary()  # ty: ignore[missing-argument]


def test_suggest_reasonable_gamma_defaults_to_75th_percentile():
    monitor = GaussianMixtureBoundary(random_state=0).fit(_two_blob_cloud())
    assert monitor.ref_points is not None
    expected = np.percentile(monitor.pdf(monitor.ref_points), 75.0)
    assert monitor.suggest_reasonable_gamma() == pytest.approx(expected)
    assert monitor.candidate_gamma == pytest.approx(expected)


def test_unfitted_gaussian_mixture_falls_back_and_refuses_pdf():
    monitor = GaussianMixtureBoundary(random_state=0)
    assert monitor.suggest_reasonable_gamma() == monitor.gamma
    assert monitor._calculate_conservatism() == pytest.approx(0.5)
    with pytest.raises(RuntimeError, match="not fitted yet"):
        monitor.pdf(np.zeros((2, 3)))
