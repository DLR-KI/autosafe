# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for OneClassSVMBoundary and SVDDBoundary (odd.comparison.oneclass)."""

import warnings
from collections.abc import Callable

import numpy as np
import pytest

from autosafe.odd.comparison.base import ODDBoundaryMethod
from autosafe.odd.comparison.oneclass import (
    OneClassSVMBoundary,
    SVDDBoundary,
    _kernel_self,
    _pairwise_kernel,
    _resolve_rbf_gamma,
)


def _single_blob(seed: int = 0, loc: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """A single 2D Gaussian blob, shape (2, 40).

    Returns:
        np.ndarray: Reference points, shape (2, 40).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=list(loc), scale=0.3, size=(40, 2)).T


def _uniform_test_points(
    seed: int, low: float, high: float, n: int = 200
) -> np.ndarray:
    """Uniform 2D test points in [low, high]^2, shape (2, n).

    Returns:
        np.ndarray: Test points, shape (2, n).
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, 2)).T


def test_oneclass_svm_boundary_is_odd_boundary_method():
    assert isinstance(OneClassSVMBoundary(), ODDBoundaryMethod)


def test_oneclass_svm_boundary_shape_and_dtype_contract():
    ref = _single_blob()
    monitor = OneClassSVMBoundary(gamma=1.5, nu=0.1)
    fitted = monitor.fit(ref)
    assert fitted is monitor
    assert monitor.method_type == "oneclass_svm"

    test_points = _uniform_test_points(1, -3.0, 3.0, n=100)
    membership = monitor.evaluate_batch(test_points)
    assert membership.shape == (100,)
    assert membership.dtype == np.bool_

    scores = monitor.decision_function(test_points)
    assert scores.shape == (100,)
    assert scores.dtype == np.float64
    assert np.array_equal(membership, scores >= monitor.threshold)

    assert isinstance(monitor(test_points[:, 0]), bool)

    db = monitor.decision_boundary
    assert db["type"] == "oneclass_svm"
    conservatism = db["conservatism"]
    assert conservatism is not None
    assert 0.0 <= conservatism <= 1.0


def test_oneclass_svm_boundary_not_fitted_raises():
    monitor = OneClassSVMBoundary()
    with pytest.raises(RuntimeError):
        monitor.decision_function(np.array([[0.0], [0.0]]))
    db = monitor.decision_boundary
    assert db["coverage"] == {}
    assert db["conservatism"] is None


def test_oneclass_svm_boundary_determinism():
    ref = _single_blob()
    test_points = _uniform_test_points(1, -3.0, 3.0)

    monitor_a = OneClassSVMBoundary(gamma=1.5, nu=0.1).fit(ref)
    monitor_b = OneClassSVMBoundary(gamma=1.5, nu=0.1).fit(ref)
    scores_a = monitor_a.decision_function(test_points)
    scores_b = monitor_b.decision_function(test_points)
    assert np.array_equal(scores_a, scores_b)


def test_oneclass_svm_boundary_auto_select_mirrors_benchmark_grid():
    ref = _single_blob()
    monitor = OneClassSVMBoundary(auto_select=True).fit(ref)
    # Selected from the (nu, gamma) grid mirroring the "ocsvm" branch of
    # experiments/benchmark/run_baseline_comparison.py::_score.
    assert monitor.nu in {0.01, 0.05, 0.1}
    assert monitor.gamma in {"scale", 1.0, 10.0}


def test_svdd_boundary_is_odd_boundary_method():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monitor = SVDDBoundary()
    assert isinstance(monitor, ODDBoundaryMethod)


def test_svdd_boundary_warns_on_rbf_kernel():
    with pytest.warns(UserWarning, match="equivalent"):
        SVDDBoundary(kernel="rbf")


def test_svdd_boundary_no_warning_for_poly_kernel():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SVDDBoundary(kernel="poly")  # must not raise/warn


def test_svdd_boundary_shape_and_dtype_contract():
    ref = _single_blob()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monitor = SVDDBoundary(gamma=1.5, nu=0.1)
    fitted = monitor.fit(ref)
    assert fitted is monitor
    assert monitor.method_type == "svdd"

    test_points = _uniform_test_points(1, -3.0, 3.0, n=100)
    membership = monitor.evaluate_batch(test_points)
    assert membership.shape == (100,)
    assert membership.dtype == np.bool_

    scores = monitor.decision_function(test_points)
    assert scores.shape == (100,)
    assert scores.dtype == np.float64
    assert np.array_equal(membership, scores >= monitor.threshold)

    db = monitor.decision_boundary
    assert db["type"] == "svdd"
    assert db["parameters"]["kernel"] == "rbf"
    conservatism = db["conservatism"]
    assert conservatism is not None
    assert 0.0 <= conservatism <= 1.0


def test_svdd_boundary_not_fitted_raises():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monitor = SVDDBoundary()
    with pytest.raises(RuntimeError):
        monitor.decision_function(np.array([[0.0], [0.0]]))


def test_svdd_boundary_determinism():
    ref = _single_blob()
    test_points = _uniform_test_points(1, -3.0, 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monitor_a = SVDDBoundary(gamma=1.5, nu=0.1).fit(ref)
        monitor_b = SVDDBoundary(gamma=1.5, nu=0.1).fit(ref)
    scores_a = monitor_a.decision_function(test_points)
    scores_b = monitor_b.decision_function(test_points)
    assert np.array_equal(scores_a, scores_b)


def test_resolve_rbf_gamma_matches_sklearn_scale_convention():
    data = _single_blob().T
    resolved = _resolve_rbf_gamma("scale", data)
    expected = 1.0 / (data.shape[1] * float(np.var(data)))
    assert resolved == pytest.approx(expected)
    assert _resolve_rbf_gamma(2.5, data) == pytest.approx(2.5)


def test_resolve_rbf_gamma_zero_variance_fallback():
    constant_data = np.zeros((10, 2))
    assert _resolve_rbf_gamma("scale", constant_data) == pytest.approx(1.0)


def test_svdd_rbf_equivalent_to_oneclass_svm_under_matched_hyperparameters():
    """SVDD(kernel='rbf') and OneClassSVM must share a decision boundary.

    Tax & Duin (2004) / Scholkopf et al. (2001): under an RBF kernel
    (constant diagonal k(x,x)=1), the SVDD dual and the one-class SVM
    dual are the same quadratic program. Verified here on a fixed 2D
    cloud by checking the two produce identical membership over a grid
    of test points spanning well beyond the training data, and that
    their raw scores are (near-perfectly) linearly related -- the
    residual imprecision is solver-precision noise in the QP solve, not
    a modeling difference (see the equivalence derivation: score_svdd
    = 2C * score_ocsvm for C = 1/(nu * n), verified separately against
    sklearn's own dual solution to ~1e-9 in the implementation notes).
    """
    ref = _single_blob(seed=0)
    test_points = _uniform_test_points(0, -1.5, 1.5, n=200)

    nu = 0.1
    gamma = 1.5

    ocsvm = OneClassSVMBoundary(gamma=gamma, nu=nu).fit(ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svdd = SVDDBoundary(kernel="rbf", gamma=gamma, nu=nu).fit(ref)

    mem_oc = ocsvm.evaluate_batch(test_points)
    mem_svdd = svdd.evaluate_batch(test_points)
    assert np.array_equal(mem_oc, mem_svdd)

    scores_oc = ocsvm.decision_function(test_points)
    scores_svdd = svdd.decision_function(test_points)
    correlation = np.corrcoef(scores_oc, scores_svdd)[0, 1]
    assert correlation == pytest.approx(1.0, abs=1e-4)


def test_svdd_poly_kernel_differs_from_oneclass_svm():
    """A non-constant-diagonal kernel must NOT reduce to the OC-SVM.

    Uses off-center data: the polynomial kernel is not translation
    invariant (unlike RBF), so an off-origin cloud makes its boundary
    diverge sharply from the RBF one-class SVM boundary. Without this,
    the equivalence test above could pass trivially for any kernel
    choice.
    """
    ref = _single_blob(seed=1, loc=(5.0, 5.0))
    test_points = _uniform_test_points(1, 3.0, 7.0, n=300)

    nu = 0.1
    ocsvm = OneClassSVMBoundary(gamma=1.0, nu=nu).fit(ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svdd_poly = SVDDBoundary(kernel="poly", gamma=1.0, nu=nu, degree=3, coef0=1.0)
        svdd_poly.fit(ref)

    mem_oc = ocsvm.evaluate_batch(test_points)
    mem_poly = svdd_poly.evaluate_batch(test_points)
    assert np.sum(mem_oc != mem_poly) > 0


def test_svdd_linear_kernel_is_a_ball_around_the_weighted_center():
    """Linear SVDD: decision value is R^2 - ||x - a||^2 with a = sum_i alpha_i x_i."""
    ref = _single_blob(seed=5)
    svdd = SVDDBoundary(kernel="linear", nu=0.1).fit(ref)
    assert svdd.alpha is not None
    center = ref @ svdd.alpha
    test = _uniform_test_points(seed=6, low=-2.0, high=2.0, n=50)
    expected = svdd._r_squared - np.sum((test.T - center) ** 2, axis=1)
    np.testing.assert_allclose(svdd.decision_function(test), expected, atol=1e-9)
    assert svdd(center)
    assert not svdd(center + 10.0)


def test_svdd_poly_kernel_separates_center_from_far_points():
    ref = _single_blob(seed=7)
    svdd = SVDDBoundary(kernel="poly", degree=2, coef0=1.0, nu=0.1).fit(ref)
    center = ref.mean(axis=1)
    assert svdd(center)
    assert not svdd(center + 10.0)


def test_svdd_single_point_call_matches_batch():
    ref = _single_blob(seed=8)
    svdd = SVDDBoundary(kernel="linear", nu=0.1).fit(ref)
    test = _uniform_test_points(seed=9, low=-1.0, high=1.0, n=30)
    batch = svdd.evaluate_batch(test)
    singles = np.array([svdd(test[:, i]) for i in range(test.shape[1])])
    np.testing.assert_array_equal(singles, batch)
    assert batch.any()
    assert not batch.all()


def test_svdd_pairwise_kernel_rejects_unknown_kernel() -> None:
    x = np.zeros((2, 3))
    with pytest.raises(ValueError, match="unknown SVDD kernel"):
        _pairwise_kernel(x, x, kernel="bogus", gamma=1.0, degree=3, coef0=0.0)  # ty: ignore[invalid-argument-type]


def test_svdd_kernel_self_rejects_unknown_kernel() -> None:
    with pytest.raises(ValueError, match="unknown SVDD kernel"):
        _kernel_self(np.zeros((2, 3)), kernel="bogus", gamma=1.0, degree=3, coef0=0.0)  # ty: ignore[invalid-argument-type]


def _unfitted_svdd() -> SVDDBoundary:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return SVDDBoundary(kernel="linear")


@pytest.mark.parametrize(
    "make_boundary", [OneClassSVMBoundary, _unfitted_svdd], ids=["ocsvm", "svdd"]
)
def test_unfitted_boundary_reports_neutral_metadata(
    make_boundary: Callable[[], OneClassSVMBoundary | SVDDBoundary],
) -> None:
    monitor = make_boundary()
    assert monitor._estimate_coverage() == {}
    assert monitor._calculate_conservatism() == pytest.approx(0.5)
    assert monitor.decision_boundary["coverage"] == {}
