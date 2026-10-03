# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import warnings
from collections.abc import Callable

import numpy as np
import pytest

from autosafe.kernels.rbf import calibrate_rbf_scale_d_tilde, calibrate_rbf_scale_median


def test_calibration_scale_equivariance():
    rng = np.random.default_rng(0)
    d_nn = rng.exponential(scale=0.01, size=(200, 3))
    a = np.array([2.0, 0.5, 10.0])
    kappa_1, eta_1 = calibrate_rbf_scale_median(d_nn)
    kappa_2, eta_2 = calibrate_rbf_scale_median(d_nn * a)
    np.testing.assert_allclose(kappa_2, kappa_1 * a**2, rtol=1e-12)
    np.testing.assert_allclose(eta_2, eta_1 / a, rtol=1e-12)
    # The sigma law is then equivariant: sigma'(a*d) == a^2 * sigma(d).
    d = d_nn[0]
    sigma_1 = kappa_1 * np.exp(-eta_1 * d)
    sigma_2 = kappa_2 * np.exp(-eta_2 * (d * a))
    np.testing.assert_allclose(sigma_2, a**2 * sigma_1, rtol=1e-12)


def test_calibration_degenerate_dimension_floor():
    rng = np.random.default_rng(1)
    d_nn = rng.exponential(scale=0.01, size=(100, 3))
    d_nn[:, 1] = 0.0  # discrete dimension: all duplicates
    kappa, eta = calibrate_rbf_scale_median(d_nn)
    assert np.all(np.isfinite(eta))
    assert np.all(eta > 0)
    assert np.all(kappa > 0)


def test_calibration_all_zero_raises():
    with pytest.raises(
        ValueError, match="all per-dimension nn-distance medians are zero"
    ):
        calibrate_rbf_scale_median(np.zeros((10, 2)))


def test_isotropic_calibration_scale_equivariance():
    rng = np.random.default_rng(2)
    d = rng.exponential(scale=0.02, size=500)
    kappa_1, eta_1 = calibrate_rbf_scale_d_tilde(d)
    kappa_2, eta_2 = calibrate_rbf_scale_d_tilde(d * 2.0)
    assert np.isclose(kappa_2, 4.0 * kappa_1)
    assert np.isclose(eta_2, eta_1 / 2.0)


def test_isotropic_calibration_ignores_duplicates_and_raises_on_all_zero():
    d = np.array([0.0, 0.0, 1.0, 3.0])
    with pytest.warns(DeprecationWarning, match="use 'gamma'"):
        kappa, eta = calibrate_rbf_scale_d_tilde(d, c=1.0, s=1.0)
    assert np.isclose(kappa, 4.0)
    assert np.isclose(eta, 0.5)
    with pytest.raises(ValueError, match="all full-space nn distances are zero"):
        calibrate_rbf_scale_d_tilde(np.zeros(5))


def test_sigma_affine_floor_keeps_sigma_invertible():
    from autosafe.kernels.rbf import SIGMA_FLOOR_RATIO, RBFKernel

    x_i = np.zeros(3)
    x_nn = np.full(3, 1e6)  # isolated anchor
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any sigma-fix warning -> failure
        kernel = RBFKernel(x_i=x_i)
        kernel.update(x_nn=x_nn, kappa=1.0, eta=1.0)
    # sigma is floored at lam = SIGMA_FLOOR_RATIO * kappa, never singular
    assert kernel.sigma is not None
    np.testing.assert_allclose(np.diag(kernel.sigma), SIGMA_FLOOR_RATIO, rtol=1e-6)


def test_sigma_affine_floor_limits():
    from autosafe.kernels.rbf import RBFKernel

    kernel = RBFKernel(x_i=np.zeros(2))
    kernel.update(x_nn=np.zeros(2), kappa=2.0, eta=1.0)
    assert kernel.sigma is not None
    np.testing.assert_allclose(
        np.diag(kernel.sigma), 2.0, rtol=1e-12
    )  # sigma(0) == kappa
    with pytest.raises(ValueError):  # ruff:ignore[pytest-raises-too-broad]
        kernel.update(x_nn=np.zeros(2), kappa=1.0, eta=1.0, lam=2.0)


def test_lam_zero_raises():
    from autosafe.kernels.rbf import RBFKernel

    kernel = RBFKernel(x_i=np.zeros(2))
    with pytest.raises(ValueError, match="lam must be strictly positive"):
        kernel.update(x_nn=np.ones(2), kappa=1.0, eta=1.0, lam=0.0)


def _calibrate_median(**kwargs: float) -> tuple[object, object]:
    return calibrate_rbf_scale_median(np.ones((4, 2)), **kwargs)


def _calibrate_d_tilde(**kwargs: float) -> tuple[object, object]:
    return calibrate_rbf_scale_d_tilde(np.ones(4), **kwargs)


_CALIBRATORS = pytest.mark.parametrize(
    "calibrate", [_calibrate_median, _calibrate_d_tilde], ids=["median", "d_tilde"]
)


@_CALIBRATORS
def test_deprecated_c_alias_matches_gamma(
    calibrate: Callable[..., tuple[object, object]],
) -> None:
    with pytest.warns(DeprecationWarning, match="use 'gamma'"):
        kappa_c, eta_c = calibrate(c=2.0)
    kappa_g, eta_g = calibrate(gamma=2.0)
    np.testing.assert_array_equal(kappa_c, kappa_g)
    np.testing.assert_array_equal(eta_c, eta_g)


@_CALIBRATORS
def test_deprecated_c_alias_conflicting_gamma_raises(
    calibrate: Callable[..., tuple[object, object]],
) -> None:
    with pytest.raises(ValueError, match="cannot define different values"):
        calibrate(gamma=2.0, c=3.0)


@_CALIBRATORS
@pytest.mark.parametrize(("gamma", "s"), [(0.0, 3.0), (-1.0, 3.0), (1.0, 0.0)])
def test_non_positive_gamma_or_s_raises(
    calibrate: Callable[..., tuple[object, object]], gamma: float, s: float
) -> None:
    with pytest.raises(ValueError, match="gamma and s must be greater than 0"):
        calibrate(gamma=gamma, s=s)


def test_median_calibration_rejects_non_2d_input():
    with pytest.raises(ValueError, match=r"d_nn must have shape \(n_anchors, n_dims\)"):
        calibrate_rbf_scale_median(np.ones(4))  # ty: ignore[invalid-argument-type]


def test_d_tilde_calibration_rejects_non_1d_input():
    with pytest.raises(ValueError, match=r"d_nn_l2 must have shape \(n_anchors,\)"):
        calibrate_rbf_scale_d_tilde(np.ones((4, 2)))  # ty: ignore[invalid-argument-type]
