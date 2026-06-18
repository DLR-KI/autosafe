# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import warnings

import numpy as np
import pytest

from autosafe.kernels.rbf import calibrate_rbf_scale, calibrate_rbf_scale_isotropic


def test_calibration_scale_equivariance():
    rng = np.random.default_rng(0)
    d_nn = rng.exponential(scale=0.01, size=(200, 3))
    a = np.array([2.0, 0.5, 10.0])
    kappa1, eta1 = calibrate_rbf_scale(d_nn)
    kappa2, eta2 = calibrate_rbf_scale(d_nn * a)
    np.testing.assert_allclose(kappa2, kappa1 * a**2, rtol=1e-12)
    np.testing.assert_allclose(eta2, eta1 / a, rtol=1e-12)
    # The sigma law is then equivariant: sigma'(a*d) == a^2 * sigma(d).
    d = d_nn[0]
    sigma1 = kappa1 * np.exp(-eta1 * d)
    sigma2 = kappa2 * np.exp(-eta2 * (d * a))
    np.testing.assert_allclose(sigma2, a**2 * sigma1, rtol=1e-12)


def test_calibration_degenerate_dimension_floor():
    rng = np.random.default_rng(1)
    d_nn = rng.exponential(scale=0.01, size=(100, 3))
    d_nn[:, 1] = 0.0  # discrete dimension: all duplicates
    kappa, eta = calibrate_rbf_scale(d_nn)
    assert np.all(np.isfinite(eta))
    assert np.all(eta > 0)
    assert np.all(kappa > 0)


def test_calibration_all_zero_raises():
    with pytest.raises(ValueError):  # noqa: PT011
        calibrate_rbf_scale(np.zeros((10, 2)))


def test_isotropic_calibration_scale_equivariance():
    rng = np.random.default_rng(2)
    d = rng.exponential(scale=0.02, size=500)
    k1, e1 = calibrate_rbf_scale_isotropic(d)
    k2, e2 = calibrate_rbf_scale_isotropic(d * 2.0)
    assert np.isclose(k2, 4.0 * k1)
    assert np.isclose(e2, e1 / 2.0)


def test_isotropic_calibration_ignores_duplicates_and_raises_on_all_zero():
    d = np.array([0.0, 0.0, 1.0, 3.0])
    k, e = calibrate_rbf_scale_isotropic(d, c=1.0, s=1.0)
    assert np.isclose(k, 4.0)
    assert np.isclose(e, 0.5)
    with pytest.raises(ValueError):  # noqa: PT011
        calibrate_rbf_scale_isotropic(np.zeros(5))


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
    with pytest.raises(ValueError):  # noqa: PT011
        kernel.update(x_nn=np.zeros(2), kappa=1.0, eta=1.0, lam=2.0)


def test_lam_zero_raises():
    from autosafe.kernels.rbf import RBFKernel

    kernel = RBFKernel(x_i=np.zeros(2))
    with pytest.raises(ValueError, match="lam must be strictly positive"):
        kernel.update(x_nn=np.ones(2), kappa=1.0, eta=1.0, lam=0.0)
