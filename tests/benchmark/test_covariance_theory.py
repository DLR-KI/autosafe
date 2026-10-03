# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Numerical verification of the closed-form covariance-structure claims.

These are not tests of ``src/autosafe`` but of the *mathematics* the paper's
benchmark suite and the camera-ready appendix rely on. They exist so that a
claim like "the penalty is exactly ``sqrt((cond+1)/2)`` at 45 degrees" cannot
silently rot while the paper cites it.

Covers: Lemma 2.1 (extremal directional bound), the ``tau <= w`` Cauchy-Schwarz
ordering (Eq. 2.1), and Corollary 2.1 (a), (c'), (d).
"""

import numpy as np
import pytest
from scipy.linalg import hadamard
from scipy.optimize import minimize

RNG_SEED = 0


def _w(sigma: np.ndarray, u: np.ndarray) -> float:
    """Directional width ``w_Sigma(u) = sqrt(u^T Sigma u)``.

    Args:
        sigma (np.ndarray): (n, n) SPD covariance.
        u (np.ndarray): (n,) unit direction.

    Returns:
        float: The directional width.
    """
    return float(np.sqrt(u @ sigma @ u))


def _tau(sigma: np.ndarray, u: np.ndarray) -> float:
    """Mahalanobis half-width ``tau_Sigma(u) = (u^T Sigma^-1 u)^-1/2``.

    Args:
        sigma (np.ndarray): (n, n) SPD covariance.
        u (np.ndarray): (n,) unit direction.

    Returns:
        float: The Mahalanobis half-width.
    """
    return float(1.0 / np.sqrt(u @ np.linalg.inv(sigma) @ u))


def _random_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(n, n))
    return a @ a.T + 0.5 * np.eye(n)


@pytest.mark.parametrize("trial", range(5))
def test_lemma_2_1_extremal_directional_bound(trial: int) -> None:
    """min{v^T Sigma^-1 v : u.v >= delta} == delta^2 / (u^T Sigma u), attained."""
    rng = np.random.default_rng(RNG_SEED + trial)
    n = 4
    sigma = _random_spd(n, rng)
    sigma_inv = np.linalg.inv(sigma)
    u = rng.normal(size=n)
    u /= np.linalg.norm(u)
    delta = 1.7

    res = minimize(
        lambda v: v @ sigma_inv @ v,
        x0=u * delta,
        constraints=[{"type": "ineq", "fun": lambda v: u @ v - delta}],
    )
    closed_form = delta**2 / (u @ sigma @ u)
    assert res.fun == pytest.approx(closed_form, rel=1e-6)

    # The minimizer is v* = (delta / w^2) Sigma u.
    v_star = (delta / (u @ sigma @ u)) * (sigma @ u)
    assert v_star @ sigma_inv @ v_star == pytest.approx(closed_form, rel=1e-12)
    assert u @ v_star == pytest.approx(delta, rel=1e-12)


@pytest.mark.parametrize("trial", range(5))
def test_tau_le_w_with_equality_iff_eigenvector(trial: int) -> None:
    """Eq. (2.1): tau <= w always, with equality exactly on eigenvectors."""
    rng = np.random.default_rng(RNG_SEED + trial)
    n = 4
    sigma = _random_spd(n, rng)
    u = rng.normal(size=n)
    u /= np.linalg.norm(u)
    assert _tau(sigma, u) <= _w(sigma, u) + 1e-12

    eigvecs = np.linalg.eigh(sigma)[1]
    for j in range(n):
        e = eigvecs[:, j]
        assert _tau(sigma, e) == pytest.approx(_w(sigma, e), rel=1e-10)


def test_cor_2_1a_axis_aligned_boundary_costs_nothing() -> None:
    """P(e_j) == 1 exactly: a diagonal kernel pays no penalty on an axis-aligned face."""
    rng = np.random.default_rng(RNG_SEED)
    n = 5
    sigma = _random_spd(n, rng)
    diag = np.diag(np.diag(sigma))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        assert _w(diag, e) / _w(sigma, e) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("cond", [2.0, 5.0, 20.0, 100.0])
def test_cor_2_1d_45_degree_band(cond: float) -> None:
    """2D at 45 deg: Diag(Sigma) is isotropic and P == sqrt((cond+1)/2)."""
    theta = np.pi / 4
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    sigma = rot @ np.diag([cond, 1.0]) @ rot.T
    diag = np.diag(np.diag(sigma))
    # The band normal is the minor eigenvector.
    u = rot @ np.array([0.0, 1.0])

    # At 45 degrees the best axis-aligned fit is exactly isotropic -- the punchline.
    assert np.allclose(diag, diag[0, 0] * np.eye(2))
    assert _w(diag, u) / _w(sigma, u) == pytest.approx(
        np.sqrt((cond + 1) / 2), rel=1e-10
    )


@pytest.mark.parametrize("n", [2, 4, 8, 16, 64])
def test_cor_2_1c_prime_worst_case_and_cond_never_attained(n: int) -> None:
    """P^2 == ((n-1)cond+1)/n in a balanced basis, strictly below cond, rising to it."""
    cond = 50.0
    q = hadamard(n) / np.sqrt(n)  # orthonormal, all entries +-1/sqrt(n)
    lam = np.array([cond] * (n - 1) + [1.0])
    sigma = q @ np.diag(lam) @ q.T
    diag = np.diag(np.diag(sigma))
    u = q[:, -1]  # minor eigenvector

    # Balanced basis: every diagonal entry equals tr(Sigma)/n.
    assert np.allclose(np.diag(sigma), np.trace(sigma) / n)

    p_sq = (_w(diag, u) / _w(sigma, u)) ** 2
    assert p_sq == pytest.approx(((n - 1) * cond + 1) / n, rel=1e-10)
    # sqrt(cond) is an upper bound that is never attained at finite n.
    assert p_sq < cond


def test_cor_2_1b_cond_is_a_uniform_upper_bound() -> None:
    """P(u) <= sqrt(cond(Sigma)) for random Sigma and random directions."""
    rng = np.random.default_rng(RNG_SEED + 99)
    for _ in range(200):
        n = int(rng.integers(2, 7))
        sigma = _random_spd(n, rng)
        diag = np.diag(np.diag(sigma))
        u = rng.normal(size=n)
        u /= np.linalg.norm(u)
        eig = np.linalg.eigvalsh(sigma)
        cond = eig[-1] / eig[0]
        assert (_w(diag, u) / _w(sigma, u)) ** 2 <= cond * (1 + 1e-10)


def test_prop_4_noisy_or_coverage_bound() -> None:
    """Prop. 4 (4.1): alpha >= 1-(1-e^-0.5)^m, shape-independent, with the quoted values."""
    base = 1.0 - np.exp(-0.5)
    assert base == pytest.approx(0.393469, abs=1e-6)
    assert 1.0 - base**5 == pytest.approx(0.9906, abs=1e-4)
    assert 1.0 - base**10 == pytest.approx(0.9999, abs=1e-4)

    # Shape independence: any Sigma, any anchors on its unit-Mahalanobis sphere.
    rng = np.random.default_rng(RNG_SEED + 7)
    for _ in range(20):
        n = 3
        sigma = _random_spd(n, rng)
        sigma_inv = np.linalg.inv(sigma)
        m = 6
        chol = np.linalg.cholesky(sigma)
        dirs = rng.normal(size=(m, n))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        # Points at exactly Mahalanobis distance 1 from the origin.
        offsets = dirs @ chol.T
        alphas = np.exp(-0.5 * np.einsum("md,de,me->m", offsets, sigma_inv, offsets))
        alpha = 1.0 - np.prod(1.0 - alphas)
        assert alpha >= 1.0 - base**m - 1e-12
