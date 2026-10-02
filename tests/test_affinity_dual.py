# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
import numpy as np
import pytest

from autosafe._affinity import (
    affinity_diag,
    affinity_diag_dual,
    affinity_full_dense_dual,
)
from autosafe.typing import (
    Matrix,
    NPMatrix,
    NPSquareMatrix,
    NPVector,
    SquareMatrix,
    Vector,
)


def _dual(
    anchors: Matrix | NPMatrix,
    inv_diag: SquareMatrix | NPSquareMatrix,
    x: Vector | NPMatrix,
) -> tuple[NPVector, NPVector]:
    a, s = affinity_diag_dual(
        jnp.asarray(anchors), jnp.asarray(inv_diag), jnp.asarray(x)
    )
    return np.asarray(a), np.asarray(s)


def test_dual_matches_linear_and_log_identity_small():
    rng = np.random.default_rng(0)
    anchors = rng.normal(size=(20, 3))
    inv_diag = np.full((20, 3), 4.0)
    x = rng.normal(size=(50, 3))
    a_ref = np.asarray(
        affinity_diag(jnp.asarray(anchors), jnp.asarray(inv_diag), jnp.asarray(x))
    )
    alpha, surv = _dual(anchors, inv_diag, x)
    np.testing.assert_allclose(alpha, a_ref, rtol=0, atol=1e-15)
    np.testing.assert_allclose(alpha, -np.expm1(surv), rtol=0, atol=1e-12)


def test_linear_saturates_log_discriminates_many_anchors():
    # Regression for the full-dataset bug: many domain-wide kernels.
    rng = np.random.default_rng(1)
    anchors = rng.uniform(-1.0, 1.0, size=(5000, 5))
    inv_diag = np.ones((5000, 5))
    x = rng.uniform(-1.0, 1.0, size=(100, 5))
    alpha, surv = _dual(anchors, inv_diag, x)
    assert np.allclose(alpha, 1.0)
    assert np.all(np.isfinite(surv))
    assert np.all(surv < 0.0)
    assert np.unique(surv).size > 50


@pytest.mark.filterwarnings("ignore: divide by zero encountered in log1p")
def test_zeta_one_selects_exact_anchor_hits_only():
    rng = np.random.default_rng(2)
    anchors = rng.normal(size=(10, 3))
    inv_diag = np.full((10, 3), 2.0)
    x = np.vstack([anchors[0], rng.normal(size=(5, 3))])
    _, surv = _dual(anchors, inv_diag, x)
    limit = np.log1p(-1.0)  # == -inf
    in_odd = surv <= limit
    assert in_odd[0]
    assert not in_odd[1:].any()


def _reference_dual(
    anchors: NPMatrix, sigma_inv: np.ndarray, x: NPMatrix
) -> tuple[NPVector, NPVector]:
    """NumPy reference: alpha = 1 - prod(1 - k_i), survival = sum log(1 - k_i).

    Returns:
        tuple[NPVector, NPVector]: (alpha, survival), each (M,).
    """
    diff = x[None, :, :] - anchors[:, None, :]
    mahal = np.einsum("nmd,nde,nme->nm", diff, sigma_inv, diff)
    k = np.exp(-0.5 * mahal)
    return 1.0 - np.prod(1.0 - k, axis=0), np.sum(np.log1p(-k), axis=0)


def _random_spd_inverses(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    a = rng.normal(size=(n, d, d))
    return np.einsum("nij,nkj->nik", a, a) + 0.5 * np.eye(d)


def test_full_dense_dual_matches_numpy_reference():
    # Chunk sizes that divide neither N nor M exercise the padding of
    # both the anchor and the point tiles.
    rng = np.random.default_rng(3)
    anchors = rng.normal(size=(37, 3))
    sigma_inv = _random_spd_inverses(rng, 37, 3)
    x = rng.normal(size=(21, 3))
    alpha, surv = affinity_full_dense_dual(
        jnp.asarray(anchors),
        jnp.asarray(sigma_inv),
        jnp.asarray(x),
        anchor_chunk=16,
        point_chunk=8,
    )
    alpha_ref, surv_ref = _reference_dual(anchors, sigma_inv, x)
    np.testing.assert_allclose(np.asarray(alpha), alpha_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(surv), surv_ref, rtol=1e-10)


def test_full_dense_dual_with_diagonal_inverse_matches_diag_dual():
    rng = np.random.default_rng(4)
    anchors = rng.normal(size=(30, 4))
    inv_diag = rng.uniform(0.5, 3.0, size=(30, 4))
    x = rng.normal(size=(40, 4))
    stack = np.einsum("nd,de->nde", inv_diag, np.eye(4))
    alpha_full, surv_full = affinity_full_dense_dual(
        jnp.asarray(anchors), jnp.asarray(stack), jnp.asarray(x)
    )
    alpha_diag, surv_diag = _dual(anchors, inv_diag, x)
    np.testing.assert_allclose(np.asarray(alpha_full), alpha_diag, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(surv_full), surv_diag, rtol=1e-12)
