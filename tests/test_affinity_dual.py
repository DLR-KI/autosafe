# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import jax.numpy as jnp
import numpy as np

from autosafe._affinity import affinity_diag, affinity_diag_dual
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
