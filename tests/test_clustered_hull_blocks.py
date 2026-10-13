# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Blocked facet evaluation of the clustered convex-hull baseline."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from autosafe.odd.comparison import clustered_hull
from autosafe.odd.comparison.clustered_hull import ClusteredConvexHulls, _inside_hull


def test_blocked_hull_matches_full_product(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    hull = ConvexHull(rng.normal(size=(200, 4)))
    points = rng.normal(scale=0.8, size=(300, 4))
    eq = hull.equations
    full = np.all(points @ eq[:, :-1].T + eq[:, -1] <= np.finfo(float).eps, axis=1)
    # Force many tiny blocks: one facet per block for 300 points.
    monkeypatch.setattr(clustered_hull, "_MAX_BLOCK_ENTRIES", 300)
    np.testing.assert_array_equal(_inside_hull(points, eq), full)
    assert full.any()
    assert not full.all()


def test_evaluate_batch_accepts_device_arrays() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(3, 120))
    monitor = ClusteredConvexHulls(n_clusters=2)
    monitor.fit(data)
    queries = rng.normal(size=(3, 50))
    host = monitor.evaluate_batch(queries)
    device = monitor.evaluate_batch(jnp.asarray(queries))
    np.testing.assert_array_equal(host, device)
