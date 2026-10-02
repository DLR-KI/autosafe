# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the synthetic ODD registry."""

import numpy as np
import pytest

from experiments.benchmark.synthetic_odds import get_odd, registry_names


def test_registry_complete() -> None:
    """All named ODDs are registered (corr2d added for the covariance-structure ablation)."""
    assert set(registry_names()) == {
        "linear2d",
        "annulus2d",
        "twoblobs2d",
        "banana2d",
        "poly5d",
        "poly10d",
        "corr2d",
    }


@pytest.mark.parametrize("name", registry_names())
def test_contains_shape_and_dtype(name: str) -> None:
    """contains() returns a (M,) boolean array."""
    odd = get_odd(name)
    rng = np.random.default_rng(0)
    pts = rng.uniform(odd.lower, odd.upper, size=(50, odd.dim))
    labels = odd.contains(pts)
    assert labels.shape == (50,)
    assert labels.dtype == bool


@pytest.mark.parametrize("name", registry_names())
def test_sample_id_is_inside(name: str) -> None:
    """sample_id() returns points that all satisfy contains()."""
    odd = get_odd(name)
    rng = np.random.default_rng(1)
    pts = odd.sample_id(200, rng)
    assert pts.shape == (200, odd.dim)
    assert odd.contains(pts).all()


def test_annulus_hole_and_ring() -> None:
    """The annulus excludes its hole and includes the ring."""
    odd = get_odd("annulus2d")
    assert not odd.contains(np.array([[0.0, 0.0]]))[0]  # in the hole
    assert odd.contains(np.array([[0.0, 2.5]]))[0]  # on the ring


def test_twoblobs_disconnected() -> None:
    """Two-blobs includes both centers, excludes the midpoint."""
    odd = get_odd("twoblobs2d")
    assert odd.contains(np.array([[-2.5, -2.5]]))[0]
    assert odd.contains(np.array([[2.5, 2.5]]))[0]
    assert not odd.contains(np.array([[0.0, 0.0]]))[0]
