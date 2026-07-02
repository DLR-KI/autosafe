# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the OOD consistency adjustment."""

import copy
from pathlib import Path

import numpy as np
import pytest

import autosafe
from autosafe.sample import Sample
from autosafe.samples import Samples


def _build_fixture() -> tuple[Samples, np.ndarray]:
    """Build a small 2D ODD (two Gaussian blobs) plus OOD points.

    Returns:
        tuple[Samples, np.ndarray]: The constructed ODD and 5 OOD points
            placed between/near the blobs.
    """
    rng = np.random.default_rng(7)
    blob_a = rng.normal(loc=[-1.5, -1.5], scale=0.2, size=(20, 2))
    blob_b = rng.normal(loc=[1.5, 1.5], scale=0.2, size=(20, 2))
    anchors = np.vstack([blob_a, blob_b])
    odd = Samples(
        [Sample(x=row.copy()) for row in anchors],
        closest_sample_mode="global",
        kernel_cls="RBF",
    )
    # OOD points between and just outside the blobs.
    ood = np.array([
        [0.0, 0.0],
        [0.5, -0.5],
        [-0.5, 0.5],
        [-1.5, 1.5],
        [1.5, -1.5],
    ])
    return odd, ood


def test_constraint_satisfied() -> None:
    """After enforcement, all OOD affinities are <= xi and work was done."""
    odd, ood = _build_fixture()
    summary = odd.enforce_ood_consistency(ood, xi=0.05)
    assert np.all(np.asarray(odd(ood)) <= 0.05 + 1e-12)
    assert int(summary["iterations"]) > 0
    assert float(summary["max_ood_affinity"]) <= 0.05 + 1e-12


def test_idempotent() -> None:
    """A second call does nothing and changes no sigma."""
    odd, ood = _build_fixture()
    odd.enforce_ood_consistency(ood, xi=0.05)
    sigmas_before = [np.array(s.kernel.sigma, copy=True) for s in odd.samples]
    summary2 = odd.enforce_ood_consistency(ood, xi=0.05)
    assert int(summary2["iterations"]) == 0
    for before, s in zip(sigmas_before, odd.samples, strict=True):
        assert np.array_equal(before, np.asarray(s.kernel.sigma))


def test_order_independent() -> None:
    """Reversing the OOD order yields bitwise-identical kernel sigmas."""
    odd_fwd, ood = _build_fixture()
    odd_rev = copy.deepcopy(odd_fwd)
    odd_fwd.enforce_ood_consistency(ood, xi=0.05)
    odd_rev.enforce_ood_consistency(ood[::-1], xi=0.05)
    for s_fwd, s_rev in zip(odd_fwd.samples, odd_rev.samples, strict=True):
        assert np.array_equal(
            np.asarray(s_fwd.kernel.sigma), np.asarray(s_rev.kernel.sigma)
        )


def test_log_space_consistent() -> None:
    """The log-space survival view agrees with alpha <= xi."""
    odd, ood = _build_fixture()
    odd.enforce_ood_consistency(ood, xi=0.05)
    survival = np.asarray(odd.affinity_dual(ood)[1])
    assert np.all(survival >= np.log1p(-0.05) - 1e-12)


def test_validation_errors() -> None:
    """Out-of-range xi / shrink_factor raise, and a tiny cap raises."""
    odd, ood = _build_fixture()
    with pytest.raises(ValueError, match="xi must be in"):
        odd.enforce_ood_consistency(ood, xi=0.0)
    with pytest.raises(ValueError, match="shrink_factor must be in"):
        odd.enforce_ood_consistency(ood, xi=0.5, shrink_factor=1.0)
    with pytest.raises(RuntimeError, match="OOD consistency not reached"):
        odd.enforce_ood_consistency(ood, xi=1e-6, max_iterations=2)


def test_empty_ood_is_noop() -> None:
    """An empty OOD set returns zero iterations without error."""
    odd, _ = _build_fixture()
    summary = odd.enforce_ood_consistency(np.empty((0, 2)), xi=0.05)
    assert int(summary["iterations"]) == 0


def test_serialization_roundtrip(tmp_path: Path) -> None:
    """The adjusted sigma is persisted: reloaded affinities are unchanged."""
    odd, ood = _build_fixture()
    odd.enforce_ood_consistency(ood, xi=0.05)
    alpha_before = np.asarray(odd(ood))
    path = tmp_path / "adjusted_odd.json"
    autosafe.to_json(odd, path)
    reloaded = autosafe.from_json(path)
    alpha_after = np.asarray(reloaded(ood))
    assert np.allclose(alpha_before, alpha_after, atol=1e-9)
