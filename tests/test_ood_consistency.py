# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the OOD consistency adjustment."""

import copy
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from loguru import logger

import autosafe
from autosafe.exceptions import (
    KernelSaturationError,
    NearAnchorOODWarning,
    OODAnchorCoincidenceError,
    OODConsistencyNotReachedError,
    OODDimensionMismatchError,
    RowShapeMismatchError,
)
from autosafe.kernels.rbf import RBFKernel as _RBFKernel
from autosafe.ood_consistency import (
    batch_shrink_count,
    invert_covariance,
    kernel_over_points,
)
from autosafe.sample import Sample
from autosafe.samples import Samples, rows_in

if TYPE_CHECKING:
    from autosafe.kernels.rbf import RBFKernel


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


def _build_separable_fixture() -> tuple[Samples, np.ndarray]:
    """Build a 2D ODD of well-separated anchors plus one OOD point each.

    Each OOD point is close to exactly one anchor and negligibly close to
    the others, so a single kernel dominates every violation. This is the
    regime in which the closed-form batched jump applies.

    Returns:
        tuple[Samples, np.ndarray]: The constructed ODD and 5 OOD points.
    """
    sep = 6.0
    anchors = np.array([[i * sep, 0.0] for i in range(5)], dtype=float)
    odd = Samples(
        [Sample(x=row.copy()) for row in anchors],
        closest_sample_mode="global",
        kernel_cls="RBF",
    )
    ood = np.array([[i * sep + 0.02, 0.0] for i in range(5)], dtype=float)
    return odd, ood


def test_constraint_satisfied() -> None:
    """After enforcement, all OOD affinities are <= xi and work was done."""
    odd, ood = _build_fixture()
    summary = odd.enforce_ood_consistency(ood, xi=0.05)
    assert np.all(np.asarray(odd(ood)) <= 0.05 + 1e-12)
    assert int(cast("int", summary["iterations"])) > 0
    assert float(cast("float", summary["max_ood_affinity"])) <= 0.05 + 1e-12


def test_idempotent() -> None:
    """A second call does nothing and changes no sigma."""
    odd, ood = _build_fixture()
    odd.enforce_ood_consistency(ood, xi=0.05)
    sigmas_before = [np.array(s.kernel.sigma, copy=True) for s in odd.samples]  # ty: ignore[unresolved-attribute]
    summary2 = odd.enforce_ood_consistency(ood, xi=0.05)
    assert int(cast("int", summary2["iterations"])) == 0
    for before, s in zip(sigmas_before, odd.samples, strict=True):
        assert np.array_equal(before, np.asarray(s.kernel.sigma))  # ty: ignore[unresolved-attribute]


def test_order_independent() -> None:
    """Reversing the OOD order yields bitwise-identical kernel sigmas."""
    odd_fwd, ood = _build_fixture()
    odd_rev = copy.deepcopy(odd_fwd)
    odd_fwd.enforce_ood_consistency(ood, xi=0.05)
    odd_rev.enforce_ood_consistency(ood[::-1], xi=0.05)
    for s_fwd, s_rev in zip(odd_fwd.samples, odd_rev.samples, strict=True):
        assert np.array_equal(
            np.asarray(s_fwd.kernel.sigma),  # ty: ignore[unresolved-attribute]
            np.asarray(s_rev.kernel.sigma),  # ty: ignore[unresolved-attribute]
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
    assert int(cast("int", summary["iterations"])) == 0


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


def test_rows_in_exact_and_edges() -> None:
    """Exact row membership, empty inputs, -0.0, and column mismatch."""
    ref = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    query = np.array([[3.0, 4.0], [9.0, 9.0], [1.0, 2.0]])
    assert list(rows_in(query, ref)) == [True, False, True]
    assert rows_in(np.empty((0, 2)), ref).shape == (0,)
    assert not rows_in(query, np.empty((0, 2))).any()
    # np.unique compares by value, so -0.0 and 0.0 are the same point.
    assert rows_in(np.array([[-0.0, 0.0]]), np.array([[0.0, 0.0]]))[0]
    # A row differing in a single dimension is not a member.
    assert not rows_in(np.array([[3.0, 4.000000001]]), ref)[0]
    with pytest.raises(RowShapeMismatchError, match="column mismatch"):
        rows_in(np.zeros((2, 3)), ref)


def test_coincident_ood_raises() -> None:
    """An OOD point equal to an anchor fails fast, naming the cause.

    This is the ``eval-vcas-rbf-ood`` misconfiguration: the affinity at
    an anchor is 1 for every covariance, so the loop cannot converge.
    """
    odd, ood = _build_fixture()
    poisoned = np.vstack([ood, np.asarray(odd.samples[3].x, dtype=float)])
    with pytest.raises(OODAnchorCoincidenceError) as excinfo:
        odd.enforce_ood_consistency(poisoned, xi=0.05)
    message = str(excinfo.value)
    assert "1 of 6 OOD points coincide exactly" in message
    assert "disjoint from the OOD set" in message
    assert "index 5" in message
    # Subclasses ValueError, so existing callers still catch it.
    assert isinstance(excinfo.value, ValueError)


def test_wrong_ood_dimension_raises() -> None:
    """OOD points with the wrong column count are rejected."""
    odd, _ = _build_fixture()
    with pytest.raises(OODDimensionMismatchError, match="columns"):
        odd.enforce_ood_consistency(np.zeros((3, 5)), xi=0.05)


def test_incremental_matches_full_recompute() -> None:
    """Never refreshing agrees with always refreshing, to round-off.

    Guards the incremental log-survival update: a wrong update would
    diverge from the exact recomputation.
    """
    odd_exact, ood = _build_fixture()
    odd_incr = copy.deepcopy(odd_exact)
    summary_exact = odd_exact.enforce_ood_consistency(ood, xi=0.05, refresh_interval=1)
    summary_incr = odd_incr.enforce_ood_consistency(
        ood, xi=0.05, refresh_interval=10**9
    )
    assert summary_exact["iterations"] == summary_incr["iterations"]
    # Two exact sweeps only: the initial one and the exit-path
    # verification (the loop never exits on a drifted estimate). That is
    # the whole point, the other ~370 iterations are incremental.
    assert int(cast("int", summary_incr["exact_recomputations"])) == 2
    assert int(cast("int", summary_incr["exact_recomputations"])) < int(
        cast("int", summary_incr["iterations"])
    )
    for s_exact, s_incr in zip(odd_exact.samples, odd_incr.samples, strict=True):
        assert np.allclose(
            np.asarray(s_exact.kernel.sigma),  # ty: ignore[unresolved-attribute]
            np.asarray(s_incr.kernel.sigma),  # ty: ignore[unresolved-attribute]
            rtol=1e-9,
            atol=0.0,
        )


def test_final_survival_matches_summary() -> None:
    """The reported max affinity agrees with an independent recompute."""
    odd, ood = _build_fixture()
    summary = odd.enforce_ood_consistency(ood, xi=0.05)
    alpha = -np.expm1(np.asarray(odd.affinity_dual(ood)[1]))
    assert float(np.max(alpha)) <= 0.05 + 1e-12
    assert float(np.max(alpha)) == pytest.approx(
        float(cast("float", summary["max_ood_affinity"])), abs=1e-12
    )


@pytest.mark.filterwarnings("ignore:overflow encountered in divide:RuntimeWarning")
def test_saturation_guard_raises() -> None:
    """A kernel driven to float64 saturation raises instead of NaN-ing."""
    # The anchor sits at the ORIGIN so that a 1e-160 offset is actually
    # representable; added to a coordinate of order 1 it would round away
    # and the point would be exactly coincident instead.
    anchors = np.array([[0.0, 0.0], [10.0, 10.0]])
    odd = Samples(
        [Sample(x=row.copy()) for row in anchors],
        closest_sample_mode="global",
        kernel_cls="RBF",
    )
    kern = cast("RBFKernel", odd.samples[0].kernel)
    kern.sigma = np.eye(2) * 1e-300
    kern.sigma_inv = np.eye(2) * 1e300
    kern._refresh_sigma_cache()
    odd.invalidate_batch_cache()
    # q = 1e-320 / 1e-300 = 1e-20, so k ~ 1 and kernel 0 is dominant, but
    # the point is NOT coincident so the precondition check passes.
    # sigma_inv overflows after log(1e8)/log(1/0.9) ~ 175 shrinks.
    ood = np.array([[1e-160, 0.0]])
    with pytest.raises(KernelSaturationError, match="saturated"):
        odd.enforce_ood_consistency(ood, xi=1e-9, max_iterations=100_000)


def test_batch_jump_satisfies_constraint() -> None:
    """The opt-in closed-form jump still satisfies alpha <= xi."""
    odd, ood = _build_fixture()
    summary = odd.enforce_ood_consistency(ood, xi=0.05, batch_jump=True)
    assert np.all(np.asarray(odd(ood)) <= 0.05 + 1e-12)
    assert summary["batch_jump"] is True


def test_batch_jump_reduces_iterations_when_separable() -> None:
    """The closed-form jump collapses a run when one kernel dominates.

    The formula solves for the shrinks of a SINGLE kernel that fix the
    selected point, so it only bites when that kernel alone can fix it.
    Here each OOD point sits next to one well-separated anchor, and the
    whole run collapses to one iteration per point.
    """
    odd_plain, ood = _build_separable_fixture()
    odd_jump = copy.deepcopy(odd_plain)
    plain = odd_plain.enforce_ood_consistency(ood, xi=0.05)
    jump = odd_jump.enforce_ood_consistency(ood, xi=0.05, batch_jump=True)
    assert int(cast("int", jump["iterations"])) < int(cast("int", plain["iterations"]))
    assert int(cast("int", jump["iterations"])) == len(ood)
    assert np.all(np.asarray(odd_jump(ood)) <= 0.05 + 1e-12)


def test_batch_jump_is_a_noop_in_the_dense_regime() -> None:
    """With overlapping kernels the jump degrades gracefully to c^1.

    When the kernels OTHER than the dominant one already push the point
    above xi, no single-kernel closed form exists; the implementation
    falls back to a plain Algorithm 1 step rather than over-shrinking.
    """
    odd_plain, ood = _build_fixture()
    odd_jump = copy.deepcopy(odd_plain)
    plain = odd_plain.enforce_ood_consistency(ood, xi=0.05)
    jump = odd_jump.enforce_ood_consistency(ood, xi=0.05, batch_jump=True)
    assert int(cast("int", jump["iterations"])) <= int(cast("int", plain["iterations"]))
    assert np.all(np.asarray(odd_jump(ood)) <= 0.05 + 1e-12)


def test_progress_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    """The loop reports progress instead of running silently."""
    odd, ood = _build_fixture()
    records: list[str] = []
    sink_id = logger.add(records.append, level="INFO", format="{message}")
    try:
        odd.enforce_ood_consistency(ood, xi=0.05, log_interval=1)
    finally:
        logger.remove(sink_id)

    assert any("enforce_ood_consistency: start" in m for m in records)
    assert any("enforce_ood_consistency: done" in m for m in records)
    written = capsys.readouterr().out
    assert "enforce_ood_consistency: iter=" in written
    assert "worst_alpha" in written


def test_progress_reporting_can_be_disabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """log_interval <= 0 suppresses the per-iteration lines."""
    odd, ood = _build_fixture()
    odd.enforce_ood_consistency(ood, xi=0.05, log_interval=0)
    assert "enforce_ood_consistency: iter=" not in capsys.readouterr().out


def test_sigma_and_sigma_inv_stay_consistent() -> None:
    """sigma_inv is recomputed from sigma, not scaled alongside it.

    Scaling both independently by c and 1/c lets them drift apart, since
    the two are not exact reciprocals in binary floating point.
    """
    odd, ood = _build_fixture()
    odd.enforce_ood_consistency(ood, xi=0.05)
    for sample in odd.samples:
        kern = cast("RBFKernel", sample.kernel)
        sigma = np.asarray(kern.sigma, dtype=float)
        sigma_inv = np.asarray(kern.sigma_inv, dtype=float)
        assert np.allclose(sigma @ sigma_inv, np.eye(sigma.shape[0]), atol=1e-12)


def test_near_anchor_ood_warns() -> None:
    """OOD points numerically on top of an anchor are called out.

    The point is distinct from the anchor, so the exact coincidence
    check passes, but the squared distance underflows to 0 and the
    log-survival is -inf. That is not convergeable either, so the run
    warns first and then fails on its own terms.
    """
    anchors = np.array([[0.0, 0.0], [10.0, 10.0]])
    odd = Samples(
        [Sample(x=row.copy()) for row in anchors],
        closest_sample_mode="global",
        kernel_cls="RBF",
    )
    ood = np.array([[1e-170, 0.0]])
    assert not rows_in(ood, anchors)[0]  # not an exact coincidence
    with (
        pytest.warns(NearAnchorOODWarning, match="log-survival -inf"),
        pytest.raises((KernelSaturationError, OODConsistencyNotReachedError)),
    ):
        odd.enforce_ood_consistency(ood, xi=0.5, max_iterations=500)


def test_max_iterations_message_is_honest() -> None:
    """The cap error no longer blames xi alone."""
    odd, ood = _build_fixture()
    with pytest.raises(OODConsistencyNotReachedError) as excinfo:
        odd.enforce_ood_consistency(ood, xi=1e-6, max_iterations=2)
    message = str(excinfo.value)
    assert "OOD consistency not reached" in message
    assert "raise max_iterations" in message
    assert "(near-)coincident" in message
    assert "xi may be too small" not in message


def test_full_covariance_constraint_satisfied() -> None:
    """The adjustment also holds for rotated (non-diagonal) kernels."""
    odd, ood = _build_fixture()
    sigma = np.array([[1.0, 0.6], [0.6, 1.0]])
    for s in odd.samples:
        assert isinstance(s.kernel, _RBFKernel)
        s.kernel.update(sigma=sigma)
    odd.invalidate_batch_cache()
    assert not odd.batch_arrays()[2]

    xi = 0.3
    assert float(np.max(np.asarray(odd.affinity_dual(ood)[0]))) > xi
    summary = odd.enforce_ood_consistency(ood, xi=xi, shrink_factor=0.9)
    assert cast("int", summary["iterations"]) > 0
    alpha, _ = odd.affinity_dual(ood)
    assert float(np.max(np.asarray(alpha))) <= xi + 1e-9
    # Shrinking keeps sigma and sigma_inv mutually consistent.
    for s in odd.samples:
        assert isinstance(s.kernel, _RBFKernel)
        assert s.kernel.sigma is not None
        assert s.kernel.sigma_inv is not None
        np.testing.assert_allclose(
            s.kernel.sigma @ s.kernel.sigma_inv, np.eye(2), atol=1e-8
        )


def test_invert_covariance_dense_and_singular() -> None:
    sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    np.testing.assert_allclose(invert_covariance(sigma), np.linalg.inv(sigma))
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert np.all(np.isinf(invert_covariance(singular)))


def test_kernel_over_points_requires_sigma_inv() -> None:
    with pytest.raises(RuntimeError, match="kernel has no sigma_inv"):
        kernel_over_points(_RBFKernel(x_i=np.zeros(2)), np.zeros((3, 2)))


@pytest.mark.parametrize(
    ("k_old_worst", "survival_worst", "log_target"),
    [
        (0.0, -1.0, -0.5),  # kernel value already zero
        (1.0, -1.0, -0.5),  # exact hit: no finite Mahalanobis distance
        # Other kernels contribute nothing (L(x*) == log(1 - k_i*)) and xi
        # is so close to 1 that 1 - exp(log_target) rounds to 1: b == 0.
        (0.5, float(np.log1p(-0.5)), -50.0),
    ],
    ids=["k_zero", "k_one", "b_underflow"],
)
def test_batch_shrink_count_degenerate_inputs_fall_back_to_one(
    k_old_worst: float, survival_worst: float, log_target: float
) -> None:
    sigma = np.eye(2)
    count = batch_shrink_count(
        survival_worst=survival_worst,
        log1m_old_worst=float(np.log1p(-k_old_worst)) if k_old_worst < 1 else -np.inf,
        k_old_worst=k_old_worst,
        k_vals=np.array([k_old_worst, 0.0]),
        i_star=0,
        log_target=log_target,
        shrink_factor=0.9,
        sigma=sigma,
        sigma_inv=sigma,
    )
    assert count == 1


def test_non_rbf_kernel_is_rejected() -> None:
    """OOD consistency only knows how to shrink RBF kernels."""
    odd = Samples(
        [Sample(x=np.array(p)) for p in ([0.0, 0.0], [1.0, 0.0])],
        closest_sample_mode="global",
        kernel_cls="Laplacian",
        kernel_kwargs={"alpha": 0.5},
    )
    with pytest.raises(TypeError, match="only supports RBFKernel"):
        odd.enforce_ood_consistency(np.array([[0.5, 0.1]]), xi=0.1)
