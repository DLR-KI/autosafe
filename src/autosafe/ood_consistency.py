# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Out-of-distribution consistency adjustment for a kernel-based ODD.

Enforces ``alpha(x) <= xi`` on a set of OOD points by repeatedly
shrinking the covariance of the kernel that dominates the most-violated
point. See docs/ood-consistency.md for the mathematics, the
preconditions this depends on, and the cost analysis.

The whole algorithm lives here rather than on ``Samples`` because
nothing in it is reused elsewhere: the class exposes only the
public ``enforce_ood_consistency`` entry point, which delegates to
:func:`enforce_ood_consistency` below.
"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import tqdm.rich
from loguru import logger

from autosafe.exceptions import (
    KernelSaturationError,
    NearAnchorOODWarning,
    OODAnchorCoincidenceError,
    OODConsistencyNotReachedError,
    OODDimensionMismatchError,
)
from autosafe.kernels.rbf import RBFKernel
from autosafe.pointsets import rows_in

if TYPE_CHECKING:
    from autosafe.samples import Samples
    from autosafe.typing import Matrix, NPMatrix

DEFAULT_SHRINK_FACTOR = 0.9
DEFAULT_MAX_ITERATIONS = 1_000_000
DEFAULT_REFRESH_INTERVAL = 1000
DEFAULT_LOG_INTERVAL = 1000


def invert_covariance(
    sigma: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Invert a covariance matrix, recomputed rather than rescaled.

    The adjustment scales ``Sigma`` by ``c``, and it would be cheaper to
    scale ``Sigma^-1`` by ``1/c`` in step. It is not done that way:
    ``c`` and ``1/c`` are not exact reciprocals in binary floating
    point, so two independently-scaled arrays drift apart over many
    iterations and stop being inverses of each other, while the kernel
    reads only ``sigma_inv`` and the serializer persists both.
    Recomputing keeps the pair consistent by construction. For the
    diagonal case---what the sigma law always produces---this is one
    reciprocal per dimension, so the cost is negligible.

    Args:
        sigma (npt.NDArray[np.float64]): Covariance, shape (n, n).

    Returns:
        npt.NDArray[np.float64]: Its inverse, shape (n, n). Entries may
            be non-finite if ``sigma`` has underflowed; the caller is
            expected to check.
    """
    diagonal = np.diag(np.diag(sigma))
    if np.array_equal(sigma, diagonal):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return np.diag(1.0 / np.diag(sigma))
    try:
        return np.asarray(np.linalg.inv(sigma), dtype=float)
    except np.linalg.LinAlgError:
        # Singular after shrinking: report as saturation upstream.
        return np.full_like(sigma, np.inf)


def kernel_over_points(
    kern: RBFKernel, points: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate one kernel at many points, in linear and log space.

    Pure NumPy and shape-unambiguous: ``points`` is always read as
    (M, n_dims), unlike ``RBFKernel.__call__`` which guesses the
    orientation. The ``log(1 - k)`` branch mirrors the JAX tile in
    ``_affinity._build_tile("diag_dual")`` so the incremental update and
    the exact refresh agree to round-off.

    Args:
        kern (RBFKernel): The kernel to evaluate.
        points (npt.NDArray[np.float64]): Query points, (M, n_dims).

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            ``(k, log1m_k)``, each of shape (M,), where
            ``log1m_k = log(1 - k)`` (``-inf`` on an exact hit).

    Raises:
        RuntimeError: If the kernel has no sigma_inv.
    """
    if kern.sigma_inv is None:
        raise RuntimeError("kernel has no sigma_inv")
    diff = np.asarray(points, dtype=float) - np.asarray(kern.x_i, dtype=float)
    inv = np.asarray(kern.sigma_inv, dtype=float)
    if kern._sigma_is_diagonal:  # ruff:ignore[private-member-access]
        mahal = np.einsum("md,d->m", diff * diff, np.diag(inv))
    else:
        mahal = np.einsum("md,de,me->m", diff, inv, diff)
    mahal = np.maximum(mahal, 0.0)
    z = 0.5 * mahal
    k = np.exp(-z)
    with np.errstate(divide="ignore", invalid="ignore"):
        log1m_k = np.where(
            z <= np.log(2.0),
            np.log(-np.expm1(-z)),
            np.log1p(-np.exp(-z)),
        )
    return k, log1m_k


def exact_survival(
    odd: "Samples", ood: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Recompute the log-survival L(x) = log(1 - alpha(x)).

    The result is an owned, WRITABLE array: ``affinity_dual`` returns
    JAX arrays, which are immutable, and ``np.asarray`` on one yields a
    read-only view that the in-place incremental update cannot write to.

    Args:
        odd (Samples): The ODD to evaluate.
        ood (npt.NDArray[np.float64]): Points, (M, n_dims).

    Returns:
        npt.NDArray[np.float64]: Log-survival, shape (M,).
    """
    return np.array(odd.affinity_dual(ood)[1], dtype=float, copy=True)


def kernel_values_at(
    odd: "Samples", x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Evaluate every local kernel k_i(x) at a single point.

    Uses the cached batch arrays on the all-diagonal-RBF fast path and
    falls back to per-kernel evaluation otherwise. Needed for the
    dominant-kernel argmax, where a Python loop over all kernels per
    iteration would be unusable at production scale.

    Args:
        odd (Samples): The ODD whose kernels to evaluate.
        x (npt.NDArray[np.float64]): Query point, shape (n_dims,).

    Returns:
        npt.NDArray[np.float64]: Kernel values, shape (n_anchors,).
    """
    anchors, inv_diag, all_diagonal = odd.batch_arrays()
    if all_diagonal and inv_diag is not None:
        diff = anchors - np.asarray(x, dtype=float)[None, :]
        mahal = np.einsum("nd,nd->n", diff * diff, inv_diag)
        return np.exp(-0.5 * np.maximum(mahal, 0.0))
    return np.array([float(s(np.asarray(x))) for s in odd.samples])


def _validate_parameters(xi: float, shrink_factor: float) -> None:
    """Check the two scalar parameters are in range.

    Args:
        xi (float): Maximum allowed OOD affinity.
        shrink_factor (float): Covariance scale factor c.

    Raises:
        ValueError: If either falls outside (0, 1).
    """
    if not 0.0 < xi < 1.0:
        raise ValueError("xi must be in (0, 1)")
    if not 0.0 < shrink_factor < 1.0:
        raise ValueError("shrink_factor must be in (0, 1)")


def _validate_disjoint(odd: "Samples", ood: npt.NDArray[np.float64]) -> None:
    """Check the OOD set is shaped right and disjoint from the anchors.

    Termination requires ``q_{i*} > 0``, i.e. the anchor set and the OOD
    set must be disjoint. If an OOD point coincides with an anchor then
    ``alpha(x*) = exp(0) = 1`` for EVERY covariance, so no amount of
    shrinking can satisfy ``max alpha <= xi`` and the loop cannot
    terminate. See docs/ood-consistency.md section 2.

    Args:
        odd (Samples): The ODD providing the anchors.
        ood (npt.NDArray[np.float64]): OOD points, (M, n_dims).

    Raises:
        OODDimensionMismatchError: On a column-count mismatch.
        OODAnchorCoincidenceError: If any OOD point equals an anchor.
    """
    anchors, _, _ = odd.batch_arrays()
    if ood.shape[1] != anchors.shape[1]:
        raise OODDimensionMismatchError(ood.shape[1], anchors.shape[1])
    coincident = rows_in(ood, anchors)
    n_coincident = int(coincident.sum())
    if n_coincident:
        raise OODAnchorCoincidenceError(
            n_coincident, ood.shape[0], int(np.flatnonzero(coincident)[0])
        )


def _initial_survival(
    odd: "Samples", ood: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the starting log-survival and warn on saturated points.

    Args:
        odd (Samples): The ODD to evaluate.
        ood (npt.NDArray[np.float64]): OOD points, (M, n_dims).

    Returns:
        npt.NDArray[np.float64]: Log-survival, shape (M,).
    """
    survival = exact_survival(odd, ood)
    n_saturated = int(np.sum(np.isneginf(survival)))
    if n_saturated:
        warnings.warn(NearAnchorOODWarning(n_saturated), stacklevel=3)
    return survival


def _dominant_kernel(
    odd: "Samples", point: npt.NDArray[np.float64]
) -> tuple[int, RBFKernel, npt.NDArray[np.float64]]:
    """Find the kernel with the largest value at ``point``.

    Args:
        odd (Samples): The ODD whose kernels to search.
        point (npt.NDArray[np.float64]): Query point, (n_dims,).

    Returns:
        tuple[int, RBFKernel, npt.NDArray[np.float64]]: The kernel index
            (lowest index wins ties, so selection is deterministic), the
            kernel itself, and all kernel values at ``point``.

    Raises:
        RuntimeError: If the selected kernel is undefined.
        TypeError: If the selected kernel is not an RBFKernel.
    """
    k_vals = kernel_values_at(odd, point)
    i_star = int(np.argmax(k_vals))
    kern = odd.samples[i_star].kernel
    if kern is None:
        raise RuntimeError(f"Kernel {i_star} is not defined.")
    if not isinstance(kern, RBFKernel):
        raise TypeError(
            f"Kernel {i_star} is not an RBFKernel; "
            "OOD consistency only supports RBFKernel."
        )
    return i_star, kern, k_vals


def _shrink_kernel(
    kern: RBFKernel, factor: float, i_star: int, n_shrinks: int, worst: int
) -> None:
    """Scale one kernel's covariance, refusing to saturate float64.

    Both candidate arrays are computed before either is committed, so a
    kernel that would underflow to 0 (or whose inverse would overflow to
    inf) leaves the ODD untouched and raises instead. Without this the
    loop goes on to compute ``0 * inf = NaN`` and selects ``argmax``
    over NaN. See docs/ood-consistency.md section 3.

    Args:
        kern (RBFKernel): The kernel to shrink, mutated in place.
        factor (float): Multiplicative factor for sigma, in (0, 1).
        i_star (int): The kernel's index, for the error message.
        n_shrinks (int): Shrinks already applied, for the error message.
        worst (int): The selected OOD index, for the error message.

    Raises:
        KernelSaturationError: If the result exhausts the float64 range.
    """
    new_sigma = np.asarray(kern.sigma, dtype=float) * factor
    new_sigma_inv = invert_covariance(new_sigma)
    if not (
        np.all(np.isfinite(new_sigma_inv))
        and np.all(np.isfinite(new_sigma))
        and np.all(np.diag(new_sigma) > 0.0)
    ):
        raise KernelSaturationError(i_star, n_shrinks, worst)
    kern.sigma = new_sigma
    kern.sigma_inv = new_sigma_inv
    kern._refresh_sigma_cache()  # ruff:ignore[private-member-access]


def batch_shrink_count(  # ruff:ignore[too-many-arguments]
    *,
    survival_worst: float,
    log1m_old_worst: float,
    k_old_worst: float,
    k_vals: npt.NDArray[np.float64],
    i_star: int,
    log_target: float,
    shrink_factor: float,
    sigma: npt.NDArray[np.float64],
    sigma_inv: npt.NDArray[np.float64],
) -> int:
    """Number of shrinks of kernel ``i_star`` to fix the selected point.

    Closed form for the repeated application of the single-shrink step
    while the selected point and its dominant kernel stay the same; see
    docs/ood-consistency.md section 5. Always returns at least 1, and
    never enough to make ``i_star`` non-dominant or to overflow float64.

    Args:
        survival_worst (float): ``L(x*)`` at the selected point.
        log1m_old_worst (float): ``log(1 - k_{i*}(x*))``, pre-shrink.
        k_old_worst (float): ``k_{i*}(x*)`` before shrinking.
        k_vals (npt.NDArray[np.float64]): All kernel values at ``x*``.
        i_star (int): Index of the dominant kernel.
        log_target (float): ``log1p(-xi)``.
        shrink_factor (float): ``c`` in (0, 1).
        sigma (npt.NDArray[np.float64]): Covariance of ``i_star``.
        sigma_inv (npt.NDArray[np.float64]): Its inverse.

    Returns:
        int: Number of shrinks to apply, >= 1.
    """
    if not np.isfinite(k_old_worst) or k_old_worst <= 0.0:
        return 1
    q = -2.0 * float(np.log(k_old_worst))
    if not np.isfinite(q) or q <= 0.0:
        return 1
    log_c = float(np.log(shrink_factor))
    l_rest = survival_worst - log1m_old_worst
    r = log_target - l_rest
    if r >= 0.0:
        # Even k_new = 0 cannot satisfy the point: the other kernels
        # alone already violate. Fall back to a plain single shrink.
        return 1
    b = -float(np.log(-np.expm1(r)))
    if not np.isfinite(b) or b <= 0.0:
        return 1
    t_need = int(np.ceil(np.log(q / (2.0 * b)) / log_c))
    t_cap = _shrink_count_cap(
        q=q,
        k_vals=k_vals,
        i_star=i_star,
        log_c=log_c,
        sigma=sigma,
        sigma_inv=sigma_inv,
    )
    return max(1, min(t_need, max(1, t_cap)))


def _shrink_count_cap(  # ruff:ignore[too-many-arguments]
    *,
    q: float,
    k_vals: npt.NDArray[np.float64],
    i_star: int,
    log_c: float,
    sigma: npt.NDArray[np.float64],
    sigma_inv: npt.NDArray[np.float64],
) -> int:
    """Upper bound on a batched shrink count.

    Two independent caps: keep ``i_star`` dominant at the selected point
    (past that, unbatched selection would switch kernels), and keep
    sigma positive and its inverse finite in float64.

    Args:
        q (float): Current Mahalanobis distance at the selected point.
        k_vals (npt.NDArray[np.float64]): All kernel values there.
        i_star (int): Index of the dominant kernel.
        log_c (float): ``log(shrink_factor)``, negative.
        sigma (npt.NDArray[np.float64]): Covariance of ``i_star``.
        sigma_inv (npt.NDArray[np.float64]): Its inverse.

    Returns:
        int: The binding cap, or a very large value if neither binds.
    """
    others = np.delete(np.asarray(k_vals, dtype=float), i_star)
    t_dom = np.inf
    if others.size:
        k_second = float(np.max(others))
        if 0.0 < k_second < 1.0:
            t_dom = np.log(q / (-2.0 * np.log(k_second))) / log_c

    smallest_sigma = float(np.min(np.diag(sigma)))
    largest_inv = float(np.max(np.abs(sigma_inv)))
    t_fp = np.inf
    if smallest_sigma > 0.0 and largest_inv > 0.0:
        t_fp = min(
            np.log(np.finfo(float).tiny / smallest_sigma) / log_c,
            np.log(np.finfo(float).max / largest_inv) / -log_c,
        )

    binding = min(t_dom, t_fp)
    return int(np.floor(binding)) if np.isfinite(binding) else DEFAULT_MAX_ITERATIONS


class _LoopState:
    """Mutable state carried through the adjustment loop.

    Bundled so the loop body reads as a sequence of small steps instead
    of threading eight values through every helper.

    Attributes:
        survival (npt.NDArray[np.float64]): Per-point log-survival.
        exact (bool): Whether ``survival`` is an exact recomputation.
        recomputations (int): Number of exact sweeps so far.
        adjusted (dict[int, int]): Kernel index -> shrink count.
        iterations (int): Adjustments applied so far.
        worst_alpha (float): Affinity of the most-violated point.
    """

    def __init__(self, survival: npt.NDArray[np.float64]) -> None:
        self.survival = survival
        self.exact = True
        self.recomputations = 1
        self.adjusted: dict[int, int] = {}
        self.iterations = 0
        self.worst_alpha = 0.0

    def refresh(self, odd: "Samples", ood: npt.NDArray[np.float64]) -> None:
        """Replace the running survival with an exact recomputation.

        Args:
            odd (Samples): The ODD to evaluate.
            ood (npt.NDArray[np.float64]): OOD points, (M, n_dims).
        """
        self.survival = exact_survival(odd, ood)
        self.exact = True
        self.recomputations += 1

    def apply_delta(
        self,
        log1m_new: npt.NDArray[np.float64],
        log1m_old: npt.NDArray[np.float64],
    ) -> None:
        """Fold one kernel's change into the running survival.

        Args:
            log1m_new (npt.NDArray[np.float64]): ``log(1 - k)`` after.
            log1m_old (npt.NDArray[np.float64]): ``log(1 - k)`` before.
        """
        with np.errstate(invalid="ignore"):
            # (-inf) - (-inf) is nan; the caller repairs it.
            self.survival += log1m_new - log1m_old
        self.exact = False


def enforce_ood_consistency(  # ruff:ignore[too-many-arguments]
    odd: "Samples",
    ood_points: "Matrix | NPMatrix",
    xi: float,
    shrink_factor: float = DEFAULT_SHRINK_FACTOR,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    *,
    refresh_interval: int = DEFAULT_REFRESH_INTERVAL,
    log_interval: int = DEFAULT_LOG_INTERVAL,
    batch_jump: bool = False,
) -> dict[str, object]:
    """Enforce alpha(x) <= xi on all OOD points.

    See :meth:`autosafe.samples.Samples.enforce_ood_consistency` for the
    full contract; this is the implementation it delegates to.

    Args:
        odd (Samples): The ODD to adjust, mutated in place.
        ood_points (Matrix | NPMatrix): OOD samples, (M, n_dims), in the
            SAME normalized coordinate system as the anchors.
        xi (float): Maximum allowed OOD affinity, in (0, 1).
        shrink_factor (float): Covariance scale factor, in (0, 1).
        max_iterations (int): Safety cap; exceeding it raises.
        refresh_interval (int): Exact-refresh period; <= 0 disables.
        log_interval (int): Progress-line period; <= 0 disables.
        batch_jump (bool): Use the closed-form shrink count per
            iteration instead of one.

    Returns:
        dict[str, object]: Summary with keys ``iterations``,
            ``max_ood_affinity``, ``adjusted_kernels``,
            ``exact_recomputations`` and ``batch_jump``.
    """
    _validate_parameters(xi, shrink_factor)
    ood = np.atleast_2d(np.asarray(ood_points, dtype=float))
    if ood.shape[0] == 0:
        return {
            "iterations": 0,
            "max_ood_affinity": 0.0,
            "adjusted_kernels": {},
            "exact_recomputations": 0,
            "batch_jump": batch_jump,
        }

    _validate_disjoint(odd, ood)
    state = _LoopState(_initial_survival(odd, ood))
    logger.info(
        f"enforce_ood_consistency: start M={ood.shape[0]} "
        f"N={len(odd.samples)} xi={xi:g} c={shrink_factor:g} "
        f"max_iter={max_iterations} batch_jump={batch_jump}"
    )

    # The loop has no meaningful total---max_iterations is a cap, not an
    # expectation---but the bar still reports elapsed time and rate,
    # which is what distinguishes "working" from "stuck".
    progress = tqdm.rich.tqdm(
        total=max_iterations,
        desc="Enforcing OOD consistency",
        unit="it",
    )
    try:
        _adjust_loop(
            odd=odd,
            ood=ood,
            state=state,
            progress=progress,
            xi=xi,
            log_target=float(np.log1p(-xi)),
            shrink_factor=shrink_factor,
            max_iterations=max_iterations,
            refresh_interval=refresh_interval,
            log_interval=log_interval,
            batch_jump=batch_jump,
        )
    finally:
        progress.close()

    logger.info(
        f"enforce_ood_consistency: done iterations={state.iterations} "
        f"max_ood_affinity={state.worst_alpha:.6g} "
        f"kernels_adjusted={len(state.adjusted)} "
        f"exact_recomputations={state.recomputations}"
    )
    return {
        "iterations": state.iterations,
        "max_ood_affinity": state.worst_alpha,
        "adjusted_kernels": state.adjusted,
        "exact_recomputations": state.recomputations,
        "batch_jump": batch_jump,
    }


def _adjust_loop(  # ruff:ignore[too-many-arguments]
    *,
    odd: "Samples",
    ood: npt.NDArray[np.float64],
    state: _LoopState,
    progress: "tqdm.rich.tqdm[object]",
    xi: float,
    log_target: float,
    shrink_factor: float,
    max_iterations: int,
    refresh_interval: int,
    log_interval: int,
    batch_jump: bool,
) -> None:
    """Shrink kernels until every OOD affinity is at or below xi.

    Args:
        odd (Samples): The ODD to adjust, mutated in place.
        ood (npt.NDArray[np.float64]): OOD points, (M, n_dims).
        state (_LoopState): Running state, mutated in place.
        progress (tqdm.rich.tqdm[object]): Progress bar to advance.
        xi (float): Maximum allowed OOD affinity.
        log_target (float): ``log1p(-xi)``.
        shrink_factor (float): Covariance scale factor c.
        max_iterations (int): Safety cap.
        refresh_interval (int): Exact-refresh period.
        log_interval (int): Progress-line period.
        batch_jump (bool): Whether to use the closed-form jump.

    Raises:
        OODConsistencyNotReachedError: If max_iterations is exceeded.
    """
    n_anchors = len(odd.samples)
    while True:
        worst = int(np.argmin(state.survival))  # first min -> deterministic
        state.worst_alpha = float(-np.expm1(state.survival[worst]))
        if state.survival[worst] >= log_target:
            if state.exact:
                return
            # Never exit on a drifted estimate: refresh and re-test.
            state.refresh(odd, ood)
            continue
        if log_interval > 0 and state.iterations % log_interval == 0:
            # tqdm.write, not the logger: the progress bar owns the
            # terminal while this loop runs.
            tqdm.rich.tqdm.write(
                f"enforce_ood_consistency: iter={state.iterations} "
                f"worst_alpha={state.worst_alpha:.6g} "
                f"worst_ood_index={worst} "
                f"kernels_adjusted={len(state.adjusted)}"
            )
        if state.iterations >= max_iterations:
            raise OODConsistencyNotReachedError(
                max_iterations=max_iterations,
                worst_alpha=state.worst_alpha,
                xi=xi,
                worst_index=worst,
                n_adjusted=len(state.adjusted),
                n_anchors=n_anchors,
            )

        _adjust_once(
            odd=odd,
            ood=ood,
            state=state,
            worst=worst,
            log_target=log_target,
            shrink_factor=shrink_factor,
            batch_jump=batch_jump,
        )
        progress.update(1)
        if refresh_interval > 0 and state.iterations % refresh_interval == 0:
            state.refresh(odd, ood)


def _adjust_once(  # ruff:ignore[too-many-arguments]
    *,
    odd: "Samples",
    ood: npt.NDArray[np.float64],
    state: _LoopState,
    worst: int,
    log_target: float,
    shrink_factor: float,
    batch_jump: bool,
) -> None:
    """Shrink the kernel dominating ``ood[worst]`` exactly once.

    "Once" is one iteration; with ``batch_jump`` it may apply several
    shrinks of the same kernel in a single step.

    Args:
        odd (Samples): The ODD to adjust, mutated in place.
        ood (npt.NDArray[np.float64]): OOD points, (M, n_dims).
        state (_LoopState): Running state, mutated in place.
        worst (int): Index of the most-violated OOD point.
        log_target (float): ``log1p(-xi)``.
        shrink_factor (float): Covariance scale factor c.
        batch_jump (bool): Whether to use the closed-form jump.
    """
    i_star, kern, k_vals = _dominant_kernel(odd, ood[worst])

    # k_old MUST be taken before the mutation and k_new after it;
    # swapping them inverts the incremental update.
    k_old, log1m_old = kernel_over_points(kern, ood)
    n_shrinks = 1
    if batch_jump:
        n_shrinks = batch_shrink_count(
            survival_worst=float(state.survival[worst]),
            log1m_old_worst=float(log1m_old[worst]),
            k_old_worst=float(k_old[worst]),
            k_vals=k_vals,
            i_star=i_star,
            log_target=log_target,
            shrink_factor=shrink_factor,
            sigma=np.asarray(kern.sigma, dtype=float),
            sigma_inv=np.asarray(kern.sigma_inv, dtype=float),
        )

    _shrink_kernel(
        kern,
        shrink_factor**n_shrinks,
        i_star,
        state.adjusted.get(i_star, 0),
        worst,
    )

    _, log1m_new = kernel_over_points(kern, ood)
    state.apply_delta(log1m_new, log1m_old)
    if not np.all(np.isfinite(state.survival)):
        # A -inf term cannot be undone by subtraction.
        state.refresh(odd, ood)

    odd.sync_kernel_cache(i_star)
    state.adjusted[i_star] = state.adjusted.get(i_star, 0) + n_shrinks
    state.iterations += 1
