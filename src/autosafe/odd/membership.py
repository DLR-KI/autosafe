# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Membership-threshold calibration for autoSAFE ODDs."""

import math

import numpy as np
import numpy.typing as npt


def conformal_membership_threshold(
    log_survival: npt.ArrayLike,
    false_exclusion_rate: float,
) -> tuple[float, float]:
    """Calibrate a lower ID-membership threshold in log space.

    The conformity score is ``-log(1 - alpha)``. Given ``n`` held-out,
    exchangeable ID scores, the selected order statistic bounds marginal
    false exclusion at ``false_exclusion_rate``.

    Args:
        log_survival (npt.ArrayLike): Held-out ID values of
            ``log(1 - alpha)``, shape ``(n,)``.
        false_exclusion_rate (float): Target rate in ``(0, 1)``.

    Returns:
        tuple[float, float]: ``(zeta, log_survival_threshold)``. The log
            threshold is authoritative near ``zeta=1``.

    Raises:
        ValueError: If inputs are invalid or the calibration set is too
            small to certify the requested false-exclusion rate.
    """
    epsilon = float(false_exclusion_rate)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("false_exclusion_rate must be in (0, 1)")

    survival = np.asarray(log_survival, dtype=np.float64)
    if survival.ndim != 1:
        raise ValueError("log_survival must be one-dimensional")
    if survival.size == 0:
        raise ValueError("log_survival must not be empty")
    if np.isnan(survival).any() or np.isposinf(survival).any():
        raise ValueError("log_survival contains invalid values")

    rank = math.floor(epsilon * (survival.size + 1))
    if rank == 0:
        raise ValueError(
            "calibration set is too small for the requested false-exclusion rate"
        )

    scores = np.sort(-survival)
    score_threshold = float(scores[rank - 1])
    if score_threshold <= 0.0:
        raise ValueError(
            "conformal calibration produced zeta=0; no strict membership "
            "threshold is certifiable"
        )
    log_threshold = -score_threshold
    with np.errstate(over="ignore", under="ignore"):
        zeta = float(-np.expm1(log_threshold))
    return zeta, log_threshold


__all__ = ["conformal_membership_threshold"]
