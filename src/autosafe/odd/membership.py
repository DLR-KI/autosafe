# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Membership-threshold calibration for autoSAFE ODDs."""

import math

import numpy as np
import numpy.typing as npt

from autosafe.exceptions import (
    CalibrationSetTooSmallError,
    EmptyParameterError,
    InvalidLogSurvivalError,
    ParameterOutOfRangeError,
    ShapeMismatchError,
    UncertifiableThresholdError,
)


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
        ParameterOutOfRangeError: If ``false_exclusion_rate`` is not in
            ``(0, 1)``.
        ShapeMismatchError: If ``log_survival`` is not 1-D.
        EmptyParameterError: If ``log_survival`` is empty.
        InvalidLogSurvivalError: If ``log_survival`` contains NaN or
            ``+inf``.
        CalibrationSetTooSmallError: If the calibration set is too small
            to certify the requested false-exclusion rate.
        UncertifiableThresholdError: If the calibrated threshold is 0.
    """
    epsilon = float(false_exclusion_rate)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ParameterOutOfRangeError("false_exclusion_rate", epsilon, 0.0, 1.0)

    survival = np.asarray(log_survival, dtype=np.float64)
    if survival.ndim != 1:
        raise ShapeMismatchError("log_survival", survival.shape, (None,))
    if survival.size == 0:
        raise EmptyParameterError("log_survival")
    if np.isnan(survival).any() or np.isposinf(survival).any():
        raise InvalidLogSurvivalError

    rank = math.floor(epsilon * (survival.size + 1))
    if rank == 0:
        raise CalibrationSetTooSmallError(survival.size, epsilon)

    scores = np.sort(-survival)
    score_threshold = float(scores[rank - 1])
    if score_threshold <= 0.0:
        raise UncertifiableThresholdError
    log_threshold = -score_threshold
    with np.errstate(over="ignore", under="ignore"):
        zeta = float(-np.expm1(log_threshold))
    return zeta, log_threshold


__all__ = ["conformal_membership_threshold"]
