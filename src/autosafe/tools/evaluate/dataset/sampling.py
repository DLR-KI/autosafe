# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Test-point sampling around an ODD's anchor set."""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from autosafe.typing import (
    Matrix,
    NPMatrix,
)


def _sample_points_around_odd(
    anchor_points: Matrix | NPMatrix,
    n_samples: int,
    seed: int = 0,
    local_noise_std: float | None = None,
) -> npt.NDArray[np.float64]:
    """Sample points in and around an affinity ODD.

    Args:
        anchor_points (Matrix | NPMatrix): Affinity ODD anchors.
        n_samples (int): Number of test points to sample.
        seed (int): PRNG seed.
        local_noise_std (float | None): Per-dimension std for the local
            perturbation half. If None, uses 0.05 * per-dim span
            (legacy default).

    Returns:
        Matrix: Sampled points (n_samples, n_dims).
    """
    rng = np.random.default_rng(seed)

    mins = anchor_points.min(axis=0)
    maxs = anchor_points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)

    n_uniform = n_samples // 2
    n_local = n_samples - n_uniform

    uniform_points = rng.uniform(
        mins - 0.1 * span,
        maxs + 0.1 * span,
        size=(n_uniform, anchor_points.shape[1]),
    )

    noise_scale = 0.05 * span if local_noise_std is None else local_noise_std
    indices = rng.integers(0, anchor_points.shape[0], size=n_local)
    local_points = anchor_points[indices] + rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=(n_local, anchor_points.shape[1]),
    )

    return np.vstack([uniform_points, local_points])


def _sample_points_with_bounds(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    anchor_points: Matrix | NPMatrix,
    lower_bounds: npt.NDArray[np.float64],
    upper_bounds: npt.NDArray[np.float64],
    n_samples: int,
    seed: int = 0,
    local_noise_std: float | None = None,
    discrete_values: Mapping[int, npt.NDArray[np.float64]] | None = None,
    wrapped_columns: Sequence[int] | None = None,
) -> npt.NDArray[np.float64]:
    """Sample points using explicit lower/upper bounds from YAML.

    Uses the same mixed strategy as `_sample_points_around_odd`: half
    uniform samples in a slightly expanded box and half local samples
    around anchors.

    Discrete coordinates stay on valid codes: the uniform half draws
    them uniformly from the code set, and the local half keeps the code
    of its source anchor instead of perturbing it. The continuous
    coordinates are drawn first, from the same stream as without
    discrete coordinates, so they do not depend on `discrete_values`.

    Periodic (wrapped-range) coordinates have no outside: the uniform
    half is rescaled from the expanded interval onto exactly one period,
    and the local half is wrapped into ``[lower, upper)``.

    Args:
        anchor_points (Matrix | NPMatrix): Affinity ODD anchors.
        lower_bounds (npt.NDArray): Lower bounds for sampling.
        upper_bounds (npt.NDArray): Upper bounds for sampling.
        n_samples (int): Number of test points to sample.
        seed (int): PRNG seed.
        local_noise_std (float | None): Per-dimension std for the local
            perturbation half. If None, uses 0.05 * per-dim span
            (legacy default).
        discrete_values (Mapping[int, npt.NDArray[np.float64]] | None):
            Column index to the valid codes of that discrete variable,
            in the same (normalized) space as `anchor_points`.
        wrapped_columns (Sequence[int] | None): Indices of periodic
            columns whose period is ``upper_bounds - lower_bounds``.

    Returns:
        Matrix: Array of sampled points with shape (n_samples, n_dims).
    """
    rng = np.random.default_rng(seed)

    mins = np.asarray(lower_bounds, dtype=float)
    maxs = np.asarray(upper_bounds, dtype=float)
    span = np.maximum(maxs - mins, 1e-9)

    n_uniform = n_samples // 2
    n_local = n_samples - n_uniform

    uniform_points = rng.uniform(
        mins - 0.1 * span,
        maxs + 0.1 * span,
        size=(n_uniform, anchor_points.shape[1]),
    )

    noise_scale = 0.05 * span if local_noise_std is None else local_noise_std
    indices = rng.integers(0, anchor_points.shape[0], size=n_local)
    local_points = anchor_points[indices] + rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=(n_local, anchor_points.shape[1]),
    )

    anchors = np.asarray(anchor_points, dtype=float)
    for index, codes in (discrete_values or {}).items():
        valid = np.asarray(codes, dtype=float)
        uniform_points[:, index] = rng.choice(valid, size=n_uniform)
        local_points[:, index] = anchors[indices, index]

    for index in wrapped_columns or ():
        low, period = mins[index], maxs[index] - mins[index]
        if period <= 0:
            continue
        expanded_low = low - 0.1 * span[index]
        uniform_points[:, index] = low + (uniform_points[:, index] - expanded_low) / 1.2
        local_points[:, index] = low + np.mod(local_points[:, index] - low, period)

    return np.vstack([uniform_points, local_points])
