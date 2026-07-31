# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Test-point sampling around an ODD's anchor set."""

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
) -> npt.NDArray[np.float64]:
    """Sample points using explicit lower/upper bounds from YAML.

    Uses the same mixed strategy as `_sample_points_around_odd`: half
    uniform samples in a slightly expanded box and half local samples
    around anchors.

    Args:
        anchor_points (Matrix | NPMatrix): Affinity ODD anchors.
        lower_bounds (npt.NDArray): Lower bounds for sampling.
        upper_bounds (npt.NDArray): Upper bounds for sampling.
        n_samples (int): Number of test points to sample.
        seed (int): PRNG seed.
        local_noise_std (float | None): Per-dimension std for the local
            perturbation half. If None, uses 0.05 * per-dim span
            (legacy default).

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

    return np.vstack([uniform_points, local_points])
