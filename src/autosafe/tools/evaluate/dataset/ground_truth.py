# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Ground-truth ODD membership labels for sampled test points."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from autosafe.preprocessing import RangeNormalizer
from autosafe.tools.evaluate.dataset.normalization import (
    _denormalize_points_to_yaml_space,
)
from autosafe.tools.evaluate.dataset.yaml_spec import (
    _discrete_values_from_yaml,
    _legacy_limits_yaml_to_odd_config,
    _periodic_columns_from_yaml,
)
from autosafe.tools.monte_carlo.inequality_utils import ODDFactory, load_yaml_odd_config
from autosafe.typing import (
    Matrix,
    NPMatrix,
)

# Tolerance for "on a valid code", relative to the smallest code
# spacing; it absorbs the normalize/denormalize round trip only.
_DISCRETE_CODE_RTOL = 1e-6


def _on_valid_code(
    values: npt.NDArray[np.float64], codes: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Return whether each value equals one of the valid codes.

    Args:
        values (npt.NDArray[np.float64]): Coordinate values, YAML space.
        codes (npt.NDArray[np.float64]): Sorted unique valid codes.

    Returns:
        npt.NDArray[np.bool_]: Per-value membership in the code set.
    """
    spacing = float(np.min(np.diff(codes))) if codes.size > 1 else 1.0
    tol = _DISCRETE_CODE_RTOL * spacing
    nearest = np.clip(np.searchsorted(codes, values), 1, max(codes.size - 1, 1))
    lower = codes[nearest - 1]
    upper = codes[np.minimum(nearest, codes.size - 1)]
    distance = np.minimum(np.abs(values - lower), np.abs(values - upper))
    return np.asarray(distance <= tol, dtype=bool)


def _wrap_columns(
    points: npt.NDArray[np.float64],
    periodic: dict[int, tuple[float, float]],
) -> npt.NDArray[np.float64]:
    """Wrap periodic columns into ``[low, low + period)`` in place.

    Args:
        points (npt.NDArray[np.float64]): Points, shape (n, n_dims).
        periodic (dict[int, tuple[float, float]]): Column index to its
            lower end and period.

    Returns:
        npt.NDArray[np.float64]: ``points`` with wrapped columns.
    """
    for index, (low, period) in periodic.items():
        wrapped = low + np.mod(points[:, index] - low, period)
        # Round-trip noise just below `low` must land on `low`, not on
        # `low + period` (which is not a valid code of a wrapped set).
        # Clamp instead of subtracting the period, so a value just below
        # `low + period` never ends up below `low`, outside the box.
        fold = wrapped >= low + period * (1.0 - _DISCRETE_CODE_RTOL)
        wrapped[fold] = low
        points[:, index] = wrapped
    return points


def _ground_truth_labels_from_yaml(
    ground_truth_yaml: Path,
    test_points: Matrix | NPMatrix,
    normalizer: RangeNormalizer | None = None,
) -> npt.NDArray[np.bool_]:
    """Compute ground-truth ODD membership labels from YAML spec.

    If normalizer provided, test_points are in normalized space and
    denormalized back to YAML space before checking membership.

    Args:
        ground_truth_yaml (Path): YAML ODD specification path.
        test_points (Matrix): Query points (normalized if normalizer
            provided).
        normalizer (RangeNormalizer | None): Optional normalizer to
            denormalize points before checking YAML membership. If None,
            test_points are used as-is.

    Returns:
        npt.NDArray[np.bool_]: Boolean membership labels. Variables a
            limits YAML declares as `type: discrete` must additionally
            lie on one of their valid codes (likewise `wrapped_discrete`
            and `categorical`); `wrapped_range` and `wrapped_discrete`
            variables are wrapped into their period first.
    """
    config = load_yaml_odd_config(ground_truth_yaml)
    if "type" not in config and "limits" in config:
        config = _legacy_limits_yaml_to_odd_config(config)
    region, _ = ODDFactory(config).create_odd()

    # Denormalize points to YAML space if normalizer provided
    if normalizer is not None:
        test_points = _denormalize_points_to_yaml_space(test_points, normalizer)

    points = np.array(test_points, dtype=float)
    points = _wrap_columns(points, _periodic_columns_from_yaml(ground_truth_yaml))
    labels = np.asarray(region.contains(points.T), dtype=bool)
    for index, codes in _discrete_values_from_yaml(ground_truth_yaml).items():
        labels &= _on_valid_code(points[:, index], codes)
    return labels
