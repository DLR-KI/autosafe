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
    _legacy_limits_yaml_to_odd_config,
)
from autosafe.tools.monte_carlo.inequality_utils import ODDFactory, load_yaml_odd_config
from autosafe.typing import (
    Matrix,
    NPMatrix,
)


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
        npt.NDArray[np.bool_]: Boolean membership labels.
    """
    config = load_yaml_odd_config(ground_truth_yaml)
    if "type" not in config and "limits" in config:
        config = _legacy_limits_yaml_to_odd_config(config)
    region, _ = ODDFactory(config).create_odd()

    # Denormalize points to YAML space if normalizer provided
    if normalizer is not None:
        test_points = _denormalize_points_to_yaml_space(test_points, normalizer)

    return np.asarray(region.contains(test_points.T), dtype=bool)
