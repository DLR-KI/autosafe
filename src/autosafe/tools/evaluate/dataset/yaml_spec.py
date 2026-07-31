# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Reading ODD specifications out of YAML sidecar files.

Pure parsing: nothing here knows about normalization, so both the
normalizer and the ground-truth labeller can depend on it.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from autosafe.tools.monte_carlo.inequality_utils import load_yaml_odd_config


def _infer_ground_truth_yaml(dataset_path: Path) -> Path | None:
    """Infer sibling YAML ground-truth ODD spec from dataset path.

    Args:
        dataset_path (Path): Input dataset path.

    Returns:
        Path | None: Existing sibling YAML path if found.
    """
    for suffix in (".yml", ".yaml"):
        candidate = dataset_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _sampling_bounds_from_yaml(
    ground_truth_yaml: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Extract box-like sampling bounds from a ground-truth YAML.

    Supports legacy `limits` YAML and explicit `type: box` YAML. For
    non-box ODDs this returns None, and caller can fall back to anchor-
    based sampling.

    Args:
        ground_truth_yaml (Path): Path to the ground-truth ODD YAML
            spec.

    Returns:
        tuple[npt.NDArray, npt.NDArray] | None: Lower and upper bounds
            arrays if box-type ODD, else None.
    """
    config = load_yaml_odd_config(ground_truth_yaml)
    if "type" not in config and "limits" in config:
        config = _legacy_limits_yaml_to_odd_config(config)

    if config.get("type") != "box":
        return None

    lower = config.get("lower_bounds")
    upper = config.get("upper_bounds")
    if not isinstance(lower, list) or not isinstance(upper, list):
        return None

    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    if lower_arr.shape != upper_arr.shape:
        return None

    return lower_arr, upper_arr


def _legacy_limits_yaml_to_odd_config(config: dict[str, Any]) -> dict[str, object]:
    """Convert legacy state-variable limits YAML into an ODD config.

    Some sibling YAML files in the dataset folder only describe variable
    limits via a top-level `limits` mapping. Those files are not ODD
    specifications, so we convert them into a box-style ODD definition
    by taking the min/max of each limit entry.

    Args:
        config (dict[str, object]): Loaded legacy limits configuration.

    Returns:
        dict[str, object]: ODD-style configuration with `type`, `dim`,
            `lower_bounds`, and `upper_bounds` fields.

    Raises:
        TypeError: If the legacy config is malformed.
        ValueError: If any limit entry is empty.
    """
    limits = config.get("limits")
    if not isinstance(limits, dict) or not limits:
        return config

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for name, entry in limits.items():
        if not isinstance(entry, dict):
            raise TypeError(f"Limit entry for {name!r} must be a mapping")

        values = entry.get("values")
        if not isinstance(values, list):
            raise TypeError(f"Limit entry for {name!r} must be a list")
        if not values:
            raise ValueError(f"Limit entry for {name!r} must contain values")

        lower_bounds.append(float(min(values)))
        upper_bounds.append(float(max(values)))

    return {
        "type": "box",
        "dim": len(lower_bounds),
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
