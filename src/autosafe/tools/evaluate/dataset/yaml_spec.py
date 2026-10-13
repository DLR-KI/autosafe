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

# Limit types with a finite set of valid values; categorical values are
# strings encoded as their YAML-order index.
_FINITE_TYPES = ("discrete", "wrapped_discrete", "categorical")


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


def _discrete_values_from_yaml(
    ground_truth_yaml: Path,
) -> dict[int, npt.NDArray[np.float64]]:
    """Extract the valid codes of finite-valued limits-YAML variables.

    Legacy `limits` entries of `type: discrete`, `wrapped_discrete` or
    `categorical` define a finite set of values, not an interval. The
    box conversion only keeps their min/max, so the sampler and the
    ground-truth labeller need the set itself to keep these coordinates
    on valid codes. Categorical values are strings; their codes are
    their indices in YAML order (see `_categories_from_yaml`).

    Args:
        ground_truth_yaml (Path): Path to the ground-truth ODD YAML
            spec.

    Returns:
        dict[int, npt.NDArray[np.float64]]: Column index to the sorted
            unique valid codes. Empty for YAMLs without a `limits`
            mapping or without finite-valued entries.
    """
    codes: dict[int, npt.NDArray[np.float64]] = {}
    for limit_type in _FINITE_TYPES:
        for index, entry in _typed_limit_entries(ground_truth_yaml, limit_type):
            codes[index] = _entry_codes(entry)
    return dict(sorted(codes.items()))


def _categories_from_yaml(ground_truth_yaml: Path) -> dict[str, list[str]]:
    """Return the categories of every `type: categorical` limit.

    Args:
        ground_truth_yaml (Path): Path to the ground-truth ODD YAML
            spec.

    Returns:
        dict[str, list[str]]: Column name to its categories in code
            order (the YAML order); empty without categorical entries.
    """
    config = load_yaml_odd_config(ground_truth_yaml)
    limits = config.get("limits") if "type" not in config else None
    if not isinstance(limits, dict):
        return {}
    return {
        str(name): [str(value) for value in entry["values"]]
        for name, entry in limits.items()
        if isinstance(entry, dict) and entry.get("type") == "categorical"
    }


def _periodic_columns_from_yaml(
    ground_truth_yaml: Path,
) -> dict[int, tuple[float, float]]:
    """Return ``(low, period)`` for every periodic limits-YAML column.

    A `wrapped_range` ``[lo, hi]`` has period ``hi - lo``. A
    `wrapped_discrete` code set has period ``max - min`` plus its
    smallest code spacing (24 for the hours 0 to 23), unless the entry
    gives `period` explicitly.

    Args:
        ground_truth_yaml (Path): Path to the ground-truth ODD YAML
            spec.

    Returns:
        dict[int, tuple[float, float]]: Column index to the lower end
            and the period, in YAML space.

    Raises:
        ValueError: If a period is not positive.
    """
    periodic: dict[int, tuple[float, float]] = {}
    for index, entry in _typed_limit_entries(ground_truth_yaml, "wrapped_range"):
        values = np.asarray(entry["values"], dtype=float)
        periodic[index] = (float(values.min()), float(values.max() - values.min()))
    for index, entry in _typed_limit_entries(ground_truth_yaml, "wrapped_discrete"):
        codes = _entry_codes(entry)
        if "period" in entry:
            period = float(entry["period"])
        else:
            step = float(np.min(np.diff(codes))) if codes.size > 1 else 1.0
            period = float(codes[-1] - codes[0]) + step
        periodic[index] = (float(codes[0]), period)
    for index, (_, period) in periodic.items():
        if not period > 0:
            raise ValueError(f"Periodic limit in column {index} needs period > 0")
    return dict(sorted(periodic.items()))


def _entry_codes(entry: dict[str, Any]) -> npt.NDArray[np.float64]:
    """Return the sorted numeric codes of a finite-valued limit entry.

    Args:
        entry (dict[str, Any]): One `limits` entry.

    Returns:
        npt.NDArray[np.float64]: ``0..k-1`` for categorical entries,
            otherwise the sorted unique values.
    """
    if entry.get("type") == "categorical":
        return np.arange(len(entry["values"]), dtype=float)
    return np.unique(np.asarray(entry["values"], dtype=float))


def _wrapped_columns_from_yaml(ground_truth_yaml: Path) -> list[int]:
    """Return the columns a limits YAML declares as wrapped ranges.

    A `type: wrapped_range` entry ``[lo, hi]`` is periodic with period
    ``hi - lo`` (e.g. an angle in ``[-180, 180]``): a value outside it
    denotes the same state as its wrapped image, so it is never outside
    the box.
    The interval itself is still read from `values` by the box
    conversion, exactly as for `type: range`.

    Args:
        ground_truth_yaml (Path): Path to the ground-truth ODD YAML
            spec.

    Returns:
        list[int]: Sorted column indices of the wrapped ranges; empty
            for YAMLs without a `limits` mapping.
    """
    entries = _typed_limit_entries(ground_truth_yaml, "wrapped_range")
    return [index for index, _ in entries]


def _typed_limit_entries(
    ground_truth_yaml: Path, limit_type: str
) -> list[tuple[int, dict[str, Any]]]:
    """Return the (column index, entry) pairs of one limits-YAML type.

    Args:
        ground_truth_yaml (Path): Path to the ground-truth ODD YAML
            spec.
        limit_type (str): The `type` value to select.

    Returns:
        list[tuple[int, dict[str, Any]]]: Matching entries in column
            order; empty for YAMLs without a `limits` mapping.
    """
    config = load_yaml_odd_config(ground_truth_yaml)
    limits = config.get("limits") if "type" not in config else None
    if not isinstance(limits, dict):
        return []
    return [
        (index, entry)
        for index, entry in enumerate(limits.values())
        if isinstance(entry, dict) and entry.get("type") == limit_type
    ]


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

        if entry.get("type") in _FINITE_TYPES:
            codes = _entry_codes(entry)
            lower_bounds.append(float(codes[0]))
            upper_bounds.append(float(codes[-1]))
        else:
            lower_bounds.append(float(min(values)))
            upper_bounds.append(float(max(values)))

    return {
        "type": "box",
        "dim": len(lower_bounds),
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }
