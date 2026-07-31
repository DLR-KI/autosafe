# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Coordinate-system transforms between raw and normalized space."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from autosafe.preprocessing import RangeNormalizer, create_robust_normalization_pipeline
from autosafe.tools.evaluate.dataset.yaml_spec import (
    _sampling_bounds_from_yaml,
)
from autosafe.tools.experiments.utils import DatasetLoadOptions, load_dataset
from autosafe.typing import (
    Matrix,
    NPMatrix,
)


def _numeric_array(df: pl.DataFrame) -> npt.NDArray[np.float64]:
    """Return the numeric columns of ``df`` as a float64 array."""
    cols = [c for c, dt in zip(df.columns, df.dtypes, strict=False) if dt.is_numeric()]
    return df.select(cols).to_numpy().astype(float)


def _normalizer_from_yaml_bounds(
    ground_truth_yaml: Path,
) -> RangeNormalizer | None:
    """Create a RangeNormalizer initialized on YAML ground-truth bounds.

    Extracts lower/upper bounds from YAML ODD config and creates a
    normalizer fit to those bounds. This normalizer can be used to:
    - Normalize datasets using YAML-derived bounds (consistent space)
    - Denormalize YAML bounds to normalized coordinate space
    - Denormalize sampled/comparison points back to raw space

    Args:
        ground_truth_yaml (Path): YAML ODD specification path.

    Returns:
        RangeNormalizer | None: Fitted normalizer, or None if YAML
            defines non-box ODD or bounds extraction fails.
    """
    bounds = _sampling_bounds_from_yaml(ground_truth_yaml)
    if bounds is None:
        return None

    lower_bounds, upper_bounds = bounds
    # Create synthetic data spanning the bounds for normalizer fitting
    # Normalizer learns bounds from data, so we give it min/max points
    synthetic_data = np.vstack([lower_bounds, upper_bounds])

    normalizer = create_robust_normalization_pipeline(
        target_range=(-1.0, 1.0),
        method="minmax",  # Use minmax for consistent YAML-based bounds
    )
    normalizer.fit(synthetic_data)
    return normalizer


def _normalize_ood_points(
    ood_path: Path,
    dataset_path: Path,
    *,
    yaml_normalizer: RangeNormalizer | None,
    normalize_data: bool,
) -> npt.NDArray[np.float64]:
    """Load OOD points and map them into the anchors' coordinate system.

    Mirrors exactly how the anchors are normalized in
    ``_build_or_load_affinity_odd``: a YAML-bounds normalizer when
    present, otherwise the built-in IQR pipeline fit on the raw dataset.

    Args:
        ood_path (Path): CSV of OOD samples (same numeric columns as the
            dataset).
        dataset_path (Path): The anchor dataset, used to refit the IQR
            normalizer when no YAML normalizer is supplied.
        yaml_normalizer (RangeNormalizer | None): External normalizer
            fit on YAML bounds, if any.
        normalize_data (bool): Whether the anchor pipeline normalized
            the data (always ``True`` in dataset mode).

    Returns:
        npt.NDArray[np.float64]: OOD points in normalized space,
            shape (M, n_dims).
    """
    raw_df, _ = load_dataset(ood_path, options=DatasetLoadOptions(normalize=False))
    ood_raw = _numeric_array(raw_df)

    if yaml_normalizer is not None:
        return np.asarray(yaml_normalizer.transform(ood_raw), dtype=float)
    if not normalize_data:
        return ood_raw

    # Reproduce the built-in IQR normalization (see normalize_dataset):
    # fit on the raw dataset's numeric columns, then transform the OOD.
    ds_df, _ = load_dataset(dataset_path, options=DatasetLoadOptions(normalize=False))
    normalizer = create_robust_normalization_pipeline(
        target_range=(-1.0, 1.0), method="iqr"
    )
    normalizer.fit(_numeric_array(ds_df))
    return np.asarray(normalizer.transform(ood_raw), dtype=float)


def _denormalize_bounds_to_yaml_space(
    lower_bounds: npt.NDArray[np.float64],
    upper_bounds: npt.NDArray[np.float64],
    normalizer: RangeNormalizer,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Denormalize normalized bounds back to YAML space.

    Args:
        lower_bounds (npt.NDArray[np.float64]): Normalized lower bounds.
        upper_bounds (npt.NDArray[np.float64]): Normalized upper bounds.
        normalizer (RangeNormalizer): Fitted normalizer for inverse.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            Denormalized (lower_bounds, upper_bounds) in YAML space.
    """
    lower_denorm = np.asarray(
        normalizer.inverse_transform(lower_bounds.reshape(1, -1))[0]
    )
    upper_denorm = np.asarray(
        normalizer.inverse_transform(upper_bounds.reshape(1, -1))[0]
    )
    return lower_denorm, upper_denorm


def _denormalize_points_to_yaml_space(
    points: Matrix | NPMatrix,
    normalizer: RangeNormalizer,
) -> Matrix | NPMatrix:
    """Denormalize points from normalized space to YAML space.

    Args:
        points (Matrix): Points in normalized space.
        normalizer (RangeNormalizer): Fitted normalizer for inverse.

    Returns:
        Matrix: Points in YAML (raw) space.
    """
    return normalizer.inverse_transform(points)
