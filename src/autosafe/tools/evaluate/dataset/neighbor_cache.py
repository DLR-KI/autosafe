# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Nearest-neighbor index cache shared by ODD builds."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from autosafe.kernels import KernelDict
from autosafe.samples import (
    Samples,
    find_closest_vectors_by_index,
    find_closest_vectors_by_index_per_dimension,
)
from autosafe.tools.evaluate.dataset.odd_cache import (
    _ODDCacheSpec,
    _resolve_kernel_calibration,
)
from autosafe.typing import (
    Matrix,
    NPMatrix,
)

if TYPE_CHECKING:
    from autosafe.typing import ClosestSampleModeType

from autosafe.tools.evaluate.dataset.anchors import _extract_anchor_points


def _default_neighbor_cache_path(
    dataset_path: Path,
    *,
    closest_sample_mode: "ClosestSampleModeType",
    filename_tag: str = "",
) -> Path:
    """Build deterministic path for nearest-neighbor cache.

    Args:
        dataset_path (Path): Path to the dataset file.
        closest_sample_mode (ClosestSampleModeType): Mode for nearest-
            neighbor assignment, e.g., "global" or "per_dimension".
        filename_tag (str): Optional tag to separate subsampled from
            full-data caches (indices are into the anchor array---a
            full-data npz applied to a subsampled array yields wrong
            kernels).

    Returns:
        Path: File path used for nearest-neighbor index cache.
    """
    return dataset_path.with_name(
        f"{dataset_path.stem}-nn-{closest_sample_mode}{filename_tag}.npz"
    )


def _load_neighbor_indices(
    cache_path: Path,
    *,
    n_points: int,
    n_dims: int,
    closest_sample_mode: "ClosestSampleModeType",
) -> npt.NDArray[np.int64] | None:
    """Load cached nearest-neighbor indices if shape and mode match.

    Args:
        cache_path (Path): Path to load nearest-neighbor indices from.
        n_points (int): Expected number of points in the reference data.
        n_dims (int): Expected number of dimensions in the reference
            data.
        closest_sample_mode (ClosestSampleModeType): Mode for nearest-
            neighbor assignment, e.g., "global" or "per_dimension".

    Returns:
        npt.NDArray[np.int64] | None: Cached indices, if valid.
    """
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as cache:
        cached_mode = str(cache["mode"])
        if cached_mode != closest_sample_mode:
            return None

        cached_indices = np.asarray(cache["indices"], dtype=np.int64)

    if closest_sample_mode == "global":
        expected_shape = (n_points,)
    else:
        expected_shape = (n_dims, n_points)

    if cached_indices.shape != expected_shape:
        return None
    return cached_indices


def _compute_neighbor_indices(
    reference_points: Matrix | NPMatrix,
    *,
    closest_sample_mode: "ClosestSampleModeType",
) -> npt.NDArray[np.int64]:
    """Compute nearest-neighbor index assignments for the given mode.

    Args:
        reference_points (Matrix | NPMatrix): Reference points for which
            to compute nearest neighbors.
        closest_sample_mode (ClosestSampleModeType): Mode for nearest-
            neighbor assignment, e.g., "global" or "per_dimension".

    Returns:
        npt.NDArray[np.int64]: Computed nearest-neighbor indices.
    """
    n_points, n_dims = reference_points.shape
    if n_points <= 1:
        if closest_sample_mode == "global":
            return np.zeros((n_points,), dtype=np.int64)
        return np.zeros((n_dims, n_points), dtype=np.int64)

    if closest_sample_mode == "global":
        return find_closest_vectors_by_index(reference_points)
    return find_closest_vectors_by_index_per_dimension(reference_points)


def _get_or_create_neighbor_indices(
    reference_points: Matrix | NPMatrix,
    *,
    cache_path: Path,
    closest_sample_mode: "ClosestSampleModeType",
) -> npt.NDArray[np.int64]:
    """Load nearest-neighbor indices from cache or compute and save.

    Args:
        reference_points (Matrix | NPMatrix): Reference points for which
            to compute nearest neighbors.
        cache_path (Path): Path to load/save nearest-neighbor indices.
        closest_sample_mode (ClosestSampleModeType): Mode for nearest-
                neighbor assignment, e.g., "global" or "per_dimension".

    Returns:
        npt.NDArray[np.int64]: Nearest-neighbor indices for all points.
    """
    n_points, n_dims = reference_points.shape
    cached = _load_neighbor_indices(
        cache_path,
        n_points=n_points,
        n_dims=n_dims,
        closest_sample_mode=closest_sample_mode,
    )
    if cached is not None:
        return cached

    indices = _compute_neighbor_indices(
        reference_points,
        closest_sample_mode=closest_sample_mode,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        mode=np.asarray(closest_sample_mode),
        indices=indices,
    )
    return indices


def _assign_cached_neighbors(
    odd: Samples,
    indices: npt.NDArray[np.int64],
    *,
    closest_sample_mode: "ClosestSampleModeType",
) -> None:
    """Assign closest-sample links from cached NN indices.

    Args:
        odd (Samples): ODD object to update with closest-sample links.
        indices (npt.NDArray[np.int64]): Nearest-neighbor indices to
            assign.
        closest_sample_mode (ClosestSampleModeType): Mode for nearest-
            neighbor assignment, e.g., "global" or "per_dimension".
    """
    for idx, sample in enumerate(odd.samples):
        if closest_sample_mode == "global":
            sample.closest_sample = [odd.samples[int(indices[idx])]]
        else:
            sample.closest_sample = [
                odd.samples[int(closest_idx)] for closest_idx in indices[:, idx]
            ]


def _refresh_odd_kernels_from_neighbor_cache(
    odd: Samples,
    *,
    dataset_path: Path,
    cache_spec: _ODDCacheSpec,
    filename_tag: str = "",
) -> Samples:
    """Refresh kernel state using cached nearest-neighbor assignments.

    Args:
        odd (Samples): ODD object to refresh with cached neighbor state.
        dataset_path (Path): Path to the dataset file, used for cache
            path derivation.
        cache_spec (_ODDCacheSpec): Cache specification to determine
            cache paths and settings for neighbor assignment and kernel
            configuration.
        filename_tag (str): Optional tag for cache file separation (e.g.
            subsampled vs. full-data).

    Returns:
        Samples: Updated ODD object with refreshed kernel state.
    """
    reference_points = _extract_anchor_points(odd)
    neighbor_cache_path = _default_neighbor_cache_path(
        dataset_path,
        closest_sample_mode=cache_spec.closest_sample_mode,
        filename_tag=filename_tag,
    )
    indices = _get_or_create_neighbor_indices(
        reference_points,
        cache_path=neighbor_cache_path,
        closest_sample_mode=cache_spec.closest_sample_mode,
    )

    odd.closest_sample_mode = cache_spec.closest_sample_mode
    odd.kernel_cls_str = cache_spec.kernel_type
    odd.kernel_cls = KernelDict[cache_spec.kernel_type]
    odd.kernel_kwargs = dict(
        cache_spec.kernel_kwargs
    )  # keep meta form for cache equality

    resolved_kwargs = _resolve_kernel_calibration(
        dict(cache_spec.kernel_kwargs),
        reference_points,
        indices,
        closest_sample_mode=cache_spec.closest_sample_mode,
    )

    _assign_cached_neighbors(
        odd,
        indices,
        closest_sample_mode=cache_spec.closest_sample_mode,
    )
    odd.refresh_kernels(kernel_kwargs_override=resolved_kwargs)
    return odd
