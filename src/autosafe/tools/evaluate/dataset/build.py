# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Construction and caching of the affinity ODD from a dataset."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

import autosafe
from autosafe.exceptions import EmptyAnchorPoolError
from autosafe.preprocessing import RangeNormalizer
from autosafe.sample import Sample
from autosafe.samples import (
    Samples,
    rows_in,
)
from autosafe.tools.evaluate.dataset.neighbor_cache import (
    _refresh_odd_kernels_from_neighbor_cache,
)
from autosafe.tools.evaluate.dataset.odd_cache import (
    _default_odd_json_path,
    _odd_matches_cache_spec,
    _ODDCacheSpec,
)
from autosafe.tools.experiments.utils import DatasetLoadOptions, load_dataset


def _build_or_load_affinity_odd(  # ruff:ignore[complex-structure, too-many-branches, too-many-arguments]
    dataset_path: Path,
    *,
    odd_json: Path | None,
    odd_json_out: Path | None,
    cache_spec: _ODDCacheSpec,
    normalizer: RangeNormalizer | None = None,
    subsample_anchors: int | None = None,
    seed: int = 0,
    exclude_points: npt.NDArray[np.float64] | None = None,
    extra_filename_tag: str = "",
) -> tuple["Samples", Path]:
    """Build affinity ODD from dataset or load existing ODD JSON.

    Args:
        dataset_path (Path): Source dataset path.
        odd_json (Path | None): Existing ODD JSON path.
        odd_json_out (Path | None): Destination path for exported ODD
            JSON.
        cache_spec (_ODDCacheSpec): ODD cache and kernel parameters.
        normalizer (RangeNormalizer | None): Optional external
            normalizer (e.g. fitted on YAML bounds). When provided it
            is applied to the raw dataset instead of the built-in IQR
            normalization path.
        subsample_anchors (int | None): If set, randomly subsample the
            anchor set to this many rows before building the ODD. Useful
            for fast evaluation runs. Subsampled runs get separate cache
            files to avoid mixing full-data and subsampled caches.
        seed (int): PRNG seed used for subsampling.
        exclude_points (npt.NDArray[np.float64] | None): Rows to remove
            from the anchor pool before subsampling, in the SAME
            (normalized) coordinate system as the anchors. Used to
            keep the anchor and OOD sets disjoint, a precondition of the
            OOD consistency loop's termination: an anchor that is also
            an OOD point holds affinity 1.0 for every covariance, so the
            loop can never converge (docs/ood-consistency.md).
        extra_filename_tag (str): Appended to ``filename_tag``. MUST be
            set whenever ``exclude_points`` is, because ``filename_tag``
            keys both the ODD JSON and the nearest-neighbor ``.npz``
            cache, and neither validates the anchor set it was built
            from---an untagged run would silently reuse caches built
            from the unfiltered pool.

    Returns:
        tuple[Samples, Path]: odd_object, odd_json_path.

    Raises:
        EmptyAnchorPoolError: If every dataset row is excluded.
    """
    filename_tag = (
        f"-sub{subsample_anchors}-seed{seed}" if subsample_anchors is not None else ""
    ) + extra_filename_tag

    if odd_json is not None:
        odd = autosafe.from_json(odd_json)
        if isinstance(odd, Samples) and not _odd_matches_cache_spec(odd, cache_spec):
            odd = _refresh_odd_kernels_from_neighbor_cache(
                odd,
                dataset_path=dataset_path,
                cache_spec=cache_spec,
                filename_tag=filename_tag,
            )
            autosafe.to_json(odd, odd_json)
        return odd, odd_json

    export_path = odd_json_out
    if export_path is None:
        export_path = _default_odd_json_path(
            dataset_path,
            closest_sample_mode=cache_spec.closest_sample_mode,
            kernel_type=cache_spec.kernel_type,
            kernel_kwargs=cache_spec.kernel_kwargs,
            normalize_data=cache_spec.normalize_data,
            yaml_normalize=cache_spec.yaml_normalize,
            filename_tag=filename_tag,
        )

    if export_path.exists():
        odd = autosafe.from_json(export_path)
        if isinstance(odd, Samples) and not _odd_matches_cache_spec(odd, cache_spec):
            odd = _refresh_odd_kernels_from_neighbor_cache(
                odd,
                dataset_path=dataset_path,
                cache_spec=cache_spec,
                filename_tag=filename_tag,
            )
            autosafe.to_json(odd, export_path)
        return odd, export_path

    if normalizer is not None:
        # External normalizer (e.g. YAML-bounds): load raw data, then
        # apply it.
        df, _ = load_dataset(
            dataset_path,
            options=DatasetLoadOptions(normalize=False),
        )
        raw_array = np.asarray(df.to_numpy(), dtype=float)
        if raw_array.ndim == 0:
            raw_array = raw_array.reshape(1, 1)
        elif raw_array.ndim == 1 and df.width == 1:
            raw_array = raw_array[:, np.newaxis]
        elif raw_array.ndim == 1:
            raw_array = raw_array.reshape(1, -1)
        base_array = np.asarray(normalizer.transform(raw_array), dtype=float)
    else:
        df, _ = load_dataset(
            dataset_path,
            options=DatasetLoadOptions(normalize=cache_spec.normalize_data),
        )
        base_array = np.asarray(df.to_numpy(), dtype=float)
        if base_array.ndim == 0:
            base_array = base_array.reshape(1, 1)
        elif base_array.ndim == 1 and df.width == 1:
            base_array = base_array[:, np.newaxis]
        elif base_array.ndim == 1:
            base_array = base_array.reshape(1, -1)

    if exclude_points is not None and np.asarray(exclude_points).size:
        keep = ~rows_in(base_array, np.asarray(exclude_points, dtype=float))
        base_array = base_array[keep]
        if base_array.shape[0] == 0:
            raise EmptyAnchorPoolError(dataset_path)

    if subsample_anchors is not None and base_array.shape[0] > subsample_anchors:
        rng = np.random.default_rng(seed)
        keep = np.sort(
            rng.choice(base_array.shape[0], size=subsample_anchors, replace=False)
        )
        base_array = base_array[keep]

    base_samples = [
        Sample(x=np.asarray(row, dtype=float).reshape(-1))
        for row in np.atleast_2d(base_array)
    ]
    odd = Samples(
        samples=base_samples,
        closest_sample_mode=cache_spec.closest_sample_mode,
        kernel_cls=cache_spec.kernel_type,
        kernel_kwargs=dict(cache_spec.kernel_kwargs),
        skip_updates=True,
    )
    odd = _refresh_odd_kernels_from_neighbor_cache(
        odd,
        dataset_path=dataset_path,
        cache_spec=cache_spec,
        filename_tag=filename_tag,
    )

    autosafe.to_json(odd, export_path)
    return odd, export_path
