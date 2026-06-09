# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Unified evaluation workflows for Monte Carlo and real-data modes."""

import hashlib
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl
import scipy.spatial
import tqdm.rich
import typer

import autosafe
from autosafe import ROOT_FOLDER
from autosafe.kernels import KernelDict
from autosafe.preprocessing import RangeNormalizer, create_robust_normalization_pipeline
from autosafe.sample import Sample
from autosafe.samples import (
    Samples,
    find_closest_vectors_by_index,
    find_closest_vectors_by_index_per_dimension,
)
from autosafe.tools.evaluate.comparison import (
    FastHullApproximation,
    create_comparison_monitor,
    validate_method_names,
)
from autosafe.tools.evaluate.core import process_files
from autosafe.tools.evaluate.metrics import (
    build_affinity_thresholds,
    evaluate_affinity_metrics,
    save_metrics_csv,
)
from autosafe.tools.experiments.utils import DatasetLoadOptions, load_dataset
from autosafe.tools.monte_carlo.inequality_utils import ODDFactory, load_yaml_odd_config
from autosafe.typing import (
    FloatType,
    Matrix,
    NPAffinityVector,
    NPMatrix,
)

# Threshold constants for hull membership decision
_HULL_FAST_DIMS = 3
_HULL_FAST_POINTS = 500

# Chunk processing threshold for memory management
_CHUNK_THRESHOLD = 5000
_MAX_DATASET_EVAL_SAMPLES = 1_000_000

DEFAULT_DATASET_BASELINES = [
    "hull_single",
    "hull_clustered",
    "knn",
    "kmeans",
    "density_single",
    "density_clustered",
    "dbscan_cluster",
]

if TYPE_CHECKING:
    from autosafe.typing import ClosestSampleModeType, KernelType


def _extract_anchor_points(odd: "Samples") -> NPMatrix:
    """Extract anchor points from an autoSAFE ODD object.

    Args:
        odd (Samples): Affinity ODD object.

    Returns:
        NPMatrix: Anchor points as `(n_points, n_dims)` array.
    """
    return np.array([np.array(sample.x, dtype=float) for sample in odd.samples])


def _extract_mc_samples(
    data: pl.DataFrame,
) -> tuple[NPMatrix, NPAffinityVector, npt.NDArray[np.bool_]]:
    """Extract sample coordinates, affinities, and labels from MC JSON.

    Args:
        data (pl.DataFrame): Loaded MC JSON DataFrame.

    Returns:
        tuple[NPMatrix, NPAffinityVector, npt.NDArray[np.bool_]]:
            Coordinates, affinities, in_odd_labels.
    """
    sampling = pl.DataFrame(data["sampling_results"][0]).unnest([
        col_name
        for col_name, dtype in pl.DataFrame(data["sampling_results"][0]).schema.items()
        if dtype == pl.Struct
    ])

    coordinates = np.array(sampling["coordinates"].to_list(), dtype=float)
    affinities = np.array(sampling["affinity"].to_list(), dtype=float)
    in_odd = np.array(sampling["in_odd"].to_list(), dtype=bool)
    return coordinates, affinities, in_odd


def _hull_membership(
    reference_points: Matrix | NPMatrix,
    test_points: Matrix | NPMatrix,
) -> npt.NDArray[np.bool_]:
    """Compute convex hull membership for test points.

    For large high-dimensional datasets, this can be computationally
    expensive. In such cases, provides a fast approximation for
    membership determination.

    Args:
        reference_points (Matrix | NPMatrix): Hull reference points.
        test_points (Matrix | NPMatrix): Points to classify.

    Returns:
        npt.NDArray[np.bool_]: Boolean membership vector for
            `test_points`.
    """
    n_points, n_dims = reference_points.shape

    # For large/high-dim datasets, conical hull can be too slow.
    # Use fast heuristics instead
    if n_points > _HULL_FAST_POINTS or n_dims > _HULL_FAST_DIMS:
        # Fast high-dimensional approximation
        # In high dimensions, most points are near the boundary anyway
        center = np.mean(reference_points, axis=0)
        max_radius = np.max(np.linalg.norm(reference_points - center, axis=1))

        # In high-D, use simple distance-based approximation
        # This is much faster but still reasonable
        difffromcenter = np.linalg.norm(test_points - center, axis=1)
        return difffromcenter <= max_radius + np.finfo(float).eps

    # Traditional convex hull for smaller datasets
    hull = scipy.spatial.ConvexHull(reference_points)
    a_hull, b_hull = hull.equations[:, :-1], hull.equations[:, -1]
    eps = np.finfo(float).eps
    return np.all(test_points @ a_hull.T + b_hull.T <= eps, axis=1)


def _evaluate_monitor_membership(
    monitor: "_ComparisonMonitor",
    test_points_t: Matrix | NPMatrix,
    n_test_points: int,
    *,
    chunk_size: int,
) -> npt.NDArray[np.bool_]:
    """Evaluate monitor in one batch or in chunks.

    Args:
        monitor (_ComparisonMonitor): Fitted comparison monitor to
            evaluate.
        test_points_t (Matrix | NPMatrix): Test points transposed to
            (n_dims, n_points).
        n_test_points (int): Number of test points.
        chunk_size (int): Size of chunks for evaluation when processing
            in batches. Ignored if `n_test_points` is below threshold.

    Returns:
        npt.NDArray[np.bool_]: Boolean membership predictions.
    """
    if n_test_points <= _CHUNK_THRESHOLD:
        return np.asarray(
            monitor.evaluate_batch(jnp.asarray(test_points_t, FloatType)), dtype=bool
        )

    result_chunks: list[npt.NDArray[np.bool_]] = []
    for i in range(0, n_test_points, chunk_size):
        chunk = test_points_t[:, i : i + chunk_size]
        result_chunks.append(
            np.asarray(
                monitor.evaluate_batch(jnp.asarray(chunk, FloatType)), dtype=bool
            )
        )
    return np.concatenate(result_chunks).astype(bool)


def _kernel_kwargs_digest(kernel_kwargs: dict[str, object]) -> str:
    """Create a stable digest for kernel configuration parameters.

    Args:
        kernel_kwargs (dict[str, object]): Kernel configuration
            parameters.

    Returns:
        str: Short stable digest of the kernel parameter mapping.
    """
    payload = json.dumps(kernel_kwargs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _default_odd_json_path(  # noqa: PLR0913
    dataset_path: Path,
    *,
    closest_sample_mode: "ClosestSampleModeType",
    kernel_type: "KernelType",
    kernel_kwargs: dict[str, object],
    normalize_data: bool,
    yaml_normalize: bool = False,
) -> Path:
    """Build a deterministic ODD cache path.

    The path is derived from the dataset and kernel configuration.

    Args:
        dataset_path (Path): Path to the dataset file.
        closest_sample_mode (ClosestSampleModeType): Mode for nearest-
            neighbor assignment, e.g., "global" or "per_dimension".
        kernel_type (KernelType): Type of kernel used.
        kernel_kwargs (dict[str, object]): Kernel configuration
            parameters.
        normalize_data (bool): Whether the data is normalized.
        yaml_normalize (bool): Whether normalization used YAML bounds
            rather than the data's own range.

    Returns:
        Path: Default cache path for the ODD JSON.
    """
    kernel_digest = _kernel_kwargs_digest(kernel_kwargs)
    norm_tag = "yaml" if yaml_normalize else str(int(normalize_data))
    return dataset_path.with_name(
        f"{dataset_path.stem}-odd-{closest_sample_mode}-{kernel_type}-"
        f"norm-{norm_tag}-{kernel_digest}.json"
    )


class _ODDCacheSpec(NamedTuple):
    """Container for ODD cache selection parameters.

    Attributes:
        closest_sample_mode (ClosestSampleModeType): Mode for nearest-
            neighbor assignment, e.g., "global" or "per_dimension".
        kernel_type (KernelType): Type of kernel used.
        kernel_kwargs (dict[str, object]): Kernel configuration
            parameters.
        normalize_data (bool): Whether the data is normalized.
        yaml_normalize (bool): Whether normalization used YAML bounds
            rather than the data's own range.
    """

    closest_sample_mode: "ClosestSampleModeType"
    kernel_type: "KernelType"
    kernel_kwargs: dict[str, object]
    normalize_data: bool
    yaml_normalize: bool = False


def _default_neighbor_cache_path(
    dataset_path: Path,
    *,
    closest_sample_mode: "ClosestSampleModeType",
) -> Path:
    """Build deterministic path for nearest-neighbor cache.

    Args:
        dataset_path (Path): Path to the dataset file.
        closest_sample_mode (ClosestSampleModeType): Mode for nearest-
            neighbor assignment, e.g., "global" or "per_dimension".

    Returns:
        Path: File path used for nearest-neighbor index cache.
    """
    return dataset_path.with_name(f"{dataset_path.stem}-nn-{closest_sample_mode}.npz")


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


def _odd_matches_cache_spec(odd: Samples, cache_spec: _ODDCacheSpec) -> bool:
    """Check if loaded ODD matches requested kernel/sample settings.

    Args:
        odd (Samples): Loaded ODD object to check.
        cache_spec (_ODDCacheSpec): Cache specification to compare
            against.

    Returns:
        bool: True when loaded settings match requested cache spec.
    """
    return (
        odd.closest_sample_mode == cache_spec.closest_sample_mode
        and odd.kernel_cls_str == cache_spec.kernel_type
        and odd.kernel_kwargs == cache_spec.kernel_kwargs
    )


def _refresh_odd_kernels_from_neighbor_cache(
    odd: Samples,
    *,
    dataset_path: Path,
    cache_spec: _ODDCacheSpec,
) -> Samples:
    """Refresh kernel state using cached nearest-neighbor assignments.

    Args:
        odd (Samples): ODD object to refresh with cached neighbor state.
        dataset_path (Path): Path to the dataset file, used for cache
            path derivation.
        cache_spec (_ODDCacheSpec): Cache specification to determine
            cache paths and settings for neighbor assignment and kernel
            configuration.

    Returns:
        Samples: Updated ODD object with refreshed kernel state.
    """
    reference_points = _extract_anchor_points(odd)
    neighbor_cache_path = _default_neighbor_cache_path(
        dataset_path,
        closest_sample_mode=cache_spec.closest_sample_mode,
    )
    indices = _get_or_create_neighbor_indices(
        reference_points,
        cache_path=neighbor_cache_path,
        closest_sample_mode=cache_spec.closest_sample_mode,
    )

    odd.closest_sample_mode = cache_spec.closest_sample_mode
    odd.kernel_cls_str = cache_spec.kernel_type
    odd.kernel_cls = KernelDict[cache_spec.kernel_type]
    odd.kernel_kwargs = dict(cache_spec.kernel_kwargs)

    _assign_cached_neighbors(
        odd,
        indices,
        closest_sample_mode=cache_spec.closest_sample_mode,
    )
    odd.refresh_kernels()
    return odd


class _BaselineEvaluationData(NamedTuple):
    """Container for shared baseline-evaluation inputs.

    Attributes:
        reference_points (Matrix | NPMatrix): Reference points for ODD
            boundary estimation.
        test_points (Matrix | NPMatrix): Test points to evaluate
            against the ODD boundary.
        ref_points_t (Matrix): Transposed reference points for monitor
            fitting.
        test_points_t (Matrix): Transposed test points for monitor
            evaluation.
        n_test_points (int): Number of test points, used for evaluation
            management.
    """

    reference_points: Matrix | NPMatrix
    test_points: Matrix | NPMatrix
    ref_points_t: Matrix
    test_points_t: Matrix
    n_test_points: int


class _ComparisonMonitor(Protocol):
    """Protocol for comparison monitors used in evaluation."""

    def fit(self, points: Matrix, /) -> object:
        """Fit monitor to reference points.

        Args:
            points (Matrix): Reference points to fit on.
        """

    def evaluate_batch(
        self,
        test_points: Matrix,
        /,
    ) -> npt.NDArray[np.bool_]:
        """Return membership predictions for a batch.

        Args:
            test_points (Matrix): Test points to evaluate.
        """


def _compute_method_membership(
    method: str,
    data: _BaselineEvaluationData,
) -> npt.NDArray[np.bool_] | None:
    """Compute memberships for one method.

    Args:
        method (str): Method name to compute membership for.
        data (_BaselineEvaluationData): Shared input data for membership
            computation, including reference points, test points, and
            transposed versions for monitor fitting and evaluation.

    Returns:
        npt.NDArray[np.bool_] | None: Membership result or None on
            recoverable method-specific failure.
    """

    def _fit_monitor(monitor: _ComparisonMonitor, points: Matrix) -> object:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Only \d+/\d+ clusters have >=\d+ points\.",
                category=UserWarning,
            )
            return monitor.fit(points)

    try:  # noqa: PLW0717
        if method == "hull_single":
            return _hull_membership(data.reference_points, data.test_points)

        if method == "fast_hull_approx":
            monitor = FastHullApproximation()
            monitor.fit(data.ref_points_t)
            return _evaluate_monitor_membership(
                monitor,
                data.test_points_t,
                data.n_test_points,
                chunk_size=5000,
            )

        monitor = create_comparison_monitor(method)
        _fit_monitor(monitor, data.ref_points_t)
        chunk_size = 1000 if method == "knn" else 5000
        return _evaluate_monitor_membership(
            monitor,
            data.test_points_t,
            data.n_test_points,
            chunk_size=chunk_size,
        )
    except (
        ValueError,
        TypeError,
        RuntimeError,
        np.linalg.LinAlgError,
        scipy.spatial.QhullError,
    ) as error:
        typer.echo(f"Warning: {method} failed with error: {error}")
        return None


def _baseline_memberships(
    reference_points: Matrix | NPMatrix,
    test_points: Matrix | NPMatrix,
    methods: list[str],
) -> dict[str, npt.NDArray[np.bool_]]:
    """Compute baseline memberships for requested comparison methods.

    This enhanced version uses the new comparison framework that
    supports configurable methods with smart automatic selection.

    Args:
        reference_points (Matrix | NPMatrix): Reference anchors.
        test_points (Matrix | NPMatrix): Query points.
        methods (list[str]): Baseline method names from experiment spec.

    Returns:
        dict[str, npt.NDArray[np.bool_]]: Mapping from method name to
            boolean membership vector.
    """
    labels: dict[str, npt.NDArray[np.bool_]] = {}
    ref_points_t = reference_points.T
    test_points_t = test_points.T

    # Validate and resolve method specifications from experiment spec
    try:
        validate_method_names(methods)
    except ValueError as error:
        typer.echo(f"Critical error in baseline membership computation: {error}")
        return {}

    methods_str = ", ".join(methods)
    typer.echo(f"Computing baseline memberships: {methods_str}")

    n_test_points = test_points_t.shape[1]
    eval_data = _BaselineEvaluationData(
        reference_points=reference_points,
        test_points=test_points,
        ref_points_t=ref_points_t,
        test_points_t=test_points_t,
        n_test_points=n_test_points,
    )

    for method in tqdm.rich.tqdm(methods):
        membership = _compute_method_membership(method, eval_data)
        if membership is not None:
            labels[method] = membership

    return labels


def _sample_points_around_odd(
    anchor_points: Matrix | NPMatrix,
    n_samples: int,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Sample points in and around an affinity ODD.

    Args:
        anchor_points (Matrix | NPMatrix): Affinity ODD anchors.
        n_samples (int): Number of test points to sample.
        seed (int): PRNG seed.

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

    indices = rng.integers(0, anchor_points.shape[0], size=n_local)
    local_points = anchor_points[indices] + rng.normal(
        loc=0.0,
        scale=0.05 * span,
        size=(n_local, anchor_points.shape[1]),
    )

    return np.vstack([uniform_points, local_points])


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


def _sample_points_with_bounds(
    anchor_points: Matrix | NPMatrix,
    lower_bounds: npt.NDArray[np.float64],
    upper_bounds: npt.NDArray[np.float64],
    n_samples: int,
    seed: int = 0,
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

    indices = rng.integers(0, anchor_points.shape[0], size=n_local)
    local_points = anchor_points[indices] + rng.normal(
        loc=0.0,
        scale=0.05 * span,
        size=(n_local, anchor_points.shape[1]),
    )

    return np.vstack([uniform_points, local_points])


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
        ValueError: If the legacy config is malformed.
    """
    limits = config.get("limits")
    if not isinstance(limits, dict) or not limits:
        return config

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for name, entry in limits.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Limit entry for {name!r} must be a mapping")

        values = entry.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Limit entry for {name!r} must contain values")

        lower_bounds.append(float(min(values)))
        upper_bounds.append(float(max(values)))

    return {
        "type": "box",
        "dim": len(lower_bounds),
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
    }


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


def evaluate_monte_carlo_results(
    files: list[Path],
    *,
    threshold_mode: str = "linear",
    threshold_count: int = 100,
    references: list[str] | None = None,
    csv_output: Path | None = None,
) -> pl.DataFrame:
    """Evaluate Monte Carlo result JSON files across thresholds.

    Args:
        files (list[Path]): MC results JSON files.
        threshold_mode (str): Threshold spacing mode.
        threshold_count (int): Number of threshold values.
        references (list[str] | None): Reference sets to evaluate
            against.
        csv_output (Path | None): Optional CSV output path.

    Returns:
        pl.DataFrame: Aggregated metrics.

    Raises:
        ValueError: If no reference labels are requested.
    """
    references = references or [
        "ground_truth",
        "hull_single",
        "hull_clustered",
        "knn",
        "kmeans",
        "density_single",
        "density_clustered",
        "dbscan_cluster",
    ]
    thresholds = build_affinity_thresholds(threshold_mode, threshold_count)

    all_rows: list[pl.DataFrame] = []

    for file_path in files:
        data = pl.read_json(file_path)
        coordinates, affinities, in_odd = _extract_mc_samples(data)

        reference_labels: dict[str, np.ndarray] = {}
        if "ground_truth" in references:
            reference_labels["ground_truth"] = in_odd

        anchor_points = np.array(data["anchors"][0], dtype=float)
        baseline_methods = [method for method in references if method != "ground_truth"]
        reference_labels.update(
            _baseline_memberships(anchor_points, coordinates, baseline_methods),
        )

        if not reference_labels:
            raise ValueError("No reference labels requested for evaluation")

        samples_df = pl.DataFrame({"affinity": affinities})
        all_rows.append(
            evaluate_affinity_metrics(
                samples_df=samples_df,
                reference_labels=reference_labels,
                thresholds=thresholds,
                source=file_path.name,
            ),
        )

    result = pl.concat(all_rows, how="vertical") if all_rows else pl.DataFrame()

    if csv_output is None:
        csv_output = ROOT_FOLDER / "evaluation_results_monte_carlo.csv"
    save_metrics_csv(result, csv_output)
    return result


def _build_or_load_affinity_odd(  # noqa: C901, PLR0912
    dataset_path: Path,
    *,
    odd_json: Path | None,
    odd_json_out: Path | None,
    cache_spec: _ODDCacheSpec,
    normalizer: RangeNormalizer | None = None,
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

    Returns:
        tuple[Samples, Path]: odd_object, odd_json_path.
    """
    if odd_json is not None:
        odd = autosafe.from_json(odd_json)
        if isinstance(odd, Samples) and not _odd_matches_cache_spec(odd, cache_spec):
            odd = _refresh_odd_kernels_from_neighbor_cache(
                odd,
                dataset_path=dataset_path,
                cache_spec=cache_spec,
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
        )

    if export_path.exists():
        odd = autosafe.from_json(export_path)
        if isinstance(odd, Samples) and not _odd_matches_cache_spec(odd, cache_spec):
            odd = _refresh_odd_kernels_from_neighbor_cache(
                odd,
                dataset_path=dataset_path,
                cache_spec=cache_spec,
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
    )

    autosafe.to_json(odd, export_path)
    return odd, export_path


def evaluate_dataset_mode(  # noqa: PLR0913, PLR0914
    dataset_path: Path,
    *,
    odd_json: Path | None = None,
    odd_json_out: Path | None = None,
    ground_truth_yaml: Path | None = None,
    threshold_mode: str = "linear",
    threshold_count: int = 100,
    references: list[str] | None = None,
    n_samples: int = 200_000,
    closest_sample_mode: "ClosestSampleModeType" = "per_dimension",
    kernel_type: "KernelType" = "RBF",
    kernel_kwargs: dict[str, object] | None = None,
    seed: int = 0,
    csv_output: Path | None = None,
) -> tuple[pl.DataFrame, Path, Path]:
    """Evaluate real-data workflow with optional ground truth YAML.

    Args:
        dataset_path (Path): Input dataset path.
        odd_json (Path | None): Optional existing affinity ODD JSON
            path.
        odd_json_out (Path | None): Optional output path for generated
            ODD JSON.
        ground_truth_yaml (Path | None): Optional YAML ground-truth ODD
            definition.
            If omitted, sibling ``.yml``/``.yaml`` next to dataset
            is used when present.
        threshold_mode (str): Threshold spacing mode.
        threshold_count (int): Number of thresholds.
        references (list[str] | None): Baseline references to evaluate.
        n_samples (int): Number of sampled test points.
        closest_sample_mode (ClosestSampleModeType): Affinity ODD
            construction mode.
        kernel_type (KernelType): Affinity kernel type.
        kernel_kwargs (dict[str, object] | None): Kernel constructor
            parameters.
        seed (int): PRNG seed.
        csv_output (Path | None): Optional CSV output path.

    Returns:
        tuple[pl.DataFrame, Path, Path]: metrics_df, csv_path,
            odd_json_path.

    Raises:
        ValueError: If no reference labels are available.
    """
    if n_samples > _MAX_DATASET_EVAL_SAMPLES:
        raise ValueError(
            "Dataset evaluation samples are too large for in-memory evaluation: "
            f"{n_samples}. Use <= {_MAX_DATASET_EVAL_SAMPLES} or split runs."
        )

    kernel_kwargs = kernel_kwargs or {}
    effective_ground_truth_yaml = ground_truth_yaml or _infer_ground_truth_yaml(
        dataset_path,
    )

    yaml_normalizer = None
    if effective_ground_truth_yaml is not None:
        yaml_normalizer = _normalizer_from_yaml_bounds(effective_ground_truth_yaml)

    yaml_norm = yaml_normalizer is not None
    normalize_data = True

    cache_spec = _ODDCacheSpec(
        closest_sample_mode=closest_sample_mode,
        kernel_type=kernel_type,
        kernel_kwargs=kernel_kwargs,
        normalize_data=normalize_data,
        yaml_normalize=yaml_norm,
    )

    odd, odd_export_path = _build_or_load_affinity_odd(
        dataset_path,
        odd_json=odd_json,
        odd_json_out=odd_json_out,
        cache_spec=cache_spec,
        normalizer=yaml_normalizer,
    )

    anchor_points = _extract_anchor_points(odd)

    sampling_bounds = None
    if effective_ground_truth_yaml is not None:
        sampling_bounds = _sampling_bounds_from_yaml(effective_ground_truth_yaml)

    if (
        sampling_bounds is not None
        and sampling_bounds[0].shape[0] == anchor_points.shape[1]
    ):
        if yaml_normalizer is not None:
            lower_norm = np.asarray(
                yaml_normalizer.transform(sampling_bounds[0].reshape(1, -1))
            ).flatten()
            upper_norm = np.asarray(
                yaml_normalizer.transform(sampling_bounds[1].reshape(1, -1))
            ).flatten()
        else:
            lower_norm = sampling_bounds[0]
            upper_norm = sampling_bounds[1]
        test_points = _sample_points_with_bounds(
            anchor_points,
            lower_norm,
            upper_norm,
            n_samples=n_samples,
            seed=seed,
        )
    else:
        test_points = _sample_points_around_odd(
            anchor_points,
            n_samples=n_samples,
            seed=seed,
        )

    # Fake progress bar for affinity calculation and ODD membership
    # This is done to have a consistent output format
    for _ in tqdm.rich.tqdm(
        range(1),
        desc="Calculating affinities and ODD memberships for "
        f"{len(test_points)} test points",
    ):
        affinities = np.asarray(odd(test_points))

    thresholds = build_affinity_thresholds(threshold_mode, threshold_count)
    reference_labels = _build_dataset_reference_labels(
        dataset_path=dataset_path,
        anchor_points=anchor_points,
        test_points=test_points,
        references=references,
        ground_truth_yaml=effective_ground_truth_yaml,
        normalizer=yaml_normalizer,
    )

    if not reference_labels:
        raise ValueError("No reference labels available for dataset evaluation")

    samples_df = pl.DataFrame({"affinity": affinities})
    results = evaluate_affinity_metrics(
        samples_df=samples_df,
        reference_labels=reference_labels,
        thresholds=thresholds,
        source=dataset_path.name,
    )

    if csv_output is None:
        csv_output = dataset_path.with_stem(
            f"{dataset_path.stem}-evaluation-{threshold_mode}"
        ).with_suffix(".csv")

    csv_path = save_metrics_csv(results, csv_output)
    return results, csv_path, odd_export_path


def _build_dataset_reference_labels(  # noqa: PLR0913
    *,
    dataset_path: Path,
    anchor_points: Matrix | NPMatrix,
    test_points: Matrix | NPMatrix,
    references: list[str] | None,
    ground_truth_yaml: Path | None,
    normalizer: RangeNormalizer | None = None,
) -> dict[str, npt.NDArray[np.bool_]]:
    """Build baseline and ground-truth labels for dataset evaluation.

    Args:
        dataset_path (Path): Input dataset path.
        anchor_points (Matrix): Reference anchor points from ODD.
        test_points (Matrix): Test points to evaluate.
        references (list[str] | None): Baseline reference methods.
        ground_truth_yaml (Path | None): Path to YAML GT.
        normalizer (RangeNormalizer | None): Optional normalizer.
            If provided, denormalization is applied when checking
            YAML ground-truth membership.

    Returns:
        dict[str, npt.NDArray[np.bool_]]: Reference labels for
            dataset evaluation.
    """
    effective_ground_truth_yaml = ground_truth_yaml or _infer_ground_truth_yaml(
        dataset_path,
    )

    requested_baselines = references or DEFAULT_DATASET_BASELINES
    requested_baselines = [ref for ref in requested_baselines if ref != "ground_truth"]

    if effective_ground_truth_yaml is not None:
        baseline_methods = list(
            dict.fromkeys([*DEFAULT_DATASET_BASELINES, *requested_baselines]),
        )
    else:
        baseline_methods = requested_baselines

    reference_labels = _baseline_memberships(
        anchor_points,
        test_points,
        baseline_methods,
    )
    if effective_ground_truth_yaml is not None:
        reference_labels["ground_truth"] = _ground_truth_labels_from_yaml(
            effective_ground_truth_yaml,
            test_points,
            normalizer=normalizer,
        )
    return reference_labels


def collect_monte_carlo_files(inputs: list[str]) -> list[Path]:
    """Collect MC result files from paths or directories.

    Args:
        inputs (list[str]): Files or directories.

    Returns:
        list[Path]: Expanded list of result JSON files.
    """
    files: list[Path] = []
    for file_input in inputs:
        files.extend(process_files(file_input))
    return files


__all__ = [
    "collect_monte_carlo_files",
    "evaluate_dataset_mode",
    "evaluate_monte_carlo_results",
]
