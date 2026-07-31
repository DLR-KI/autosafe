# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Comparison-baseline membership for dataset-mode evaluation."""

import warnings
from pathlib import Path
from typing import NamedTuple, Protocol

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import scipy.spatial
import tqdm.rich
import typer

from autosafe.preprocessing import RangeNormalizer
from autosafe.tools.evaluate.comparison import (
    FastHullApproximation,
    create_comparison_monitor,
    validate_method_names,
)
from autosafe.tools.evaluate.dataset.ground_truth import (
    _ground_truth_labels_from_yaml,
)
from autosafe.tools.evaluate.dataset.yaml_spec import _infer_ground_truth_yaml
from autosafe.typing import (
    FloatType,
    Matrix,
    NPMatrix,
)

_HULL_FAST_DIMS = 3
_HULL_FAST_POINTS = 500

# Chunk processing threshold for memory management
_CHUNK_THRESHOLD = 5000


DEFAULT_DATASET_BASELINES = [
    "hull_single",
    "hull_clustered",
    "knn",
    "kmeans",
    "density_single",
    "density_clustered",
    "dbscan_cluster",
]


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
        method_params (dict[str, dict[str, object]]): Per-method
            parameter overrides passed to create_comparison_monitor.
    """

    reference_points: Matrix | NPMatrix
    test_points: Matrix | NPMatrix
    ref_points_t: Matrix
    test_points_t: Matrix
    n_test_points: int
    method_params: dict[str, dict[str, object]] | None = None


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

    try:  # ruff:ignore[too-many-statements-in-try-clause]
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

        monitor = create_comparison_monitor(
            method, **(data.method_params or {}).get(method, {})
        )
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
    method_params: dict[str, dict[str, object]] | None = None,
) -> dict[str, npt.NDArray[np.bool_]]:
    """Compute baseline memberships for requested comparison methods.

    This enhanced version uses the new comparison framework that
    supports configurable methods with smart automatic selection.

    Args:
        reference_points (Matrix | NPMatrix): Reference anchors.
        test_points (Matrix | NPMatrix): Query points.
        methods (list[str]): Baseline method names from experiment spec.
        method_params (dict[str, dict[str, object]] | None): Per-method
            parameter overrides forwarded to create_comparison_monitor.

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
        method_params=method_params or {},
    )

    for method in tqdm.rich.tqdm(methods):
        membership = _compute_method_membership(method, eval_data)
        if membership is not None:
            labels[method] = membership

    return labels


def _build_dataset_reference_labels(  # ruff:ignore[too-many-arguments]
    *,
    dataset_path: Path,
    anchor_points: Matrix | NPMatrix,
    test_points: Matrix | NPMatrix,
    references: list[str] | None,
    ground_truth_yaml: Path | None,
    normalizer: RangeNormalizer | None = None,
    method_params: dict[str, dict[str, object]] | None = None,
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
        method_params (dict[str, dict[str, object]] | None): Per-method
            parameter overrides forwarded to baseline monitor creation.

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
        method_params=method_params,
    )
    if effective_ground_truth_yaml is not None:
        reference_labels["ground_truth"] = _ground_truth_labels_from_yaml(
            effective_ground_truth_yaml,
            test_points,
            normalizer=normalizer,
        )
    return reference_labels
