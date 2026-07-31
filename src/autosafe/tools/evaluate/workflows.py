# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Unified evaluation workflows for Monte Carlo and real-data modes."""

import hashlib
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import polars as pl
import scipy.spatial
import tqdm.rich
import typer

import autosafe
from autosafe import ROOT_FOLDER
from autosafe.kernels.rbf import (
    SIGMA_FLOOR_RATIO,
    RBFKernel,
)
from autosafe.samples import (
    rows_in,
)
from autosafe.tools.evaluate.core import process_files
from autosafe.tools.evaluate.dataset.anchors import _extract_anchor_points
from autosafe.tools.evaluate.dataset.baselines import (
    DEFAULT_DATASET_BASELINES,
    _baseline_memberships,
    _BaselineEvaluationData,
    _build_dataset_reference_labels,
    _ComparisonMonitor,
    _compute_method_membership,
    _evaluate_monitor_membership,
    _hull_membership,
)
from autosafe.tools.evaluate.dataset.build import _build_or_load_affinity_odd
from autosafe.tools.evaluate.dataset.ground_truth import (
    _ground_truth_labels_from_yaml,
)
from autosafe.tools.evaluate.dataset.neighbor_cache import (
    _assign_cached_neighbors,
    _compute_neighbor_indices,
    _default_neighbor_cache_path,
    _get_or_create_neighbor_indices,
    _load_neighbor_indices,
    _refresh_odd_kernels_from_neighbor_cache,
)
from autosafe.tools.evaluate.dataset.normalization import (
    _denormalize_bounds_to_yaml_space,
    _denormalize_points_to_yaml_space,
    _normalize_ood_points,
    _normalizer_from_yaml_bounds,
    _numeric_array,
)
from autosafe.tools.evaluate.dataset.odd_cache import (
    _CALIBRATION_VERSION,
    _default_odd_json_path,
    _kernel_kwargs_digest,
    _odd_matches_cache_spec,
    _ODDCacheSpec,
    _resolve_kernel_calibration,
)
from autosafe.tools.evaluate.dataset.sampling import (
    _sample_points_around_odd,
    _sample_points_with_bounds,
)
from autosafe.tools.evaluate.dataset.yaml_spec import (
    _infer_ground_truth_yaml,
    _sampling_bounds_from_yaml,
)
from autosafe.tools.evaluate.metrics import (
    build_affinity_thresholds,
    build_threshold_pairs,
    evaluate_affinity_metrics,
    save_metrics_csv,
)
from autosafe.typing import (
    NPAffinityVector,
    NPMatrix,
)

# Threshold constants for hull membership decision

# Chunk processing threshold for memory management


if TYPE_CHECKING:
    from autosafe.typing import ClosestSampleModeType, KernelType


_MAX_DATASET_EVAL_SAMPLES = 1_000_000


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


def evaluate_dataset_mode(  # ruff:ignore[complex-structure, too-many-branches, too-many-arguments, too-many-locals, too-many-statements]
    dataset_path: Path,
    *,
    odd_json: Path | None = None,
    odd_json_out: Path | None = None,
    ground_truth_yaml: Path | None = None,
    threshold_mode: str | None = None,
    threshold_count: int = 100,
    references: list[str] | None = None,
    n_samples: int = 200_000,
    closest_sample_mode: "ClosestSampleModeType" = "per_dimension",
    kernel_type: "KernelType" = "RBF",
    kernel_kwargs: dict[str, object] | None = None,
    seed: int = 0,
    csv_output: Path | None = None,
    subsample_anchors: int | None = None,
    baseline_params: dict[str, object] | None = None,
    local_noise_mode: str = "span",
    local_noise_multiplier: float = 3.0,
    ood_path: Path | None = None,
    ood_xi: float | None = None,
    ood_shrink_factor: float = 0.9,
    ood_max_iterations: int = 1_000_000,
    ood_batch_jump: bool = False,
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
        threshold_mode (str | None): Deprecated and ignored. Dataset
            mode always sweeps the adaptive two-sided ``"edges"`` pair
            grid (see ``build_threshold_pairs``); passing a non-``None``
            value only triggers a ``DeprecationWarning``.
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
        subsample_anchors (int | None): If set, subsample anchors to
            this count before building the ODD (fast evaluation path).
        baseline_params (dict[str, object] | None): Per-method baseline
            parameters and optional ``auto_scale`` key. May contain
            per-method dicts like ``{"knn": {"gamma": 0.1}}`` and/or
            ``{"auto_scale": true}`` to auto-derive scale from the
            anchor spacing.
        local_noise_mode (str): Noise mode for the local half of test
            points. ``"span"`` (default) uses legacy 0.05*per-dim span;
            ``"nn"`` uses per-dim std = multiplier*d/sqrt(D) matched to
            the calibrated kernel scale.
        local_noise_multiplier (float): Multiplier m in the ``"nn"``
            mode formula (default 3.0).
        ood_path (Path | None): Optional CSV of OOD samples. When set,
            the OOD consistency adjustment is applied and the adjusted
            ODD is cached separately (ood-tagged JSON); requires
            ``ood_xi``. These rows are also SUBTRACTED from the anchor
            pool, so the anchors are drawn from the in-distribution set
            only. The adjustment loop can only terminate when the anchor
            and OOD sets are disjoint: an anchor that is also an OOD
            point holds affinity 1.0 for every covariance. See
            docs/ood-consistency.md. Because the anchor set then depends
            on the OOD file, the ODD JSON and the nearest-neighbor cache
            both carry an ``-ex<digest>`` tag.
        ood_xi (float | None): Maximum allowed OOD affinity in (0, 1).
            Required when ``ood_path`` is set.
        ood_shrink_factor (float): Covariance shrink factor c in (0, 1)
            for the OOD adjustment (default 0.9).
        ood_max_iterations (int): Safety cap for the adjustment loop.
            The worst-case number of adjustments grows with the anchor
            count and at N = 20000 already exceeds the 1e6 default, so
            production-scale runs may need to raise this. See
            docs/ood-consistency.md.
        ood_batch_jump (bool): Apply the closed-form number of shrinks
            per iteration instead of one. Faster but may over-shrink
            relative to the one-shrink-per-iteration procedure; off by
            default.

    Returns:
        tuple[pl.DataFrame, Path, Path]: metrics_df, csv_path,
            odd_json_path.

    Raises:
        ValueError: If no reference labels are available, if
            local_noise_mode is unknown, or if ood_path is set without
            ood_xi.
    """
    if threshold_mode is not None:
        warnings.warn(
            "threshold_mode is deprecated and ignored for dataset-mode "
            "evaluation; the adaptive two-sided 'edges' grid is always used.",
            DeprecationWarning,
            stacklevel=2,
        )
    if ood_path is not None and ood_xi is None:
        raise ValueError("ood_path requires ood_xi to be set")
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

    # The OOD points are needed BEFORE the ODD is built, because the
    # anchor pool is the complement of the OOD set (see
    # docs/ood-consistency.md). Two digests: the anchor set depends only
    # on WHICH rows are excluded, the adjusted ODD also on (xi, c).
    ood_norm: npt.NDArray[np.float64] | None = None
    ood_file_tag = ""
    ood_run_tag = ""
    if ood_path is not None:
        # ood_xi is guaranteed non-None by the early validation above.
        ood_xi = cast("float", ood_xi)
        ood_bytes = ood_path.read_bytes()
        ood_file_tag = f"-ex{hashlib.sha256(ood_bytes).hexdigest()[:8]}"
        ood_run_tag = (
            "-ood"
            + hashlib.sha256(
                ood_bytes + f":{ood_xi}:{ood_shrink_factor}".encode()
            ).hexdigest()[:8]
        )
        ood_norm = _normalize_ood_points(
            ood_path,
            dataset_path,
            yaml_normalizer=yaml_normalizer,
            normalize_data=normalize_data,
        )

    odd, odd_export_path = _build_or_load_affinity_odd(
        dataset_path,
        odd_json=odd_json,
        odd_json_out=odd_json_out,
        cache_spec=cache_spec,
        normalizer=yaml_normalizer,
        subsample_anchors=subsample_anchors,
        seed=seed,
        exclude_points=ood_norm,
        extra_filename_tag=ood_file_tag,
    )

    # OOD consistency adjustment.
    # The adjusted ODD is cached under an ood-tagged path so it never
    # overwrites the unadjusted cache; on a cache hit, enforcement is a
    # no-op (idempotent), so the file is not rewritten.
    ood_summary: dict[str, object] | None = None
    if ood_norm is not None:
        ood_xi = cast("float", ood_xi)
        adjusted_path = odd_export_path.with_name(
            f"{odd_export_path.stem}{ood_run_tag}{odd_export_path.suffix}"
        )
        if adjusted_path.exists():
            odd = autosafe.from_json(adjusted_path)
            ood_summary = odd.enforce_ood_consistency(
                ood_norm,
                xi=ood_xi,
                shrink_factor=ood_shrink_factor,
                max_iterations=ood_max_iterations,
                batch_jump=ood_batch_jump,
            )
        else:
            ood_summary = odd.enforce_ood_consistency(
                ood_norm,
                xi=ood_xi,
                shrink_factor=ood_shrink_factor,
                max_iterations=ood_max_iterations,
                batch_jump=ood_batch_jump,
            )
            autosafe.to_json(odd, adjusted_path)
        odd_export_path = adjusted_path

    anchor_points = _extract_anchor_points(odd)
    anchor_arr = np.asarray(anchor_points, dtype=float)

    # Compute median full-space NN distance once for auto_scale and nn
    # noise.
    median_nn: float | None = None
    if local_noise_mode == "nn" or (baseline_params or {}).get("auto_scale"):
        tree = scipy.spatial.KDTree(anchor_arr, compact_nodes=True, balanced_tree=True)
        nn_d, _ = tree.query(anchor_arr, k=2)
        positive = nn_d[:, 1][nn_d[:, 1] > 0]
        if positive.size == 0:
            raise ValueError("all anchors are exact duplicates")
        median_nn = float(np.median(positive))

    local_noise_std: float | None = None
    if local_noise_mode == "nn":
        if median_nn is None:
            raise ValueError(
                "Cannot compute local noise std without a valid median NN distance"
            )
        local_noise_std = float(
            local_noise_multiplier * median_nn / np.sqrt(anchor_arr.shape[1])
        )
    elif local_noise_mode != "span":
        raise ValueError(f"unknown local_noise_mode {local_noise_mode!r}")

    # Resolve baseline scale parameters
    bp = baseline_params or {}
    resolved_baseline_params: dict[str, dict[str, object]] = {
        k: dict(cast("dict[str, object]", v))
        for k, v in bp.items()
        if isinstance(v, dict)
    }
    if bp.get("auto_scale"):
        if median_nn is None:
            raise ValueError("Cannot auto_scale without a valid median NN distance")
        raw_mult = bp.get("auto_scale_multiplier", 3.0)
        multiplier = float(raw_mult) if isinstance(raw_mult, (int, float)) else 3.0
        scale = multiplier * median_nn
        for method, key in (
            ("knn", "gamma"),
            ("density_single", "gamma"),
            ("density_clustered", "gamma"),
            ("dbscan_cluster", "eps"),
        ):
            resolved_baseline_params.setdefault(method, {}).setdefault(key, scale)

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
            local_noise_std=local_noise_std,
        )
    else:
        test_points = _sample_points_around_odd(
            anchor_points,
            n_samples=n_samples,
            seed=seed,
            local_noise_std=local_noise_std,
        )

    # Dual affinity computation (7d)
    for _ in tqdm.rich.tqdm(
        range(1),
        desc="Calculating affinities and ODD memberships for "
        f"{len(test_points)} test points",
    ):
        alpha_lin, survival = odd.affinity_dual(test_points)
        affinities = np.asarray(alpha_lin)
        survival_np = np.asarray(survival)

    thresholds = build_threshold_pairs(threshold_count, affinities, survival_np)
    reference_labels = _build_dataset_reference_labels(
        dataset_path=dataset_path,
        anchor_points=anchor_points,
        test_points=test_points,
        references=references,
        ground_truth_yaml=effective_ground_truth_yaml,
        normalizer=yaml_normalizer,
        method_params=resolved_baseline_params,
    )

    if not reference_labels:
        raise ValueError("No reference labels available for dataset evaluation")

    samples_df = pl.DataFrame({"affinity": affinities, "survival": survival_np})
    results = evaluate_affinity_metrics(
        samples_df=samples_df,
        reference_labels=reference_labels,
        thresholds=thresholds,
        source=dataset_path.name,
    )

    if csv_output is None:
        # ood_run_tag keeps an OOD run from overwriting the CSV and
        # sidecar of the otherwise identically-configured non-OOD run.
        tag = (
            f"-sub{subsample_anchors}-seed{seed}"
            if subsample_anchors is not None
            else ""
        ) + ood_run_tag
        csv_output = dataset_path.with_name(
            f"{dataset_path.stem}-evaluation-edges{tag}.csv"
        )

    csv_path = save_metrics_csv(results, csv_output)

    # Sidecar JSON
    sidecar_path = csv_path.with_name(csv_path.stem + "-params.json")
    try:  # ruff:ignore[too-many-statements-in-try-clause]
        kernel_scale_realized: object = None
        if odd.samples:
            k = odd.samples[0].kernel
            k = cast("RBFKernel", k)
            ksr: dict[str, object] = {
                "kappa": list(np.atleast_1d(k.kappa)),
                "eta": list(np.atleast_1d(k.eta)),
            }
            lam_val = getattr(k, "lam", None)
            if lam_val is not None:
                ksr["lam"] = list(np.atleast_1d(lam_val))
                ksr["lam_source"] = "explicit"
            else:
                ksr["lam"] = list(
                    np.atleast_1d(SIGMA_FLOOR_RATIO * np.asarray(k.kappa))
                )
                ksr["lam_source"] = "default_ratio"
            kernel_scale_realized = ksr
    except AttributeError:
        kernel_scale_realized = None
    sidecar = {
        "dataset": str(dataset_path),
        "seed": seed,
        "n_samples": n_samples,
        "subsample_anchors": subsample_anchors,
        "closest_sample_mode": closest_sample_mode,
        "kernel_type": kernel_type,
        "kernel_kwargs_spec": kernel_kwargs,
        "kernel_scale_realized": kernel_scale_realized,
        "median_nn_distance": median_nn,
        "median_nn_distance_note": (
            "calibration uses the FAISS float32 neighbor cache; "
            "this value is the float64 cKDTree median (sub-percent difference)"
        ),
        "local_noise_mode": local_noise_mode,
        "local_noise_multiplier": local_noise_multiplier,
        "ood_path": str(ood_path) if ood_path is not None else None,
        "ood_xi": ood_xi,
        "ood_shrink_factor": ood_shrink_factor if ood_path is not None else None,
        "ood_iterations": (
            ood_summary["iterations"] if ood_summary is not None else None
        ),
        "ood_max_affinity_final": (
            ood_summary["max_ood_affinity"] if ood_summary is not None else None
        ),
        "ood_adjusted_kernel_count": (
            len(cast("dict[int, int]", ood_summary["adjusted_kernels"]))
            if ood_summary is not None
            else None
        ),
        "n_anchors": int(anchor_points.shape[0]),
        "ood_n_points": int(ood_norm.shape[0]) if ood_norm is not None else None,
        "ood_anchor_exclusion_tag": ood_file_tag or None,
        # Audit number: must be 0 on every OOD run. Computed post hoc so
        # it is correct on the cache-hit path too.
        "ood_anchor_coincidences": (
            int(rows_in(anchor_points, ood_norm).sum())
            if ood_norm is not None
            else None
        ),
        "ood_max_iterations": (ood_max_iterations if ood_path is not None else None),
        "ood_batch_jump": ood_batch_jump if ood_path is not None else None,
        "calibration_version": _CALIBRATION_VERSION,
        "baseline_params_resolved": resolved_baseline_params,
        "threshold_mode": "edges",
        "threshold_count": threshold_count,
        "threshold_grid_stats": {
            "smallest_positive_affinity": (
                float(np.min(affinities[affinities > 0]))
                if np.any(affinities > 0)
                else None
            ),
            "min_survival": (
                float(np.min(survival_np[np.isfinite(survival_np)]))
                if np.any(np.isfinite(survival_np))
                else None
            ),
        },
        "notes": {
            "ground_truth_prevalence": (
                "Uniform test points are drawn in a +/-10% expanded box, so only "
                "(1/1.2)^D of them lie inside the YAML box; anchors outside the "
                "YAML bounds further reduce prevalence. This is expected."
            )
        },
    }
    try:
        sidecar_path.write_text(json.dumps(sidecar, default=str, indent=2))
    except OSError as exc:
        typer.echo(f"warning: could not write sidecar: {exc}")

    return results, csv_path, odd_export_path


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
    "DEFAULT_DATASET_BASELINES",
    "_CALIBRATION_VERSION",
    "_BaselineEvaluationData",
    "_ComparisonMonitor",
    "_ODDCacheSpec",
    "_assign_cached_neighbors",
    "_baseline_memberships",
    "_build_dataset_reference_labels",
    "_build_or_load_affinity_odd",
    "_compute_method_membership",
    "_compute_neighbor_indices",
    "_default_neighbor_cache_path",
    "_default_odd_json_path",
    "_denormalize_bounds_to_yaml_space",
    "_denormalize_points_to_yaml_space",
    "_evaluate_monitor_membership",
    "_extract_anchor_points",
    "_get_or_create_neighbor_indices",
    "_ground_truth_labels_from_yaml",
    "_hull_membership",
    "_infer_ground_truth_yaml",
    "_kernel_kwargs_digest",
    "_load_neighbor_indices",
    "_normalize_ood_points",
    "_normalizer_from_yaml_bounds",
    "_numeric_array",
    "_odd_matches_cache_spec",
    "_refresh_odd_kernels_from_neighbor_cache",
    "_resolve_kernel_calibration",
    "_sample_points_around_odd",
    "_sample_points_with_bounds",
    "_sampling_bounds_from_yaml",
    "collect_monte_carlo_files",
    "evaluate_dataset_mode",
    "evaluate_monte_carlo_results",
]
