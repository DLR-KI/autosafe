# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Cache-key construction for exported affinity ODD JSON files."""

import hashlib
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from autosafe.deduplication import DEDUP_ALGORITHM_VERSION, DeduplicationPolicy
from autosafe.exceptions import (
    AutoCalibrationModeError,
    ConflictingParametersError,
    ParameterOutOfRangeError,
    ParameterTypeError,
    UnknownOptionError,
)
from autosafe.kernels.rbf import (
    WIDTH_SCHEDULE_SQRT_LOG_N,
    calibrate_rbf_scale_d_tilde,
    resolve_width_multiple,
)
from autosafe.samples import (
    Samples,
)

if TYPE_CHECKING:
    from autosafe.typing import (
        NPMatrix,
    )

if TYPE_CHECKING:
    from autosafe.typing import ClosestSampleModeType, KernelType


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


def _default_odd_json_path(  # ruff:ignore[too-many-arguments]
    dataset_path: Path,
    *,
    closest_sample_mode: "ClosestSampleModeType",
    kernel_type: "KernelType",
    kernel_kwargs: dict[str, object],
    normalize_data: bool,
    yaml_normalize: bool = False,
    filename_tag: str = "",
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
        filename_tag (str): Optional tag inserted before ".json" to
            separate subsampled from full-data caches.

    Returns:
        Path: Default cache path for the ODD JSON.
    """
    digest_kwargs = dict(kernel_kwargs)
    if "calibration" in digest_kwargs:
        digest_kwargs["_calibration_version"] = _CALIBRATION_VERSION
    kernel_digest = _kernel_kwargs_digest(digest_kwargs)
    norm_tag = "yaml" if yaml_normalize else str(int(normalize_data))
    return dataset_path.with_name(
        f"{dataset_path.stem}-odd-{closest_sample_mode}-{kernel_type}-"
        f"norm-{norm_tag}-{kernel_digest}{filename_tag}.json"
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


def _dedup_filename_tag(
    policy: DeduplicationPolicy,
    representative_points: "NPMatrix",
) -> str:
    """Build the ``-dedup<digest>`` cache tag for a de-duplication run.

    De-duplication changes the anchor set, and neither
    ``_load_neighbor_indices`` (mode/shape only) nor
    ``_odd_matches_cache_spec`` (kernel settings only) would notice that
    change, so an unversioned anchor-set change could be silently
    absorbed by an existing cache. This follows the same
    ``-ex<digest>``/``-ood<digest>`` precedent used for OOD exclusion:
    the tag binds the algorithm version, the fully resolved policy,
    and a digest of the resulting representative coordinates into the
    cache path itself, so a different de-duplication outcome always
    resolves to a different file.

    Args:
        policy (DeduplicationPolicy): Fully resolved de-duplication
            policy.
        representative_points (NPMatrix): Shape ``(n_cells, n_dims)``
            retained representative coordinates.

    Returns:
        str: Filename tag of the form ``-dedup<8 hex chars>``.
    """
    pts = np.ascontiguousarray(np.asarray(representative_points, dtype=np.float64))
    payload = f"v{DEDUP_ALGORITHM_VERSION}:{policy.digest()}:".encode() + pts.tobytes()
    return f"-dedup{hashlib.sha256(payload).hexdigest()[:8]}"


# Bump whenever the calibration implementation changes: it is hashed
# into the ODD cache digest so stale calibrated caches are never
# silently reused. Version 4: the default width multiple changed from
# s = 3 to the sqrt(ln N) schedule.
_CALIBRATION_VERSION = 4

_CALIBRATION_META_KEYS = (
    "calibration",
    "calibration_c",
    "calibration_gamma",
    "calibration_s",
    "calibration_lambda_rel",
)


def _calibration_width_spec(kernel_kwargs: dict[str, object]) -> object | None:
    """Return the requested width multiple of a calibrated kernel.

    Args:
        kernel_kwargs (dict[str, object]): Kernel kwargs, possibly
            containing calibration meta-keys.

    Returns:
        object | None: The ``calibration_s`` value (the schedule name
            when unset), or None when the kernel is not calibrated.
    """
    if str(kernel_kwargs.get("calibration", "manual")) != "auto":
        return None
    return kernel_kwargs.get("calibration_s", WIDTH_SCHEDULE_SQRT_LOG_N)


def _width_filename_tag(kernel_kwargs: dict[str, object]) -> str:
    """Filename tag naming the width multiple of a calibrated run.

    Output CSVs and sidecars are not keyed by the kernel digest, so
    without this tag a run with a different width multiple would
    overwrite the results of an otherwise identical run.

    Args:
        kernel_kwargs (dict[str, object]): Kernel kwargs, possibly
            containing calibration meta-keys.

    Returns:
        str: ``-width-sqrt-log-n`` for the schedule, ``-width-<s>`` for
            a fixed multiple, or an empty string for manual kernels.
    """
    spec = _calibration_width_spec(kernel_kwargs)
    if spec is None:
        return ""
    if isinstance(spec, str):
        return "-width-" + spec.strip().casefold().replace("_", "-")
    return f"-width-{float(cast('float', spec)):g}"


def _resolve_kernel_calibration(
    kernel_kwargs: dict[str, object],
    reference_points: "NPMatrix",
    indices: "npt.NDArray[np.int64]",
    *,
    closest_sample_mode: "ClosestSampleModeType",
) -> dict[str, object]:
    """Resolve calibration meta-kwargs into concrete kernel kwargs.

    Returns kwargs safe to pass to RBFKernel.update: meta keys are
    stripped; in 'auto' mode isotropic kappa/eta scalars are derived
    from the full-space L2 NN distances (see
    docs/bandwidth-calibration.md). Requires closest_sample_mode:
    global.

    Args:
        kernel_kwargs (dict[str, object]): Kernel kwargs, possibly
            containing calibration meta-keys.
        reference_points (NPMatrix): (N, D) anchor positions.
        indices (npt.NDArray[np.int64]): Nearest-neighbor index array
            from the neighbor cache.
        closest_sample_mode (ClosestSampleModeType): Mode for nearest-
            neighbor assignment, "global" or "per_dimension".

    Returns:
        dict[str, object]: Resolved kwargs without calibration
            meta-keys.

    Raises:
        ConflictingParametersError: If both calibration_gamma and the
            deprecated calibration_c are given.
        UnknownOptionError: If an unknown calibration mode is specified.
        AutoCalibrationModeError: If automatic calibration is requested
            without the global closest-sample mode.
        ParameterTypeError: If a relative variance floor is not
            numeric.
        ParameterOutOfRangeError: If a relative variance floor is
            outside (0, 1).
    """
    meta = dict(kernel_kwargs)
    mode = str(meta.pop("calibration", "manual"))
    raw_gamma = meta.pop("calibration_gamma", None)
    legacy_c = meta.pop("calibration_c", None)
    if raw_gamma is not None and legacy_c is not None:
        raise ConflictingParametersError("calibration_gamma", "calibration_c")
    if legacy_c is not None:
        warnings.warn(
            "calibration_c is deprecated; use calibration_gamma",
            DeprecationWarning,
            stacklevel=2,
        )
        raw_gamma = legacy_c
    gamma = float(raw_gamma) if isinstance(raw_gamma, (int, float)) else 1.0
    raw_s = meta.pop("calibration_s", WIDTH_SCHEDULE_SQRT_LOG_N)
    raw_lambda_rel = meta.pop("calibration_lambda_rel", None)
    if mode == "manual":
        return meta
    if mode != "auto":
        raise UnknownOptionError("calibration", mode, ("manual", "auto"))
    if closest_sample_mode != "global":
        raise AutoCalibrationModeError(closest_sample_mode)
    pts = np.asarray(reference_points, dtype=float)
    idx = np.asarray(indices)  # global mode: shape (N,)
    d_l2 = np.linalg.norm(pts[idx] - pts, axis=1)
    # Numbers and the schedule name are resolved against the number of
    # anchors that instantiate kernels; anything else raises instead of
    # silently falling back to a default width.
    s = resolve_width_multiple(cast("float | str", raw_s), pts.shape[0])
    kappa, eta = calibrate_rbf_scale_d_tilde(d_l2, gamma=gamma, s=s)
    meta["kappa"] = kappa
    meta["eta"] = eta
    if raw_lambda_rel is not None:
        if not isinstance(raw_lambda_rel, (int, float)):
            raise ParameterTypeError(
                "calibration_lambda_rel", raw_lambda_rel, (int, float)
            )
        lambda_rel = float(raw_lambda_rel)
        if not 0.0 < lambda_rel < 1.0:
            raise ParameterOutOfRangeError(
                "calibration_lambda_rel", lambda_rel, 0.0, 1.0
            )
        meta["lam"] = lambda_rel * kappa
    return meta
