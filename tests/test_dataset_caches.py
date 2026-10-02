# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Real-file tests for the dataset-mode caches and normalization helpers.

Unlike the mocked workflow tests, these build actual ODDs from CSV files
so that cache hits, cache rejection and kernel refreshes are exercised
end to end.
"""

from pathlib import Path

import numpy as np
import pytest

import autosafe
from autosafe.exceptions import EmptyAnchorPoolError
from autosafe.kernels.rbf import RBFKernel
from autosafe.preprocessing import RangeNormalizer
from autosafe.samples import Samples
from autosafe.tools.evaluate.dataset import neighbor_cache
from autosafe.tools.evaluate.dataset.build import _build_or_load_affinity_odd
from autosafe.tools.evaluate.dataset.neighbor_cache import (
    _compute_neighbor_indices,
    _get_or_create_neighbor_indices,
    _load_neighbor_indices,
)
from autosafe.tools.evaluate.dataset.normalization import (
    _denormalize_bounds_to_yaml_space,
    _normalize_ood_points,
)
from autosafe.tools.evaluate.dataset.odd_cache import (
    _ODDCacheSpec,
    _resolve_kernel_calibration,
)
from autosafe.tools.experiments.utils import DatasetLoadOptions, load_dataset
from autosafe.typing import ClosestSampleModeType


def _write_csv(path: Path, points: np.ndarray) -> Path:
    header = ",".join(f"x{i}" for i in range(points.shape[1]))
    np.savetxt(path, points, delimiter=",", header=header, comments="")
    return path


def _dataset(tmp_path: Path, n: int = 30, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    return _write_csv(tmp_path / "data.csv", rng.normal(size=(n, 2)) * [3.0, 0.5])


def _spec(kernel_kwargs: dict[str, object] | None = None) -> _ODDCacheSpec:
    return _ODDCacheSpec(
        closest_sample_mode="global",
        kernel_type="RBF",
        kernel_kwargs=kernel_kwargs if kernel_kwargs is not None else {},
        normalize_data=True,
    )


def _sigmas(odd: Samples) -> np.ndarray:
    sigmas = []
    for s in odd.samples:
        assert isinstance(s.kernel, RBFKernel)
        sigmas.append(np.asarray(s.kernel.sigma))
    return np.array(sigmas)


@pytest.mark.parametrize("mode", ["global", "per_dimension"])
def test_neighbor_cache_hit_skips_recomputation(
    tmp_path: Path, mode: ClosestSampleModeType, monkeypatch: pytest.MonkeyPatch
) -> None:
    points = np.random.default_rng(1).normal(size=(12, 3))
    cache = tmp_path / "nn.npz"
    first = _get_or_create_neighbor_indices(
        points, cache_path=cache, closest_sample_mode=mode
    )
    assert cache.exists()

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cache hit must not recompute")

    monkeypatch.setattr(neighbor_cache, "_compute_neighbor_indices", _fail)
    second = _get_or_create_neighbor_indices(
        points, cache_path=cache, closest_sample_mode=mode
    )
    np.testing.assert_array_equal(first, second)


def test_neighbor_cache_rejects_other_mode_and_shape(tmp_path: Path) -> None:
    points = np.random.default_rng(2).normal(size=(10, 2))
    cache = tmp_path / "nn.npz"
    _get_or_create_neighbor_indices(
        points, cache_path=cache, closest_sample_mode="global"
    )
    assert (
        _load_neighbor_indices(
            cache, n_points=10, n_dims=2, closest_sample_mode="per_dimension"
        )
        is None
    )
    assert (
        _load_neighbor_indices(
            cache, n_points=11, n_dims=2, closest_sample_mode="global"
        )
        is None
    )
    assert (
        _load_neighbor_indices(
            cache, n_points=10, n_dims=2, closest_sample_mode="global"
        )
        is not None
    )


def test_neighbor_indices_for_a_single_point() -> None:
    one = np.zeros((1, 3))
    assert _compute_neighbor_indices(one, closest_sample_mode="global").shape == (1,)
    assert _compute_neighbor_indices(
        one, closest_sample_mode="per_dimension"
    ).shape == (3, 1)


@pytest.mark.parametrize("via", ["odd_json", "odd_json_out"])
def test_cached_odd_with_other_kernel_settings_is_refreshed(
    tmp_path: Path, via: str
) -> None:
    """A cached ODD built with other kernel kwargs must not be reused as is."""
    data = _dataset(tmp_path)
    cached_path = tmp_path / "cached.json"
    _build_or_load_affinity_odd(
        data, odd_json=None, odd_json_out=cached_path, cache_spec=_spec()
    )

    new_kwargs: dict[str, object] = {"kappa": 2.0, "eta": 0.5}
    fresh, _ = _build_or_load_affinity_odd(
        data,
        odd_json=None,
        odd_json_out=tmp_path / "fresh.json",
        cache_spec=_spec(new_kwargs),
    )
    refreshed, path = _build_or_load_affinity_odd(
        data,
        odd_json=cached_path if via == "odd_json" else None,
        odd_json_out=cached_path if via == "odd_json_out" else None,
        cache_spec=_spec(new_kwargs),
    )

    assert path == cached_path
    assert isinstance(refreshed, Samples)
    assert refreshed.kernel_kwargs == new_kwargs
    np.testing.assert_allclose(_sigmas(refreshed), _sigmas(fresh))
    # The refreshed ODD is written back, so the next run is a plain hit.
    reloaded = autosafe.from_json(cached_path)
    assert isinstance(reloaded, Samples)
    assert reloaded.kernel_kwargs == new_kwargs


def test_excluding_every_row_raises(tmp_path: Path) -> None:
    data = _dataset(tmp_path, n=6)
    df, _ = load_dataset(data, options=DatasetLoadOptions(normalize=True))
    with pytest.raises(EmptyAnchorPoolError, match="no anchors"):
        _build_or_load_affinity_odd(
            data,
            odd_json=None,
            odd_json_out=tmp_path / "odd.json",
            cache_spec=_spec(),
            exclude_points=np.asarray(df.to_numpy(), dtype=float),
            extra_filename_tag="-exall",
        )


def test_ood_points_use_the_anchors_iqr_normalization(tmp_path: Path) -> None:
    """OOD rows taken from the dataset land exactly on the normalized anchors."""
    data = _dataset(tmp_path)
    raw, _ = load_dataset(data, options=DatasetLoadOptions(normalize=False))
    normalized, _ = load_dataset(data, options=DatasetLoadOptions(normalize=True))
    ood = _write_csv(tmp_path / "ood.csv", np.asarray(raw.to_numpy())[:5])

    ood_n = _normalize_ood_points(ood, data, yaml_normalizer=None, normalize_data=True)
    np.testing.assert_allclose(ood_n, np.asarray(normalized.to_numpy())[:5])

    ood_raw = _normalize_ood_points(
        ood, data, yaml_normalizer=None, normalize_data=False
    )
    np.testing.assert_allclose(ood_raw, np.asarray(raw.to_numpy())[:5])


def test_denormalize_bounds_inverts_the_normalizer() -> None:
    lower, upper = np.array([-4.0, 10.0]), np.array([6.0, 30.0])
    normalizer = RangeNormalizer()
    normalizer.fit(np.vstack([lower, upper]))
    lo_n = np.asarray(normalizer.transform(lower.reshape(1, -1))[0])
    up_n = np.asarray(normalizer.transform(upper.reshape(1, -1))[0])
    lo, up = _denormalize_bounds_to_yaml_space(lo_n, up_n, normalizer)
    np.testing.assert_allclose(lo, lower)
    np.testing.assert_allclose(up, upper)


def _calibrate(
    kwargs: dict[str, object], mode: ClosestSampleModeType = "global"
) -> dict[str, object]:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [3.0, 2.0]])
    idx = _compute_neighbor_indices(pts, closest_sample_mode="global")
    return _resolve_kernel_calibration(kwargs, pts, idx, closest_sample_mode=mode)


def test_calibration_legacy_c_matches_gamma() -> None:
    with pytest.warns(DeprecationWarning, match="calibration_c is deprecated"):
        legacy = _calibrate({"calibration": "auto", "calibration_c": 2.0})
    assert legacy == _calibrate({"calibration": "auto", "calibration_gamma": 2.0})


def test_calibration_lambda_rel_sets_relative_floor() -> None:
    out = _calibrate({"calibration": "auto", "calibration_lambda_rel": 0.25})
    kappa = out["kappa"]
    assert isinstance(kappa, float)
    assert out["lam"] == pytest.approx(0.25 * kappa)


@pytest.mark.parametrize(
    ("kwargs", "mode", "error", "match"),
    [
        (
            {"calibration": "auto", "calibration_gamma": 1.0, "calibration_c": 1.0},
            "global",
            ValueError,
            "not both",
        ),
        ({"calibration": "bogus"}, "global", ValueError, "unknown calibration mode"),
        ({"calibration": "auto"}, "per_dimension", ValueError, "requires"),
        (
            {"calibration": "auto", "calibration_lambda_rel": "0.1"},
            "global",
            TypeError,
            "must be numeric",
        ),
        (
            {"calibration": "auto", "calibration_lambda_rel": 1.0},
            "global",
            ValueError,
            r"must be in \(0, 1\)",
        ),
    ],
    ids=["gamma_and_c", "unknown_mode", "per_dimension", "lambda_type", "lambda_range"],
)
def test_calibration_rejects_invalid_settings(
    kwargs: dict[str, object],
    mode: ClosestSampleModeType,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _calibrate(kwargs, mode)
