# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# ruff:file-ignore[ambiguous-unicode-character-string]
"""Tests for the array-based construction of global-mode RBF samples."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from autosafe.exceptions import (
    ParameterOrderError,
    ShapeMismatchError,
    TooFewAnchorsError,
    UnknownOptionError,
)
from autosafe.kernels.rbf import RBFKernel, rbf_diagonal_sigma
from autosafe.odd import AutoSafeConfig, AutoSafeODD
from autosafe.sample import Sample
from autosafe.samples import Samples
from autosafe.tools.exporters import to_json
from autosafe.tools.importers import from_json

_QUERY_COUNT = 200


def _legacy(points: npt.NDArray[np.float64], kwargs: dict[str, Any]) -> Samples:
    return Samples(
        [Sample(row) for row in points],
        closest_sample_mode="global",
        kernel_cls="RBF",
        kernel_kwargs=dict(kwargs),
        show_progress=False,
    )


def _normal(seed: int, rows: int, dim: int) -> npt.NDArray[np.float64]:
    return np.random.default_rng(seed).normal(size=(rows, dim))


def _duplicated() -> npt.NDArray[np.float64]:
    return np.repeat(_normal(1, 150, 5), 2, axis=0)


def _with_far_rows() -> npt.NDArray[np.float64]:
    base = _normal(2, 200, 4)
    far = np.full((10, 4), 50.0)
    far[::2] *= -1.0
    far += np.random.default_rng(3).normal(scale=0.01, size=far.shape)
    return np.vstack([base, far])


_DATASETS = {
    "normal": _normal(0, 300, 5),
    "duplicated": _duplicated(),
    "far_rows": _with_far_rows(),
    "single_feature": _normal(4, 150, 1),
}


def _kwargs_for(dim: int, kind: str) -> dict[str, Any]:
    if kind == "scalar":
        return {"kappa": 0.3, "eta": 2.5, "lam": 0.3 * 1e-4}
    return {"kappa": np.full(dim, 0.4), "eta": np.linspace(1.0, 3.0, dim)}


def _queries(points: npt.NDArray[np.float64], seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    half = _QUERY_COUNT // 2
    near = points[rng.integers(0, len(points), half)] + rng.normal(
        scale=0.05, size=(half, points.shape[1])
    )
    far = rng.normal(scale=5.0, size=(_QUERY_COUNT - half, points.shape[1]))
    return np.vstack([near, far])


def _same_param(first: object, second: object) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return bool(np.array_equal(first, second))


def _assert_dicts_equal(first: dict[str, object], second: dict[str, object]) -> None:
    assert first.keys() == second.keys()
    for key, value in first.items():
        other = second[key]
        if isinstance(value, np.ndarray) or isinstance(other, np.ndarray):
            assert np.array_equal(value, other), key
        else:
            assert value == other, key


@pytest.mark.parametrize("kind", ["scalar", "vector"])
@pytest.mark.parametrize("name", list(_DATASETS))
def test_array_path_matches_per_anchor_kernels(name: str, kind: str) -> None:
    points = _DATASETS[name]
    kwargs = _kwargs_for(points.shape[1], kind)
    array = Samples.from_array(points, kernel_kwargs=kwargs)
    legacy = _legacy(points, kwargs)
    queries = _queries(points, 10)

    array_arrays = array.batch_arrays()
    legacy_arrays = legacy.batch_arrays()
    assert np.array_equal(array_arrays[0], legacy_arrays[0])
    assert np.array_equal(array_arrays[1], legacy_arrays[1])
    assert array.kernels_materialized is False

    array_dual = array.affinity_dual(queries)
    legacy_dual = legacy.affinity_dual(queries)
    assert np.array_equal(array_dual[0], legacy_dual[0])
    assert np.array_equal(array_dual[1], legacy_dual[1])
    assert np.array_equal(array(queries), legacy(queries))
    assert len(array) == len(legacy)
    assert array.shape == legacy.shape
    assert array.kernels_materialized is False

    _ = array.samples
    assert array.kernels_materialized is True
    for i in range(len(legacy)):
        mine, theirs = array[i], legacy[i]
        assert np.array_equal(mine.x, theirs.x)
        assert mine.kernel is not None
        assert theirs.kernel is not None
        assert np.array_equal(mine.kernel.sigma, theirs.kernel.sigma)
        assert np.array_equal(mine.kernel.sigma_inv, theirs.kernel.sigma_inv)
        assert mine.kernel.sigma_is_diagonal
        assert _same_param(mine.kernel.kappa, theirs.kernel.kappa)
        assert _same_param(mine.kernel.eta, theirs.kernel.eta)
        assert _same_param(mine.kernel.lam, theirs.kernel.lam)
        assert mine.closest_sample is not None
        assert theirs.closest_sample is not None
        assert np.array_equal(mine.closest_sample[0].x, theirs.closest_sample[0].x)
    assert array == legacy
    assert hash(array) == hash(legacy)
    after = array.batch_arrays()
    assert np.array_equal(after[0], legacy_arrays[0])
    assert np.array_equal(after[1], legacy_arrays[1])


def test_serialization_matches(tmp_path: Path) -> None:
    points = _DATASETS["normal"]
    kwargs = _kwargs_for(points.shape[1], "scalar")
    array = Samples.from_array(points, kernel_kwargs=kwargs)
    legacy = _legacy(points, kwargs)

    array_file = tmp_path / "array.json"
    legacy_file = tmp_path / "legacy.json"
    to_json(array, array_file)
    to_json(legacy, legacy_file)

    assert array_file.read_bytes() == legacy_file.read_bytes()
    assert from_json(array_file) == legacy


def test_ood_adjustment_matches() -> None:
    points = _DATASETS["normal"]
    kwargs = _kwargs_for(points.shape[1], "scalar")
    rng = np.random.default_rng(5)
    picks = points[rng.choice(len(points), size=20, replace=False)]
    directions = rng.normal(size=picks.shape)
    directions *= 0.05 / np.linalg.norm(directions, axis=1, keepdims=True)
    ood = picks + directions
    queries = _queries(points, 11)

    array = Samples.from_array(points, kernel_kwargs=kwargs)
    legacy = _legacy(points, kwargs)
    array_result = array.enforce_ood_consistency(
        ood, xi=0.2, shrink_factor=0.9, show_progress=False
    )
    legacy_result = legacy.enforce_ood_consistency(
        ood, xi=0.2, shrink_factor=0.9, show_progress=False
    )

    _assert_dicts_equal(array_result, legacy_result)
    array_dual = array.affinity_dual(queries)
    legacy_dual = legacy.affinity_dual(queries)
    assert np.array_equal(array_dual[0], legacy_dual[0])
    assert np.array_equal(array_dual[1], legacy_dual[1])


def _calibrated_config() -> AutoSafeConfig:
    return AutoSafeConfig.from_mapping({
        "γ": 1.2,
        "s": 2.0,
        "λ_rel": 1e-4,
        "ζ": 0.5,
    })


def test_fit_keeps_kernels_unmaterialized() -> None:
    points = _normal(6, 500, 6)
    queries = _queries(points, 12)

    model = AutoSafeODD.fit(points, config=_calibrated_config())
    model.contains(queries)
    model.affinity(queries)
    survival = model.log_survival(queries)
    assert model.samples.kernels_materialized is False

    legacy = _legacy(model.normalized_in_distribution_data, model.samples.kernel_kwargs)
    assert model.normalizer is not None
    transformed = np.asarray(model.normalizer.transform(queries), dtype=np.float64)
    assert np.array_equal(survival, legacy.affinity_dual(transformed)[1])


def test_fit_per_dimension_mode_uses_objects() -> None:
    points = _normal(7, 100, 3)
    config = {
        "κ": 1.0,
        "η": 1.0,
        "λ_rel": 1e-4,
        "ζ": 0.5,
        "nearest_neighbor_mode": "per_dimension",
    }

    model = AutoSafeODD.fit(points, config=config)

    assert model.samples.kernels_materialized is True


def test_fit_with_ood_matches_legacy() -> None:
    points = _normal(8, 400, 3)
    rng = np.random.default_rng(9)
    picks = points[rng.choice(len(points), size=15, replace=False)]
    directions = rng.normal(size=picks.shape)
    directions *= 0.05 / np.linalg.norm(directions, axis=1, keepdims=True)
    ood = picks + directions
    queries = _queries(points, 13)
    config = {
        "κ": 1.0,
        "η": 1.0,
        "λ_rel": 1e-4,
        "ζ": 0.5,
        "ξ": 0.1,
        "c": 0.9,
        "ood_batch_jump": True,
        "normalize": False,
    }

    model = AutoSafeODD.fit(points, out_of_distribution_data=ood, config=config)
    ood_config = model.config.ood
    assert ood_config is not None
    ood_normalized = model.normalized_out_of_distribution_data
    assert ood_normalized is not None
    legacy = _legacy(model.normalized_in_distribution_data, model.samples.kernel_kwargs)
    legacy.enforce_ood_consistency(
        ood_normalized,
        xi=ood_config.max_affinity,
        shrink_factor=ood_config.covariance_shrink_factor,
        max_iterations=ood_config.max_iterations,
        refresh_interval=ood_config.refresh_interval,
        log_interval=ood_config.log_interval,
        batch_jump=ood_config.batch_jump,
        show_progress=False,
    )

    assert model.samples.kernels_materialized is True
    survival = model.log_survival(queries)
    assert np.array_equal(survival, legacy.affinity_dual(queries)[1])


def test_from_array_validates() -> None:
    points = _normal(14, 20, 3)

    with pytest.raises(ShapeMismatchError):
        Samples.from_array(points[:, 0])
    with pytest.raises(TooFewAnchorsError):
        Samples.from_array(points[:1])
    with pytest.raises(ShapeMismatchError):
        Samples.from_array(points, neighbor_indices=np.zeros(5, dtype=np.int64))
    with pytest.raises(UnknownOptionError):
        Samples.from_array(points, kernel_kwargs={"sigma": 1.0})
    with pytest.raises(ParameterOrderError):
        Samples.from_array(points, kernel_kwargs={"kappa": 0.5, "lam": 0.5})


def test_from_array_copies_input() -> None:
    points = _normal(15, 30, 2)
    original = points.copy()
    array = Samples.from_array(points, kernel_kwargs={"kappa": 0.5, "eta": 1.0})
    before = array.batch_arrays()[0].copy()

    points += 10.0

    assert np.array_equal(array.batch_arrays()[0], before)
    assert np.array_equal(array[0].x, original[0])


@pytest.mark.parametrize("with_lam", [True, False])
def test_rbf_from_diagonal_matches_update(*, with_lam: bool) -> None:
    rng = np.random.default_rng(16)
    x = rng.normal(size=4)
    neighbor = rng.normal(size=4)
    kwargs: dict[str, Any] = {"kappa": 0.6, "eta": np.linspace(0.5, 2.0, 4)}
    if with_lam:
        kwargs["lam"] = 6e-5

    reference = RBFKernel(x_i=x, **kwargs)
    reference.update(x_nn=neighbor, **kwargs)
    sigma, sigma_inv = rbf_diagonal_sigma(x[None], neighbor[None], **kwargs)
    built = RBFKernel.from_diagonal(x, sigma[0], sigma_inv[0], **kwargs)

    assert np.array_equal(built.sigma, reference.sigma)
    assert np.array_equal(built.sigma_inv, reference.sigma_inv)
    assert built.sigma_is_diagonal == reference.sigma_is_diagonal
    assert built == reference
