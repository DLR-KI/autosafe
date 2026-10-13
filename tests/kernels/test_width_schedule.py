# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from autosafe.exceptions import (
    NonFiniteParameterError,
    NonPositiveParameterError,
    ParameterTypeError,
    TooFewAnchorsError,
    UnknownOptionError,
)
from autosafe.kernels.rbf import (
    WIDTH_SCHEDULE_SQRT_LOG_N,
    calibrate_rbf_scale_d_tilde,
    calibrate_rbf_scale_median,
    resolve_width_multiple,
)
from autosafe.odd import AutoSafeConfig, AutoSafeODD, CalibratedRBFConfig
from autosafe.tools.evaluate.dataset.odd_cache import (
    _calibration_width_spec,
    _resolve_kernel_calibration,
    _width_filename_tag,
)


@pytest.mark.parametrize("n_anchors", [2, 10, 1000, 622_110])
def test_schedule_is_sqrt_log_n(n_anchors: int) -> None:
    s = resolve_width_multiple(WIDTH_SCHEDULE_SQRT_LOG_N, n_anchors)
    assert s == pytest.approx(np.sqrt(np.log(n_anchors)), rel=0, abs=0)


def test_fixed_width_is_returned_unchanged() -> None:
    assert resolve_width_multiple(3.0, 10) == pytest.approx(3.0, rel=0, abs=0)
    assert resolve_width_multiple(2, 10) == pytest.approx(2.0, rel=0, abs=0)


@pytest.mark.parametrize(
    ("s", "n_anchors", "error", "match"),
    [
        ("sqrt_n", 10, UnknownOptionError, "unknown s 'sqrt_n'"),
        (WIDTH_SCHEDULE_SQRT_LOG_N, 1, TooFewAnchorsError, "at least 2 anchors"),
        (0.0, 10, NonPositiveParameterError, "s must be greater than 0"),
        (float("inf"), 10, NonFiniteParameterError, "s must contain only finite"),
        (True, 10, ParameterTypeError, "s must be of type"),
        (None, 10, ParameterTypeError, "s must be of type"),
    ],
)
def test_invalid_width_multiple_raises(
    s: object, n_anchors: int, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        resolve_width_multiple(s, n_anchors)  # ty: ignore[invalid-argument-type]


def test_calibrators_default_to_the_schedule() -> None:
    rng = np.random.default_rng(0)
    d_l2 = rng.exponential(scale=0.1, size=500)
    s_n = np.sqrt(np.log(500))
    kappa, eta = calibrate_rbf_scale_d_tilde(d_l2)
    kappa_fixed, eta_fixed = calibrate_rbf_scale_d_tilde(d_l2, s=float(s_n))
    assert kappa == kappa_fixed
    assert eta == eta_fixed
    d_nn = rng.exponential(scale=0.1, size=(500, 3))
    kappa_m, _ = calibrate_rbf_scale_median(d_nn)
    kappa_m_fixed, _ = calibrate_rbf_scale_median(d_nn, s=float(s_n))
    np.testing.assert_array_equal(kappa_m, kappa_m_fixed)


def test_config_defaults_to_schedule_and_accepts_numbers() -> None:
    assert CalibratedRBFConfig().width_in_median_gaps == WIDTH_SCHEDULE_SQRT_LOG_N
    fixed = CalibratedRBFConfig(width_in_median_gaps=3).width_in_median_gaps
    assert fixed == pytest.approx(3.0, rel=0, abs=0)
    parsed = AutoSafeConfig.from_mapping({"s": " SQRT_LOG_N ", "zeta": 0.5})
    assert isinstance(parsed.kernel, CalibratedRBFConfig)
    assert parsed.kernel.width_in_median_gaps == WIDTH_SCHEDULE_SQRT_LOG_N
    assert parsed.to_mapping()["kernel"] == {
        "mode": "calibrated",
        "decay_per_median_gap": 1.0,
        "width_in_median_gaps": WIDTH_SCHEDULE_SQRT_LOG_N,
        "relative_variance_floor": parsed.kernel.relative_variance_floor,
    }
    with pytest.raises(UnknownOptionError, match="unknown width_in_median_gaps "):
        CalibratedRBFConfig(width_in_median_gaps="sqrt_n")  # ty: ignore[invalid-argument-type]


def test_resolved_config_records_the_realized_width() -> None:
    rng = np.random.default_rng(1)
    anchors = rng.uniform(size=(200, 2))
    odd = AutoSafeODD.fit(anchors, config={"zeta": 0.5})
    assert odd.resolved_config.kernel.width_in_median_gaps == pytest.approx(
        np.sqrt(np.log(200))
    )


def test_dataset_calibration_resolves_and_rejects_widths() -> None:
    rng = np.random.default_rng(2)
    pts = rng.uniform(size=(100, 2))
    idx = np.array([int(np.argsort(np.linalg.norm(pts - p, axis=1))[1]) for p in pts])
    d_l2 = np.linalg.norm(pts[idx] - pts, axis=1)
    default = _resolve_kernel_calibration(
        {"calibration": "auto"}, pts, idx, closest_sample_mode="global"
    )
    expected_kappa, _ = calibrate_rbf_scale_d_tilde(d_l2, s=float(np.sqrt(np.log(100))))
    assert default["kappa"] == pytest.approx(expected_kappa, rel=1e-12)
    fixed = _resolve_kernel_calibration(
        {"calibration": "auto", "calibration_s": 3.0},
        pts,
        idx,
        closest_sample_mode="global",
    )
    assert fixed["kappa"] == pytest.approx(
        calibrate_rbf_scale_d_tilde(d_l2, s=3.0)[0], rel=1e-12
    )
    with pytest.raises(UnknownOptionError, match="unknown s "):
        _resolve_kernel_calibration(
            {"calibration": "auto", "calibration_s": "three"},
            pts,
            idx,
            closest_sample_mode="global",
        )


@pytest.mark.parametrize(
    ("kernel_kwargs", "spec", "tag"),
    [
        ({}, None, ""),
        ({"calibration": "manual", "calibration_s": 3.0}, None, ""),
        ({"calibration": "auto"}, WIDTH_SCHEDULE_SQRT_LOG_N, "-width-sqrt-log-n"),
        (
            {"calibration": "auto", "calibration_s": "sqrt_log_n"},
            "sqrt_log_n",
            "-width-sqrt-log-n",
        ),
        ({"calibration": "auto", "calibration_s": 3.0}, 3.0, "-width-3"),
        ({"calibration": "auto", "calibration_s": 2.5}, 2.5, "-width-2.5"),
    ],
)
def test_width_filename_tag(
    kernel_kwargs: dict[str, object], spec: object, tag: str
) -> None:
    assert _calibration_width_spec(kernel_kwargs) == spec
    assert _width_filename_tag(kernel_kwargs) == tag
