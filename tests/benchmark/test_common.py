# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the benchmark experiment helpers."""

from pathlib import Path

import numpy as np
import pytest

from experiments.benchmark.common import (
    boundary_error,
    build_odd,
    classification_metrics,
    confusion_at,
    curve_r2,
    default_thresholds,
    iou,
    level_set_mask,
    normalize_fit_apply,
    pr_curves,
    score_autosafe,
    write_dat,
)
from experiments.benchmark.synthetic_odds import get_odd


def _linear2d_anchors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return get_odd("linear2d").sample_id(n, rng)


def test_boundary_error_known_distance() -> None:
    """A single FP's boundary distance equals its gap to the nearest inside point."""
    x = np.array([[-2.0, 0.0], [-1.0, 0.0], [0.4, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = x[:, 0] < 0.0  # true region: left half-plane
    pred = x[:, 0] < 0.5  # over-approximates by the point at x = 0.4
    be = boundary_error(x, labels, pred)
    assert be["n_fp"] == 1
    assert be["n_fn"] == 0
    assert be["be_fp_mean"] == pytest.approx(1.4)  # (0.4, 0) -> (-1, 0)
    assert be["be_fn_mean"] == pytest.approx(0.0)
    # No misclassification -> all-zero report.
    clean = boundary_error(x, labels, labels)
    assert clean["n_fp"] == 0
    assert clean["be_fp_mean"] == pytest.approx(0.0)
    # Degenerate single-class grid -> zeros, no crash.
    assert boundary_error(x, np.ones(5, dtype=bool), pred)["n_fp"] == 0


def test_build_odd_both_modes_run() -> None:
    """Both modes build a working ODD with affinities in [0, 1]."""
    pts = _linear2d_anchors(40)
    val = np.array([[0.0, 0.0], [4.0, -4.0], [-4.0, 4.0]])
    for mode in ("fixed", "calibrated"):
        odd = build_odd(pts, mode=mode)
        alpha = score_autosafe(odd, val)
        assert alpha.shape == (3,)
        assert np.all(alpha >= 0.0)
        assert np.all(alpha <= 1.0)


def test_metrics_separable_is_perfect() -> None:
    """A trivially separable score gives AUPR ~ 1."""
    labels = np.array([True] * 50 + [False] * 50)
    scores = np.concatenate([np.ones(50), np.zeros(50)])
    m = classification_metrics(scores, labels)
    assert m["aupr"] > 0.99
    assert m["auroc"] > 0.99


def test_metrics_single_class_is_nan() -> None:
    """All-positive labels yield NaN metrics rather than raising."""
    m = classification_metrics(np.random.default_rng(0).random(20), np.ones(20, bool))
    assert np.isnan(m["aupr"])


def test_curve_r2_self_is_one() -> None:
    """R^2 of a non-constant curve against itself is 1.0."""
    c = np.linspace(0.1, 0.9, 20)
    assert curve_r2(c, c) == pytest.approx(1.0)


def test_pr_curves_monotone_endpoints() -> None:
    """At threshold 0 recall is 1; precision is in [0, 1] throughout."""
    pts = _linear2d_anchors(60)
    odd = build_odd(pts, mode="calibrated")
    odd_obj = get_odd("linear2d")
    rng = np.random.default_rng(3)
    val, labels = odd_obj.sample_validation(2000, rng)
    scores = score_autosafe(odd, val)
    prec, rec = pr_curves(scores, labels, default_thresholds(64))
    assert rec[0] == pytest.approx(1.0)
    assert np.all((prec >= 0.0) & (prec <= 1.0))


def test_normalize_fit_apply_range() -> None:
    """ID points map into [-1, 1] per dimension."""
    pts = _linear2d_anchors(50)
    _, (pts_n,) = normalize_fit_apply(pts)
    assert pts_n.min() >= -1.0 - 1e-9  # ty: ignore[invalid-argument-type]
    assert pts_n.max() <= 1.0 + 1e-9  # ty: ignore[invalid-argument-type]


def test_iou_bounds() -> None:
    """IoU is 1 for identical masks, 0 for disjoint, in-between otherwise."""
    m = np.array([True, True, False, False])
    assert iou(m, m) == pytest.approx(1.0)
    assert iou(np.ones(4, bool), np.zeros(4, bool)) == pytest.approx(0.0)
    assert iou(np.zeros(4, bool), np.zeros(4, bool)) == pytest.approx(1.0)
    a = np.array([True, True, False])
    b = np.array([True, False, False])
    assert iou(a, b) == pytest.approx(0.5)  # 1 shared / 2 union


def test_level_set_mask_zeta_zero_all_inside() -> None:
    """At zeta=0 every point is inside (alpha >= 0 always holds)."""
    pts = _linear2d_anchors(40)
    odd = build_odd(pts, mode="calibrated")
    val = np.array([[0.0, 0.0], [9.0, -9.0]])
    mask = level_set_mask(odd, val, 0.0)
    assert mask.dtype == np.bool_
    assert bool(np.all(mask))


def test_confusion_at_separable() -> None:
    """Fixed-threshold confusion is perfect on a separable score."""
    labels = np.array([True] * 50 + [False] * 50)
    scores = np.concatenate([np.ones(50), np.zeros(50)])
    c = confusion_at(scores, labels, 0.5)
    assert c["tp"] == 50
    assert c["fp"] == 0
    assert c["fn"] == 0
    assert c["precision"] == pytest.approx(1.0)
    assert c["recall"] == pytest.approx(1.0)
    assert c["iou"] == pytest.approx(1.0)
    assert c["fpr"] == pytest.approx(0.0)


def test_write_dat_roundtrip(tmp_path: Path) -> None:
    """write_dat produces a numpy-loadable table with a header."""
    p = tmp_path / "x.dat"
    write_dat(p, {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    loaded = np.loadtxt(p, skiprows=1)
    assert loaded.shape == (3, 2)
    assert p.read_text().splitlines()[0] == "a b"
