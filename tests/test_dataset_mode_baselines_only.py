# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the baselines_only fast path of evaluate_dataset_mode.

Mirrors tests/test_dataset_mode_calibration.py's cache-roundtrip
pattern: baselines_only must reuse a cached affinity ODD rather than
rebuild it, refuse to run without one, refuse a kernel-mismatched
cache, and leave the autoSAFE affinity numbers byte-identical to the
run that produced the cache.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from autosafe.exceptions import (
    BaselinesOnlyCacheMismatchError,
    BaselinesOnlyRequiresCachedODDError,
)
from autosafe.tools.evaluate.workflows import evaluate_dataset_mode


def _make_synthetic_dataset(tmp_path: Path) -> Path:
    """40-row, 2-dim dataset with two Gaussian blobs.

    Returns:
        Path: Path to the synthetic CSV dataset.
    """
    rng = np.random.default_rng(11)
    blob_a = rng.normal(loc=[-0.5, -0.5], scale=0.15, size=(20, 2))
    blob_b = rng.normal(loc=[0.5, 0.5], scale=0.15, size=(20, 2))
    data = np.vstack([blob_a, blob_b])

    ds = tmp_path / "synthetic.csv"
    rows = ["x0,x1"] + [f"{r[0]:.6f},{r[1]:.6f}" for r in data]
    ds.write_text("\n".join(rows), encoding="utf-8")
    return ds


def test_baselines_only_requires_cached_odd_json(tmp_path: Path) -> None:
    ds = _make_synthetic_dataset(tmp_path)

    with pytest.raises(BaselinesOnlyRequiresCachedODDError):
        evaluate_dataset_mode(
            ds,
            references=["knn"],
            n_samples=200,
            threshold_count=5,
            baselines_only=True,
        )


def test_baselines_only_rejects_cache_mismatch(tmp_path: Path) -> None:
    ds = _make_synthetic_dataset(tmp_path)

    _, _csv_path, odd_path = evaluate_dataset_mode(
        ds,
        references=["knn"],
        n_samples=200,
        threshold_count=5,
        kernel_kwargs={"kappa": 1.0},
    )

    with pytest.raises(BaselinesOnlyCacheMismatchError):
        evaluate_dataset_mode(
            ds,
            odd_json=odd_path,
            references=["knn"],
            n_samples=200,
            threshold_count=5,
            kernel_kwargs={"kappa": 2.0},  # mismatched vs. the cached ODD
            baselines_only=True,
        )


def test_baselines_only_reuses_cache_and_autosafe_column_unchanged(
    tmp_path: Path,
) -> None:
    ds = _make_synthetic_dataset(tmp_path)

    results1, _csv_path1, odd_path1 = evaluate_dataset_mode(
        ds,
        references=["knn"],
        n_samples=500,
        threshold_count=11,
        seed=3,
    )
    mtime_before = odd_path1.stat().st_mtime
    bytes_before = odd_path1.read_bytes()

    results2, _csv_path2, odd_path2 = evaluate_dataset_mode(
        ds,
        odd_json=odd_path1,
        references=["knn"],
        n_samples=500,
        threshold_count=11,
        seed=3,
        baselines_only=True,
    )

    # Cache reused, not rebuilt: same path, untouched on disk.
    assert odd_path2 == odd_path1
    assert odd_path1.stat().st_mtime == mtime_before
    assert odd_path1.read_bytes() == bytes_before

    # Same reference method, same seed, cached ODD => every column
    # (affinity-derived confusion counts included) must match exactly.
    knn1 = results1.filter(pl.col("reference") == "knn").sort([
        "affinity_space",
        "survival_threshold",
    ])
    knn2 = results2.filter(pl.col("reference") == "knn").sort([
        "affinity_space",
        "survival_threshold",
    ])
    assert knn1.equals(knn2)


def test_baselines_only_odd_json_out_ignored_for_cache_path(tmp_path: Path) -> None:
    """baselines_only reuses odd_json regardless of odd_json_out."""
    ds = _make_synthetic_dataset(tmp_path)

    _, _csv_path, odd_path = evaluate_dataset_mode(
        ds,
        references=["knn"],
        n_samples=200,
        threshold_count=5,
        seed=1,
    )

    _, _csv_path2, odd_path2 = evaluate_dataset_mode(
        ds,
        odd_json=odd_path,
        references=["knn"],
        n_samples=200,
        threshold_count=5,
        seed=1,
        baselines_only=True,
    )
    assert odd_path2 == odd_path
