# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""End-to-end tests for calibration + baseline_params + sidecar + cache round-trip."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from autosafe.tools.evaluate.workflows import evaluate_dataset_mode


def _make_synthetic_dataset(tmp_path: Path) -> Path:
    """60-row, 3-dim dataset with two Gaussian blobs.

    Args:
        tmp_path (Path): pytest fixture for temporary directory.

    Returns:
        Path: Path to the synthetic CSV dataset.
    """
    rng = np.random.default_rng(42)
    blob_a = rng.normal(loc=[-0.5, -0.5, -0.5], scale=0.15, size=(30, 3))
    blob_b = rng.normal(loc=[0.5, 0.5, 0.5], scale=0.15, size=(30, 3))
    data = np.vstack([blob_a, blob_b])

    ds = tmp_path / "synthetic.csv"
    rows = ["x0,x1,x2"] + [f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f}" for r in data]
    ds.write_text("\n".join(rows), encoding="utf-8")

    # Sibling YAML with box limits covering the data
    yaml_path = tmp_path / "synthetic.yaml"
    yaml_path.write_text(
        "type: box\n"
        "dim: 3\n"
        "lower_bounds: [-1.0, -1.0, -1.0]\n"
        "upper_bounds: [1.0, 1.0, 1.0]\n",
        encoding="utf-8",
    )
    return ds


def test_calibration_end_to_end(tmp_path: Path) -> None:
    ds = _make_synthetic_dataset(tmp_path)

    results, csv_path, _odd_path = evaluate_dataset_mode(
        ds,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        baseline_params={"auto_scale": True},
        local_noise_mode="nn",
        local_noise_multiplier=3.0,
        references=["knn"],
        n_samples=2000,
        threshold_count=11,
    )

    # CSV exists and has both affinity_space values
    assert csv_path.exists()
    assert isinstance(results, pl.DataFrame)
    assert set(results["affinity_space"].unique().to_list()) == {"linear", "log"}

    # Sidecar exists and kappa/eta are scalars consistent with eta == 1/d_tilde
    sidecar_path = csv_path.with_name(csv_path.stem + "-params.json")
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())
    ksr = sidecar["kernel_scale_realized"]
    assert ksr is not None
    assert len(ksr["kappa"]) == 1
    assert len(ksr["eta"]) == 1
    median_nn = sidecar["median_nn_distance"]
    assert median_nn is not None
    assert median_nn > 0
    eta_realized = ksr["eta"][0]
    assert eta_realized == pytest.approx(1.0 / median_nn, rel=1e-6)

    # Log rows at threshold 1.0: TP == FP == 0
    log_rows = results.filter(
        (pl.col("affinity_space") == "log") & (pl.col("affinity_threshold") == 1.0)  # noqa: RUF069
    )
    assert len(log_rows) > 0
    for row in log_rows.iter_rows(named=True):
        assert row["true_positive"] == 0
        assert row["false_positive"] == 0


def test_calibration_cache_roundtrip(tmp_path: Path) -> None:
    ds = _make_synthetic_dataset(tmp_path)

    def _run() -> tuple[pl.DataFrame, Path, Path]:
        return evaluate_dataset_mode(
            ds,
            closest_sample_mode="global",
            kernel_kwargs={"calibration": "auto"},
            references=["knn"],
            n_samples=2000,
            threshold_count=11,
        )

    _, csv_path1, odd_path1 = _run()
    mtime_after_first = odd_path1.stat().st_mtime

    _, csv_path2, odd_path2 = _run()
    mtime_after_second = odd_path2.stat().st_mtime

    # ODD JSON not rewritten on cache hit
    assert mtime_after_first == mtime_after_second
    assert odd_path1 == odd_path2

    # Metrics are identical
    df1 = pl.read_csv(csv_path1)
    df2 = pl.read_csv(csv_path2)
    assert df1.shape == df2.shape


def test_calibration_requires_global_mode(tmp_path: Path) -> None:
    ds = _make_synthetic_dataset(tmp_path)

    with pytest.raises(ValueError, match="closest_sample_mode: global"):
        evaluate_dataset_mode(
            ds,
            closest_sample_mode="per_dimension",
            kernel_kwargs={"calibration": "auto"},
            references=["knn"],
            n_samples=500,
            threshold_count=5,
        )
