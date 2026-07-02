# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""End-to-end tests for OOD consistency in the dataset-mode workflow."""

import json
from pathlib import Path

import numpy as np

from autosafe.tools.evaluate.workflows import evaluate_dataset_mode


def _make_dataset_with_ood(tmp_path: Path) -> tuple[Path, Path]:
    """Build a 3D two-blob dataset, sibling YAML, and a 3-row OOD CSV.

    Args:
        tmp_path (Path): pytest temporary directory.

    Returns:
        tuple[Path, Path]: (dataset CSV path, OOD CSV path).
    """
    rng = np.random.default_rng(42)
    blob_a = rng.normal(loc=[-0.5, -0.5, -0.5], scale=0.12, size=(30, 3))
    blob_b = rng.normal(loc=[0.5, 0.5, 0.5], scale=0.12, size=(30, 3))
    data = np.vstack([blob_a, blob_b])

    ds = tmp_path / "synthetic.csv"
    ds.write_text(
        "\n".join(["x0,x1,x2"] + [f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f}" for r in data]),
        encoding="utf-8",
    )
    (tmp_path / "synthetic.yaml").write_text(
        "type: box\n"
        "dim: 3\n"
        "lower_bounds: [-1.0, -1.0, -1.0]\n"
        "upper_bounds: [1.0, 1.0, 1.0]\n",
        encoding="utf-8",
    )

    # OOD points sitting near the blobs -> high initial affinity, so the
    # adjustment loop is exercised.
    ood = tmp_path / "ood.csv"
    ood.write_text(
        "x0,x1,x2\n0.5,0.5,0.5\n-0.5,-0.5,-0.5\n0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    return ds, ood


def _sidecar(csv_path: Path) -> dict:
    return json.loads(csv_path.with_name(csv_path.stem + "-params.json").read_text())


def test_ood_adjustment_triggers_and_caches(tmp_path: Path) -> None:
    """OOD run adjusts the ODD, satisfies xi, and tags the cache path."""
    ds, ood = _make_dataset_with_ood(tmp_path)
    xi = 0.1

    _, csv_path, odd_path = evaluate_dataset_mode(
        ds,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        references=["knn"],
        n_samples=2000,
        threshold_count=11,
        ood_path=ood,
        ood_xi=xi,
        ood_shrink_factor=0.9,
    )

    sc = _sidecar(csv_path)
    assert sc["ood_path"] is not None
    assert sc["ood_xi"] == xi
    assert int(sc["ood_iterations"]) >= 1
    assert float(sc["ood_max_affinity_final"]) <= xi + 1e-12
    assert int(sc["ood_adjusted_kernel_count"]) >= 1
    assert "-ood" in odd_path.name


def test_ood_cache_hit_is_idempotent(tmp_path: Path) -> None:
    """A second identical OOD run is a cache hit: no rewrite, 0 iterations."""
    ds, ood = _make_dataset_with_ood(tmp_path)

    def _run() -> tuple:
        return evaluate_dataset_mode(
            ds,
            closest_sample_mode="global",
            kernel_kwargs={"calibration": "auto"},
            references=["knn"],
            n_samples=1500,
            threshold_count=7,
            ood_path=ood,
            ood_xi=0.1,
        )

    _, _csv1, odd1 = _run()
    mtime1 = odd1.stat().st_mtime
    _, csv2, odd2 = _run()

    assert odd1 == odd2
    assert odd1.stat().st_mtime == mtime1  # adjusted ODD not rewritten
    assert int(_sidecar(csv2)["ood_iterations"]) == 0


def test_no_ood_keys_leave_base_cache_untagged(tmp_path: Path) -> None:
    """Without OOD keys, the ODD path is untagged and sidecar OOD is None."""
    ds, _ = _make_dataset_with_ood(tmp_path)

    _, csv_path, odd_path = evaluate_dataset_mode(
        ds,
        closest_sample_mode="global",
        kernel_kwargs={"calibration": "auto"},
        references=["knn"],
        n_samples=1500,
        threshold_count=7,
    )

    assert "-ood" not in odd_path.name
    assert _sidecar(csv_path)["ood_iterations"] is None
