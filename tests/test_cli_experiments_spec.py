# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for run-spec overrides in experiments CLI."""

from pathlib import Path
from typing import Any, cast

import pytest

from autosafe.cli.experiments import glob_run_dataset, glob_run_mc_sample


def test_glob_run_dataset_uses_evaluation_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use `evaluation_samples` from spec for dataset workflow."""
    captured: dict[str, Any] = {}

    def _fake_eval(**kwargs: object) -> tuple[object, Path, Path]:
        captured["n_samples"] = cast("int", kwargs["n_samples"])
        return object(), Path("out.csv"), Path("odd.json")

    monkeypatch.setattr(
        "autosafe.cli.experiments.evaluate_dataset_mode",
        _fake_eval,
    )

    _ = glob_run_dataset({
        "mode": "dataset",
        "dataset_path": "data/iris.csv",
        "evaluation_samples": 12345,
    })

    assert captured["n_samples"] == 12345


def test_glob_run_mc_sample_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec-level MC settings should override config-file values."""
    cfg = {
        "dim": 2,
        "odd_type": "box",
        "odd_lower_limits": -1.0,
        "odd_upper_limits": 1.0,
        "box_lower_limits": -2.0,
        "box_upper_limits": 2.0,
        "odd_anchors": 5,
        "kernel_config": {"type": "RBF", "params": {}},
        "samples": 100,
        "filename": Path("sampling_results.json"),
        "custom_odd_config": None,
    }

    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        "autosafe.cli.experiments.create_config",
        lambda **_kwargs: cfg,
    )

    def _fake_run(config: dict[str, Any]) -> None:
        seen.update(config)

    monkeypatch.setattr("autosafe.cli.experiments.run_single_sampling", _fake_run)

    result = glob_run_mc_sample({
        "mode": "mc-sample",
        "config_file": "experiments/dim_2d/sampling_config.yaml",
        "samples": 2000,
        "kernel_type": "Laplacian",
        "kernel_kwargs": {"alpha": 0.7},
    })

    assert seen["samples"] == 2000
    assert seen["kernel_config"]["type"] == "Laplacian"
    assert seen["kernel_config"]["params"] == {"alpha": 0.7}
    assert result["mode"] == "mc-sample"


def test_glob_run_mc_sample_rejects_invalid_kernel_kwargs() -> None:
    """`kernel_kwargs` must be a mapping in run-spec items."""
    with pytest.raises(ValueError, match="kernel_kwargs must be a mapping"):
        _ = glob_run_mc_sample({
            "mode": "mc-sample",
            "config_file": "experiments/dim_2d/sampling_config.yaml",
            "kernel_kwargs": "not-a-dict",
        })
