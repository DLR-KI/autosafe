# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Coverage tests for autosafe.tools.monte_carlo.comparison package."""

import math
import pathlib
import tempfile

import numpy as np
import pytest

from autosafe.tools.monte_carlo.comparison import (
    ComparisonExperimentConfig,
    _create_test_grid,
    evaluate_comparison,
)


def test_comparison_experiment_config_defaults():
    """Test ComparisonExperimentConfig default values."""
    config = ComparisonExperimentConfig()
    assert config.methods is None
    assert config.knn_k == 3
    assert math.isclose(config.knn_gamma, 0.5)
    assert config.kmeans_clusters == 3
    assert math.isclose(config.density_gamma, 0.01)
    assert config.export_path is None


def test_comparison_experiment_config_custom():
    """Test ComparisonExperimentConfig with custom values."""
    with tempfile.NamedTemporaryFile() as file:
        config = ComparisonExperimentConfig(
            methods=["hull_single"],
            knn_k=5,
            knn_gamma=1.0,
            kmeans_clusters=10,
            density_gamma=0.1,
            export_path=str(file.name),
        )
        assert config.methods == ["hull_single"]
        assert config.knn_k == 5
        assert math.isclose(config.knn_gamma, 1.0)
        assert config.kmeans_clusters == 10
        assert math.isclose(config.density_gamma, 0.1)
        assert config.export_path == str(file.name)


def test_create_test_grid():
    """Test _create_test_grid generates expected grid."""
    ref_points = np.array([[0.0, 1.0], [0.0, 1.0]])  # (n_features=2, n_samples=2)
    grid = _create_test_grid(ref_points, resolution=10)
    # Shape is (n_features, n_test_samples) = (2, 10)
    assert grid.shape[0] == 2  # 2D features
    assert grid.shape[1] == 10  # resolution points


def test_evaluate_comparison(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test evaluate_comparison runs without errors."""
    # Create a minimal CSV dataset
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("x,y\n0.0,0.0\n1.0,1.0\n0.5,0.5\n")

    # Mock at the function level in the monte_carlo.comparison module
    from autosafe.tools.monte_carlo import comparison as mc_comparison

    def mock_evaluate_methods(**_kwargs: dict) -> dict:
        return {
            "dataset": "mocked",
            "reference_points": 2,
            "test_points": 10,
            "comparison_methods": {},
            "summary": {
                "most_conservative": "mocked",
                "best_coverage": "mocked",
                "method_count": 0,
            },
        }

    monkeypatch.setattr(
        mc_comparison, "_evaluate_comparison_methods", mock_evaluate_methods
    )

    config = ComparisonExperimentConfig(methods=["hull_single"])
    result = evaluate_comparison(dataset_path, config)
    assert result["summary"]["most_conservative"] == "mocked"
