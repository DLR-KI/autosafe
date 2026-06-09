# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Coverage tests for autosafe.tools.monte_carlo.evaluate package."""

import pathlib

import pytest

from autosafe.tools.monte_carlo.evaluate import evaluate


def test_evaluate_raises_file_not_found():
    """Test evaluate raises FileNotFoundError when no files provided."""
    with pytest.raises(FileNotFoundError, match="No file"):
        evaluate(file=None)


def test_evaluate(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Test evaluate processes files."""
    # Create a test JSON file
    json_file = tmp_path / "test.json"
    json_file.write_text("{}")

    def mock_process_files(path: pathlib.Path) -> list[pathlib.Path]:  # noqa: ARG001
        return [json_file]

    def mock_evaluate_mc_results(files: list[pathlib.Path], **kwargs: dict) -> None:
        pass

    monkeypatch.setattr(
        "autosafe.tools.monte_carlo.evaluate.process_files", mock_process_files
    )
    monkeypatch.setattr(
        "autosafe.tools.monte_carlo.evaluate.evaluate_monte_carlo_results",
        mock_evaluate_mc_results,
    )

    # Should not raise
    evaluate(file=[str(json_file)])
