# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Coverage tests for autosafe.cli.experiments module."""

import pathlib
import tempfile
from collections.abc import Callable
from typing import NoReturn

import pytest

from autosafe.cli.experiments import (
    EXP_APP,
    SETTINGS,
    _configure_experiments_cli,
    get_app,
    glob_run_dataset,
    glob_run_mc_results,
    glob_run_mc_sample,
    list_datasets,
    run_benchmark_command,
    run_evaluation_command,
    run_pipeline_command,
    run_spec,
    show_info,
)


def test_get_app():
    """Test get_app returns the EXP_APP instance."""
    app = get_app()
    assert app is EXP_APP


def test_show_info(capsys: pytest.CaptureFixture[str]):
    """Test show_info command displays experiment types."""
    show_info()
    captured = capsys.readouterr()
    assert "Available experiment types" in captured.out
    assert "evaluate" in captured.out


def test_list_datasets(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    """Test list_datasets command."""
    # Create some test files
    (tmp_path / "test.csv").write_text("x\n1\n")
    (tmp_path / "test.json").write_text("{}")
    (tmp_path / "ignored.txt").write_text("ignore")

    list_datasets(str(tmp_path))
    captured = capsys.readouterr()
    assert "test.csv" in captured.out
    assert "test.json" in captured.out


def test_list_datasets_empty_dir(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
):
    """Test list_datasets with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    list_datasets(str(empty_dir))
    captured = capsys.readouterr()
    assert "No datasets found" in captured.out


def test_list_datasets_not_found():
    """Test list_datasets raises error for non-existent directory."""
    with pytest.raises(Exception):  # typer.BadParameter  # noqa: B017, PT011
        list_datasets("/nonexistent/path/xyz123")


def test_run_evaluation_command(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Test run_evaluation_command."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("x,y\n0,0\n1,1\n")

    # Mock ExperimentManager
    class MockResult:
        experiment_id = "test-123"
        processing_time = 1.5
        export_paths = None

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_evaluation_experiment(
            dataset_config: object,  # noqa: ARG004
            export_dir: object = None,  # noqa: ARG004
        ) -> MockResult:
            return MockResult()

    monkeypatch.setattr("autosafe.cli.experiments.ExperimentManager", MockManager)

    run_evaluation_command(str(dataset_path), n_samples=100)
    captured = capsys.readouterr()
    assert "completed" in captured.out


def test_run_evaluation_command_verbose(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Test run_evaluation_command with verbose output."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("x,y\n0,0\n1,1\n")

    class MockResult:
        experiment_id = "test-123"
        processing_time = 1.5
        export_paths = None

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_evaluation_experiment(
            dataset_config: object,  # noqa: ARG004
            export_dir: object = None,  # noqa: ARG004
        ) -> MockResult:
            return MockResult()

    monkeypatch.setattr("autosafe.cli.experiments.ExperimentManager", MockManager)

    # Set verbose mode
    from autosafe.cli.experiments import SETTINGS

    SETTINGS["verbose"] = True

    try:
        run_evaluation_command(str(dataset_path), n_samples=100)
        captured = capsys.readouterr()
        assert "Starting evaluation" in captured.out or "completed" in captured.out
    finally:
        SETTINGS["verbose"] = False


def test_run_benchmark_command(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Test run_benchmark_command."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("x,y\n0,0\n1,1\n")

    class MockResult:
        experiment_id = "test-123"
        processing_time = 1.5
        export_paths = None

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_benchmark_experiment(
            dataset_config: object,  # noqa: ARG004
            export_dir: object = None,  # noqa: ARG004
        ) -> MockResult:
            return MockResult()

    monkeypatch.setattr("autosafe.cli.experiments.ExperimentManager", MockManager)

    run_benchmark_command(str(dataset_path), n_samples=100)
    captured = capsys.readouterr()
    assert "completed" in captured.out


def test_run_benchmark_command_verbose(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Test run_benchmark_command with verbose output."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("x,y\n0,0\n1,1\n")

    class MockResult:
        experiment_id = "test-123"
        processing_time = 1.5
        export_paths = None

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_benchmark_experiment(
            dataset_config: object,  # noqa: ARG004
            export_dir: object = None,  # noqa: ARG004
        ) -> MockResult:
            return MockResult()

    monkeypatch.setattr("autosafe.cli.experiments.ExperimentManager", MockManager)

    from autosafe.cli.experiments import SETTINGS

    SETTINGS["verbose"] = True

    try:
        run_benchmark_command(str(dataset_path), n_samples=100)
        captured = capsys.readouterr()
        assert "Starting benchmark" in captured.out or "completed" in captured.out
    finally:
        SETTINGS["verbose"] = False


def test_run_pipeline_command(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Test run_pipeline_command."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("x,y\n0,0\n1,1\n")

    class MockResult:
        experiment_id = "test-123"
        processing_time = 1.5

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_benchmark_experiment(
            dataset_config: object,  # noqa: ARG004
            export_dir: object = None,  # noqa: ARG004
        ) -> MockResult:
            return MockResult()

    monkeypatch.setattr("autosafe.cli.experiments.ExperimentManager", MockManager)

    run_pipeline_command([str(dataset_path)], n_samples=100)
    captured = capsys.readouterr()
    assert "Processed" in captured.out


def test_run_pipeline_command_verbose(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Test run_pipeline_command with verbose output."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("x,y\n0,0\n1,1\n")

    class MockResult:
        experiment_id = "test-123"
        processing_time = 1.5

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_benchmark_experiment(
            dataset_config: object,  # noqa: ARG004
        ) -> MockResult:
            return MockResult()

    monkeypatch.setattr("autosafe.cli.experiments.ExperimentManager", MockManager)

    from autosafe.cli.experiments import SETTINGS

    SETTINGS["verbose"] = True

    try:
        run_pipeline_command([str(dataset_path)], n_samples=100)
        captured = capsys.readouterr()
        assert "Running pipeline" in captured.out or "Processed" in captured.out
    finally:
        SETTINGS["verbose"] = False


def test_run_pipeline_command_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Test run_pipeline_command handles errors."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("x,y\n0,0\n1,1\n")

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_benchmark_experiment(
            dataset_config: dict,  # noqa: ARG004
        ) -> NoReturn:
            raise RuntimeError("Test error")

    monkeypatch.setattr(
        "autosafe.cli.experiments.ExperimentManager",
        MockManager,
    )

    run_pipeline_command([str(dataset_path)], n_samples=100)
    captured = capsys.readouterr()
    # Error messages go to stderr via typer.echo(..., err=True)
    assert "Error" in captured.err


def test_glob_run_mc_sample_missing_config():
    """Test glob_run_mc_sample raises error without config_file."""
    with pytest.raises(ValueError, match="config_file"):
        glob_run_mc_sample({"mode": "mc-sample"})


def test_glob_run_mc_sample_invalid_kwargs(tmp_path: pathlib.Path):
    """Test glob_run_mc_sample with invalid kernel_kwargs."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("n_samples: 100\nfilename: test.json\n")

    with pytest.raises(ValueError, match="kernel_kwargs must be a mapping"):
        glob_run_mc_sample({
            "config_file": str(config_file),
            "kernel_kwargs": "invalid",
        })


def test_glob_run_mc_sample_invalid_kernel_type(tmp_path: pathlib.Path):
    """Test glob_run_mc_sample with invalid kernel_type."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("n_samples: 100\nfilename: test.json\n")

    with pytest.raises(ValueError, match="kernel_type must be one of"):
        glob_run_mc_sample({"config_file": str(config_file), "kernel_type": "Invalid"})


def test_glob_run_mc_sample(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test glob_run_mc_sample."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("n_samples: 100\nfilename: test.json\n")

    def mock_run_single_sampling(config: dict) -> None:
        pass

    monkeypatch.setattr(
        "autosafe.cli.experiments.run_single_sampling", mock_run_single_sampling
    )

    result = glob_run_mc_sample({
        "config_file": str(config_file),
        "n_samples": 50,
    })
    assert result["mode"] == "mc-sample"


def test_glob_run_dataset_invalid_comparison_methods(tmp_path: pathlib.Path):
    """Test glob_run_dataset with invalid comparison_methods."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("x,y\n0,0\n")

    with pytest.raises(ValueError, match="comparison_methods must be a list"):
        glob_run_dataset({
            "dataset_path": str(dataset_path),
            "comparison_methods": "invalid",
        })


def test_glob_run_dataset_invalid_kernel_kwargs(tmp_path: pathlib.Path):
    """Test glob_run_dataset with invalid kernel_kwargs."""
    dataset_path = tmp_path / "test.csv"
    dataset_path.write_text("x,y\n0,0\n")

    with pytest.raises(ValueError, match="kernel_kwargs must be a mapping"):
        glob_run_dataset({
            "dataset_path": str(dataset_path),
            "kernel_kwargs": "invalid",
        })


def test_glob_run_dataset(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test glob_run_dataset."""
    with (
        tempfile.NamedTemporaryFile() as output_file,
        tempfile.NamedTemporaryFile() as odd_file,
    ):
        dataset_path = tmp_path / "test.csv"
        dataset_path.write_text("x,y\n0,0\n1,1\n")

        def mock_evaluate_dataset_mode(
            **kwargs: dict,  # noqa: ARG001
        ) -> tuple[object, pathlib.Path, pathlib.Path]:
            return (
                None,
                pathlib.Path(output_file.name),
                pathlib.Path(odd_file.name),
            )

        monkeypatch.setattr(
            "autosafe.cli.experiments.evaluate_dataset_mode",
            mock_evaluate_dataset_mode,
        )

        result = glob_run_dataset({
            "dataset_path": str(dataset_path),
            "comparison_methods": ["hull_single"],
        })
        assert result["mode"] == "dataset"


def test_glob_run_dataset_global_mode(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test glob_run_dataset with global closest_sample_mode."""
    with (
        tempfile.NamedTemporaryFile() as output_file,
        tempfile.NamedTemporaryFile() as odd_file,
    ):
        dataset_path = tmp_path / "test.csv"
        dataset_path.write_text("x,y\n0,0\n1,1\n")

        def mock_evaluate_dataset_mode(
            **kwargs: dict,  # noqa: ARG001
        ) -> tuple[object, pathlib.Path, pathlib.Path]:
            return (
                None,
                pathlib.Path(output_file.name),
                pathlib.Path(odd_file.name),
            )

        monkeypatch.setattr(
            "autosafe.cli.experiments.evaluate_dataset_mode",
            mock_evaluate_dataset_mode,
        )

        result = glob_run_dataset({
            "dataset_path": str(dataset_path),
            "closest_sample_mode": "global",
        })
        assert result["mode"] == "dataset"


def test_glob_run_dataset_laplacian_kernel(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test glob_run_dataset with Laplacian kernel_type."""
    with (
        tempfile.NamedTemporaryFile() as output_file,
        tempfile.NamedTemporaryFile() as odd_file,
    ):
        dataset_path = tmp_path / "test.csv"
        dataset_path.write_text("x,y\n0,0\n1,1\n")

        def mock_evaluate_dataset_mode(
            **kwargs: dict,  # noqa: ARG001
        ) -> tuple[object, pathlib.Path, pathlib.Path]:
            return (
                None,
                pathlib.Path(output_file.name),
                pathlib.Path(odd_file.name),
            )

        monkeypatch.setattr(
            "autosafe.cli.experiments.evaluate_dataset_mode",
            mock_evaluate_dataset_mode,
        )

        result = glob_run_dataset({
            "dataset_path": str(dataset_path),
            "kernel_type": "Laplacian",
        })
        assert result["mode"] == "dataset"


def test_glob_run_mc_results_empty_inputs():
    """Test glob_run_mc_results with empty inputs."""
    with pytest.raises(ValueError, match="inputs"):
        glob_run_mc_results({"inputs": []})


def test_glob_run_mc_results_invalid_inputs():
    """Test glob_run_mc_results with non-list inputs."""
    with pytest.raises(ValueError, match="inputs"):
        glob_run_mc_results({"inputs": "not a list"})


def test_glob_run_mc_results(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test glob_run_mc_results."""
    import polars as pl

    mock_df = pl.DataFrame({"col": [1, 2, 3]})

    def mock_collect_files(inputs: list[str]) -> list[pathlib.Path]:
        return [pathlib.Path(f) for f in inputs]

    def mock_evaluate_mc_results(
        files: list[pathlib.Path],  # noqa: ARG001
        **kwargs: dict,  # noqa: ARG001
    ) -> pl.DataFrame:
        return mock_df

    monkeypatch.setattr(
        "autosafe.cli.experiments.collect_monte_carlo_files", mock_collect_files
    )
    monkeypatch.setattr(
        "autosafe.cli.experiments.evaluate_monte_carlo_results",
        mock_evaluate_mc_results,
    )

    result = glob_run_mc_results({
        "inputs": [str(tmp_path / "test.json")],
    })
    assert result["mode"] == "mc-results"
    assert result["rows"] == 3


def test_run_spec_file_not_found():
    """Test run_spec raises error for non-existent spec file."""
    with pytest.raises(Exception):  # typer.BadParameter  # noqa: B017, PT011
        run_spec("/nonexistent/spec.yaml")


def test_run_spec(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Test run_spec."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        "experiments:\n  - mode: mc-sample\n    config_file: test.yaml\n"
    )

    config_file = tmp_path / "test.yaml"
    config_file.write_text("samples: 100\n")

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_batch_spec(
            spec_path: str,  # noqa: ARG004
            run_item: Callable[[object], dict],  # noqa: ARG004
            state_path: pathlib.Path | None = None,
            resume: bool = True,  # noqa: ARG004, FBT001, FBT002
            stop_on_error: bool = False,  # noqa: ARG004, FBT001, FBT002
        ):
            return {
                "completed_count": 1,
                "failed_count": 0,
                "state_path": str(state_path) if state_path else "None",
            }

    def mock_run_item(
        item: object,  # noqa: ARG001
    ) -> dict:
        return {"mode": "mc-sample"}

    monkeypatch.setattr(
        "autosafe.cli.experiments.ExperimentManager",
        MockManager,
    )

    run_spec(str(spec_file))
    captured = capsys.readouterr()
    assert "Batch finished" in captured.out


def test_run_spec_with_state(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Test run_spec with state file."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        "experiments:\n  - mode: dataset\n    dataset_path: test.csv\n"
    )

    dataset_file = tmp_path / "test.csv"
    dataset_file.write_text("x,y\n0,0\n")

    state_file = tmp_path / "state.json"

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_batch_spec(
            spec_path: str,  # noqa: ARG004
            run_item: Callable[[object], dict],  # noqa: ARG004
            state_path: pathlib.Path | None = None,
            resume: bool = True,  # noqa: ARG004, FBT001, FBT002
            stop_on_error: bool = False,  # noqa: ARG004, FBT001, FBT002
        ):
            return {
                "completed_count": 1,
                "failed_count": 0,
                "state_path": str(state_path) if state_path else "None",
            }

    monkeypatch.setattr(
        "autosafe.cli.experiments.ExperimentManager",
        MockManager,
    )

    run_spec(str(spec_file), state_path=str(state_file))
    captured = capsys.readouterr()
    assert "Batch finished" in captured.out


def test_run_spec_stop_on_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Test run_spec with stop_on_error=True."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        "experiments:\n  - mode: mc-results\n    inputs: [test.json]\n"
    )

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_batch_spec(
            spec_path: str,  # noqa: ARG004
            run_item: Callable[[object], dict],  # noqa: ARG004
            state_path: pathlib.Path | None = None,
            resume: bool = True,  # noqa: ARG004, FBT001, FBT002
            stop_on_error: bool = False,  # noqa: ARG004, FBT001, FBT002
        ):
            return {
                "completed_count": 0,
                "failed_count": 1,
                "state_path": str(state_path) if state_path else "None",
            }

    monkeypatch.setattr(
        "autosafe.cli.experiments.ExperimentManager",
        MockManager,
    )

    run_spec(str(spec_file), stop_on_error=True)
    captured = capsys.readouterr()
    assert "Batch finished" in captured.out


def test_run_spec_unsupported_mode(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test run_spec raises error for unsupported mode."""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("experiments:\n  - mode: invalid-mode\n")

    class MockManager:
        def __init__(self, config: object) -> None:
            pass

        @staticmethod
        def run_batch_spec(
            spec_path: str,  # noqa: ARG004
            run_item: Callable[[object], dict],
            state_path: pathlib.Path | None = None,
            resume: bool = True,  # noqa: ARG004, FBT001, FBT002
            stop_on_error: bool = False,  # noqa: ARG004, FBT001, FBT002
        ):
            # Simulate the runner raising an error for unsupported mode
            try:
                run_item({"mode": "invalid-mode"})
            except ValueError:
                raise
            return {
                "completed_count": 0,
                "failed_count": 1,
                "state_path": str(state_path) if state_path else "None",
            }

    monkeypatch.setattr(
        "autosafe.cli.experiments.ExperimentManager",
        MockManager,
    )

    with pytest.raises(ValueError, match="Unsupported spec mode"):
        run_spec(str(spec_file))


def test_configure_experiments_cli_verbose():
    """Test _configure_experiments_cli callback with verbose flag."""
    # Reset SETTINGS first
    SETTINGS["verbose"] = False

    # Call with verbose_count > 0
    _configure_experiments_cli(verbose_count=2)
    assert SETTINGS["verbose"] is True

    # Call with verbose_count = 0
    _configure_experiments_cli(verbose_count=0)
    assert SETTINGS["verbose"] is False
