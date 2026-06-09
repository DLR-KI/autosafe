# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Basic tests for experiments module functionality."""

import datetime
import pathlib
import tempfile
import types
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
import yaml

from autosafe.tools.experiments import core as core_mod
from autosafe.tools.experiments import evaluation as eval_mod
from autosafe.tools.experiments.core import (
    BenchmarkResult,
    DatasetConfig,
    DatasetType,
    EvaluationResult,
    ExperimentManager,
    ExperimentType,
    KernelExperimentConfig,
    create_experiment_manager,
)
from autosafe.tools.experiments.evaluation import (
    BenchmarkExperimentConfig,
    EvaluationExperimentConfig,
    generate_experiment_id,
    run_monte_carlo_evaluation,
)
from autosafe.tools.experiments.utils import (
    DatasetLoadOptions,
    apply_filters,
    calculate_dataset_statistics,
    construct_experiment_id,
    create_dataset_config,
    find_dataset_bounds,
    get_dataset_type,
    load_dataset,
    load_experiment_results,
    normalize_dataset,
    save_results,
    setup_logging,
)


class _DummyOdd:
    def __init__(self, points: np.ndarray) -> None:
        self.samples = [types.SimpleNamespace(x=row) for row in points]
        self.shape = points.shape

    def __call__(self, points: np.ndarray) -> np.ndarray:
        # Deterministic pseudo-affinity in [0, 1]
        return np.clip(np.mean(points, axis=1), 0.0, 1.0)


def test_experiment_type_enum():
    """Test ExperimentType enum values."""
    assert ExperimentType.EVALUATION.value == "evaluation"
    assert ExperimentType.BENCHMARK.value == "benchmark"
    assert ExperimentType.CUSTOM.value == "custom"

    # Check that these are the only values
    assert len(list(ExperimentType)) == 3


def test_experiment_id_generation():
    """Test experiment ID generation functions."""
    # Test basic ID generation
    exp_id = generate_experiment_id()
    assert exp_id.startswith("exp_")
    assert len(exp_id) > 20  # Contains timestamp + random number

    # Test structured ID generation
    test_path = pathlib.Path("test_dataset.csv")
    structured_id = construct_experiment_id(
        ExperimentType.EVALUATION,
        test_path,
        timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        suffix="test",
    )
    assert "evaluation" in structured_id
    assert "20260101" in structured_id
    assert "test_dataset" in structured_id.replace("-", "_")
    assert "_test" in structured_id


def test_dataset_config_creation():
    """Test dataset configuration creation and validation."""
    # Test with mock CSV file (we can't create real files in tests, so test validation logic)
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp_file:
        test_file = pathlib.Path(tmp_file.name)
        config = DatasetConfig(file_path=test_file, dataset_type=DatasetType.CSV)

        assert config.file_path == test_file
        assert config.dataset_type == DatasetType.CSV
        assert np.allclose(config.range_extension, 0.5)


def test_dataset_type_detection():
    """Test automatic dataset type detection."""
    assert get_dataset_type(pathlib.Path("test.csv")) == "csv"
    assert get_dataset_type(pathlib.Path("test.json")) == "json"
    assert get_dataset_type(pathlib.Path("test.parquet")) == "parquet"
    assert get_dataset_type(pathlib.Path("test.npy")) == "numpy"

    # Test error on unknown type
    with pytest.raises(ValueError, match="Unknown dataset type"):
        get_dataset_type(pathlib.Path("test.unknown"))


def test_dataset_config_validation():
    """Test dataset configuration validation logic."""
    # Test format validation in DatasetConfig - test basic creation
    config = DatasetConfig(file_path=pathlib.Path("test.csv"))
    assert config.file_path.suffix == ".csv"

    # Test that a valid CSV path creates valid config
    assert (
        str(config.dataset_type) == "DatasetType.CSV"
        or config.dataset_type.value == "csv"
    )

    # Test suffix property works
    config.file_path = pathlib.Path("outlier.npy")
    assert config.file_path.suffix == ".npy" in [".npy", ".numpy"]


def test_experiment_manager_creation():
    """Test experiment manager initialization."""
    config = KernelExperimentConfig(
        kernel_type="RBF",
        kernel_kwargs={},
        n_samples=1000,
        evaluation_samples=100,
    )

    manager = ExperimentManager(config)

    assert manager.config == config
    assert manager.results == []
    assert manager.experiment_count == 0
    assert manager.last_result is None


def test_experiment_manager_id_generation():
    """Test experiment ID generation in manager."""
    config = KernelExperimentConfig()
    manager = ExperimentManager(config)

    # Generate IDs
    id1 = manager.create_experiment_id()
    id2 = manager.create_experiment_id()

    assert id1.startswith("exp_")
    assert id2.startswith("exp_")
    assert id1 != id2  # IDs should be unique
    assert manager.experiment_count == 2


def test_benchmark_result_creation():
    """Test benchmark result creation."""
    test_path = pathlib.Path("test_dataset.csv")
    result = BenchmarkResult(
        experiment_id="test_001",
        experiment_type=ExperimentType.BENCHMARK,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        dataset_path=test_path,
        dataset_size=1000,
        dataset_dimensions=5,
        config={},
        total_samples=10000,
        processing_time=10.5,
    )

    assert result.experiment_id == "test_001"
    assert result.experiment_type == ExperimentType.BENCHMARK
    assert len(result.to_dict()) > 0


def test_evaluation_result_creation():
    """Test evaluation result creation."""
    test_path = pathlib.Path("test_dataset.json")
    result = EvaluationResult(
        experiment_id="eval_001",
        experiment_type=ExperimentType.EVALUATION,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        dataset_path=test_path,
        dataset_size=5000,
        dataset_dimensions=7,
        config={"kernel": "RBF"},
        total_samples=100000,
        processing_time=25.8,
    )

    assert result.experiment_id == "eval_001"
    assert result.experiment_type == ExperimentType.EVALUATION
    assert result.export_paths == []


def test_result_export_functionality():
    """Test result export functionality."""
    result = BenchmarkResult(
        experiment_id="test_export",
        experiment_type=ExperimentType.BENCHMARK,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        dataset_path=pathlib.Path("test.csv"),
        dataset_size=100,
        dataset_dimensions=2,
        config={},
        total_samples=1000,
        processing_time=2.0,
    )

    # Test add_export method
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
        export_path = pathlib.Path(tmp_file.name)
        result.add_export(export_path)

        assert len(result.export_paths) == 1
        assert result.export_paths[0] == export_path


def test_normalize_dataset_basic():
    """Test basic dataset normalization without specific datasets."""
    import polars as pl

    # Test with basic dataframe
    df = pl.DataFrame({
        "col1": [1.0, 2.0, 3.0],
        "col2": [4.0, 5.0, 6.0],
    })

    # Should not fail and should keep the shape while normalizing numerics.
    normalized = normalize_dataset(df, "minmax")
    assert normalized.shape == df.shape
    assert normalized.select(pl.col("col1").min()).item() == pytest.approx(-1.0)
    assert normalized.select(pl.col("col1").max()).item() == pytest.approx(1.0)
    assert normalized.select(pl.col("col2").min()).item() == pytest.approx(-1.0)
    assert normalized.select(pl.col("col2").max()).item() == pytest.approx(1.0)


def test_experiment_utils_filters_statistics_and_bounds():
    """Test filtering, statistics, and bounds helpers."""
    import polars as pl

    df = pl.DataFrame(
        {
            "group": ["keep", "keep", "skip"],
            "value": [1.0, 2.0, 3.0],
            "other": [10.0, 20.0, 30.0],
        },
    )

    filtered = apply_filters(
        df,
        {
            "group": "keep",
            "value": (1.0, 2.0),
            "missing": "ignored",
        },
    )
    assert filtered.shape == (2, 3)
    assert filtered["group"].to_list() == ["keep", "keep"]

    stats = calculate_dataset_statistics(filtered)
    assert stats["shape"] == (2, 3)
    assert stats["columns"] == ["group", "value", "other"]
    assert stats["means"]["value"][0] == pytest.approx(1.5)
    assert stats["mins"]["other"][0] == pytest.approx(10.0)

    min_values, max_values = find_dataset_bounds(filtered, ["value", "other"])
    assert min_values == pytest.approx([1.0, 10.0])
    assert max_values == pytest.approx([2.0, 20.0])


def test_load_dataset_and_save_results_round_trip(tmp_path: pathlib.Path):
    """Test loading, normalization, and JSON export helpers."""
    import polars as pl

    dataset_path = tmp_path / "dataset.csv"
    pl.DataFrame(
        {
            "group": ["keep", "keep", "skip"],
            "value": [1.0, 2.0, 3.0],
            "other": [10.0, 20.0, 30.0],
        },
    ).write_csv(dataset_path)

    loaded_df, loaded_type = load_dataset(
        dataset_path,
        DatasetLoadOptions(
            filters={"group": "keep", "value": (1.0, 2.0)},
            normalize=True,
            normalization_method="minmax",
        ),
    )

    assert loaded_type == "csv"
    assert loaded_df.shape == (2, 3)
    assert loaded_df["value"].to_list() == pytest.approx([-1.0, 1.0])
    assert loaded_df["other"].to_list() == pytest.approx([-1.0, 1.0])

    export_path = tmp_path / "results.json"

    class DummyResult:
        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"value": 42, "items": [1, 2, 3]}

    save_results(DummyResult(), export_path)
    assert load_experiment_results(export_path) == {"value": 42, "items": [1, 2, 3]}


def test_create_dataset_config_infers_and_reuses_config():
    """Test dataset config inference and config reuse."""
    parquet_path = pathlib.Path("sample.parquet")
    config = create_dataset_config(parquet_path)
    assert config.file_path == parquet_path
    assert config.dataset_type == DatasetType.POLARS

    existing = DatasetConfig(
        file_path=pathlib.Path("old.csv"),
        dataset_type=DatasetType.JSON,
    )
    reused = create_dataset_config(pathlib.Path("updated.csv"), existing)
    assert reused is existing
    assert reused.file_path == pathlib.Path("updated.csv")


def test_monte_carlo_evaluation_interface():
    """Test that Monte Carlo evaluation interface exists."""
    # This tests that the function exists and has the right signature
    # We don't actually run it to avoid side effects

    assert callable(run_monte_carlo_evaluation)


def test_config_validation():
    """Test kernel configuration validation."""
    config = KernelExperimentConfig()

    # Should not raise with valid config
    config.validate()

    # Test invalid sample count
    invalid_config = KernelExperimentConfig(n_samples=-100)
    with pytest.raises(ValueError, match="must be positive"):
        invalid_config.validate()


def test_experiment_result_serialization():
    """Test that experiment results can be serialized to dict."""
    result = EvaluationResult(
        experiment_id="serialization_test",
        experiment_type=ExperimentType.EVALUATION,
        timestamp=datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
        dataset_path=pathlib.Path("data/test.csv"),
        dataset_size=1000,
        dataset_dimensions=3,
        config={"test_param": "value"},
        total_samples=10000,
        processing_time=5.43,
    )

    # Test serialization to dict
    result_dict = result.to_dict()

    assert result_dict["experiment_id"] == "serialization_test"
    assert result_dict["experiment_type"] == "evaluation"
    assert result_dict["config"]["test_param"] == "value"


def test_dataset_and_kernel_validation_errors(tmp_path: pathlib.Path):
    """Cover validation branches for dataset and kernel config."""
    missing = DatasetConfig(file_path=tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        missing.validate()

    bad_suffix = tmp_path / "data.txt"
    bad_suffix.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        DatasetConfig(file_path=bad_suffix).validate()

    with pytest.raises(ValueError, match="Unsupported kernel type"):
        KernelExperimentConfig(kernel_type=cast("Any", "BadKernel")).validate()


def test_experiment_manager_save_results_and_factory(tmp_path: pathlib.Path):
    """Cover manager save and factory helper."""
    manager = create_experiment_manager(KernelExperimentConfig())
    assert isinstance(manager, ExperimentManager)

    result = EvaluationResult(
        experiment_id="exp_test",
        experiment_type=ExperimentType.EVALUATION,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        dataset_path=tmp_path / "dataset.csv",
        dataset_size=10,
        dataset_dimensions=2,
        config={"k": "v"},
        total_samples=100,
        processing_time=0.1,
    )

    exported = manager.save_results(result, export_dir=tmp_path / "out")
    assert exported.exists()
    payload = load_experiment_results(exported)
    assert payload["experiment_id"] == "exp_test"
    assert result.export_paths
    assert result.export_paths[0] == exported


def test_manager_run_experiments_with_monkeypatched_module(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Cover dynamic import path for evaluation and benchmark manager
    methods."""

    class _EvalCfg:
        def __init__(self, **kwargs: dict[str, Any]) -> None:
            self.kwargs = kwargs

    class _BenchCfg:
        def __init__(self, **kwargs: dict[str, Any]) -> None:
            self.kwargs = kwargs

    def _eval_impl(
        *, dataset_path: pathlib.Path, request: _EvalCfg
    ) -> EvaluationResult:
        assert dataset_path.name == "dataset.csv"
        assert request.kwargs["n_samples"] == 20
        return EvaluationResult(
            experiment_id="temp",
            experiment_type=ExperimentType.EVALUATION,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            dataset_path=dataset_path,
            dataset_size=3,
            dataset_dimensions=2,
            config={},
            total_samples=20,
            processing_time=0.01,
        )

    def _bench_impl(
        *, dataset_path: pathlib.Path, request: _BenchCfg
    ) -> BenchmarkResult:
        assert dataset_path.name == "dataset.csv"
        assert request.kwargs["n_samples"] == 50
        return BenchmarkResult(
            experiment_id="temp",
            experiment_type=ExperimentType.BENCHMARK,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            dataset_path=dataset_path,
            dataset_size=3,
            dataset_dimensions=2,
            config={},
            total_samples=50,
            processing_time=0.01,
        )

    fake_module = types.SimpleNamespace(
        EvaluationExperimentConfig=_EvalCfg,
        BenchmarkExperimentConfig=_BenchCfg,
        evaluate_experiment=_eval_impl,
        run_benchmark_experiment=_bench_impl,
    )
    monkeypatch.setattr(core_mod, "import_module", lambda _: fake_module)

    manager = ExperimentManager(
        KernelExperimentConfig(kernel_type="RBF", evaluation_samples=20, n_samples=50),
    )
    ds = DatasetConfig(file_path=tmp_path / "dataset.csv", dataset_type=DatasetType.CSV)
    ds.file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    eval_result = manager.run_evaluation_experiment(ds, export_dir=tmp_path / "x")
    bench_result = manager.run_benchmark_experiment(ds, export_dir=tmp_path / "y")

    assert eval_result.experiment_id.startswith("exp_")
    assert bench_result.experiment_id.startswith("exp_")
    assert manager.last_result is bench_result
    assert len(manager.results) == 2


def test_run_batch_spec_paths(tmp_path: pathlib.Path):
    """Cover batch-spec execution, failure handling, and resume behavior."""
    manager = ExperimentManager(KernelExperimentConfig())

    with pytest.raises(FileNotFoundError):
        manager.run_batch_spec(tmp_path / "missing.yaml", lambda item: item)

    invalid_spec = tmp_path / "invalid.yaml"
    invalid_spec.write_text("123", encoding="utf-8")
    with pytest.raises(ValueError, match="Spec must be a list"):
        manager.run_batch_spec(invalid_spec, lambda item: item)

    bad_mapping_spec = tmp_path / "bad_mapping.yaml"
    bad_mapping_spec.write_text("experiments: 123", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        manager.run_batch_spec(bad_mapping_spec, lambda item: item)

    spec_path = tmp_path / "batch.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {"id": "ok", "value": 1},
                    {"id": "boom", "value": 2},
                    "not-a-mapping",
                ]
            },
        ),
        encoding="utf-8",
    )
    seen: list[str] = []

    def run_item(item: dict[str, object]) -> dict[str, object]:
        seen.append(str(item["id"]))
        if item["id"] == "boom":
            raise RuntimeError("failed")
        return {"ok": True}

    summary = manager.run_batch_spec(
        spec_path, run_item, resume=True, stop_on_error=False
    )
    assert summary["completed_count"] == 1
    assert summary["failed_count"] == 2
    assert "ok" in seen
    assert "boom" in seen

    # Resume should skip completed "ok" and retry failures.
    seen.clear()
    summary_retry = manager.run_batch_spec(
        spec_path, run_item, resume=True, stop_on_error=False
    )
    assert "ok" not in seen
    assert "boom" in seen
    assert summary_retry["completed_count"] == 1

    with pytest.raises(RuntimeError, match="failed"):
        manager.run_batch_spec(spec_path, run_item, resume=False, stop_on_error=True)


def test_evaluation_module_paths_and_helpers(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Cover evaluation internals, both .json and tabular dataset paths."""
    points = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    odd = _DummyOdd(points)

    monkeypatch.setattr(eval_mod.autosafe, "from_json", lambda _: odd)
    loaded = eval_mod._load_dataset_context(
        tmp_path / "odd.json",
        kernel_type="RBF",
        kernel_kwargs={},
        closest_sample_mode="global",
    )
    assert loaded[1] == 2
    assert loaded[2] == 2
    assert np.allclose(loaded[3], [0.0, 0.0])
    assert np.allclose(loaded[4], [1.0, 1.0])

    monkeypatch.setattr(
        eval_mod,
        "load_dataset",
        lambda _: (pl.DataFrame({"x": [0.0, 1.0], "y": [0.5, 1.5]}), "csv"),
    )
    monkeypatch.setattr(
        eval_mod.autosafe,
        "from_polars",
        lambda *args, **kwargs: odd,  # noqa: ARG005
    )
    monkeypatch.setattr(
        eval_mod,
        "find_dataset_bounds",
        lambda df: ([0.0, 0.5], [1.0, 1.5]),  # noqa: ARG005
    )

    loaded_tabular = eval_mod._load_dataset_context(
        tmp_path / "data.csv",
        kernel_type="RBF",
        kernel_kwargs={},
        closest_sample_mode="per_dimension",
    )
    assert loaded_tabular[1] == 2
    assert loaded_tabular[2] == 2

    # Exercise sampling/benchmark helpers.
    affinities = eval_mod._sample_affinities(
        cast("Any", odd),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        dataset_dimensions=2,
        n_samples=8,
    )
    assert affinities.shape == (8,)

    timings = eval_mod._benchmark_timings(
        cast("Any", odd),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        dataset_dimensions=2,
        n_samples=250,
    )
    assert "1" in timings
    assert "100" in timings
    assert "1000" not in timings


def test_evaluation_public_functions_and_export_request(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Cover evaluate/benchmark/pipeline/summary and conditional save
    helper."""
    odd = _DummyOdd(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float))
    monkeypatch.setattr(
        eval_mod,
        "_load_dataset_context",
        lambda *args, **kwargs: (  # noqa: ARG005
            odd,
            2,
            2,
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
        ),
    )
    monkeypatch.setattr(
        eval_mod,
        "_sample_affinities",
        lambda *args, **kwargs: np.array([0.1, 0.9]),  # noqa: ARG005
    )
    monkeypatch.setattr(
        eval_mod,
        "_benchmark_timings",
        lambda *args, **kwargs: {"1": 0.0, "10": 0.0},  # noqa: ARG005
    )

    result_eval = eval_mod.evaluate_experiment(
        dataset_path=tmp_path / "x.csv",
        request=EvaluationExperimentConfig(n_samples=5),
    )
    assert result_eval.total_samples == 5
    assert result_eval.affinity_statistics is not None
    assert result_eval.affinity_statistics["mean"] == pytest.approx(0.5)

    result_bench = eval_mod.run_benchmark_experiment(
        dataset_path=tmp_path / "x.csv",
        request=BenchmarkExperimentConfig(n_samples=123),
    )
    assert result_bench.total_samples == 123
    assert "10" in result_bench.timing_by_sample_size

    called: list[pathlib.Path] = []
    monkeypatch.setattr(
        eval_mod,
        "evaluate_experiment",
        lambda dataset_path, request=None: called.append(dataset_path) or result_eval,  # noqa: ARG005
    )
    pipeline = eval_mod.run_evaluation_pipeline([
        tmp_path / "a.csv",
        tmp_path / "b.csv",
    ])
    assert pipeline == [ExperimentType.EVALUATION, ExperimentType.EVALUATION]
    assert len(called) == 2

    monkeypatch.setattr(
        eval_mod,
        "evaluate_experiment",
        lambda dataset_path, request=None: result_eval,  # noqa: ARG005
    )
    summary = eval_mod.run_monte_carlo_evaluation(tmp_path / "x.csv", n_samples=7)
    assert summary["success"] is True
    assert summary["dataset_dimensions"] == 2

    # _save_requested_result should be a no-op when export_dir is None,
    # and should delegate to save_results otherwise.
    eval_mod._save_requested_result(result_eval, None)
    captured: list[pathlib.Path] = []
    monkeypatch.setattr(
        eval_mod,
        "save_results",
        lambda result, path: captured.append(path),  # noqa: ARG005
    )
    eval_mod._save_requested_result(result_eval, tmp_path)
    assert captured
    assert captured[0].name.endswith(".json")


def test_utils_extra_file_types_and_logging(tmp_path: pathlib.Path):
    """Cover extra load_dataset branches and setup_logging behavior."""
    csv_path = tmp_path / "data.csv"
    jsonl_path = tmp_path / "data.jsonl"
    parquet_path = tmp_path / "data.parquet"

    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "tag": ["x", "y"]})
    df.write_csv(csv_path)
    df.write_ndjson(jsonl_path)
    df.write_parquet(parquet_path)

    loaded_csv, kind_csv = load_dataset(csv_path, DatasetLoadOptions(normalize=False))
    assert kind_csv == "csv"
    assert loaded_csv.shape == (2, 3)

    loaded_json, kind_json = load_dataset(jsonl_path)
    assert kind_json == "jsonl"
    assert loaded_json.shape[1] == 3

    loaded_parquet, kind_parquet = load_dataset(parquet_path)
    assert kind_parquet == "parquet"
    assert loaded_parquet.shape[0] == 2

    # Explicitly trigger NotImplemented and unsupported format branches.
    with pytest.raises(NotImplementedError):
        load_dataset(csv_path, DatasetLoadOptions(file_type="npy"))
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataset(csv_path, DatasetLoadOptions(file_type="txt"))

    # apply_filters with non-matching keys should return input unchanged.
    unchanged = apply_filters(df, {"missing": 1})
    assert unchanged.shape == df.shape

    # normalize_dataset with only non-numeric columns should be a no-op.
    text_df = pl.DataFrame({"name": ["a", "b"]})
    normalized_text = normalize_dataset(text_df)
    assert normalized_text.to_dicts() == text_df.to_dicts()

    log_file = tmp_path / "logs" / "exp.log"
    logger = setup_logging(log_file=log_file, name="autosafe.tests.coverage")
    logger.info("hello")
    assert log_file.exists()
