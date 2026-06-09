# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Coverage tests for autosafe.tools.evaluate package."""

import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from scipy.spatial import QhullError
from typer.testing import CliRunner

from autosafe.samples import Samples
from autosafe.tools.evaluate import cli as eval_cli
from autosafe.tools.evaluate.core import (
    ConvexHullError,
    calculate_confusion_matrix,
    calculate_performance_metrics,
    create_convex_hull,
    process_files,
)
from autosafe.tools.evaluate.metrics import (
    build_affinity_thresholds,
    evaluate_affinity_metrics,
    save_metrics_csv,
)
from autosafe.tools.evaluate.workflows import (
    _baseline_memberships,
    _build_or_load_affinity_odd,
    _extract_anchor_points,
    _extract_mc_samples,
    _ground_truth_labels_from_yaml,
    _hull_membership,
    _infer_ground_truth_yaml,
    _ODDCacheSpec,
    _sample_points_around_odd,
    collect_monte_carlo_files,
    evaluate_dataset_mode,
    evaluate_monte_carlo_results,
)


class _DummyOdd:
    def __init__(self) -> None:
        self.samples = [
            type("S", (), {"x": np.array([0.0, 0.0])})(),
            type("S", (), {"x": np.array([1.0, 1.0])})(),
        ]

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return np.clip(points.mean(axis=1), 0.0, 1.0)


def _mc_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sampling_results": [
                [
                    {"coordinates": [0.0, 0.1], "affinity": 0.2, "in_odd": True},
                    {"coordinates": [0.5, 0.6], "affinity": 0.8, "in_odd": False},
                ]
            ],
            "anchors": [[[0.0, 0.0], [1.0, 1.0]]],
            "config": [{"dim": 2}],
            "autosafe_odd": ["array([0.0, 0.0]) array([1.0, 1.0])"],
        },
    )


def test_core_confusion_and_metrics():
    positive = pl.DataFrame({"affinity": [0.9, 0.8, 0.1]})
    negative = pl.DataFrame({"affinity": [0.9, 0.2, 0.1]})

    cm = calculate_confusion_matrix(positive, negative, np.float64(0.5))
    assert cm["true_positive"] == 2
    assert cm["false_positive"] == 1

    metrics = calculate_performance_metrics(cm)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0

    zeros = calculate_performance_metrics(
        {
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
        },
    )
    assert zeros["accuracy"] == pytest.approx(0.0)
    assert zeros["precision"] == pytest.approx(0.0)


def test_core_process_files_and_hull_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    json_file = tmp_path / "one.json"
    json_file.write_text("{}", encoding="utf-8")

    assert process_files(str(json_file)) == [json_file]

    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a.json").write_text("{}", encoding="utf-8")
    files = process_files(str(folder))
    assert len(files) == 1

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        process_files(str(empty))

    calls: list[str | None] = []

    class _Hull:
        pass

    def _convex(points: np.ndarray, qhull_options: str | None = None) -> _Hull:
        _ = points
        calls.append(qhull_options)
        if qhull_options is None:
            raise QhullError("fail first")
        return _Hull()

    monkeypatch.setattr(
        "autosafe.tools.evaluate.core.scipy.spatial.ConvexHull", _convex
    )
    hull = create_convex_hull(_mc_df())
    assert isinstance(hull, _Hull)
    assert calls[0] is None
    assert calls[1] == "QJ"


def test_core_create_convex_hull_fallback_and_failure(monkeypatch: pytest.MonkeyPatch):
    df_no_dim = pl.DataFrame(
        {
            "sampling_results": [
                [{"coordinates": [0.0, 1.0], "affinity": 0.5, "in_odd": True}]
            ],
            "anchors": [[[0.0, 0.0], [1.0, 1.0]]],
            "config": [{"x": 2}],
            "autosafe_odd": ["array([0.0, 0.0]) array([1.0, 1.0])"],
        },
    )

    def _ok_hull(_p: np.ndarray, qhull_options: str | None = None) -> object:
        _ = qhull_options
        return type("H", (), {})()

    monkeypatch.setattr(
        "autosafe.tools.evaluate.core.scipy.spatial.ConvexHull", _ok_hull
    )
    _ = create_convex_hull(df_no_dim)

    df_fallback = pl.DataFrame(
        {
            "sampling_results": [
                [{"coordinates": [0.0, 1.0], "affinity": 0.5, "in_odd": True}]
            ],
            "anchors": [[]],
            "config": [{"dim": 2}],
            "autosafe_odd": ["array([0.0, 0.0]) array([1.0, 1.0])"],
        },
    )
    _ = create_convex_hull(df_fallback)

    def _always_fail(_p: np.ndarray, qhull_options: str | None = None) -> None:
        _ = qhull_options
        raise QhullError("always")

    monkeypatch.setattr(
        "autosafe.tools.evaluate.core.scipy.spatial.ConvexHull",
        _always_fail,
    )
    with pytest.raises(ConvexHullError):
        create_convex_hull(_mc_df())


def test_metrics_module(tmp_path: Path):
    assert len(build_affinity_thresholds("linear", 3)) == 3
    assert len(build_affinity_thresholds("log", 4)) == 4

    with pytest.raises(ValueError, match="count must be positive"):
        build_affinity_thresholds("linear", 0)
    with pytest.raises(ValueError, match="mode must be"):
        build_affinity_thresholds("unknown", 2)

    with pytest.raises(ValueError, match="affinity"):
        evaluate_affinity_metrics(
            pl.DataFrame({"x": [1]}), {"r": np.array([True])}, np.array([0.5]), "s"
        )

    with pytest.raises(ValueError, match="length does not match"):
        evaluate_affinity_metrics(
            pl.DataFrame({"affinity": [0.1, 0.2]}),
            {"r": np.array([True])},
            np.array([0.5]),
            "s",
        )

    out = evaluate_affinity_metrics(
        pl.DataFrame({"affinity": [0.9, 0.1]}),
        {"r": np.array([True, False])},
        np.array([0.2, 0.8]),
        "source",
    )
    assert out.height == 2
    csv_path = save_metrics_csv(out, tmp_path / "m" / "results.csv")
    assert csv_path.exists()


def test_workflows_extractors_and_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    odd = _DummyOdd()
    anchors = _extract_anchor_points(cast("Any", odd))
    assert anchors.shape == (2, 2)

    coords, aff, in_odd = _extract_mc_samples(_mc_df())
    assert coords.shape == (2, 2)
    assert aff.shape == (2,)
    assert in_odd.dtype == np.bool_

    ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    test = np.array([[0.2, 0.2], [2.0, 2.0]])
    membership = _hull_membership(ref, test)
    assert membership.shape == (2,)

    sampled = _sample_points_around_odd(np.array([[0.0, 0.0], [1.0, 1.0]]), 20, seed=1)
    assert sampled.shape == (20, 2)

    gt_yaml = tmp_path / "gt.yml"
    gt_yaml.write_text("odd: true", encoding="utf-8")

    class _Region:
        def contains(self, points_t: np.ndarray) -> np.ndarray:
            _ = self
            return np.ones(points_t.shape[1], dtype=bool)

    class _Factory:
        def __init__(self, _cfg: dict[str, object]) -> None:
            pass

        def create_odd(self) -> tuple[_Region, str]:
            _ = self
            return _Region(), "desc"

    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.load_yaml_odd_config", lambda _p: {"x": 1}
    )
    monkeypatch.setattr("autosafe.tools.evaluate.workflows.ODDFactory", _Factory)
    gt = _ground_truth_labels_from_yaml(gt_yaml, np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert gt.dtype == np.bool_

    data_path = tmp_path / "data.csv"
    data_path.write_text("x,y\n0,0\n", encoding="utf-8")
    sibling = data_path.with_suffix(".yaml")
    sibling.write_text("x: 1", encoding="utf-8")
    assert _infer_ground_truth_yaml(data_path) == sibling


def test_workflows_baselines_and_build_or_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    points = np.array([[0.1, 0.2], [0.3, 0.4]])

    # Filter expected warning about cluster size
    warnings.filterwarnings(
        "ignore",
        message=r"Only \d+/\d+ clusters have >= \d+ points",
        category=UserWarning,
    )

    class _Mon:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def fit(self, _x: np.ndarray) -> "_Mon":
            return self

        def evaluate_batch(self, x: np.ndarray) -> np.ndarray:
            _ = self
            return np.ones(x.shape[1], dtype=bool)

    def mock_create_comparison_monitor(
        _method_name: str,
        **_kwargs: dict[str, Any],
    ) -> _Mon:
        return _Mon()

    def mock_get_available_method_names() -> list[str]:
        return [
            "hull_single",
            "hull_clustered",
            "knn",
            "kmeans",
            "density_single",
            "density_clustered",
            "dbscan_cluster",
            "fast_hull_approx",
        ]

    monkeypatch.setattr(
        "autosafe.tools.evaluate.comparison.create_comparison_monitor",
        mock_create_comparison_monitor,
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.comparison.get_available_method_names",
        mock_get_available_method_names,
    )

    labels = _baseline_memberships(
        ref,
        points,
        [
            "hull_single",
            "density_single",
            "knn",
            "kmeans",
            "hull_clustered",
            "density_clustered",
            "dbscan_cluster",
        ],
    )
    assert "hull_single" in labels
    assert "density_single" in labels
    assert labels["knn"].shape == (2,)

    odd_json = tmp_path / "odd.json"
    odd_json.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.autosafe.from_json", lambda _p: "loaded"
    )
    odd_obj, odd_path = _build_or_load_affinity_odd(
        dataset_path=tmp_path / "d.csv",
        odd_json=odd_json,
        odd_json_out=None,
        cache_spec=_ODDCacheSpec(
            closest_sample_mode="global",
            kernel_type="RBF",
            kernel_kwargs={},
            normalize_data=False,
        ),
    )
    assert odd_obj == "loaded"
    assert odd_path == odd_json

    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.load_dataset",
        lambda _p, **_kw: (pl.DataFrame({"x": [0.0]}), "csv"),
    )

    written: list[Path] = []

    def _to_json(_odd: object, path: Path) -> None:
        written.append(path)

    monkeypatch.setattr("autosafe.tools.evaluate.workflows.autosafe.to_json", _to_json)

    built, built_path = _build_or_load_affinity_odd(
        dataset_path=tmp_path / "dataset.csv",
        odd_json=None,
        odd_json_out=None,
        cache_spec=_ODDCacheSpec(
            closest_sample_mode="per_dimension",
            kernel_type="RBF",
            kernel_kwargs={},
            normalize_data=False,
        ),
    )
    assert isinstance(built, Samples)
    assert built_path in written


def test_workflows_public_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mc_file = tmp_path / "mc.json"
    mc_file.write_text("{}", encoding="utf-8")

    fake_data: dict[str, object] = {
        "anchors": [np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
    }

    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.pl.read_json",
        lambda _p: fake_data,
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._extract_mc_samples",
        lambda _d: (
            np.array([[0.0, 0.1], [0.5, 0.6]]),
            np.array([0.2, 0.8]),
            np.array([True, False]),
        ),
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._baseline_memberships",
        lambda _a, _b, _m: {"knn": np.array([True, False])},
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.evaluate_affinity_metrics",
        lambda **_k: pl.DataFrame({
            "source": ["x"],
            "reference": ["knn"],
            "affinity_threshold": [0.5],
        }),
    )

    saved: list[Path] = []
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.save_metrics_csv",
        lambda _r, p: saved.append(p) or p,
    )

    res = evaluate_monte_carlo_results(
        [mc_file], references=["ground_truth", "knn"], csv_output=tmp_path / "out.csv"
    )
    assert res.height == 1
    assert saved

    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._baseline_memberships",
        lambda _a, _b, _m: {},
    )
    with pytest.raises(ValueError, match="No reference labels"):
        evaluate_monte_carlo_results(
            [mc_file], references=["unknown"], csv_output=tmp_path / "out2.csv"
        )

    def _build_stub(_dataset_path: Path, **_kwargs: object) -> tuple[_DummyOdd, Path]:
        return _DummyOdd(), tmp_path / "odd.json"

    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._build_or_load_affinity_odd", _build_stub
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._infer_ground_truth_yaml", lambda _p: None
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._baseline_memberships",
        lambda _a, _b, _m: {"knn": np.array([True, False])},
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.evaluate_affinity_metrics",
        lambda **_k: pl.DataFrame({
            "source": ["d"],
            "reference": ["knn"],
            "affinity_threshold": [0.5],
        }),
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.save_metrics_csv", lambda _r, p: p
    )

    ds = tmp_path / "dataset.csv"
    ds.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")

    df, csv_path, odd_path = evaluate_dataset_mode(
        dataset_path=ds,
        n_samples=2,
        references=["knn"],
        csv_output=tmp_path / "metrics.csv",
    )
    assert df.height == 1
    assert csv_path.name == "metrics.csv"
    assert odd_path.name == "odd.json"

    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._baseline_memberships", lambda _a, _b, _m: {}
    )
    with pytest.raises(ValueError, match="No reference labels"):
        evaluate_dataset_mode(
            dataset_path=ds,
            n_samples=2,
            references=["ground_truth"],
            ground_truth_yaml=None,
        )

    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.process_files", lambda p: [Path(p)]
    )
    assert collect_monte_carlo_files([str(mc_file)]) == [mc_file]

    runner = CliRunner()

    no_file = runner.invoke(eval_cli.EVAL_APP, ["sampling-results"])
    assert no_file.exit_code != 0
    assert isinstance(no_file.exception, FileNotFoundError)
    assert "No file" in str(no_file.exception)

    monkeypatch.setattr(
        "autosafe.tools.evaluate.cli.collect_monte_carlo_files", lambda _f: [mc_file]
    )

    def _eval_mc_stub(
        _files: list[Path],
        *,
        threshold_mode: str = "linear",
        threshold_count: int = 100,
        references: list[str] | None = None,
        csv_output: Path | None = None,
    ) -> pl.DataFrame:
        _ = (threshold_mode, threshold_count, references, csv_output)
        return pl.DataFrame({"x": [1]})

    monkeypatch.setattr(
        "autosafe.tools.evaluate.cli.evaluate_monte_carlo_results", _eval_mc_stub
    )

    ok_sampling = runner.invoke(
        eval_cli.EVAL_APP, ["sampling-results", "--file", str(mc_file)]
    )
    assert ok_sampling.exit_code == 0

    missing_dataset = runner.invoke(
        eval_cli.EVAL_APP, ["dataset", str(tmp_path / "missing.csv")]
    )
    assert missing_dataset.exit_code != 0
    assert "Dataset file not found" in missing_dataset.output

    monkeypatch.setattr(
        "autosafe.tools.evaluate.cli.evaluate_dataset_mode",
        lambda **_k: (
            pl.DataFrame({"x": [1]}),
            tmp_path / "c.csv",
            tmp_path / "o.json",
        ),
    )
    ok_dataset = runner.invoke(eval_cli.EVAL_APP, ["dataset", str(ds)])
    assert ok_dataset.exit_code == 0
    assert "Evaluation completed" in ok_dataset.output


def test_evaluate_init_exports():
    import autosafe.tools.evaluate as evaluate_pkg

    assert hasattr(evaluate_pkg, "EVAL_APP")
    assert callable(evaluate_pkg.calculate_confusion_matrix)
    assert callable(evaluate_pkg.evaluate_dataset)


def test_workflows_remaining_default_path_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ds = tmp_path / "dataset.csv"
    ds.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")

    # _infer_ground_truth_yaml -> None branch
    assert _infer_ground_truth_yaml(ds) is None

    mc_file = tmp_path / "mc.json"
    mc_file.write_text("{}", encoding="utf-8")

    fake_data: dict[str, object] = {"anchors": [np.array([[0.0, 0.0], [1.0, 1.0]])]}
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.pl.read_json", lambda _p: fake_data
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._extract_mc_samples",
        lambda _d: (
            np.array([[0.0, 0.1], [0.5, 0.6]]),
            np.array([0.2, 0.8]),
            np.array([True, False]),
        ),
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._baseline_memberships", lambda _a, _b, _m: {}
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.evaluate_affinity_metrics",
        lambda **_k: pl.DataFrame({
            "source": ["x"],
            "reference": ["ground_truth"],
            "affinity_threshold": [0.5],
        }),
    )

    saved_paths: list[Path] = []
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.save_metrics_csv",
        lambda _r, p: saved_paths.append(p) or p,
    )

    _ = evaluate_monte_carlo_results([mc_file], references=None, csv_output=None)
    assert saved_paths
    assert saved_paths[0].name == "evaluation_results_monte_carlo.csv"

    # evaluate_dataset_mode branches: effective_ground_truth_yaml present,
    # add ground_truth labels, and default csv_output path.
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._build_or_load_affinity_odd",
        lambda _dataset_path, **_kwargs: (_DummyOdd(), tmp_path / "odd.json"),
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._baseline_memberships",
        lambda _a, _b, _m: {"knn": np.array([True, False])},
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows._ground_truth_labels_from_yaml",
        lambda _yaml, _points, **_kw: np.array([True, False]),
    )
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.evaluate_affinity_metrics",
        lambda **_k: pl.DataFrame({
            "source": ["d"],
            "reference": ["ground_truth"],
            "affinity_threshold": [0.5],
        }),
    )

    ds_saved: list[Path] = []
    monkeypatch.setattr(
        "autosafe.tools.evaluate.workflows.save_metrics_csv",
        lambda _r, p: ds_saved.append(p) or p,
    )

    gt = tmp_path / "ground_truth.yaml"
    gt.write_text("dummy: true", encoding="utf-8")
    _df, csv_path, _odd_path = evaluate_dataset_mode(
        dataset_path=ds,
        n_samples=2,
        references=["knn", "ground_truth"],
        ground_truth_yaml=gt,
        csv_output=None,
    )
    assert csv_path in ds_saved
    assert "evaluation-linear" in csv_path.stem
