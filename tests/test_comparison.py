# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Coverage-focused tests for odd.comparison and tools.comparison modules."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from autosafe.odd.comparison.base import (
    ODDBoundaryMethod,
    ODDComparisonConfig,
    validate_comparison_config,
)
from autosafe.odd.comparison.cluster import (
    ClusteredConvexHulls,
    DBSCANCluster,
    KMeansBoundaries,
    KNNMonitor,
    auto_detect_optimal_k,
)
from autosafe.odd.comparison.density import (
    ClusteredSuperlevelSetMonitor,
    SuperlevelSetMonitor,
)
from autosafe.tools.comparison.cli import COMP_APP, _display_comparison_summary
from autosafe.tools.comparison.core import (
    _evaluate_comparison_methods,
    _find_best_coverage,
    _find_most_conservative,
    build_comparison_results_dataframe,
    create_comparison_test_grid,
    evaluate_comparison_methods,
    evaluate_dataset_with_comparison_methods,
    setup_evaluation_framework,
)
from autosafe.typing import Matrix, Vector


class _TinyMethod(ODDBoundaryMethod):
    """Minimal concrete implementation for base-class coverage."""

    @property
    def method_type(self) -> str:
        return "tiny"

    @property
    def decision_boundary(self) -> dict[str, Any]:
        return {
            "type": "tiny",
            "parameters": {},
            "coverage": {},
            "conservatism": None,
        }

    def fit(self, reference_points: Matrix | np.ndarray) -> "_TinyMethod":
        self.reference_points = reference_points
        self.trained = True
        return self

    def __call__(self, test_point: Vector | np.ndarray) -> bool:
        return bool(np.sum(test_point) >= 0)


def _reference_points() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 0.0, 1.0, 0.4, 0.8],
            [0.0, 0.0, 1.0, 1.0, 0.7, 0.2],
        ],
        dtype=float,
    )


def _test_points() -> np.ndarray:
    return np.array(
        [
            [0.1, 0.5, 2.0, -1.0],
            [0.2, 0.6, 2.0, -1.0],
        ],
        dtype=float,
    )


def test_base_methods_and_config_validation():
    tiny = _TinyMethod().fit(_reference_points())
    decisions = tiny.evaluate_batch(_test_points())
    assert decisions.shape == (4,)
    metric = tiny.get_conservatism_metric(_reference_points())
    assert 0.0 <= metric <= 1.0
    assert ODDBoundaryMethod.get_coverage_stats()["area"] is None

    valid_cfg: ODDComparisonConfig = {
        "methods": ["knn", "kmeans"],
        "evaluate_point_grid": True,
        "grid_resolution": 10,
        "conservatism_target": np.float64(0.5),
    }
    validate_comparison_config(valid_cfg)

    invalid_method_cfg: ODDComparisonConfig = {
        "methods": ["invalid"],
        "evaluate_point_grid": False,
        "grid_resolution": 10,
        "conservatism_target": None,
    }
    validate_comparison_config(
        valid_cfg,
    )

    with pytest.raises(ValueError, match="Invalid comparison method"):
        validate_comparison_config(invalid_method_cfg)

    bad_conservatism_cfg: ODDComparisonConfig = {
        "methods": ["all"],
        "evaluate_point_grid": False,
        "grid_resolution": 10,
        "conservatism_target": np.float64(1.2),
    }
    with pytest.raises(ValueError, match="Conservatism target"):
        validate_comparison_config(bad_conservatism_cfg)

    bad_resolution_cfg: ODDComparisonConfig = {
        "methods": ["all"],
        "evaluate_point_grid": False,
        "grid_resolution": -1,
        "conservatism_target": None,
    }
    with pytest.raises(ValueError, match="Grid resolution"):
        validate_comparison_config(bad_resolution_cfg)


def test_knn_monitor_paths():
    ref = _reference_points()
    monitor = KNNMonitor(k=2)

    with pytest.raises(RuntimeError, match="not fitted"):
        monitor(np.array([0.0, 0.0]))

    # Trigger explicit ValueError branch in auto-detection.
    monitor.data = cast("Any", None)
    with pytest.raises(ValueError, match="Data must be provided"):
        monitor.auto_detect_consensus_radius()

    monitor.fit(ref)
    assert monitor.method_type == "knn"
    assert monitor.decision_boundary["type"] == "knn"
    assert monitor.evaluate_batch(_test_points()).dtype == np.bool_
    assert auto_detect_optimal_k(ref, max_k_upper=4) >= 1

    # Cover query error branch.
    class _BadTree:
        def query(
            self,
            _test_point: np.ndarray,
            _k: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            _ = self
            raise ValueError("boom")

    monitor.tree = cast("Any", _BadTree())
    with pytest.raises(RuntimeError, match="KDTree query failed"):
        monitor(np.array([0.0, 0.0]))


def test_kmeans_cluster_hulls_and_dbscan(monkeypatch: pytest.MonkeyPatch):
    ref = _reference_points()
    kmeans = KMeansBoundaries(n_clusters=2, min_cluster_size=1).fit(ref)
    assert kmeans.decision_boundary["type"] == "kmeans"
    assert isinstance(kmeans(np.array([0.2, 0.2])), bool)
    assert kmeans._calculate_centroid_distances().shape == (2, 2)

    # Cover fallback/exception path for silhouette score.
    def _raise_silhouette(_x: np.ndarray, _labels: np.ndarray) -> float:
        raise ValueError("x")

    monkeypatch.setattr(
        "autosafe.odd.comparison.cluster.silhouette_score", _raise_silhouette
    )
    assert kmeans._calculate_silhouette_score(ref) == pytest.approx(0.0)

    # _point_in_hull 2D + 3D proxy + malformed path.
    nondeg = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    hull2d = cast("Any", __import__("scipy").spatial.ConvexHull(nondeg))
    assert isinstance(
        KMeansBoundaries._point_in_hull(np.array([0.1, 0.1]), hull2d), bool
    )

    hull3d = cast(
        "Any",
        type(
            "DummyHull", (), {"points": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
        )(),
    )
    assert KMeansBoundaries._point_in_hull(np.array([0.2, 0.0, 0.0]), hull3d)

    clustered = ClusteredConvexHulls(n_clusters=2)
    with pytest.raises(RuntimeError, match="Not fitted"):
        clustered(np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="currently supported"):
        ClusteredConvexHulls(method="dbscan").fit(ref)
    clustered.fit(ref)
    batch = clustered.evaluate_batch(_test_points())
    assert batch.shape == (4,)

    dbscan = DBSCANCluster(eps=0.5, min_samples=2)
    with pytest.raises(RuntimeError, match="Not fitted"):
        dbscan(np.array([0.0, 0.0]))
    dbscan.fit(ref)
    assert isinstance(dbscan(np.array([0.0, 0.0])), bool)


def test_density_monitors_paths(monkeypatch: pytest.MonkeyPatch):
    ref = _reference_points()
    monitor = SuperlevelSetMonitor(gamma=np.float64(0.01), bandwidth="scott")
    assert monitor.decision_boundary["type"] == "density"
    with pytest.raises(RuntimeError, match="not fitted"):
        monitor.pdf(_test_points())
    with pytest.raises(RuntimeError, match="not fitted"):
        monitor.evaluate_batch(_test_points())

    monitor.fit(ref)
    assert monitor.bandwidth_for_deflection() in {"scott", "silverman"}
    assert isinstance(monitor.suggest_reasonable_gamma(), float)
    assert "validation" in monitor.validate_hypersurface()
    assert isinstance(monitor(np.array([0.1, 0.1])), bool)
    grid_points, grid_pdf = monitor.create_visualization_grid(
        (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        resolution=6,
    )
    assert grid_points.shape[1] == 36
    assert grid_pdf.shape[0] == 36
    with pytest.raises(NotImplementedError):
        monitor.create_visualization_grid((
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ))

    # Force conservatism exception branch by monkeypatching pdf.
    monkeypatch.setattr(
        monitor,
        "pdf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert monitor._calculate_conservatism() == pytest.approx(0.5)

    clustered = ClusteredSuperlevelSetMonitor(n_clusters=3, min_cluster_size=100)
    with pytest.raises(RuntimeError, match="not fitted"):
        clustered(np.array([0.0, 0.0]))
    with pytest.raises(RuntimeError, match="not fitted"):
        clustered.evaluate_batch(_test_points())
    clustered.fit(ref)
    # No monitors due to large min_cluster_size
    assert clustered.evaluate_batch(_test_points()).sum() == 0


def test_comparison_core_end_to_end_and_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_path = tmp_path / "dataset.csv"
    pl.DataFrame({"x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 0.0, 1.0, 1.0]}).write_csv(
        data_path
    )

    framework = setup_evaluation_framework(data_path, foo="bar")
    assert framework["dataset_type"] == "csv"
    assert framework["reference_points"].shape[0] == 2

    grid = create_comparison_test_grid(framework["reference_points"], resolution=7)
    assert grid.shape == (2, 7)

    hull_result = evaluate_dataset_with_comparison_methods(
        framework["reference_points"],
        grid,
        method="hull_single",
    )
    assert hull_result["method"] == "hull_single"

    # Monkeypatch monitor classes to cover selection branches cheaply.
    class _FakeMonitor:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def fit(self, _ref: np.ndarray) -> "_FakeMonitor":
            return self

        def evaluate_batch(self, tests: np.ndarray) -> np.ndarray:
            _ = self.kwargs
            return np.ones(tests.shape[1], dtype=bool)

        @property
        def decision_boundary(self) -> dict[str, Any]:
            return {
                "type": "fake",
                "parameters": self.kwargs,
                "coverage": {},
                "conservatism": 0.0,
            }

    monkeypatch.setattr("autosafe.tools.comparison.core.KNNMonitor", _FakeMonitor)
    monkeypatch.setattr("autosafe.tools.comparison.core.KMeansBoundaries", _FakeMonitor)
    monkeypatch.setattr(
        "autosafe.tools.comparison.core.SuperlevelSetMonitor", _FakeMonitor
    )
    monkeypatch.setattr(
        "autosafe.tools.comparison.core.ClusteredConvexHulls", _FakeMonitor
    )
    monkeypatch.setattr(
        "autosafe.tools.comparison.core.ClusteredSuperlevelSetMonitor", _FakeMonitor
    )
    monkeypatch.setattr("autosafe.tools.comparison.core.DBSCANCluster", _FakeMonitor)

    for method in [
        "knn",
        "kmeans",
        "density_single",
        "hull_clustered",
        "density_clustered",
        "dbscan_cluster",
    ]:
        res = evaluate_dataset_with_comparison_methods(
            framework["reference_points"],
            grid,
            method=cast("Any", method),
        )
        assert res["coverage_ratio"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Unknown method"):
        evaluate_dataset_with_comparison_methods(
            framework["reference_points"], grid, method=cast("Any", "bad")
        )

    def _patched_grid(_ref_points: np.ndarray, resolution: int = 50) -> np.ndarray:
        _ = resolution
        return np.asarray(grid)

    monkeypatch.setattr(
        "autosafe.tools.comparison.core.create_comparison_test_grid", _patched_grid
    )

    full = evaluate_comparison_methods(
        dataset_path=data_path,
        methods=["hull_single", "knn"],
        export_path=str(tmp_path / "res.json"),
    )
    assert full["summary"]["method_count"] == 2
    assert (tmp_path / "res.json").exists()

    with pytest.raises(ValueError, match="Unknown comparison method"):
        evaluate_comparison_methods(
            dataset_path=data_path, methods=[cast("Any", "bad")]
        )

    compat = _evaluate_comparison_methods(
        dataset_path=data_path, methods=["hull_single"]
    )
    assert compat["summary"]["method_count"] == 1

    assert _find_most_conservative({}) == "unknown"
    assert _find_best_coverage({}) == "unknown"
    df = build_comparison_results_dataframe(cast("Any", full))
    assert df.height == 2


def test_comparison_cli_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    runner = CliRunner()

    result_info = runner.invoke(COMP_APP, ["info"])
    assert result_info.exit_code == 0
    assert "Available ODD Comparison Methods" in result_info.output

    missing = runner.invoke(COMP_APP, ["evaluate", str(tmp_path / "missing.csv")])
    assert missing.exit_code != 0
    assert "Dataset file not found" in missing.output

    fake_results = {
        "dataset": "d.csv",
        "reference_points": 4,
        "test_points": 8,
        "comparison_methods": {
            "knn": {
                "method": "knn",
                "coverage_ratio": 0.5,
                "conservatism": 0.2,
                "parameters": {"k": 3},
                "decision_boundary": {
                    "type": "knn",
                    "parameters": {},
                    "coverage": {},
                    "conservatism": 0.2,
                },
            },
        },
        "summary": {
            "most_conservative": "knn (conservatism: 0.2)",
            "best_coverage": "knn (coverage: 0.5)",
            "method_count": 1,
        },
    }

    def _ok_eval(**_kwargs: object) -> dict[str, Any]:
        return fake_results

    monkeypatch.setattr(
        "autosafe.tools.comparison.cli._evaluate_comparison_methods", _ok_eval
    )

    data_path = tmp_path / "d.csv"
    data_path.write_text("x,y\n0,0\n1,1\n", encoding="utf-8")

    ok = runner.invoke(COMP_APP, ["evaluate", str(data_path), "--verbose"])
    assert ok.exit_code == 0
    assert "ODD Comparison Result Summary" in ok.output
    assert "Parameters" in ok.output

    quick = runner.invoke(COMP_APP, ["quick", str(data_path)])
    assert quick.exit_code == 0

    # Error path in evaluate handler.
    def _failing_eval(**_kwargs: object) -> dict[str, Any]:
        raise ValueError("bad")

    monkeypatch.setattr(
        "autosafe.tools.comparison.cli._evaluate_comparison_methods", _failing_eval
    )
    err = runner.invoke(COMP_APP, ["evaluate", str(data_path)])
    assert err.exit_code != 0
    assert "Evaluation error" in err.output

    # Direct coverage of summary printer.
    _display_comparison_summary(cast("Any", fake_results), verbose=False)


def test_cluster_remaining_knn_branches():
    knn = KNNMonitor(k=2)
    assert knn.decision_boundary["type"] == "knn"
    assert knn._estimate_coverage() == {}
    assert knn.compute_conservatism_metric() == pytest.approx(0.5)

    knn.trained = True
    knn.gamma = np.float64(1.0)
    knn.consensus_radius = np.float64(1e-12)
    assert knn.compute_conservatism_metric() == pytest.approx(1.0)

    knn.consensus_radius = np.float64(1.0)
    knn.gamma = np.float64(2.0)
    assert knn.compute_conservatism_metric() == pytest.approx(0.0)

    knn.tree = None
    with pytest.raises(RuntimeError, match="KDTree not initialized"):
        knn(np.array([0.0, 0.0]))

    with pytest.raises(RuntimeError, match="Method not fitted"):
        KNNMonitor().evaluate_batch(_test_points())


def test_cluster_remaining_kmeans_branches(monkeypatch: pytest.MonkeyPatch):
    ref = _reference_points()

    knn_mid = KNNMonitor(k=1)
    knn_mid.trained = True
    knn_mid.consensus_radius = np.float64(2.0)
    knn_mid.gamma = np.float64(1.0)
    assert knn_mid.compute_conservatism_metric() == pytest.approx(0.5)

    km = KMeansBoundaries()
    assert km.method_type == "kmeans"
    assert km.decision_boundary["type"] == "kmeans"
    assert km._estimate_coverage() == {}
    assert km._calculate_conservatism() == pytest.approx(0.5)
    km.silhouette = np.float64(0.4)
    # Cover branch without cluster_sizes attribute.
    assert 0.1 <= km._calculate_conservatism() <= 1.0
    with pytest.raises(RuntimeError, match="not fitted"):
        km(np.array([0.0, 0.0]))

    with pytest.warns(UserWarning, match="Only"):
        KMeansBoundaries(n_clusters=4, min_cluster_size=10).fit(ref)

    km_exc = KMeansBoundaries(n_clusters=1, min_cluster_size=1)
    km_exc.labels_ = np.zeros(ref.shape[1], dtype=int)

    def _raise_hull(_pts: np.ndarray) -> None:
        raise ValueError("hull")

    monkeypatch.setattr("autosafe.odd.comparison.cluster.ConvexHull", _raise_hull)
    km_exc._create_cluster_convex_hulls(ref)
    assert km_exc.hulls == [None]

    bad_hull = type(
        "BadHull",
        (),
        {"equations": property(lambda _self: (_ for _ in ()).throw(ValueError("bad")))},
    )()
    km_bad = KMeansBoundaries()
    km_bad.trained = True
    km_bad.hulls = [cast("Any", bad_hull)]
    assert km_bad(np.array([0.0, 0.0])) is False
    degenerate_hull = cast(
        "Any",
        type("DegenerateHull", (), {"points": np.array([[0.0, 0.0]])})(),
    )
    assert (
        KMeansBoundaries._point_in_hull(np.array([0.0, 0.0]), degenerate_hull) is False
    )

    raw_km = KMeansBoundaries.__new__(KMeansBoundaries)
    assert raw_km.get_cluster_info() == {}
    assert KMeansBoundaries()._calculate_centroid_distances().shape == (0, 0)

    km_info = KMeansBoundaries(n_clusters=2, min_cluster_size=1).fit(ref)
    info = km_info.get_cluster_info()
    assert "centroids" in info
    assert "cluster_sizes" in info


def test_cluster_remaining_clustered_hulls_and_dbscan(monkeypatch: pytest.MonkeyPatch):
    ref = _reference_points()

    ch = ClusteredConvexHulls()
    assert ch.method_type == "clustered_hulls"
    assert ch.decision_boundary["type"] == "clustered_hulls"

    def _raise_hull(_pts: np.ndarray) -> None:
        raise ValueError("hull")

    monkeypatch.setattr("autosafe.odd.comparison.cluster.ConvexHull", _raise_hull)
    ch_exc = ClusteredConvexHulls(n_clusters=1)
    ch_exc.fit(ref)
    assert ch_exc.hulls == [None]

    # Restore actual hull behavior for positive membership path.
    monkeypatch.undo()
    ch_true = ClusteredConvexHulls(n_clusters=1).fit(ref)
    assert ch_true(np.array([0.2, 0.2])) is True

    db = DBSCANCluster(eps=0.01, min_samples=10).fit(ref)
    assert db.method_type == "dbscan"
    assert db.decision_boundary["type"] == "dbscan"
    assert db(np.array([0.0, 0.0])) is False
    assert db.evaluate_batch(_test_points()).shape == (4,)


def test_density_remaining_core_branches(monkeypatch: pytest.MonkeyPatch):
    ref = _reference_points()

    m = SuperlevelSetMonitor(gamma=np.float64(0.2), bandwidth="scott")
    m.trained = False
    assert m.method_type == "density"
    assert m._estimate_coverage() == {}
    with pytest.raises(RuntimeError, match="not fitted"):
        m.pdf(_test_points())

    m.trained = True
    m.kde = None
    with pytest.raises(RuntimeError, match="KDE not initialized"):
        m.pdf(_test_points())
    assert m.suggest_reasonable_gamma() == pytest.approx(0.2)

    m.ref_points = ref
    m.kde = cast("Any", object())
    monkeypatch.setattr(m, "pdf", lambda *_a, **_k: np.array([]))
    assert m.suggest_reasonable_gamma() == pytest.approx(0.2)

    m2 = SuperlevelSetMonitor()
    m2.trained = False
    assert m2.validate_hypersurface()["valid"] is False
    m2.ref_points = None
    m2.bandwidth = None
    assert m2.bandwidth_for_deflection() == "scott"


def test_density_remaining_fit_and_conservatism_branches(
    monkeypatch: pytest.MonkeyPatch,
):
    ref = _reference_points()

    m3 = SuperlevelSetMonitor(gamma=np.float64(0.5), bandwidth=cast("Any", 0.3))
    m3.fit(ref)
    assert m3.kde is not None

    # Cover fit branch where gamma == 0 and suggest_reasonable_gamma is called.
    m4 = SuperlevelSetMonitor(gamma=np.float64(0.0), bandwidth="scott")
    m4.trained = True
    m4.fit(ref)
    assert m4.gamma is not None
    m5 = SuperlevelSetMonitor()
    m5.trained = False
    assert m5._calculate_conservatism() == pytest.approx(0.5)

    m5.trained = True
    m5.kde = cast("Any", object())
    m5.ref_points = None
    assert m5._calculate_conservatism() == pytest.approx(0.5)

    m5.ref_points = ref
    monkeypatch.setattr(m5, "pdf", lambda *_a, **_k: np.array([]))
    assert m5._calculate_conservatism() == pytest.approx(0.5)
    monkeypatch.setattr(m5, "pdf", lambda *_a, **_k: np.array([1e-12, 1e-15]))
    assert m5._calculate_conservatism() == pytest.approx(0.5)


def test_density_remaining_call_and_clustered_branches(monkeypatch: pytest.MonkeyPatch):
    ref = _reference_points()

    m6 = SuperlevelSetMonitor()
    m6.trained = False
    with pytest.raises(RuntimeError, match="not fitted"):
        m6(np.array([0.0, 0.0]))

    m6.trained = True
    m6.gamma = np.float64(0.1)
    monkeypatch.setattr(m6, "pdf", lambda *_a, **_k: np.float64(0.2))
    assert m6(np.array([0.0, 0.0])) is True

    m7 = SuperlevelSetMonitor()
    m7.trained = False
    with pytest.raises(RuntimeError, match="Method not fitted"):
        m7.evaluate_batch(_test_points())

    m8 = SuperlevelSetMonitor(gamma=np.float64(0.1))
    m8.trained = True
    m8.kde = cast("Any", object())
    m8.ref_points = ref
    monkeypatch.setattr(m8, "pdf", lambda *_a, **_k: np.full(ref.shape[1], 0.2))
    assert m8.decision_boundary["conservatism"] is not None

    cs = ClusteredSuperlevelSetMonitor(n_clusters=1, min_cluster_size=1)
    assert cs.method_type == "density_clustered"
    assert cs.decision_boundary["type"] == "density_clustered"
    cs.fit(ref)
    assert len(cs.monitors) >= 1
    assert isinstance(cs(np.array([0.1, 0.1])), bool)
    assert cs.evaluate_batch(_test_points()).shape == (4,)


def test_core_default_dispatch_all_methods(monkeypatch: pytest.MonkeyPatch):
    ref = _reference_points()
    grid = _test_points()

    monkeypatch.setattr(
        "autosafe.tools.comparison.core.setup_evaluation_framework",
        lambda _dataset_path: {"reference_points": ref},
    )

    def _grid_fn(_ref_points: np.ndarray, resolution: int = 50) -> np.ndarray:
        _ = resolution
        return grid

    monkeypatch.setattr(
        "autosafe.tools.comparison.core.create_comparison_test_grid", _grid_fn
    )

    seen: list[str] = []

    def _fake_eval(
        _ref_points: np.ndarray,
        _test_points: np.ndarray,
        method: str,
        **kwargs: object,
    ) -> dict[str, Any]:
        _ = kwargs
        seen.append(method)
        return {
            "method": method,
            "coverage_ratio": 0.5,
            "conservatism": 0.1,
            "parameters": {},
            "decision_boundary": {
                "type": method,
                "parameters": {},
                "coverage": {},
                "conservatism": 0.1,
            },
        }

    monkeypatch.setattr(
        "autosafe.tools.comparison.core.evaluate_dataset_with_comparison_methods",
        _fake_eval,
    )

    results = evaluate_comparison_methods(
        dataset_path=cast("Any", "dummy.csv"),
        methods=None,
        export_path=None,
    )

    assert results["summary"]["method_count"] == 7
    assert seen == [
        "hull_single",
        "knn",
        "kmeans",
        "density_single",
        "hull_clustered",
        "density_clustered",
        "dbscan_cluster",
    ]


def test_knn_evaluate_batch_tree_none():
    knn = KNNMonitor(k=2)
    knn.trained = True
    knn.tree = None
    with pytest.raises(RuntimeError, match="KDTree not initialized"):
        knn.evaluate_batch(_test_points())


def test_kmeans_empty_cluster_and_evaluate_batch():
    ref = _reference_points()

    # Empty-cluster path in _create_cluster_convex_hulls:
    # force all points into cluster 0 so cluster 1 has 0 points.
    km = KMeansBoundaries(n_clusters=2, min_cluster_size=1)
    km.labels_ = np.zeros(ref.shape[1], dtype=int)
    km._create_cluster_convex_hulls(ref)
    assert km.hulls[1] is None
    assert km._cluster_balls[1] is None

    # evaluate_batch not-fitted guard.
    km_unfitted = KMeansBoundaries()
    with pytest.raises(RuntimeError, match="KMeansBoundaries not fitted yet"):
        km_unfitted.evaluate_batch(_test_points())

    # evaluate_batch success path (hull exists).
    km_fitted = KMeansBoundaries(n_clusters=2, min_cluster_size=1).fit(ref)
    result = km_fitted.evaluate_batch(_test_points())
    assert result.shape == (_test_points().shape[1],)

    # evaluate_batch exception path: hull raises on batch equations access.
    bad_hull = type(
        "BadHull",
        (),
        {"equations": property(lambda _self: (_ for _ in ()).throw(ValueError("bad")))},
    )()
    km_bad = KMeansBoundaries()
    km_bad.trained = True
    km_bad.hulls = [cast("Any", bad_hull)]
    km_bad._cluster_balls = [None]
    km_bad.evaluate_batch(_test_points())


def test_kmeans_call_point_outside_all_hulls():
    ref = _reference_points()
    km = KMeansBoundaries(n_clusters=2, min_cluster_size=1).fit(ref)
    # A point far outside exercises the "continue" branch (hull valid, point not inside).
    result = km(np.array([100.0, 100.0]))
    assert result is False


def test_clustered_hulls_evaluate_batch_and_ball_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    ref = _reference_points()

    # evaluate_batch not-fitted guard.
    ch = ClusteredConvexHulls(n_clusters=1)
    ch.trained = False
    with pytest.raises(RuntimeError, match="Not fitted"):
        ch.evaluate_batch(_test_points())

    # Bounding-ball fallback: hull=None but ball exists.
    # Monkeypatch ConvexHull to fail so hull stays None while ball is kept.
    monkeypatch.setattr(
        "autosafe.odd.comparison.cluster.ConvexHull",
        lambda _pts: (_ for _ in ()).throw(ValueError("hull")),
    )
    ch_ball = ClusteredConvexHulls(n_clusters=1).fit(ref)
    assert ch_ball.hulls == [None]
    # Point clearly inside the bounding ball (centre of ref data).
    centre = ref.T.mean(axis=0)
    assert ch_ball(centre) is True

    # Exception path in both __call__ and evaluate_batch when hull.equations raises.
    monkeypatch.undo()
    bad_hull = type(
        "BadHull",
        (),
        {"equations": property(lambda _self: (_ for _ in ()).throw(ValueError("bad")))},
    )()
    ch_exc = ClusteredConvexHulls.__new__(ClusteredConvexHulls)
    ch_exc.trained = True
    ch_exc.n_clusters = 1
    ch_exc.hulls = [cast("Any", bad_hull)]
    ch_exc._cluster_balls = [None]
    # __call__ exception path (lines 837-838): hull raises -> except -> fall through
    assert ch_exc(np.array([0.0, 0.0])) is False
    # evaluate_batch exception path (lines 880-885): same hull in batch context
    ch_exc.evaluate_batch(_test_points())

    # Outside-hull path: hull is valid but point is far outside -> continue -> return False.
    ch_fitted = ClusteredConvexHulls(n_clusters=1).fit(ref)
    assert ch_fitted(np.array([100.0, 100.0])) is False

    # evaluate_batch fitted success path.
    batch = ch_fitted.evaluate_batch(_test_points())
    assert batch.shape == (_test_points().shape[1],)


def test_dbscan_k_less_than_1_and_evaluate_batch():
    # k < 1 path: single-point dataset forces k = min(min_samples, n_pts-1) = 0.
    ref_1pt = np.array([[0.5], [0.5]], dtype=float)
    db_auto = DBSCANCluster(eps=None, min_samples=2)
    db_auto.fit(ref_1pt)
    assert db_auto.eps == pytest.approx(0.5)

    # evaluate_batch not-fitted guard.
    db = DBSCANCluster()
    with pytest.raises(RuntimeError, match="Not fitted"):
        db.evaluate_batch(_test_points())

    # evaluate_batch with core tree (needs actual core points).
    ref = _reference_points()
    db_core = DBSCANCluster(eps=0.5, min_samples=2).fit(ref)
    if db_core._core_tree is not None:
        result = db_core.evaluate_batch(_test_points())
        assert result.shape == (_test_points().shape[1],)


def test_clustered_hulls_empty_cluster(monkeypatch: pytest.MonkeyPatch):
    ref = _reference_points()

    # Force all points into cluster 0 so cluster 1 has 0 points by mocking KMeans.
    class _FakeKMeans:
        def __init__(self, **_kwargs: object) -> None:
            pass

        @staticmethod
        def fit_predict(data: np.ndarray) -> np.ndarray:
            return np.zeros(data.shape[0], dtype=int)

    monkeypatch.setattr("autosafe.odd.comparison.cluster.KMeans", _FakeKMeans)
    ch = ClusteredConvexHulls(n_clusters=2).fit(ref)
    assert ch.hulls[1] is None
    assert ch._cluster_balls[1] is None
