# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the archive-bundle packaging step.

Uses a small synthetic results tree (never the real one, which is 2.4 MB and
untracked) to exercise the filesystem walk, the PNG exclusion, the numeric
artifact count guard, and the zip step in well under a second.
"""

import zipfile
from pathlib import Path

import pytest

from experiments.benchmark import make_archive_bundle as mab


def _make_fixture_results(root: Path, *, n_numeric: int, n_png: int) -> Path:
    """A tiny results tree with the requested number of numeric/PNG files.

    Returns:
        Path: the created ``results`` directory.
    """
    results_dir = root / "results"
    for i in range(n_numeric):
        d = results_dir / f"experiment_{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
    for i in range(n_png):
        d = results_dir / f"experiment_{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "plot.png").write_bytes(b"\x89PNG")
    return results_dir


@pytest.fixture
def support_files(tmp_path: Path) -> dict:
    run_spec = tmp_path / "run_all_spec.yaml"
    run_spec.write_text("experiments: []\n", encoding="utf-8")
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text("# lock\n", encoding="utf-8")
    license_src = tmp_path / "CC-BY-4.0.txt"
    license_src.write_text("Creative Commons Attribution 4.0\n", encoding="utf-8")
    return {"run_spec": run_spec, "uv_lock": uv_lock, "license": license_src}


def test_iter_files_walks_filesystem_not_git(tmp_path: Path) -> None:
    """The walk finds files that exist on disk, whether or not git tracks them."""
    results_dir = _make_fixture_results(tmp_path, n_numeric=3, n_png=0)
    found = mab._iter_files(results_dir)
    assert len(found) == 3
    assert all(p.suffix == ".json" for p in found)


def test_iter_files_missing_dir_raises(tmp_path: Path) -> None:
    """A nonexistent results directory is a hard error, not an empty bundle."""
    with pytest.raises(FileNotFoundError):
        mab._iter_files(tmp_path / "does-not-exist")


def test_stage_bundle_excludes_pngs_and_matches_numeric_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, support_files: dict
) -> None:
    """Staging drops PNGs and requires exactly the expected numeric count."""
    monkeypatch.setattr(mab, "EXPECTED_NUMERIC_ARTIFACT_COUNT", 5)
    results_dir = _make_fixture_results(tmp_path, n_numeric=5, n_png=2)

    copied = mab.stage_bundle(
        results_dir=results_dir,
        run_spec_path=support_files["run_spec"],
        uv_lock_path=support_files["uv_lock"],
        license_src=support_files["license"],
        staging_dir=tmp_path / "staged",
    )

    assert not any(p.suffix == ".png" for p in copied)
    numeric = [p for p in copied if p.suffix == ".json"]
    assert len(numeric) == 5
    assert (tmp_path / "staged" / "run_all_spec.yaml").exists()
    assert (tmp_path / "staged" / "uv.lock").exists()
    assert (tmp_path / "staged" / "LICENSE").exists()
    readme = (tmp_path / "staged" / "README.md").read_text(encoding="utf-8")
    assert "revision:" in readme


def test_stage_bundle_rejects_empty_results_tree(
    tmp_path: Path, support_files: dict
) -> None:
    """An empty results tree is refused rather than silently bundled as empty."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with pytest.raises(ValueError, match="no files found"):
        mab.stage_bundle(
            results_dir=results_dir,
            run_spec_path=support_files["run_spec"],
            uv_lock_path=support_files["uv_lock"],
            license_src=support_files["license"],
            staging_dir=tmp_path / "staged",
        )


def test_stage_bundle_rejects_wrong_numeric_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, support_files: dict
) -> None:
    """A numeric-artifact count mismatch is a finding, not something to ship."""
    monkeypatch.setattr(mab, "EXPECTED_NUMERIC_ARTIFACT_COUNT", 999)
    results_dir = _make_fixture_results(tmp_path, n_numeric=3, n_png=0)
    with pytest.raises(ValueError, match="expected 999"):
        mab.stage_bundle(
            results_dir=results_dir,
            run_spec_path=support_files["run_spec"],
            uv_lock_path=support_files["uv_lock"],
            license_src=support_files["license"],
            staging_dir=tmp_path / "staged",
        )


def test_make_zip_archive_contains_staged_files(tmp_path: Path) -> None:
    """The zip step mirrors the staged tree, with paths relative to it."""
    staging_dir = tmp_path / "staged"
    (staging_dir / "results" / "sample_experiment").mkdir(parents=True)
    (staging_dir / "results" / "sample_experiment" / "config.json").write_text("{}")
    (staging_dir / "README.md").write_text("hello")

    archive_path = mab.make_zip_archive(staging_dir, tmp_path / "out.zip")
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
    assert names == {"results/sample_experiment/config.json", "README.md"}


def test_real_results_tree_has_expected_numeric_count() -> None:
    """The real (unmodified) results tree matches the plan's audited count.

    Regression guard: if this ever fails, either the results tree drifted
    (a re-run happened, which the plan forbids) or a script's output
    filenames changed without updating the expectation here.
    """
    files = mab._iter_files(mab.DEFAULT_RESULTS_DIR)
    numeric = [p for p in files if p.suffix in mab.NUMERIC_EXTENSIONS]
    assert len(numeric) == mab.EXPECTED_NUMERIC_ARTIFACT_COUNT == 75
    pngs = [p for p in files if p.suffix == ".png"]
    assert len(pngs) == 16
