# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (a subprocess call to read the git
# revision); scoped per file since this is a reproduction-package
# driver, not library code.
# ruff: file-ignore[suspicious-subprocess-import, start-process-with-partial-path, too-many-arguments, too-many-positional-arguments]
r"""Package the benchmark results tree into an archive bundle.

The bundle is what gets uploaded to Zenodo/HuggingFace (Plan Sec. 3.1)
-- results are never committed to git, so this script is the only thing
that turns the local, untracked ``experiments/benchmark/results/`` tree
into a shareable artifact. It contains:

-   the results tree, minus ``*.png`` (plotting is TikZ/PGFPlots now;
    the matplotlib debug plots are local-only, see
    ``experiments/benchmark/README.md``);
-   the merged run spec (``experiments/run_all_spec.yaml``);
-   ``uv.lock`` (the exact dependency set the results were produced
    with);
-   a ``LICENSE`` file (CC-BY-4.0, matching the data licence --
    ``data/REUSE.toml``);
-   a ``README`` naming the exact repository revision the results came
    from.

Critical: the results tree is untracked, so this script reads the
filesystem directly (:func:`_iter_files`) and must never use
``git ls-files`` -- that would silently produce an empty bundle. Both
the raw file count and the expected count of 75 numeric artifacts
(``.dat``/``.csv``/``.json``/``.tex``; see Plan Sec. 3.1) are asserted
before an archive is written.

Usage::

    uv run python -m experiments.benchmark.make_archive_bundle \
        --outdir /tmp/bundle

Uploading the resulting archive to Zenodo/HuggingFace is out of scope
here -- that publication step is Johann's, after review (Plan Sec.
3.10).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "benchmark" / "results"
DEFAULT_RUN_SPEC = REPO_ROOT / "experiments" / "run_all_spec.yaml"
DEFAULT_UV_LOCK = REPO_ROOT / "uv.lock"
DEFAULT_LICENSE_SRC = REPO_ROOT / "LICENSES" / "CC-BY-4.0.txt"
DEFAULT_OUTDIR = Path(tempfile.gettempdir()) / "autosafe-benchmark-bundle"

#: `.dat`/`.csv`/`.json`/`.tex` artifacts measured in the results tree
#: at plan time (Plan Sec. 3.1: 50 + 11 + 13 + 1). A mismatch means the
#: results tree has drifted from what the paper was audited against -- a
#: finding to report, not something this script should silently accept.
EXPECTED_NUMERIC_ARTIFACT_COUNT = 75
NUMERIC_EXTENSIONS = frozenset({".dat", ".csv", ".json", ".tex"})

_README_TEMPLATE = """\
# autoSAFE benchmark results bundle

This archive contains the numeric result artifacts underlying every figure
and table in the paper's benchmark suite, packaged from the repository at:

    revision: {rev}

## Contents

- `results/` -- the per-experiment results tree
    (`experiments/benchmark/results/` in the repository), minus `*.png` debug
    plots. All paper plotting is TikZ/PGFPlots; the PNGs are local matplotlib
    debugging aids and are not part of the reproduction package.
- `run_all_spec.yaml` -- the merged experiment-manager spec used for the
    real-data (HCAS/VCAS) evaluations.
- `uv.lock` -- the exact dependency set the results were produced with.
- `LICENSE` -- CC-BY-4.0, matching `data/REUSE.toml` for the datasets these
    results derive from.

## Reproducing a figure or table from this bundle

1. Clone the repository at the revision above and run `uv sync --frozen`.
2. Point `experiments/benchmark/export_paper_data.py` at `results/` from
    this bundle (`--outdir` some scratch directory).
3. See `REPRODUCTION.md` at the repository root for the full
    figure/table -> artifact -> command mapping, and each experiment's
    `config.json` (inside `results/<experiment>/`) for the exact seed and
    configuration used.

## Numeric artifact count

{n_numeric} numeric artifacts (`.dat`/`.csv`/`.json`/`.tex`) across
{n_files} files total in this bundle.
"""


def _git_rev(repo_root: Path = REPO_ROOT) -> str:
    """Return the current git commit hash, or 'unknown'.

    Args:
        repo_root (Path): Repository root to run ``git`` in.

    Returns:
        str: The full commit hash, or ``"unknown"`` if it cannot be
            read.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _iter_files(results_dir: Path) -> list[Path]:
    """List every file under ``results_dir`` by walking the filesystem.

    Never uses ``git ls-files``: the results tree is untracked (Plan
    Sec. 3.1/3.2), so a git-based listing would silently return nothing.

    Args:
        results_dir (Path): Root of the per-experiment results tree.

    Returns:
        list[Path]: Every file found, sorted for a deterministic bundle.

    Raises:
        FileNotFoundError: If ``results_dir`` does not exist.
    """
    if not results_dir.exists():
        raise FileNotFoundError(str(results_dir))
    return sorted(p for p in results_dir.rglob("*") if p.is_file())


def stage_bundle(
    *,
    results_dir: Path,
    run_spec_path: Path,
    uv_lock_path: Path,
    license_src: Path,
    staging_dir: Path,
) -> list[Path]:
    """Assemble the bundle contents under ``staging_dir``.

    Args:
        results_dir (Path): Root of the per-experiment results tree.
        run_spec_path (Path): The merged manager spec to include.
        uv_lock_path (Path): The dependency lockfile to include.
        license_src (Path): CC-BY-4.0 license text to copy in as
            ``LICENSE``.
        staging_dir (Path): Destination directory (created if needed).

    Returns:
        list[Path]: Every file written into the staged bundle.

    Raises:
        FileNotFoundError: If ``results_dir`` does not exist, or a
            required file (run spec, lockfile, license) is missing.
        ValueError: If the filesystem walk over ``results_dir`` finds no
            files at all, or the number of numeric artifacts
            (``.dat``/``.csv``/``.json``/``.tex``) does not match
            :data:`EXPECTED_NUMERIC_ARTIFACT_COUNT` -- either is a sign
            that something is silently wrong (an empty/partial results
            tree, or drift from what the paper was audited against), not
            something to bundle and ship anyway.
    """
    for required in (run_spec_path, uv_lock_path, license_src):
        if not required.exists():
            raise FileNotFoundError(str(required))

    all_files = _iter_files(results_dir)
    if not all_files:
        raise ValueError(
            f"no files found under {results_dir} -- refusing to build an "
            "empty bundle (are you accidentally reading from git instead "
            "of the filesystem?)"
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    results_staging = staging_dir / "results"
    copied: list[Path] = []
    for src in all_files:
        if src.suffix == ".png":
            continue
        dst = results_staging / src.relative_to(results_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)

    numeric_count = sum(1 for p in copied if p.suffix in NUMERIC_EXTENSIONS)
    if numeric_count != EXPECTED_NUMERIC_ARTIFACT_COUNT:
        raise ValueError(
            f"expected {EXPECTED_NUMERIC_ARTIFACT_COUNT} numeric artifacts "
            f"(.dat/.csv/.json/.tex) under {results_dir}, found {numeric_count}"
        )

    run_spec_dst = staging_dir / "run_all_spec.yaml"
    shutil.copy2(run_spec_path, run_spec_dst)
    uv_lock_dst = staging_dir / "uv.lock"
    shutil.copy2(uv_lock_path, uv_lock_dst)
    license_dst = staging_dir / "LICENSE"
    shutil.copy2(license_src, license_dst)
    copied.extend((run_spec_dst, uv_lock_dst, license_dst))

    readme_dst = staging_dir / "README.md"
    readme_dst.write_text(
        _README_TEMPLATE.format(
            rev=_git_rev(), n_numeric=numeric_count, n_files=len(copied) + 1
        ),
        encoding="utf-8",
    )
    copied.append(readme_dst)

    return copied


def make_zip_archive(staging_dir: Path, archive_path: Path) -> Path:
    """Zip a staged bundle directory into a single archive file.

    Args:
        staging_dir (Path): Directory produced by :func:`stage_bundle`.
        archive_path (Path): Destination ``.zip`` path (parents
            created).

    Returns:
        Path: ``archive_path``, for chaining.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(staging_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(staging_dir))
    return archive_path


def main(
    outdir: Path = DEFAULT_OUTDIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    run_spec_path: Path = DEFAULT_RUN_SPEC,
    uv_lock_path: Path = DEFAULT_UV_LOCK,
    license_src: Path = DEFAULT_LICENSE_SRC,
    archive_name: str = "",
) -> None:
    """Stage and zip the benchmark results bundle.

    Args:
        outdir (Path): Destination directory; holds both the staged
            ``bundle/`` tree and the final ``.zip``.
        results_dir (Path): Root of the per-experiment results tree.
        run_spec_path (Path): The merged manager spec to include.
        uv_lock_path (Path): The dependency lockfile to include.
        license_src (Path): CC-BY-4.0 license text to copy in as
            ``LICENSE``.
        archive_name (str): Override the archive filename; defaults to
            ``autosafe-benchmark-results-<short-revision>.zip``.
    """
    outdir = Path(outdir)
    staging_dir = outdir / "bundle"
    copied = stage_bundle(
        results_dir=Path(results_dir),
        run_spec_path=Path(run_spec_path),
        uv_lock_path=Path(uv_lock_path),
        license_src=Path(license_src),
        staging_dir=staging_dir,
    )
    rev = _git_rev()
    name = archive_name or f"autosafe-benchmark-results-{rev[:12]}.zip"
    archive_path = make_zip_archive(staging_dir, outdir / name)
    typer.echo(
        f"staged {len(copied)} files at {staging_dir}; archive -> {archive_path}"
    )


if __name__ == "__main__":
    typer.run(main)
