# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (many small, tersely-documented
# pivot/filter helpers -- one per paper data file); scoped per file
# since this is a reproduction-package driver, not library code.
# ruff: file-ignore[docstring-missing-returns]
r"""Rebuild the paper's benchmark ``.dat`` files from the results tree.

The manuscript's data files under ``paper/graphics/data/benchmark/``
were originally typed by hand from the results tree (see
``PLAN_CAMERA_READY_RELEASE.md`` Sec. 2 for the recovered provenance).
This script reproduces every one of those eighteen files
programmatically from ``experiments/benchmark/results/`` so that future
runs of the benchmark suite can regenerate the paper's data without hand
transcription.

Each of the eighteen files is a deterministic transform (verbatim copy,
column pivot, join, filter, or a documented row-drop / label-shortening)
of one or more per-experiment artifacts; the recipe for each is recorded
next to its builder function below.

Usage::

    uv run python -m experiments.benchmark.export_paper_data \
        --outdir /tmp/out
    uv run python -m experiments.benchmark.export_paper_data \
        --outdir /tmp/out --verify paper/graphics/data/benchmark

The script refuses to write anywhere under ``paper/`` (see
:func:`_assert_not_under_paper`) -- that directory is the manuscript's
own copy and is compared against, never overwritten. ``--verify``
compares numerically, cell by cell, ignoring formatting differences such
as the inconsistent zero-padding of the hand-typed originals.
"""

from __future__ import annotations

import dataclasses
import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from collections.abc import Callable

Row = dict[str, str]
Table = tuple[list[str], list[Row]]

REPO_ROOT = Path(__file__).resolve().parents[2]
_PAPER_ROOT = (REPO_ROOT / "paper").resolve()
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "benchmark" / "results"
DEFAULT_OUTDIR = Path(tempfile.gettempdir()) / "autosafe-paper-data-export"

#: Label shortenings applied when carrying a dataset/method/arm name
#: from the results tree into a paper data file (Plan Sec. 2.2, "Label
#: shortening"). Anything not listed here is carried through unchanged
#: (e.g. ``poly5d``).
LABEL_MAP = {
    "linear2d": "linear",
    "annulus2d": "annulus",
    "banana2d": "banana",
    "twoblobs2d": "blobs",
    "autosafe_no_ood": "autosafe",
    "autosafe_with_ood": "autosafe_ood",
    "convex_hull": "hull",
}

#: The eighteen canonical paper data files, in the order they appear in
#: PLAN_CAMERA_READY_RELEASE.md Sec. 2.3.
EXPECTED_FILENAMES = [
    "halo.dat",
    "truncation_error.dat",
    "halo_wide.dat",
    "sensitivity_linear_n1000.dat",
    "sensitivity_annulus_n1000.dat",
    "sensitivity_poly5d_n1000.dat",
    "baselines.dat",
    "per_n.dat",
    "mcm_per_n.dat",
    "mcm_anchor_count_mean.dat",
    "subset.dat",
    "density.dat",
    "dedup.dat",
    "covariance_n100.dat",
    "deployed_covariance.dat",
    "truncation_latency.dat",
    "hole.dat",
    "conformal_n1000.dat",
]


def _shorten(label: str) -> str:
    """Apply the paper's label shortening, or pass through unchanged."""
    return LABEL_MAP.get(label, label)


def _assert_not_under_paper(path: Path) -> None:
    """Refuse to write anywhere under ``paper/`` -- it is read-only.

    Args:
        path (Path): Candidate output directory.

    Raises:
        ValueError: If ``path`` is ``paper/`` or lives underneath it.
    """
    resolved = Path(path).resolve()
    if resolved == _PAPER_ROOT or _PAPER_ROOT in resolved.parents:
        raise ValueError(f"refusing to write under paper/ (read-only): {resolved}")


def _read_dat(path: Path) -> Table:
    """Read a whitespace-separated ``.dat`` file with a header row.

    Args:
        path (Path): File to read.

    Returns:
        Table: ``(header, rows)`` with each row a
            ``{column: raw_token}`` mapping; tokens are kept as strings,
            unparsed.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file has no header/rows.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty results file: {path}")
    header = lines[0].split()
    rows = [dict(zip(header, ln.split(), strict=True)) for ln in lines[1:]]
    return header, rows


def _write_table(path: Path, header: list[str], rows: list[Row]) -> None:
    """Write a whitespace-separated table (header row, then the rows).

    Args:
        path (Path): Output path (parent directories are created).
        header (list[str]): Column names, in order.
        rows (list[Row]): Row mappings; each must contain every header
            key.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(header)]
    lines.extend(" ".join(str(row[col]) for col in header) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_seen_order(rows: list[Row], key: str) -> list[str]:
    """Distinct values of ``rows[*][key]``, in first-seen order."""
    seen: list[str] = []
    index: set[str] = set()
    for row in rows:
        v = row[key]
        if v not in index:
            index.add(v)
            seen.append(v)
    return seen


def _halo(results_dir: Path) -> Table:
    """``halo.dat`` <- ``halo_vs_anchor_count/halo.dat``, verbatim."""
    return _read_dat(results_dir / "halo_vs_anchor_count" / "halo.dat")


def _truncation_error(results_dir: Path) -> Table:
    """``truncation_error.dat``: verbatim copy.

    Source: ``kernel_truncation/truncation_error.dat``.
    """
    return _read_dat(results_dir / "kernel_truncation" / "truncation_error.dat")


def _halo_wide(results_dir: Path) -> Table:
    """``halo_wide.dat``: pivot ``arm`` to columns, keyed by ``n``.

    ``halo_max`` becomes ``{arm}_halo``; ``precision`` becomes
    ``{arm}_precision``.
    """
    _, rows = _read_dat(results_dir / "halo_vs_anchor_count" / "halo.dat")
    arms = _first_seen_order(rows, "arm")
    order_n = _first_seen_order(rows, "n")
    by_n: dict[str, dict[str, Row]] = {n: {} for n in order_n}
    for row in rows:
        by_n[row["n"]][row["arm"]] = row
    header = ["n", *[f"{a}_halo" for a in arms], *[f"{a}_precision" for a in arms]]
    out_rows = []
    for n in order_n:
        out: Row = {"n": n}
        for a in arms:
            out[f"{a}_halo"] = by_n[n][a]["halo_max"]
        for a in arms:
            out[f"{a}_precision"] = by_n[n][a]["precision"]
        out_rows.append(out)
    return header, out_rows


#: source dataset name (as it appears in the results tree) for each of
#: the three sensitivity output files. ``poly5d`` is already short.
_SENSITIVITY_SOURCES = {
    "sensitivity_linear_n1000.dat": "linear2d",
    "sensitivity_annulus_n1000.dat": "annulus2d",
    "sensitivity_poly5d_n1000.dat": "poly5d",
}


def _sensitivity(results_dir: Path, output_filename: str) -> Table:
    """``sensitivity_<ds>_n1000.dat``: keep rows with ``gamma <= 2``."""
    source_dataset = _SENSITIVITY_SOURCES[output_filename]
    sensitivity_dir = results_dir / "parameter_sensitivity"
    path = sensitivity_dir / f"sensitivity_{source_dataset}_n1000.dat"
    header, rows = _read_dat(path)
    filtered = [r for r in rows if float(r["gamma"]) <= 2.0 + 1e-9]
    return header, filtered


#: dataset row order and label for baselines.dat; poly5d is dropped
#: (Plan Sec. 2.2: it lives in the appendix table instead).
_BASELINE_DATASETS = ["linear2d", "annulus2d", "twoblobs2d", "banana2d"]
_BASELINE_METHODS = ["autosafe", "kde", "gmm", "convex_hull"]
_BASELINE_N = ["10", "1000"]


def _baselines(results_dir: Path) -> Table:
    """``baselines.dat``: pivot method x N to columns.

    ``poly5d`` is dropped.
    """
    header = ["dataset"]
    for n in _BASELINE_N:
        header.extend(f"{_shorten(m)}{n}" for m in _BASELINE_METHODS)
    out_rows = []
    for ds in _BASELINE_DATASETS:
        _, rows = _read_dat(results_dir / "baseline_comparison" / f"aupr_vs_n_{ds}.dat")
        lookup = {(r["n"], r["method"]): r["aupr"] for r in rows}
        out: Row = {"dataset": _shorten(ds)}
        for n in _BASELINE_N:
            for m in _BASELINE_METHODS:
                out[f"{_shorten(m)}{n}"] = lookup[n, m]
        out_rows.append(out)
    return header, out_rows


def _per_n(results_dir: Path) -> Table:
    """``per_n.dat``: join two per-N tables on ``n``.

    The sources are ``aupr_vs_n.dat`` and ``curve_r2.dat``.
    """
    sweep_dir = results_dir / "anchor_count_sweep"
    _, aupr_rows = _read_dat(sweep_dir / "aupr_vs_n.dat")
    _, r2_rows = _read_dat(sweep_dir / "curve_r2.dat")
    r2_by_n = {r["n"]: r for r in r2_rows}
    header = ["n", "aupr", "aupr_std", "r2_precision", "r2_recall"]
    out_rows = []
    for r in aupr_rows:
        r2 = r2_by_n[r["n"]]
        out_rows.append({
            "n": r["n"],
            "aupr": r["aupr_mean"],
            "aupr_std": r["aupr_std"],
            "r2_precision": r2["r2_precision"],
            "r2_recall": r2["r2_recall"],
        })
    return header, out_rows


def _mcm_per_n(results_dir: Path) -> Table:
    """``mcm_per_n.dat``: concatenate per-N PR sweeps, prepending ``n``.

    Keeps columns 1, 3-6 of each ``pr_n<N>.dat`` (``zeta``,
    ``precision_mean``, ``precision_std``, ``recall_mean``,
    ``recall_std``; ``log_threshold`` is dropped), in the N order of
    ``aupr_vs_n.dat``.
    """
    sweep_dir = results_dir / "anchor_count_sweep"
    _, aupr_rows = _read_dat(sweep_dir / "aupr_vs_n.dat")
    n_order = [r["n"] for r in aupr_rows]
    header = [
        "n",
        "zeta",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
    ]
    out_rows: list[Row] = []
    for n in n_order:
        n_int = int(float(n))
        _, pr_rows = _read_dat(sweep_dir / f"pr_n{n_int}.dat")
        out_rows.extend(
            {
                "n": n,
                "zeta": r["zeta"],
                "precision_mean": r["precision_mean"],
                "precision_std": r["precision_std"],
                "recall_mean": r["recall_mean"],
                "recall_std": r["recall_std"],
            }
            for r in pr_rows
        )
    return header, out_rows


def _mcm_anchor_count_mean(results_dir: Path) -> Table:
    """``mcm_anchor_count_mean.dat``: precision/recall mean over N.

    The only genuinely *computed* (not merely rearranged) value among
    the eighteen; formatted at 9 significant figures, matching the
    recovered ``awk`` recipe (Plan Sec. 2.1, 2026-08-26T09:42:26Z).
    """
    _, mcm_rows = _mcm_per_n(results_dir)
    order_zeta = _first_seen_order(mcm_rows, "zeta")
    sums: dict[str, list[float]] = {z: [0.0, 0.0] for z in order_zeta}
    counts: dict[str, int] = dict.fromkeys(order_zeta, 0)
    for r in mcm_rows:
        z = r["zeta"]
        sums[z][0] += float(r["precision_mean"])
        sums[z][1] += float(r["recall_mean"])
        counts[z] += 1
    header = ["zeta", "precision_mean", "recall_mean"]
    out_rows = [
        {
            "zeta": z,
            "precision_mean": f"{sums[z][0] / counts[z]:.9g}",
            "recall_mean": f"{sums[z][1] / counts[z]:.9g}",
        }
        for z in order_zeta
    ]
    return header, out_rows


def _subset(results_dir: Path) -> Table:
    """``subset.dat``: pivot ``dataset`` to columns, keyed by ``n``.

    ``iou_pairwise_{mean,min,max}`` become ``{dataset}_{mean,min,max}``.
    """
    subset_path = results_dir / "anchor_subset_stability" / "subset_stability.dat"
    _, rows = _read_dat(subset_path)
    datasets = _first_seen_order(rows, "dataset")
    order_n = _first_seen_order(rows, "n")
    by_n: dict[str, dict[str, Row]] = {n: {} for n in order_n}
    for row in rows:
        by_n[row["n"]][row["dataset"]] = row
    header = ["n"]
    for ds in datasets:
        short = _shorten(ds)
        header.extend((f"{short}_mean", f"{short}_min", f"{short}_max"))
    out_rows = []
    for n in order_n:
        out: Row = {"n": n}
        for ds in datasets:
            short = _shorten(ds)
            r = by_n[n][ds]
            out[f"{short}_mean"] = r["iou_pairwise_mean"]
            out[f"{short}_min"] = r["iou_pairwise_min"]
            out[f"{short}_max"] = r["iou_pairwise_max"]
        out_rows.append(out)
    return header, out_rows


def _density(results_dir: Path) -> Table:
    """``density.dat``: pivot dataset to columns, keyed by ``n``.

    Values are ``iou_true``.
    """
    density_path = results_dir / "duplicate_sensitivity" / "density_stability.dat"
    _, rows = _read_dat(density_path)
    datasets = _first_seen_order(rows, "dataset")
    order_n = _first_seen_order(rows, "n")
    by_n: dict[str, dict[str, str]] = {n: {} for n in order_n}
    for row in rows:
        by_n[row["n"]][row["dataset"]] = row["iou_true"]
    header = ["n", *[_shorten(ds) for ds in datasets]]
    out_rows = [
        {"n": n, **{_shorten(ds): by_n[n][ds] for ds in datasets}} for n in order_n
    ]
    return header, out_rows


def _dedup(results_dir: Path) -> Table:
    """``dedup.dat``: arm=calibrated only; pivot dataset by ``tol``."""
    _, rows = _read_dat(results_dir / "duplicate_sensitivity" / "dedup.dat")
    rows = [r for r in rows if r["arm"] == "calibrated"]
    datasets = _first_seen_order(rows, "dataset")
    order_tol = _first_seen_order(rows, "tol")
    by_tol: dict[str, dict[str, str]] = {t: {} for t in order_tol}
    for row in rows:
        by_tol[row["tol"]][row["dataset"]] = row["iou_vs_base"]
    header = ["tol", *[_shorten(ds) for ds in datasets]]
    out_rows = [
        {"tol": t, **{_shorten(ds): by_tol[t][ds] for ds in datasets}}
        for t in order_tol
    ]
    return header, out_rows


def _covariance_n100(results_dir: Path) -> Table:
    """``covariance_n100.dat``: filter n=100; pivot ``arm`` by ``rho``.

    Values are ``iou``.
    """
    _, rows = _read_dat(results_dir / "covariance_structure" / "anisotropy.dat")
    rows = [r for r in rows if r["n"] == "100"]
    arms = _first_seen_order(rows, "arm")
    order_rho = _first_seen_order(rows, "rho")
    by_rho: dict[str, dict[str, str]] = {rho: {} for rho in order_rho}
    for row in rows:
        by_rho[row["rho"]][row["arm"]] = row["iou"]
    header = ["rho", *arms]
    out_rows = [{"rho": rho, **{a: by_rho[rho][a] for a in arms}} for rho in order_rho]
    return header, out_rows


def _deployed_covariance(results_dir: Path) -> Table:
    """``deployed_covariance.dat``: pivot ``rho`` by ``n``.

    Only rows with arm=deployed are kept.
    """
    _, rows = _read_dat(results_dir / "covariance_structure" / "anisotropy.dat")
    rows = [r for r in rows if r["arm"] == "deployed"]
    rhos = _first_seen_order(rows, "rho")
    order_n = _first_seen_order(rows, "n")
    by_n: dict[str, dict[str, str]] = {n: {} for n in order_n}
    for row in rows:
        by_n[row["n"]][row["rho"]] = row["iou"]

    def _col(rho: str) -> str:
        return f"rho{int(float(rho))}" if float(rho).is_integer() else f"rho{rho}"

    header = ["n", *[_col(rho) for rho in rhos]]
    out_rows = [{"n": n, **{_col(rho): by_n[n][rho] for rho in rhos}} for n in order_n]
    return header, out_rows


#: the two truncation levels the paper reports (Plan Sec. 2.3).
_TRUNCATION_K_COLS = ("128", "512")


def _truncation_latency(results_dir: Path) -> Table:
    """``truncation_latency.dat``: exact vs. K=128/512 query latency.

    Columns are ``q_secs_full`` plus ``q_secs_trunc`` at K=128 and
    K=512.
    """
    latency_path = results_dir / "kernel_truncation" / "truncation_latency.dat"
    _, rows = _read_dat(latency_path)
    rows = [r for r in rows if r["n"] != "1000"]
    order_n = _first_seen_order(rows, "n")
    by_n: dict[str, dict[str, Row]] = {n: {} for n in order_n}
    for row in rows:
        by_n[row["n"]][row["k"]] = row
    header = ["n", "exact", "k128", "k512"]
    out_rows = []
    for n in order_n:
        entries = by_n[n]
        if not all(k in entries for k in _TRUNCATION_K_COLS):
            continue  # required truncation levels absent (e.g. --quick fixtures)
        exact = next(iter(entries.values()))["q_secs_full"]
        out_rows.append({
            "n": n,
            "exact": exact,
            "k128": entries["128"]["q_secs_trunc"],
            "k512": entries["512"]["q_secs_trunc"],
        })
    return header, out_rows


def _hole(results_dir: Path) -> Table:
    """``hole.dat``: ``fp_rate_in_hole`` only, labels shortened."""
    _, rows = _read_dat(results_dir / "held_out_vcas_hole" / "hole_fp.dat")
    header = ["method", "fp_rate"]
    out_rows = [
        {"method": _shorten(r["method"]), "fp_rate": r["fp_rate_in_hole"]} for r in rows
    ]
    return header, out_rows


def _conformal_n1000(results_dir: Path) -> Table:
    """``conformal_n1000.dat``: pivot dataset by ``eps``.

    Only rows with ``n == 1000`` are kept.
    """
    _, rows = _read_dat(results_dir / "conformal_threshold" / "conformal_coverage.dat")
    rows = [r for r in rows if r["n"] == "1000"]
    datasets = _first_seen_order(rows, "dataset")
    order_eps = _first_seen_order(rows, "eps")
    by_eps: dict[str, dict[str, str]] = {e: {} for e in order_eps}
    for row in rows:
        by_eps[row["eps"]][row["dataset"]] = row["emp_false_excl_mean"]
    header = ["eps", *[_shorten(ds) for ds in datasets]]
    out_rows = [
        {"eps": e, **{_shorten(ds): by_eps[e][ds] for ds in datasets}}
        for e in order_eps
    ]
    return header, out_rows


@dataclasses.dataclass(frozen=True)
class _Export:
    """One canonical output file and the builder that produces it."""

    filename: str
    build: Callable[[Path], Table]


_EXPORTS: list[_Export] = [
    _Export("halo.dat", _halo),
    _Export("truncation_error.dat", _truncation_error),
    _Export("halo_wide.dat", _halo_wide),
    _Export(
        "sensitivity_linear_n1000.dat",
        lambda rd: _sensitivity(rd, "sensitivity_linear_n1000.dat"),
    ),
    _Export(
        "sensitivity_annulus_n1000.dat",
        lambda rd: _sensitivity(rd, "sensitivity_annulus_n1000.dat"),
    ),
    _Export(
        "sensitivity_poly5d_n1000.dat",
        lambda rd: _sensitivity(rd, "sensitivity_poly5d_n1000.dat"),
    ),
    _Export("baselines.dat", _baselines),
    _Export("per_n.dat", _per_n),
    _Export("mcm_per_n.dat", _mcm_per_n),
    _Export("mcm_anchor_count_mean.dat", _mcm_anchor_count_mean),
    _Export("subset.dat", _subset),
    _Export("density.dat", _density),
    _Export("dedup.dat", _dedup),
    _Export("covariance_n100.dat", _covariance_n100),
    _Export("deployed_covariance.dat", _deployed_covariance),
    _Export("truncation_latency.dat", _truncation_latency),
    _Export("hole.dat", _hole),
    _Export("conformal_n1000.dat", _conformal_n1000),
]

if [e.filename for e in _EXPORTS] != EXPECTED_FILENAMES:
    # Module-load invariant: keeps EXPECTED_FILENAMES (used by
    # verify_against, independent of _EXPORTS) in lockstep with the
    # builder list above.
    raise AssertionError("_EXPORTS and EXPECTED_FILENAMES have drifted apart")


def run(*, results_dir: Path, outdir: Path) -> dict[str, str]:
    """Build and write every paper data file the results tree supports.

    Args:
        results_dir (Path): Root of the per-experiment results tree
            (``experiments/benchmark/results`` by default).
        outdir (Path): Destination directory; refused if under
            ``paper/``.

    Returns:
        dict[str, str]: ``{filename: status}``, status either
            ``"written (N rows)"`` or ``"skipped: <reason>"`` when a
            required source artifact is missing or incomplete (this
            happens under ``--quick`` fixtures, where a smaller sweep
            may not cover every combination a recipe needs; it is not an
            error).
    """
    _assert_not_under_paper(outdir)
    results_dir = Path(results_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for spec in _EXPORTS:
        try:
            header, rows = spec.build(results_dir)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            status[spec.filename] = f"skipped: {exc!r}"
            continue
        _write_table(outdir / spec.filename, header, rows)
        status[spec.filename] = f"written ({len(rows)} rows)"
    return status


def _numeric_cell_equal(
    a: str, b: str, *, rel_tol: float = 1e-6, abs_tol: float = 1e-9
) -> bool:
    """Compare two table cells as numbers if possible, else as text."""
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return a == b
    if math.isnan(fa) and math.isnan(fb):
        return True
    return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)


@dataclasses.dataclass
class VerifyReport:
    """Result of comparing generated files against a reference dir.

    Attributes:
        matches (list[str]): Filenames that compared equal, numerically.
        differs (dict[str, str]): Filenames that differ, mapped to a
            short description of the first difference found.
        missing (list[str]): Filenames absent from either side.
    """

    matches: list[str]
    differs: dict[str, str]
    missing: list[str]

    @property
    def ok(self) -> bool:
        """True iff all 18 canonical files matched exactly."""
        return not self.differs and not self.missing


def verify_against(generated_dir: Path, reference_dir: Path) -> VerifyReport:
    """Numerically compare the 18 canonical files, cell by cell.

    Args:
        generated_dir (Path): Directory just written by :func:`run`.
        reference_dir (Path): Reference directory to compare against
            (read-only; never written).

    Returns:
        VerifyReport: matches / differs / missing, over
            :data:`EXPECTED_FILENAMES`.
    """
    matches: list[str] = []
    differs: dict[str, str] = {}
    missing: list[str] = []
    for name in EXPECTED_FILENAMES:
        gen_path = Path(generated_dir) / name
        ref_path = Path(reference_dir) / name
        if not gen_path.exists() or not ref_path.exists():
            missing.append(name)
            continue
        gen_header, gen_rows = _read_dat(gen_path)
        ref_header, ref_rows = _read_dat(ref_path)
        if gen_header != ref_header:
            differs[name] = f"columns differ: {gen_header} vs {ref_header}"
            continue
        if len(gen_rows) != len(ref_rows):
            differs[name] = f"row count differs: {len(gen_rows)} vs {len(ref_rows)}"
            continue
        mismatch = None
        for i, (gr, rr) in enumerate(zip(gen_rows, ref_rows, strict=True)):
            for col in gen_header:
                if not _numeric_cell_equal(gr[col], rr[col]):
                    mismatch = f"row {i} column {col!r}: {gr[col]!r} vs {rr[col]!r}"
                    break
            if mismatch:
                break
        if mismatch:
            differs[name] = mismatch
        else:
            matches.append(name)
    return VerifyReport(matches=matches, differs=differs, missing=missing)


def main(
    outdir: Path = DEFAULT_OUTDIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    verify: str = "",
) -> None:
    """Rebuild the paper's benchmark ``.dat`` files from results.

    Args:
        outdir (Path): Destination directory (created if needed);
            refused if it resolves under ``paper/``.
        results_dir (Path): Root of the per-experiment results tree.
        verify (str): If non-empty, a directory (normally
            ``paper/graphics/data/benchmark``) to compare the written
            files against, numerically. Exits non-zero on any
            difference.

    Raises:
        typer.Exit: With a non-zero code if ``--verify`` found any
            difference or missing file.
    """
    status = run(results_dir=results_dir, outdir=Path(outdir))
    for name in EXPECTED_FILENAMES:
        typer.echo(f"{name}: {status[name]}")
    n_written = sum(1 for s in status.values() if s.startswith("written"))
    typer.echo(f"{n_written}/{len(EXPECTED_FILENAMES)} paper data files -> {outdir}")

    if verify:
        report = verify_against(outdir, Path(verify))
        typer.echo(
            f"verify vs {verify}: {len(report.matches)} match, "
            f"{len(report.differs)} differ, {len(report.missing)} missing"
        )
        for name, reason in report.differs.items():
            typer.echo(f"  DIFFER {name}: {reason}")
        for name in report.missing:
            typer.echo(f"  MISSING {name}")
        if not report.ok:
            raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
