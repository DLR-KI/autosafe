# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# Research-driver relaxations (typer boolean flags, a broad except
# around arbitrary experiment mains -- a failing experiment must not
# abort the batch); scoped per file since this is a reproduction-package
# driver, not library code.
# ruff: file-ignore[blind-except, boolean-type-hint-positional-argument, boolean-default-value-positional-argument, too-many-arguments, too-many-positional-arguments]
"""Run the twelve benchmark experiment scripts in a fixed order.

This is the single entry point for reproducing every result in
``experiments/benchmark/results/``: one process, one fixed order, one
``--seed``, and a local, resumable state file with the same
completed/failed/updated_at contract as
:class:`autosafe.tools.experiments.core.ExperimentManager` (the manager
used for the real-data/mc-sample spec in
``experiments/run_all_spec.yaml``) -- that manager does not cover these
twelve scripts, which keep their own documented runner.

Usage::

    uv run python -m experiments.benchmark.run_all --quick
    uv run python -m experiments.benchmark.run_all --seed 42

Never run the full-size sweep from an interactive/CI session -- each
script takes up to ~30 minutes at full size (see
``experiments/benchmark/README.md`` for per-experiment runtimes); use
``--quick`` to exercise the whole path in seconds.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

import typer

from experiments.benchmark import (
    run_anchor_count_sweep,
    run_anchor_subset_stability,
    run_baseline_comparison,
    run_conformal_threshold,
    run_covariance_structure,
    run_duplicate_sensitivity,
    run_halo_vs_anchor_count,
    run_held_out_vcas_hole,
    run_kernel_truncation,
    run_ood_adjustment,
    run_parameter_sensitivity,
    run_permutation_stability,
)

MainFn = Callable[..., None]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "experiments" / "benchmark" / "results"
DEFAULT_STATE_PATH = (
    REPO_ROOT / "experiments" / "benchmark" / "run_all.state.local.json"
)

#: The twelve scripts, in the fixed order they run in; each name is both
#: the state-file id and the results subdirectory (matching each
#: script's own ``DEFAULT_OUTDIR``). Order follows the paper's
#: presentation, not the alphabet, so a partial/resumed run finishes the
#: earlier claims first.
_EXPERIMENTS: list[tuple[str, MainFn]] = [
    ("baseline_comparison", run_baseline_comparison.main),
    ("ood_adjustment", run_ood_adjustment.main),
    ("parameter_sensitivity", run_parameter_sensitivity.main),
    ("halo_vs_anchor_count", run_halo_vs_anchor_count.main),
    ("anchor_count_sweep", run_anchor_count_sweep.main),
    ("kernel_truncation", run_kernel_truncation.main),
    ("permutation_stability", run_permutation_stability.main),
    ("held_out_vcas_hole", run_held_out_vcas_hole.main),
    ("conformal_threshold", run_conformal_threshold.main),
    ("duplicate_sensitivity", run_duplicate_sensitivity.main),
    ("anchor_subset_stability", run_anchor_subset_stability.main),
    ("covariance_structure", run_covariance_structure.main),
]

EXPERIMENT_NAMES = [name for name, _ in _EXPERIMENTS]


def _load_state(state_path: Path) -> dict:
    """Load the resumable batch state, or a fresh one if absent.

    Args:
        state_path (Path): State file path.

    Returns:
        dict: State with keys ``completed`` (list), ``failed`` (dict)
            and ``updated_at``.
    """
    if not state_path.exists():
        return {"completed": [], "failed": {}, "updated_at": None}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _save_state(state_path: Path, completed: set, failed: dict) -> None:
    """Persist batch state per the ``ExperimentManager`` contract.

    Args:
        state_path (Path): State file path (created if needed).
        completed (set): Completed experiment names.
        failed (dict): Failed experiment names mapped to their error
            message.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed": sorted(completed),
        "failed": failed,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(
    *,
    outdir: Path,
    state_path: Path,
    seed: int = 42,
    quick: bool = False,
    resume: bool = True,
    stop_on_error: bool = False,
) -> dict[str, str]:
    """Run the twelve benchmark scripts in order, with resume support.

    Args:
        outdir (Path): Root results directory; each experiment writes to
            ``outdir / <name>``, matching the layout in
            ``experiments/benchmark/results/``.
        state_path (Path): Local, resumable state file (completed/failed
            ids plus an update timestamp).
        seed (int): Random seed forwarded to every script.
        quick (bool): Forwarded to every script's own ``--quick`` smoke
            configuration.
        resume (bool): Skip experiments already marked completed in
            ``state_path``.
        stop_on_error (bool): Stop at the first failing experiment
            instead of continuing with the rest.

    Returns:
        dict[str, str]: ``{name: status}`` for every one of the twelve
            experiments, status one of
            ``"skipped (already completed)"``, ``"completed (N.Ns)"``,
            or ``"failed: <message>"``.
    """
    outdir = Path(outdir)
    state_path = Path(state_path)
    state = _load_state(state_path)
    completed: set = set(state.get("completed", []))
    failed: dict = dict(state.get("failed", {}))

    status: dict[str, str] = {}
    for name, run_main in _EXPERIMENTS:
        if resume and name in completed:
            status[name] = "skipped (already completed)"
            typer.echo(f"[skip] {name} (already completed)")
            continue

        typer.echo(f"[run ] {name} ...")
        start = time.perf_counter()
        try:
            run_main(quick=quick, seed=seed, outdir=outdir / name)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            failed[name] = str(exc)
            _save_state(state_path, completed, failed)
            status[name] = f"failed: {exc}"
            typer.echo(f"[fail] {name} after {elapsed:.1f}s: {exc}")
            if stop_on_error:
                break
            continue

        elapsed = time.perf_counter() - start
        completed.add(name)
        failed.pop(name, None)
        _save_state(state_path, completed, failed)
        status[name] = f"completed ({elapsed:.1f}s)"
        typer.echo(f"[done] {name} in {elapsed:.1f}s")

    return status


def main(
    seed: int = 42,
    outdir: Path = DEFAULT_OUTDIR,
    quick: bool = False,
    resume: bool = True,
    stop_on_error: bool = False,
    state_path: Path = DEFAULT_STATE_PATH,
) -> None:
    """Run all twelve benchmark experiments in their fixed order.

    Args:
        seed (int): Random seed forwarded to every script.
        outdir (Path): Root results directory (each script writes to
            ``outdir / <name>``).
        quick (bool): Forward ``--quick`` to every script; finishes in
            seconds and is the only mode this driver should ever be run
            in from an interactive or CI session.
        resume (bool): Skip experiments already marked completed in the
            state file (the default; matches ``ExperimentManager``).
        stop_on_error (bool): Stop at the first failing experiment.
        state_path (Path): Local, resumable state file.

    Raises:
        Exit: With a non-zero code if any experiment failed and
            ``stop_on_error`` is set.
    """
    status = run(
        outdir=Path(outdir),
        state_path=Path(state_path),
        seed=seed,
        quick=quick,
        resume=resume,
        stop_on_error=stop_on_error,
    )
    n_completed = sum(1 for s in status.values() if s.startswith("completed"))
    n_failed = sum(1 for s in status.values() if s.startswith("failed"))
    typer.echo(
        f"{n_completed}/{len(EXPERIMENT_NAMES)} completed, {n_failed} failed "
        f"-> {outdir} (state: {state_path})"
    )
    if n_failed and stop_on_error:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
