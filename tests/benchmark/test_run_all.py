# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Tests for the ``run_all`` benchmark driver.

Resume, failure-handling, and state-file-shape tests use small fake
"experiments" (monkeypatched onto ``run_all._EXPERIMENTS``) so they run in
milliseconds; one integration test exercises the real twelve scripts in
``--quick`` mode end to end, proving the wiring itself (imports, argument
names, result directory names) is correct.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import typer

from experiments.benchmark import run_all


def _fake_experiment(
    calls: list, name: str, *, fail: bool = False
) -> Callable[..., None]:
    def _main(*, quick: bool, seed: int, outdir: Path) -> None:
        calls.append((name, quick, seed, outdir))
        if fail:
            raise ValueError(f"{name} exploded")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "marker.txt").write_text("ok", encoding="utf-8")

    return _main


def test_run_completes_and_persists_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean run completes every experiment and writes a matching state file."""
    calls: list = []
    fake = [
        ("alpha", _fake_experiment(calls, "alpha")),
        ("beta", _fake_experiment(calls, "beta")),
    ]
    monkeypatch.setattr(run_all, "_EXPERIMENTS", fake)

    state_path = tmp_path / "run_all.state.local.json"
    status = run_all.run(
        outdir=tmp_path / "results", state_path=state_path, seed=7, quick=True
    )

    assert set(status) == {"alpha", "beta"}
    assert status["alpha"].startswith("completed")
    assert status["beta"].startswith("completed")
    assert [c[0] for c in calls] == ["alpha", "beta"]
    assert all(c[1] is True and c[2] == 7 for c in calls)  # quick + seed forwarded
    assert (tmp_path / "results" / "alpha" / "marker.txt").exists()

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["completed"] == ["alpha", "beta"]
    assert state["failed"] == {}
    assert state["updated_at"]


def test_resume_skips_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second run with the same state file skips everything already done."""
    calls: list = []
    fake = [("alpha", _fake_experiment(calls, "alpha"))]
    monkeypatch.setattr(run_all, "_EXPERIMENTS", fake)
    state_path = tmp_path / "state.json"

    run_all.run(outdir=tmp_path / "results", state_path=state_path, quick=True)
    assert len(calls) == 1

    status = run_all.run(outdir=tmp_path / "results", state_path=state_path, quick=True)
    assert status["alpha"] == "skipped (already completed)"
    assert len(calls) == 1  # not called again


def test_failure_is_recorded_and_does_not_abort_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing experiment is recorded, but later experiments still run."""
    calls: list = []
    fake = [
        ("bad", _fake_experiment(calls, "bad", fail=True)),
        ("good", _fake_experiment(calls, "good")),
    ]
    monkeypatch.setattr(run_all, "_EXPERIMENTS", fake)
    state_path = tmp_path / "state.json"

    status = run_all.run(outdir=tmp_path / "results", state_path=state_path, quick=True)
    assert status["bad"] == "failed: bad exploded"
    assert status["good"].startswith("completed")
    assert [c[0] for c in calls] == ["bad", "good"]  # good still ran

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["completed"] == ["good"]
    assert state["failed"] == {"bad": "bad exploded"}


def test_stop_on_error_halts_remaining_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stop_on_error`` halts the batch instead of continuing past a failure."""
    calls: list = []
    fake = [
        ("bad", _fake_experiment(calls, "bad", fail=True)),
        ("never", _fake_experiment(calls, "never")),
    ]
    monkeypatch.setattr(run_all, "_EXPERIMENTS", fake)
    state_path = tmp_path / "state.json"

    status = run_all.run(
        outdir=tmp_path / "results",
        state_path=state_path,
        quick=True,
        stop_on_error=True,
    )
    assert "bad" in status
    assert "never" not in status
    assert [c[0] for c in calls] == ["bad"]


def test_main_cli_exits_nonzero_on_failure_with_stop_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI wrapper raises ``typer.Exit`` when a run failed under stop_on_error."""
    calls: list = []
    fake = [("bad", _fake_experiment(calls, "bad", fail=True))]
    monkeypatch.setattr(run_all, "_EXPERIMENTS", fake)
    monkeypatch.setattr(run_all, "EXPERIMENT_NAMES", ["bad"])

    with pytest.raises(typer.Exit):
        run_all.main(
            outdir=tmp_path / "results",
            state_path=tmp_path / "state.json",
            quick=True,
            stop_on_error=True,
        )


def test_run_against_real_quick_experiments(tmp_path: Path) -> None:
    """The real driver wiring: all twelve scripts complete under ``--quick``."""
    status = run_all.run(
        outdir=tmp_path / "results",
        state_path=tmp_path / "run_all.state.local.json",
        seed=0,
        quick=True,
    )
    assert set(status) == set(run_all.EXPERIMENT_NAMES)
    for name, message in status.items():
        assert message.startswith("completed"), (name, message)
        result_dir = tmp_path / "results" / name
        assert result_dir.is_dir()
        assert any(result_dir.iterdir()), f"{name} wrote nothing to {result_dir}"
