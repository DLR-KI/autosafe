# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT

import pytest
from typer.testing import CliRunner

from autosafe import ROOT_FOLDER
from autosafe.cli import APP
from autosafe.tools.monte_carlo import (  # noqa: F401
    sample,  # imported to register the command
)

runner = CliRunner()

monte_carlo_testdata = [
    (None, None, None, None, None, None, None, None, None, None, None),
    (2, "box", 1.0, 3.0, 4, 100, "RBF", {"sigma": "eye"}, "output.json", None, None),
    (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        ROOT_FOLDER / "tests" / "assets" / "sampling_configs" / "pytest.json",
        None,
    ),
    (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "tests/assets/sampling_configs/pytest.json",
        None,
    ),
    (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        ROOT_FOLDER / "tests" / "assets" / "sampling_configs",
    ),
    (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "tests/assets/sampling_configs",
    ),
]


def test_cli_app_monte_carlo_help():
    """Test the CLI application help command."""
    result = runner.invoke(APP, ["montecarlo", "sample", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Monte Carlo" in result.output
    assert "--dim" in result.output
    assert "--odd-type" in result.output
    assert "--odd-limits" in result.output
    assert "--box-limits" in result.output
    assert "--odd-anchors" in result.output
    assert "--samples" in result.output
    assert "--kernel-type" in result.output
    assert "--kernel-params" in result.output
    assert "--filename" in result.output
    assert "--config-file" in result.output
    assert "--config-file-folder" in result.output


@pytest.mark.parametrize(
    (
        "dim",
        "odd_type",
        "odd_limits",
        "box_limits",
        "odd_anchors",
        "samples",
        "kernel_type",
        "kernel_params",
        "filename",
        "config_file",
        "config_file_folder",
    ),
    monte_carlo_testdata,
)
def test_cli_app_monte_carlo_runs_sampling(  # noqa: C901, PLR0913, PLR0917
    dim: int,
    odd_type: str,
    odd_limits: float,
    box_limits: float,
    odd_anchors: int,
    samples: int,
    kernel_type: str,
    kernel_params: dict,
    filename: str,
    config_file: str,
    config_file_folder: str,
):
    """Test the CLI application runs the sampling command."""
    cmd = []
    if dim is not None:
        cmd.extend(["--dim", str(dim)])
    if odd_type is not None:
        cmd.extend(["--odd-type", odd_type])
    if odd_limits is not None:
        cmd.extend(["--odd-limits", str(odd_limits)])
    if box_limits is not None:
        cmd.extend(["--box-limits", str(box_limits)])
    if odd_anchors is not None:
        cmd.extend(["--odd-anchors", str(odd_anchors)])
    if samples is not None:
        cmd.extend(["--samples", str(samples)])
    if kernel_type is not None:
        cmd.extend(["--kernel-type", kernel_type])
    if kernel_params is not None:
        cmd.extend(["--kernel-params", str(kernel_params).replace("'", '"')])
    if filename is not None:
        cmd.extend(["--filename", filename])
    if config_file is not None:
        cmd.extend(["--config-file", config_file])
    if config_file_folder is not None:
        cmd.extend(["--config-file-folder", config_file_folder])

    result = runner.invoke(
        APP,
        ["montecarlo", "sample", *cmd],
    )
    assert result.exit_code == 0
    assert "Finding anchor points within ODD" in result.output
    if odd_anchors is not None:
        assert f"{odd_anchors}/{odd_anchors}" in result.output
    assert "Finding closest samples" in result.output
    assert "Updating kernels" in result.output
    assert "Computing affinity scores" in result.output
