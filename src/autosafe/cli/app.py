# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Main CLI app."""

import typer

import autosafe.cli.experiments  # register experiments commands # noqa: F401
import autosafe.tools.evaluate.cli  # register evaluate commands # noqa: F401
import autosafe.tools.monte_carlo.evaluate  # register mc evaluate # noqa: F401
import autosafe.tools.monte_carlo.sample  # register mc sample # noqa: F401
from autosafe.cli.experiments import get_app as get_experiments_app
from autosafe.tools.comparison import COMP_APP
from autosafe.tools.evaluate import EVAL_APP
from autosafe.tools.monte_carlo import MC_APP

APP = typer.Typer()


# Register subcommands
APP.add_typer(MC_APP, name="montecarlo")
APP.add_typer(EVAL_APP, name="evaluate")
APP.add_typer(COMP_APP, name="comparison")
APP.add_typer(get_experiments_app(), name="experiments")
