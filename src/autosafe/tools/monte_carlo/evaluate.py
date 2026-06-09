# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Main entry point for the monte carlo sampling."""

from typing import TYPE_CHECKING, Annotated

import typer

from autosafe import ROOT_FOLDER
from autosafe.tools.evaluate.core import process_files
from autosafe.tools.evaluate.workflows import evaluate_monte_carlo_results
from autosafe.tools.monte_carlo import MC_APP

if TYPE_CHECKING:
    from pathlib import Path


@MC_APP.command(
    name="evaluate",
    help="""
    Legacy entry-point to evaluate Monte Carlo sampling results.
    """,
)
def evaluate(
    file: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Path(s) to the JSON file(s) containing the "
                "sampling results to evaluate."
            ),
        ),
    ] = None,
) -> None:
    """Evaluate the results of Monte Carlo sampling.

    This function processes the results stored in JSON files, computes
    relevant metrics, and provides insights into the sampling
    performance.

    Args:
        file (list[str] | None): Path(s) to the JSON file(s)
            containing the sampling results.

    Raises:
        FileNotFoundError: If no configuration files are found in the
            specified folder.
    """
    if file is None:
        raise FileNotFoundError("No file(s) provided for evaluation.")
    files: list[Path] = []
    for f in file:
        added_files = process_files(f)
        files.extend(added_files)

    evaluate_monte_carlo_results(files, threshold_mode="linear")

    # Plotting is now handled within process_data function


def main() -> None:
    """Main file for testing purposes."""
    evaluate(
        file=[
            str(
                ROOT_FOLDER
                / "data"
                / "vcas_state_variables-results-global-RBF-{}-2026-01-16T01:35:38.266737+00:00.json"  # noqa: E501
            )
        ]
    )


if __name__ == "__main__":
    main()
