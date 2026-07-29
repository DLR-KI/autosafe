# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
# pylint: disable=unused-import
"""Main CLI module."""

import autosafe.tools.monte_carlo.sample  # imported to register command # ruff:ignore[unused-import]
from autosafe.cli import APP


def main() -> None:
    """Main entry point for the CLI application."""
    APP()


if __name__ == "__main__":
    main()
