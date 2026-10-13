# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Command-line parameter errors shared by the autoSAFE CLIs.

These subclass :class:`typer.BadParameter`, so Typer reports them as
usage errors, and they live outside :mod:`autosafe.exceptions` to keep
that module free of third-party imports.
"""

from pathlib import Path

import typer

_PATH_KINDS = {"dataset": "Dataset file", "directory": "Directory", "spec": "Spec file"}


class PathNotFoundParameterError(typer.BadParameter):
    """Raised when a path given on the command line does not exist.

    ``kind`` is one of ``"dataset"``, ``"directory"`` or ``"spec"``.
    """

    def __init__(self, kind: str, path: Path) -> None:
        super().__init__(f"{_PATH_KINDS[kind]} not found: {path}")


__all__ = ["PathNotFoundParameterError"]
