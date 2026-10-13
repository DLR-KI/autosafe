# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Exceptions raised by the benchmark scripts.

Errors that also occur in the library reuse :mod:`autosafe.exceptions`;
the classes here cover conditions specific to the benchmark harness.
Each class builds its own message, so call sites pass values only.
"""

from collections.abc import Iterable
from pathlib import Path


class UnknownSyntheticODDError(KeyError):
    """Raised when a synthetic ODD name is not registered."""

    def __init__(self, name: str, available: Iterable[str]) -> None:
        super().__init__(f"unknown synthetic ODD {name!r}; have {sorted(available)}")


class EmptyResultsFileError(ValueError):
    """Raised when a benchmark results file has no header or rows."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"empty results file: {path}")


class ExportRegistryDriftError(AssertionError):
    """Raised at import when the export list and filenames diverge.

    ``EXPECTED_FILENAMES`` is used by ``main`` and the tests
    independently of ``_EXPORTS``, so the two must stay in lockstep.
    """

    def __init__(self) -> None:
        super().__init__("_EXPORTS and EXPECTED_FILENAMES have drifted apart")


__all__ = [
    "EmptyResultsFileError",
    "ExportRegistryDriftError",
    "UnknownSyntheticODDError",
]
