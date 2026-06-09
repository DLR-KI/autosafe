# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Functions to decode and encode numpy arrays."""

import pathlib
from typing import cast


def decode_path(obj: object) -> pathlib.Path:
    """Convert a string back to a pathlib.Path.

    Args:
        obj (object): The string representation of the path.

    Returns:
        pathlib.Path: The reconstructed pathlib.Path.
    """
    obj = cast("str", obj)
    return pathlib.Path(obj)


def encode_path(path: pathlib.Path) -> str:
    """Convert a pathlib.Path to a string.

    Args:
        path (pathlib.Path): The path to convert.

    Returns:
        str: The string representation of the path.
    """
    return str(path)
