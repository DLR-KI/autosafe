# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR e.V.) <https://dlr.de>
#
# SPDX-License-Identifier: MIT
"""Export functions to turn data into writable formats."""

from pathlib import Path

import msgspec.json

from autosafe.tools.serializers.msgspec import encode_hook


def to_json(obj: object, file: Path | str) -> None:
    """Export to a JSON file.

    Turns an object into a JSON file using msgspec.

    Args:
        obj (object): The object to convert to JSON.
        file (Path | str): Path to the output JSON file.
    """
    Path(file).write_bytes(msgspec.json.Encoder(enc_hook=encode_hook).encode(obj))
